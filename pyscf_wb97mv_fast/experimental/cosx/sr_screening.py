"""FROZEN (2026-09-30): gates failed, see benchmarks/experimental/cosx/FINDINGS.md section 9; not imported by mainline.

P2a: SR-specific screening with stock kernels.

Three independent pieces:

BPath
    Exact operator decomposition switch. dft.rks.get_veff builds
    K = hyb*K_full + (alpha-hyb)*K_LR via an mf-level get_jk (full) followed by
    an mf-level get_k(omega=+w). Because K_full = K_LR + K_SR exactly, a get_k
    call in the LR slot can return (stashed K_full - K_SR): get_veff then forms

        hyb*K_full + (alpha-hyb)*(K_full - K_SR)
          = alpha*K_full - (alpha-hyb)*K_SR,

    which is the B path, with the SR build (cheap, erfc-screenable) replacing
    the LR build. Tolerances and caches per kernel are untouched.

PerKernelBudget
    Give K_full a stricter DM-screening budget than the attenuated K (its
    coefficient is alpha=1, not hyb=0.15). Implemented at the mf level: the
    main SGX object only ever sees full-range calls and the RSH copies only
    attenuated ones, so each keeps a constant _itol and its cached screening
    bounds survive across cycles.

ErfcSRBounds
    EXPERIMENTAL, KNOWN-UNSOUND PREMISE -- keep enable=False. It multiplies
    SGXData._mbar_bi of SR objects by an erfc envelope of the (block, shell)
    distance. But for an SR copy, _mbar_bi is already computed with the
    attenuated int1e_grids kernel (the bounds are built inside
    with_range_coulomb(-omega)), so the erfc decay is counted twice and the
    result is not an upper bound by construction. Evidence that stock bounds
    already see erfc: on chain30, sample_pos cuts SR tasks from 23.1M to 12.7M
    while full/LR barely change (FINDINGS.md 4.1). The remaining stock-vs-
    oracle SR gap has to come from elsewhere (position-independent pair bound
    _mbar_ij/_rbar_ij, block-max products in the DM screening, vtol).
    tests/test_sr_screening.py keeps the correctness gate as an expected
    failure.
"""
import numpy as np
from pyscf import gto
from pyscf.sgx import sgx as _sgx
from pyscf.sgx import sgx_jk

from pyscf_wb97mv_fast.core import sgx_patch
from pyscf_wb97mv_fast.core.hooks import HookSet


class BPath:
    """K = alpha*K_full + (hyb-alpha)*K_SR via mf-level call interception."""

    def __init__(self, mf):
        self.mf = mf
        self.n_sr_builds = 0
        self._stash = {}
        self._hooks = HookSet()
        self._attached = False

    def attach(self):
        if self._attached:
            return self
        self._attached = True

        def jk_factory(orig):
            def get_jk(mol, dm, hermi=1, *a, **k):
                vj, vk = orig(mol, dm, hermi, *a, **k)
                # stash only the full-range build (omega absent). Attenuated
                # calls (e.g. the SR build inside BPath's own get_k) must not
                # pollute the stash: a LR slot consuming them would be wrong.
                if vk is not None and k.get('omega') is None:
                    # COPY: dft.rks.get_veff scales the returned vk in place
                    # (vk *= hyb) before it calls get_k for the LR slot.
                    self._stash[np.asarray(dm).shape] = vk.copy()
                return vj, vk
            return get_jk

        def k_factory(orig):
            def get_k(mol, dm, hermi=1, omega=None, *a, **k):
                if omega is not None and omega > 0:
                    k_full = self._stash.pop(np.asarray(dm).shape, None)
                    if k_full is not None:
                        k_sr = orig(mol, dm, hermi, omega=-abs(omega), *a, **k)
                        self.n_sr_builds += 1
                        return k_full - k_sr
                return orig(mol, dm, hermi, omega=omega, *a, **k)
            return get_k

        self._hooks.wrap(self.mf, 'get_jk', jk_factory)
        self._hooks.wrap(self.mf, 'get_k', k_factory)
        return self

    def remove(self):
        if self._attached:
            self._hooks.restore_all()
            self._stash.clear()
            self._attached = False

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.remove()


class PerKernelBudget:
    """Give K_full a stricter DM-screening budget than the attenuated K.

    Implemented on mf.with_df.get_jk (SGX.get_jk), which every K build --
    full-range via get_jk and LR/SR via get_k -- funnels through. The main SGX
    object only ever receives full-range calls and the RSH copies only
    attenuated ones, so each keeps a constant _itol and its cached screening
    bounds survive across cycles. This class OWNS direct_scf_tol: every call
    is overridden by the per-kernel rule (explicit per-call tolerances are
    ignored while it is attached).
    """

    def __init__(self, mf, tol_full=1e-11, tol_attenuated=1e-8):
        self.mf = mf
        self.tol_full = tol_full
        self.tol_attenuated = tol_attenuated
        self._hooks = HookSet()
        self._attached = False

    def attach(self):
        if self._attached:
            return self
        # RSH copies are shallow copies of mf.with_df and inherit this
        # instance-level wrapper (bound to the parent). Only the patched RSH
        # branch never calls rsh_df.get_jk, so the patch is a hard requirement.
        if _sgx.SGX.get_jk is not sgx_patch.get_jk:
            raise RuntimeError('PerKernelBudget requires sgx_patch.apply() first')
        self._attached = True
        full, att = self.tol_full, self.tol_attenuated

        def factory(orig):
            def get_jk(dm, hermi=1, vhfopt=None, with_j=True, with_k=True,
                       direct_scf_tol=None, omega=None):
                tol = att if omega not in (None, 0) else full
                return orig(dm, hermi, vhfopt, with_j, with_k, tol, omega)
            return get_jk
        self._hooks.wrap(self.mf.with_df, 'get_jk', factory)
        return self

    def remove(self):
        if self._attached:
            self._hooks.restore_all()
            self._attached = False

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.remove()


class ErfcSRBounds:
    """Scale _mbar_bi of SR builds by erfc(omega * max(d - slack, 0)), where d
    is the distance from the grid-block center to the shell center. d=0 gives
    factor 1 (no change); far blocks are cut hard. NOT a valid bound: the SR
    _mbar_bi already contains the erfc decay (see module docstring). Kept only
    for experiments; enable=False by default.
    """

    def __init__(self, slack_bohr=4.0, floor=1e-3, enable=False):
        self.slack_bohr = slack_bohr
        self.floor = floor
        self.enable = enable
        self._hooks = HookSet()
        self._attached = False

    def attach(self):
        if self._attached:
            return self
        self._attached = True
        self_ = self

        def factory(orig):
            def get_pos_dpt_ints(sgxdata, *a, **k):
                orig(sgxdata, *a, **k)
                omega = float(sgxdata.mol._env[gto.PTR_RANGE_OMEGA])
                if not (self_.enable and omega < 0):
                    return
                mb = sgxdata._mbar_bi
                coords = sgxdata.grids.coords
                nblk = mb.shape[0]
                blk = sgx_jk.SGX_BLKSIZE
                blk_center = np.array([coords[b * blk:min((b + 1) * blk, len(coords))].mean(axis=0)
                                       for b in range(nblk)])
                shell_center = np.array([sgxdata.mol.bas_coord(i)
                                         for i in range(mb.shape[1])])
                d = np.linalg.norm(blk_center[:, None, :] - shell_center[None], axis=-1)
                env = _erfc(-omega * np.maximum(d - self_.slack_bohr, 0.0))
                sgxdata._mbar_bi = mb * np.maximum(env, self_.floor)
            return get_pos_dpt_ints
        self._hooks.wrap(sgx_jk.SGXData, '_get_pos_dpt_ints', factory)
        return self

    def remove(self):
        if self._attached:
            self._hooks.restore_all()
            self._attached = False

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.remove()


def _erfc(x):
    from scipy.special import erfc
    return erfc(x)
