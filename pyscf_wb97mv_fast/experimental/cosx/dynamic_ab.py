"""FROZEN (2026-09-30): gates failed, see benchmarks/experimental/cosx/FINDINGS.md section 9; not imported by mainline.

P3: switch the A/B decomposition by SCF stage.

Early SCF (loose residual budget) runs the B path (full + SR: the erfc kernel
is the screenable one); once the orbital gradient norm r_n drops below
``switch_at`` the mf is switched back to the A path (full + LR, stock algebra)
for the tight endgame. Both decompositions evaluate the SAME total K operator
(K_full = K_LR + K_SR exactly), so the incremental vhf_last stays valid across
the switch; on the switch the screening-state rebuild is forced once via
set_sgx_tolerance to start the endgame from clean bounds.

Usage:
    from pyscf_wb97mv_fast.experimental.cosx.dynamic_ab import DynamicAB
    with DynamicAB(mf).attach():
        mf.kernel()
"""
import numpy as np

from pyscf_wb97mv_fast.core import sgx_patch
from pyscf_wb97mv_fast.core.hooks import HookSet, track_residual
from pyscf_wb97mv_fast.experimental.cosx.sr_screening import BPath
from pyscf_wb97mv_fast.experimental.cosx.tolerance import set_sgx_tolerance


class DynamicAB:
    """switch_at=None (default) derives the switch point from the SCF
    convergence target: 10 * sqrt(mf.conv_tol), i.e. 10 x PySCF's default
    conv_tol_grad, compared with the same |g| (envs['norm_gorb']) -- the stage
    flips while there are still a few cycles of endgame left."""

    def __init__(self, mf, switch_at=None, endgame_etol='auto'):
        self.mf = mf
        self.switch_at = switch_at
        self.endgame_etol = endgame_etol
        self.rn = np.inf
        self.switch_cycle = None       # first veff call that ran on the A path
        self.n_veff_calls = 0
        self._bpath = BPath(mf)
        self._hooks = HookSet()
        self._attached = False

    def _set_rn(self, rn):
        self.rn = rn

    def attach(self):
        if self._attached:
            return self
        self._attached = True
        if self.switch_at is None:
            self.switch_at = float(np.sqrt(self.mf.conv_tol)) * 10
        sgx_patch.apply()
        self._bpath.attach()
        mf = self.mf
        hooks = self._hooks

        track_residual(hooks, mf, self._set_rn)

        def veff_factory(orig):
            def get_veff(mol=None, dm=None, dm_last=None, vhf_last=None,
                         hermi=1, *a, **k):
                if self.switch_cycle is None and self.rn <= self.switch_at:
                    self._bpath.remove()
                    # clean screening state for the A-path endgame; K operator
                    # itself is identical, so incremental vhf_last stays valid
                    set_sgx_tolerance(mf, self.endgame_etol)
                    self.switch_cycle = self.n_veff_calls
                self.n_veff_calls += 1
                return orig(mol, dm, dm_last, vhf_last, hermi, *a, **k)
            return get_veff
        hooks.wrap(mf, 'get_veff', veff_factory)
        return self

    def remove(self):
        if self._attached:
            self._bpath.remove()
            self._hooks.restore_all()
            self._attached = False

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.remove()
