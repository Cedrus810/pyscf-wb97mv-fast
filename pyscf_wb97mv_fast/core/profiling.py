"""Per-component wall-time profiler for PySCF RKS / SGX SCF runs."""
import contextlib
import time
from collections import defaultdict

from pyscf import dft, gto
from pyscf.df import df_jk
from pyscf.scf import diis as scf_diis
from pyscf.sgx import sgx_jk

from pyscf_wb97mv_fast.core import sgx_patch

_MISSING = object()


def _omega_of(obj):
    return float(obj.mol._env[gto.PTR_RANGE_OMEGA])


def _k_key(omega):
    if omega == 0:
        return 'k_full'
    return 'k_lr' if omega > 0 else 'k_sr'


class ScfProfiler:
    """Accumulate wall time per SCF component while attached to an mf.

    Keys: coulomb, coulomb_rsh_waste (RI-J under an attenuated kernel, i.e. the
    upstream sgx.py:521 bug), k_full, k_lr, k_sr, sgx_get_jk (non-optk SGX
    path), dft_xc, vv10, diag, diis.
    """

    def __init__(self):
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self._saved = []

    def _wrap(self, owner, name, key_fn):
        orig = getattr(owner, name)
        own = vars(owner).get(name, _MISSING) if hasattr(owner, '__dict__') else _MISSING
        prof = self

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                key = key_fn(args)
                prof.timings[key] += time.perf_counter() - t0
                prof.counts[key] += 1

        setattr(owner, name, timed)
        self._saved.append((owner, name, own))

    def _unwrap_all(self):
        for owner, name, own in reversed(self._saved):
            if own is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, own)
        self._saved.clear()

    @contextlib.contextmanager
    def attach(self, mf):
        try:
            self._wrap(sgx_jk, 'get_k_only', lambda a: _k_key(_omega_of(a[0])))
            self._wrap(sgx_jk, 'get_jk', lambda a: 'sgx_get_jk')
            self._wrap(df_jk, 'get_j', lambda a: 'coulomb' if _omega_of(a[0]) == 0
                       else 'coulomb_rsh_waste')
            self._wrap(scf_diis.CDIIS, 'update', lambda a: 'diis')
            self._wrap(mf, 'eig', lambda a: 'diag')
            ni = getattr(mf, '_numint', None)
            if ni is not None:
                self._wrap(ni, 'nr_rks', lambda a: 'dft_xc')
                self._wrap(ni, 'nr_nlc_vxc', lambda a: 'vv10')
            yield self
        finally:
            self._unwrap_all()


def run_profile(mol, xc='wb97m-v', patched=True, conv_tol=1e-9,
                sgx_tol_energy='auto', dm0=None):
    """Run one RKS/COSX(pjs=True) SCF and return per-component timings.

    Always leaves sgx_patch reverted on return.
    """
    if patched:
        sgx_patch.apply()
    else:
        sgx_patch.revert()
    try:
        mf = dft.RKS(mol, xc=xc).COSX(pjs=True)
        mf.conv_tol = conv_tol
        mf.with_df.sgx_tol_energy = sgx_tol_energy
        prof = ScfProfiler()
        with prof.attach(mf):
            t0 = time.perf_counter()
            e = mf.kernel(dm0=dm0)
            prof.timings['scf_total'] = time.perf_counter() - t0
        return dict(e_tot=float(e), converged=bool(mf.converged), cycles=int(mf.cycles),
                    nao=int(mol.nao), patched=patched, timings=dict(prof.timings),
                    counts=dict(prof.counts), dm=mf.make_rdm1())
    finally:
        sgx_patch.revert()
