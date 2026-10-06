"""Tests for pyscf_wb97mv_fast.sgx_patch. Run: python tests/test_sgx_patch.py"""
import os
import sys

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.df import df_jk
from pyscf.sgx import sgx_jk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from pyscf_wb97mv_fast.core import sgx_patch  # noqa: E402

OMEGA = 0.3
WATER_DIMER = '''
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
'''


def make_mol():
    return gto.M(atom=WATER_DIMER, basis='def2-svp', verbose=0)


def make_sgx(mol, pjs=True):
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=pjs)
    mf.with_df.build()
    return mf, mf.with_df, mf.get_init_guess()


class CallTrace:
    """Record calls to SGX K builders and the DF J builder."""
    targets = [(sgx_jk, 'get_k_only'), (sgx_jk, 'get_jk'), (df_jk, 'get_j')]

    def __enter__(self):
        self.calls, self._saved = [], []
        for mod, name in self.targets:
            f = getattr(mod, name)
            self._saved.append((mod, name, f))

            def g(*a, _f=f, _n=name, **k):
                self.calls.append(_n)
                return _f(*a, **k)
            setattr(mod, name, g)
        return self

    def __exit__(self, *exc):
        for mod, name, f in self._saved:
            setattr(mod, name, f)


def reference_k(sgx, dm, omega):
    """K from a fresh SGX object built entirely under the attenuated kernel."""
    r = sgx.copy()
    r._rsh_df = {}
    r._vjopt = None
    r._overlap_correction_matrix = None
    with r.mol.with_range_coulomb(omega):
        r.build()
        return r.get_jk(dm, 1, None, with_j=False, with_k=True)[1]


def test_no_extra_j_build():
    sgx_patch.apply()
    try:
        _, sgx, dm = make_sgx(make_mol(), pjs=True)
        with CallTrace() as t:
            sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=OMEGA)
        assert t.calls == ['get_k_only'], t.calls
    finally:
        sgx_patch.revert()


def test_unpatched_has_extra_j_build():
    """Documents the upstream bug; fails once PySCF fixes it."""
    _, sgx, dm = make_sgx(make_mol(), pjs=True)
    with CallTrace() as t:
        sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=OMEGA)
    assert t.calls == ['get_j', 'get_k_only'], t.calls


def test_direct_scf_tol_propagates():
    sgx_patch.apply()
    try:
        _, sgx, dm = make_sgx(make_mol(), pjs=True)
        sgx.get_jk(dm, 1, None, with_j=False, with_k=True,
                   direct_scf_tol=1e-9, omega=OMEGA)
        assert sgx._rsh_df['%.6f' % OMEGA]._pjs_data._itol == 1e-9
    finally:
        sgx_patch.revert()


def test_bounds_not_shared_with_full_kernel():
    sgx_patch.apply()
    try:
        _, sgx, dm = make_sgx(make_mol(), pjs=True)
        sgx.get_jk(dm, 1, None, with_j=False, with_k=True)          # full K first
        for om in (OMEGA, -OMEGA):
            sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=om)
            assert sgx._rsh_df['%.6f' % om]._pjs_data is not sgx._pjs_data
    finally:
        sgx_patch.revert()


def test_k_matches_reference_and_decomposition():
    sgx_patch.apply()
    try:
        _, sgx, dm = make_sgx(make_mol(), pjs=True)
        kf = sgx.get_jk(dm, 1, None, with_j=False, with_k=True)[1]
        kl = sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=OMEGA)[1]
        ks = sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=-OMEGA)[1]
        assert abs(kl - reference_k(sgx, dm, OMEGA)).max() < 1e-12
        assert abs(ks - reference_k(sgx, dm, -OMEGA)).max() < 1e-12
        assert abs(kf - kl - ks).max() < 1e-9
    finally:
        sgx_patch.revert()


@pytest.mark.slow
def test_scf_energy_unchanged():
    mol = make_mol()
    e = {}
    for patched in (False, True):
        if patched:
            sgx_patch.apply()
        try:
            mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
            mf.conv_tol = 1e-10
            e[patched] = mf.kernel()
            assert mf.converged
        finally:
            sgx_patch.revert()
    assert abs(e[True] - e[False]) < 1e-8, e


def test_patch_survives_grid_rebuild():
    """SGXHF calls with_df.build() when switching grids_level_i -> grids_level_f.
    build() rebuilds the RSH copies outside with_range_coulomb; LR K must stay correct."""
    sgx_patch.apply()
    try:
        _, sgx, dm = make_sgx(make_mol(), pjs=True)
        sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=OMEGA)
        sgx.build()
        kl = sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=OMEGA)[1]
        assert abs(kl - reference_k(sgx, dm, OMEGA)).max() < 1e-12
    finally:
        sgx_patch.revert()


def test_multiple_density_matrices():
    """nset > 1 (e.g. UKS) goes through the same patched path; K is linear in dm."""
    sgx_patch.apply()
    try:
        _, sgx, dm = make_sgx(make_mol(), pjs=True)
        dms = np.stack([0.5 * dm, 0.5 * dm])
        k2 = sgx.get_jk(dms, 1, None, with_j=False, with_k=True, omega=OMEGA)[1]
        k1 = sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=OMEGA)[1]
        assert k2.shape == dms.shape
        assert abs(k2.sum(axis=0) - k1).max() < 1e-9
    finally:
        sgx_patch.revert()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'PASS {t.__name__}')
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f'FAIL {t.__name__}: {type(exc).__name__}: {exc}')
    sys.exit(1 if failed else 0)
