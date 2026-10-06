import pytest
from pyscf import dft

from pyscf_wb97mv_fast.core import sgx_patch
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.experimental.cosx.tolerance import set_sgx_tolerance, update_sgx_tolerance

OMEGA = 0.3


def test_set_sgx_tolerance_reaches_rsh_copies():
    sgx_patch.apply()
    try:
        mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
        sgx = mf.with_df
        sgx.build()
        dm = mf.get_init_guess()
        sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=OMEGA)   # creates RSH copy
        set_sgx_tolerance(mf, 1e-8)
        sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=OMEGA)
        sgx.get_jk(dm, 1, None, with_j=False, with_k=True)
        assert sgx._rsh_df['%.6f' % OMEGA]._pjs_data._etol == 1e-8
        assert sgx._pjs_data._etol == 1e-8
    finally:
        sgx_patch.revert()


def _k(sgx, dm, omega=None):
    return sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=omega)[1]


@pytest.mark.parametrize('omega', [None, OMEGA, -OMEGA])
def test_update_in_place_equals_full_reset(omega):
    """update_sgx_tolerance (DM-screen rebuild only) must give the same K as
    set_sgx_tolerance (full cache reset), for the main object and RSH copies."""
    sgx_patch.apply()
    try:
        mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
        sgx = mf.with_df
        sgx.build()
        dm = mf.get_init_guess()
        _k(sgx, dm, omega)                               # build caches at 'auto'
        obj = sgx if omega is None else sgx._rsh_df['%.6f' % omega]
        mbar_before = obj._pjs_data._mbar_ij
        update_sgx_tolerance(mf, 1e-8)
        assert obj._pjs_data._etol == 1e-8 and obj._pjs_data._vtol == 1e-4
        assert obj._pjs_data._mbar_ij is mbar_before     # integral bounds kept
        k_upd = _k(sgx, dm, omega)
        set_sgx_tolerance(mf, 1e-8)
        k_reset = _k(sgx, dm, omega)
        assert abs(k_upd - k_reset).max() < 1e-13
    finally:
        sgx_patch.revert()


def test_update_before_first_build_only_sets_attributes():
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    update_sgx_tolerance(mf, 1e-9, 1e-5)
    assert mf.with_df.sgx_tol_energy == 1e-9 and mf.with_df.sgx_tol_potential == 1e-5
    assert mf.with_df._pjs_data is None


@pytest.mark.slow
def test_gate_run_setting_smoke():
    from scf_tolerance_gate import run_setting
    mol = build_mol('water_dimer')
    ref = run_setting(mol, 'auto', None, 1e-9)
    loose = run_setting(mol, 1e-8, 'auto', 1e-9)
    assert ref['converged'] and loose['converged']
    assert loose['cleanup_cycles'] >= 1
    assert abs(loose['e_tot'] - ref['e_tot']) < 1e-8
