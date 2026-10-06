import numpy as np
import pytest
from pyscf import dft

from pyscf_wb97mv_fast.core import sgx_patch
from pyscf_wb97mv_fast.experimental.cosx.sr_screening import BPath, ErfcSRBounds, PerKernelBudget
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.experimental.cosx.tolerance import set_sgx_tolerance

OMEGA = 0.3


def _mf():
    """wb97m-v COSX mf ready for direct get_jk/get_k calls outside kernel().

    mf.build() (not just mf.with_df.build()) is required: SGXHF.get_jk uses
    mf._nsteps_direct, which only SGXHF.build()/reset() initialize.
    """
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    mf.build()
    return mf


def _k(sgx, dm, omega=None):
    return sgx.get_jk(dm, 1, None, with_j=False, with_k=True, omega=omega)[1]


def _drop_caches(mf):
    """Drop _pjs_data of the main SGX object AND of every RSH copy."""
    sgx = mf.with_df
    set_sgx_tolerance(mf, sgx.sgx_tol_energy, sgx.sgx_tol_potential)


def test_bpath_operator_identity():
    """hyb*K_full + (alpha-hyb)*K_LR == alpha*K_full - (alpha-hyb)*K_SR."""
    sgx_patch.apply()
    try:
        mf = _mf()
        sgx = mf.with_df
        dm = mf.get_init_guess()
        omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
        k_full = _k(sgx, dm)
        k_lr = _k(sgx, dm, omega=OMEGA)
        k_sr = _k(sgx, dm, omega=-OMEGA)
        vk_a = hyb * k_full + (alpha - hyb) * k_lr
        vk_b = alpha * k_full - (alpha - hyb) * k_sr
        assert abs(vk_a - vk_b).max() < 1e-10
    finally:
        sgx_patch.revert()


def test_bpath_get_k_slot_returns_kfull_minus_ksr():
    sgx_patch.apply()
    try:
        mf = _mf()
        sgx = mf.with_df
        dm = mf.get_init_guess()
        k_full = _k(sgx, dm)
        k_sr = _k(sgx, dm, omega=-OMEGA)
        with BPath(mf).attach():
            mf.get_jk(mf.mol, dm, 1)                 # stashes full K
            k_slot = mf.get_k(mf.mol, dm, 1, omega=OMEGA)   # LR slot
        assert abs(k_slot - (k_full - k_sr)).max() < 1e-12
    finally:
        sgx_patch.revert()


def test_bpath_survives_inplace_scaling_of_vk():
    """dft.rks.get_veff does `vk *= hyb` on the returned array BEFORE calling
    get_k for the LR slot; the stash must not alias that array."""
    sgx_patch.apply()
    try:
        mf = _mf()
        sgx = mf.with_df
        dm = mf.get_init_guess()
        omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
        k_full = _k(sgx, dm)
        k_sr = _k(sgx, dm, omega=-OMEGA)
        with BPath(mf).attach():
            _, vk = mf.get_jk(mf.mol, dm, 1)         # same call sequence as rks.get_veff
            vk *= hyb
            vklr = mf.get_k(mf.mol, dm, 1, omega=omega)
        assert abs(vklr - (k_full - k_sr)).max() < 1e-12
    finally:
        sgx_patch.revert()


@pytest.mark.slow
def test_bpath_scf_energy_unchanged():
    mol = build_mol('water_dimer')
    e = {}
    for use_b in (False, True):
        sgx_patch.apply()
        try:
            mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
            mf.conv_tol = 1e-9
            if use_b:
                bp = BPath(mf).attach()
                try:
                    e[use_b] = mf.kernel()
                finally:
                    bp.remove()
                assert bp.n_sr_builds >= 1
            else:
                e[use_b] = mf.kernel()
            assert mf.converged
        finally:
            sgx_patch.revert()
    assert abs(e[True] - e[False]) < 1e-8, e


def test_per_kernel_budget_routes_tol():
    sgx_patch.apply()
    try:
        mf = _mf()
        sgx = mf.with_df
        dm = mf.get_init_guess()
        pkb = PerKernelBudget(mf, tol_full=1e-11, tol_attenuated=1e-8).attach()
        try:
            mf.get_jk(mf.mol, dm, 1)
            mf.get_k(mf.mol, dm, 1, omega=OMEGA)
        finally:
            pkb.remove()
        assert sgx._pjs_data._itol == 1e-11
        assert sgx._rsh_df['%.6f' % OMEGA]._pjs_data._itol == 1e-8
    finally:
        sgx_patch.revert()


def test_per_kernel_budget_requires_patch():
    sgx_patch.revert()
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    with pytest.raises(RuntimeError):
        PerKernelBudget(mf).attach()


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason='SR _mbar_bi already contains the erfc '
                   'decay; the extra envelope double-counts it (sr_screening.py)')
def test_erfc_bounds_correctness_gate():
    """G2a correctness gate for the experimental envelope: K_SR must stay
    within 1e-9 of the stock SR build. Expected to fail."""
    sgx_patch.apply()
    try:
        mf = _mf()
        sgx = mf.with_df
        dm = mf.get_init_guess()
        k_ref = _k(sgx, dm, omega=-OMEGA)
        _drop_caches(mf)
        gate = ErfcSRBounds(slack_bohr=4.0, enable=True).attach()
        try:
            k_env = _k(sgx, dm, omega=-OMEGA)
        finally:
            gate.remove()
        assert abs(k_env - k_ref).max() < 1e-9
    finally:
        sgx_patch.revert()


def test_erfc_bounds_disabled_is_noop():
    sgx_patch.apply()
    try:
        mf = _mf()
        sgx = mf.with_df
        dm = mf.get_init_guess()
        k_ref = _k(sgx, dm, omega=-OMEGA)
        _drop_caches(mf)
        assert sgx._rsh_df['%.6f' % -OMEGA]._pjs_data is None
        with ErfcSRBounds(enable=False).attach():
            k = _k(sgx, dm, omega=-OMEGA)
        assert abs(k - k_ref).max() < 1e-14
    finally:
        sgx_patch.revert()
