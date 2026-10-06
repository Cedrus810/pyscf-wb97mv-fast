import numpy as np
import pytest
from pyscf import dft

from pyscf_wb97mv_fast.core import sgx_patch
from pyscf_wb97mv_fast.experimental.cosx.fastgrad import (attach, check_gradient_support,
                                        finite_difference_check)
from pyscf_wb97mv_fast.core.testsystems import build_mol


def test_cosx_gradient_supported_upstream():
    """pyscf 2.14 ships pyscf/sgx/grad; _SGXHF.nuc_grad_method dispatches there."""
    from pyscf.sgx.grad import rks as sgx_rks_grad
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    ok, reason = check_gradient_support(mf)
    assert ok, reason
    assert isinstance(mf.nuc_grad_method(), sgx_rks_grad.Gradients)


def test_cosx_direct_j_unsupported():
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    mf.with_df.direct_j = True
    ok, _ = check_gradient_support(mf)
    assert not ok


def test_plain_rks_gradient_supported():
    mf = dft.RKS(build_mol('water_dimer'), xc='pbe')
    ok, _ = check_gradient_support(mf)
    assert ok


def test_attach_pins_reference_tolerance():
    sgx_patch.revert()
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    mf.with_df.build()
    mf.with_df.sgx_tol_energy = 1e-8
    dm = mf.get_init_guess()
    mf.with_df.get_jk(dm, 1, None, with_j=False, with_k=True)
    assert mf.with_df._pjs_data is not None
    try:
        with attach(mf):
            assert sgx_patch._sgx.SGX.get_jk is sgx_patch.get_jk
            assert mf.with_df.sgx_tol_energy == 'auto'
            assert mf.with_df._pjs_data is None          # dropped -> rebuilt at ref tol
        mf.with_df.get_jk(dm, 1, None, with_j=False, with_k=True)
        assert mf.with_df._pjs_data._etol == mf.with_df._pjs_data._itol
    finally:
        sgx_patch.revert()


@pytest.mark.slow
def test_fd_check_on_plain_rks():
    """Sanity of the FD gate utility itself (P5 gate: <= 1e-5 Ha/bohr)."""
    mol = build_mol('water_dimer', 'sto-3g')
    mf = dft.RKS(mol, xc='pbe')
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    err, _, _ = finite_difference_check(mf, dm0=mf.make_rdm1(), disp=1e-3)
    assert err < 1e-5, err


@pytest.mark.slow
@pytest.mark.xfail(strict=False, raises=FloatingPointError,
                   reason='upstream PySCF 2.14: SGX grid response NaN from '
                          'libdft.VXCgen_grid_lko_deriv (fastgrad.py docstring)')
def test_fd_check_on_cosx_wb97mv():
    """The strict P5 gate: SGX analytic gradient (with SGX grid response) vs FD.
    XPASS here means upstream fixed the lko-derivative NaN."""
    sgx_patch.apply()
    try:
        mol = build_mol('water_dimer', 'sto-3g')
        mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
        mf.conv_tol = 1e-11
        mf.kernel()
        assert mf.converged
        err, _, _ = finite_difference_check(mf, dm0=mf.make_rdm1(), disp=1e-3)
        assert err < 1e-5, err
    finally:
        sgx_patch.revert()


@pytest.mark.slow
def test_cosx_grad_finite_without_sgx_grid_response():
    """Workaround path: finite gradient, reported FD error (no SGX grid-weight
    response, so the error is the SGX grid response, not a code bug)."""
    sgx_patch.apply()
    try:
        mol = build_mol('water_dimer', 'sto-3g')
        mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
        mf.conv_tol = 1e-11
        mf.kernel()
        assert mf.converged
        err, _, grad = finite_difference_check(mf, dm0=mf.make_rdm1(), disp=1e-3,
                                               sgx_grid_response=False)
        assert np.isfinite(grad).all()
        print(f'FD error without SGX grid response: {err:.2e} Ha/bohr')
    finally:
        sgx_patch.revert()
