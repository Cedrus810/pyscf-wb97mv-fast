import numpy as np
import pytest
from pyscf import dft

from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.experimental.cosx.vv10_staged import StagedVV10


@pytest.mark.slow
def test_staged_vv10_energy_within_gate():
    mol = build_mol('water_dimer')
    mf0 = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    mf0.conv_tol = 1e-9
    e_ref = mf0.kernel()
    assert mf0.converged

    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    mf.conv_tol = 1e-9
    with StagedVV10(mf).attach():
        e_staged = mf.kernel()
    assert mf.converged
    assert abs(e_staged - e_ref) < 1e-6, (e_staged, e_ref)


@pytest.mark.slow
def test_staged_vv10_skips_early_cycles():
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    mf.conv_tol = 1e-9
    with StagedVV10(mf).attach() as st:
        mf.kernel()
    # >= 2: the initial-guess veff alone accounts for one off call
    assert st.n_off_calls >= 2, st.n_off_calls
    assert st.ever_on                       # and it was switched on (or cleanup)


def test_zeros_shape_matches_dm():
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    st = StagedVV10(mf, switch_at=0.0).attach()
    try:
        dm = np.eye(mf.mol.nao)
        ni = mf._numint
        n, enlc, vnlc = ni.nr_nlc_vxc(mf.mol, mf.nlcgrids, 'vv10', dm)
        assert vnlc.shape == dm.shape and enlc == 0.0
    finally:
        st.remove()
