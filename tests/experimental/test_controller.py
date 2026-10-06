import pytest
from pyscf import dft

from pyscf_wb97mv_fast.core import sgx_patch
from pyscf_wb97mv_fast.experimental.cosx.controller import ErrorBudgetController, fast_path
from pyscf_wb97mv_fast.core.testsystems import build_mol


def test_budget_clamps():
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    ctl = ErrorBudgetController(mf, c=1e-5, eps_min=1e-13, eps_max=1e-8)
    for rn, want in ((1e-1, 1e-8), (1e-3, 1e-8), (1e-4, 1e-9),
                     (1e-9, 1e-13), (0.0, 1e-13)):
        ctl.rn = rn
        assert ctl.budget() == pytest.approx(want)


def test_budget_pushed_only_on_change():
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    ctl = ErrorBudgetController(mf)
    ctl.rn = 1e-2
    ctl._push_budget()
    assert ctl.applied == 1e-8 and ctl.n_budget_updates == 1
    ctl.rn = 1e-2 * 1.1                      # within rel_change: no rebuild
    ctl._push_budget()
    assert ctl.n_budget_updates == 1
    ctl.rn = 1e-5                            # crosses rel_change: rebuild
    ctl._push_budget()
    assert ctl.n_budget_updates == 2 and ctl.applied == pytest.approx(1e-10)


def test_fast_path_reference_mode_is_plain_patch():
    mf0 = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    mf = fast_path(mf0, mode='reference')
    assert mf is mf0
    assert 'kernel' not in vars(mf)             # no controller hooks attached
    assert not any(isinstance(v, ErrorBudgetController)
                   for v in vars(mf).values())


@pytest.mark.slow
def test_production_matches_reference():
    mol = build_mol('water_dimer')
    e = {}
    for mode in ('reference', 'production'):
        mf = fast_path(dft.RKS(mol, xc='wb97m-v').COSX(pjs=True), mode=mode)
        mf.conv_tol = 1e-9
        e[mode] = mf.kernel()
        assert mf.converged
    assert abs(e['production'] - e['reference']) < 1e-8, e


@pytest.mark.slow
def test_controller_reports_updates_and_cleanup():
    mf = fast_path(dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True),
                   mode='production')
    mf.conv_tol = 1e-9
    mf.kernel()
    ctl = mf._wb97mv_fast_controller
    assert ctl.cleanup_runs == 1
    assert ctl.n_budget_updates >= 1
    # The residual must be PySCF's real |g| (converged ~ conv_tol_grad), not the
    # ~1e-15 you get from the Fock that eig() just diagonalized; the latter
    # would pin every budget to eps_min.
    assert ctl.rn > 1e-10, ctl.rn
    assert min(ctl.history) > ctl.eps_min, ctl.history


def test_controller_restores_hooks_on_remove():
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    # mf.kernel is always a fresh bound method; check the instance dict instead
    ctl = ErrorBudgetController(mf).attach()
    assert 'kernel' in vars(mf) and 'callback' in vars(mf) and 'get_veff' in vars(mf)
    ctl.remove()
    assert not {'kernel', 'get_veff'} & set(vars(mf))
    assert mf.callback is None
