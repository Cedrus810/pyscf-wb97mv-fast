import pytest
from pyscf import dft
from pyscf.sgx import sgx_jk

from pyscf_wb97mv_fast.core.profiling import ScfProfiler, run_profile
from pyscf_wb97mv_fast.core.testsystems import build_mol


@pytest.mark.slow
def test_profiler_records_components_patched():
    res = run_profile(build_mol('water_dimer'), patched=True, conv_tol=1e-8)
    t = res['timings']
    assert res['converged']
    for key in ('coulomb', 'k_full', 'k_lr', 'dft_xc', 'vv10', 'diag'):
        assert t.get(key, 0) > 0, key
    assert t.get('coulomb_rsh_waste', 0) == 0
    parts = sum(v for k, v in t.items() if k != 'scf_total')
    assert parts <= 1.05 * t['scf_total']


@pytest.mark.slow
def test_profiler_sees_upstream_waste_when_unpatched():
    res = run_profile(build_mol('water_dimer'), patched=False, conv_tol=1e-6)
    assert res['timings'].get('coulomb_rsh_waste', 0) > 0


def test_profiler_restores_on_exception():
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    orig = sgx_jk.get_k_only
    with pytest.raises(RuntimeError):
        with ScfProfiler().attach(mf):
            assert sgx_jk.get_k_only is not orig
            raise RuntimeError('boom')
    assert sgx_jk.get_k_only is orig
    assert 'eig' not in vars(mf)
    assert 'nr_rks' not in vars(mf._numint)


@pytest.mark.slow
def test_profiler_on_plain_rks():
    mf = dft.RKS(build_mol('water_dimer'), xc='pbe')
    prof = ScfProfiler()
    with prof.attach(mf):
        mf.kernel()
    assert prof.timings['dft_xc'] > 0
    assert 'k_full' not in prof.timings
