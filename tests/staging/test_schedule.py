"""S2: Engine-style staged SCF (staging.schedule).

Fast tests drive the stage machine with synthetic callback envs (no SCF);
the slow tests run real water-dimer SCFs and check the spec section 6 gates.
"""
import numpy as np
import pytest
from pyscf import dft

from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends
from pyscf_wb97mv_fast.staging.schedule import StagedSCF, run_staged


@pytest.fixture(scope='module')
def cosx_mf():
    mol = build_mol('water_dimer', 'def2-svp')
    return dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)


class FakeDiis:
    """Attribute-compatible stand-in for lib.diis.CDIIS."""

    def __init__(self):
        self._buffer = {'a': 1}
        self._bookkeep = [1]
        self._head = 2
        self._H = np.eye(2)
        self._xprev = np.ones(2)
        self._err_vec_touched = True
        self._diisfile = 'x'


def _envs(cycle, norm_gorb, e_tot, last_hf_e, diis):
    return dict(cycle=cycle, norm_gorb=norm_gorb, e_tot=e_tot,
                last_hf_e=last_hf_e, mf_diis=diis)


def test_constructor_rejects_non_cosx():
    mol = build_mol('water_dimer', 'def2-svp')
    with pytest.raises(TypeError):
        StagedSCF(dft.RKS(mol, xc='wb97m-v'))


def test_constructor_rejects_bad_modes(cosx_mf):
    with pytest.raises(ValueError):
        StagedSCF(cosx_mf, vv10_mode='always')
    with pytest.raises(ValueError):
        StagedSCF(cosx_mf, s2_mode='maybe')
    with pytest.raises(ValueError):
        StagedSCF(cosx_mf, xc_levels=(1, 3))


def test_attach_applies_stage_s0_and_restore_is_complete(cosx_mf):
    mf = cosx_mf
    df = mf.with_df
    orig = dict(nlc=mf.nlc, conv_check=mf.conv_check,
                grids_level=mf.grids.level, nlcgrids_level=mf.nlcgrids.level,
                gl_i=df.grids_level_i, gl_f=df.grids_level_f,
                callback=mf.callback, conv=mf.check_convergence)
    staged = StagedSCF(mf)
    with staged:
        assert mf.grids.level == 1 and mf.nlcgrids.level == 3  # unchanged while off
        assert mf.nlc == 0                                     # VV10 off in S0
        assert (df.grids_level_i, df.grids_level_f) == (1, 1)  # SGX auto-switch off
        assert mf._nsteps_direct == 0
        assert mf.callback is not orig['callback']
        assert mf.check_convergence is not orig['conv']
    assert mf.nlc == orig['nlc']
    assert mf.conv_check == orig['conv_check']
    assert mf.grids.level == orig['grids_level']
    assert mf.nlcgrids.level == orig['nlcgrids_level']
    assert (df.grids_level_i, df.grids_level_f) == (orig['gl_i'], orig['gl_f'])
    assert mf.callback is orig['callback']
    assert mf.check_convergence is orig['conv']


def test_switch_resets_diis_fock_and_grids(cosx_mf):
    mf = cosx_mf
    df = mf.with_df
    # pre-build everything so the switch exercises the reset/rebuild paths
    mf.grids.build(with_non0tab=True)
    mf.nlcgrids.build(with_non0tab=True)
    df.build(level=1)
    diis = FakeDiis()
    staged = StagedSCF(mf)
    with staged:
        staged._on_cycle_end(_envs(0, 1e-2, -76.4, -76.5, diis))
        assert staged.stage == 0                       # |g| above tau1
        staged._on_cycle_end(_envs(1, 1e-4, -76.42, -76.4199, diis))
        assert staged.stage == 1                       # |g| < 1e-3 -> S1
        assert diis._buffer == {} and diis._bookkeep == []
        assert diis._head == 0 and diis._H is None and diis._xprev is None
        assert mf._nsteps_direct == 0                  # full rebuild queued
        assert mf.grids.coords is None                 # rebuilt on next get_veff
        assert mf.grids.level == 3
        assert mf.nlc == ''                            # VV10 on (auto from xc)
        assert mf.nlcgrids.level == 1
        assert (df.grids_level_i, df.grids_level_f) == (1, 1)
        staged._on_cycle_end(_envs(2, 1e-5, -76.423, -76.4229999, diis))
        assert staged.stage == 2                       # |dE| < 1e-6 -> S2
        assert mf.nlcgrids.level == 3
        assert mf.nlcgrids.coords is None
        assert (df.grids_level_i, df.grids_level_f) == (2, 2)


def test_switch_never_goes_backwards(cosx_mf):
    staged = StagedSCF(cosx_mf)
    with staged:
        staged._on_cycle_end(_envs(0, 1e-9, -76.4, -76.3999, FakeDiis()))
        staged._on_cycle_end(_envs(1, 1e-9, -76.4, -76.4000000001, FakeDiis()))
        staged._on_cycle_end(_envs(2, 1e-9, -76.4, -76.4000000002, FakeDiis()))
        assert staged.stage == 2
        assert len(staged.switches) == 2


def test_check_convergence_replicates_stock_criterion(cosx_mf):
    mf = cosx_mf
    staged = StagedSCF(mf, s2_mode='converge')
    conv = staged._conv_factory(mf.check_convergence)  # original is None
    mf.conv_tol, mf.conv_tol_grad = 1e-9, None
    with staged:
        # neither criterion met -> not converged (stock: AND of the two)
        assert conv(_envs(0, 1e-3, -76.4, -76.4 + 1e-8, None)) is False
        # both met (|dE| < 1e-9, |g| < sqrt(1e-9)) -> converged
        assert conv(_envs(1, 1e-6, -76.4, -76.4 + 1e-10, None)) is True
        # energy met, gradient not -> not converged
        assert conv(_envs(2, 1e-2, -76.4, -76.4 + 1e-12, None)) is False


def test_check_convergence_extra_cycle_uses_relaxed_thresholds(cosx_mf):
    mf = cosx_mf
    staged = StagedSCF(mf, s2_mode='converge')
    conv = staged._conv_factory(mf.check_convergence)
    mf.conv_tol, mf.conv_tol_grad = 1e-9, None
    with staged:
        assert conv(_envs(4, 1e-5, -76.4, -76.4 + 1e-8, None)) is False
        # second call with the same cycle number = conv_check extra cycle:
        # relaxed thresholds (x10 / x3), joined by OR
        assert conv(_envs(4, 1e-5, -76.4, -76.4 + 1e-8, None)) is True


def test_steps_mode_stops_after_s2_steps(cosx_mf):
    mf = cosx_mf
    staged = StagedSCF(mf, s2_mode='steps', s2_steps=2)
    conv = staged._conv_factory(mf.check_convergence)
    with staged:
        assert mf.conv_check is False                  # no extra cycle
        staged._on_cycle_end(_envs(0, 1e-4, -76.4, -76.3999, FakeDiis()))
        assert staged.stage == 1
        assert conv(_envs(1, 1e-5, -76.41, -76.4099, None)) is False  # S1
        staged._on_cycle_end(_envs(2, 1e-6, -76.42, -76.4199999, FakeDiis()))
        assert staged.stage == 2
        assert conv(_envs(3, 1e-7, -76.43, -76.4299, None)) is False  # S2 #1
        assert conv(_envs(4, 1e-7, -76.44, -76.4399, None)) is True   # S2 #2
    assert mf.conv_check is True


def test_nonscf_mode_keeps_vv10_off(cosx_mf):
    staged = StagedSCF(cosx_mf, vv10_mode='nonscf')
    with staged:
        assert all(not staged._vv10_on(s) for s in range(3))
        assert cosx_mf.nlc == 0


def test_info_reports_stages_and_switches(cosx_mf):
    staged = StagedSCF(cosx_mf)
    with staged:
        staged._on_cycle_end(_envs(0, 1e-4, -76.4, -76.3999, FakeDiis()))
        staged._on_cycle_end(_envs(1, 1e-6, -76.42, -76.4199999, FakeDiis()))
    info = staged.info()
    assert info['stages']['S0']['cycles'] == 1
    assert info['stages']['S1']['cycles'] == 1
    assert info['stages']['S2']['cycles'] == 0
    assert [(s['_from'], s['to']) for s in info['switches']] == [(0, 1), (1, 2)]
    assert info['total_wall'] > 0.0


@pytest.mark.slow
def test_staged_energy_matches_unstaged_reference():
    """spec section 6: staged S2 uses the reference grid set (3/3/2), so the
    staged energy must land on the plain COSX SCF energy within 1e-4 Ha."""
    mol = build_mol('water_dimer', 'def2-svp')
    ref = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    ref.conv_tol = 1e-10
    from pyscf_wb97mv_fast.core import sgx_patch
    sgx_patch.apply()
    try:
        e_ref = ref.kernel()
    finally:
        sgx_patch.revert()

    info = run_staged(mol, conv_tol=1e-10)
    assert info['converged']
    assert len(info['switches']) == 2                 # reached S2
    assert abs(info['e_tot'] - e_ref) < 1e-4


@pytest.mark.slow
def test_nonscf_matches_late_scf():
    mol = build_mol('water_dimer', 'def2-svp')
    late = run_staged(mol, conv_tol=1e-10)
    nonscf = run_staged(mol, conv_tol=1e-10, vv10_mode='nonscf')
    assert nonscf['e_nlc'] != 0.0
    assert abs(late['e_tot'] - nonscf['e_tot']) < 1e-4


@pytest.mark.slow
def test_steps_mode_runs_and_stays_within_gate():
    mol = build_mol('water_dimer', 'def2-svp')
    info = run_staged(mol, conv_tol=1e-10, s2_mode='steps', s2_steps=1)
    assert info['converged']
    staged = info  # history written per cycle; S2 must have exactly 1 cycle
    assert staged['stages']['S2']['cycles'] == 1


@pytest.mark.slow
def test_run_staged_restores_state():
    mol = build_mol('water_dimer', 'def2-svp')
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    orig = (mf.nlc, mf.conv_check, mf.grids.level, mf.nlcgrids.level,
            mf.with_df.grids_level_i, mf.with_df.grids_level_f)
    run_staged(mol, conv_tol=1e-9)
    now = (mf.nlc, mf.conv_check, mf.grids.level, mf.nlcgrids.level,
           mf.with_df.grids_level_i, mf.with_df.grids_level_f)
    assert orig == now


# -- precision per stage (spec section 6: heavy work FP32 on the GPU in the
#    early stages, the last stage on the CPU FP64 numint) -------------------

class FakeGpuHooks:
    def __init__(self, log):
        self.log = log

    def restore_all(self):
        self.log.append('restore')


@pytest.fixture
def fake_install(monkeypatch):
    from pyscf_wb97mv_fast.gpu import install
    log = []

    def fake(mf, **kwargs):
        log.append(('install', kwargs))
        return FakeGpuHooks(log)
    monkeypatch.setattr(install, 'install_gpu', fake)
    return log


def test_gpu_stages_validated(cosx_mf):
    with pytest.raises(ValueError):
        StagedSCF(cosx_mf, gpu_stages=(3,))


def test_gpu_stages_default_never_installs(cosx_mf, fake_install):
    with StagedSCF(cosx_mf) as staged:
        staged._on_cycle_end(_envs(0, 1e-4, -76.4, -76.3999, FakeDiis()))
        staged._on_cycle_end(_envs(1, 1e-5, -76.42, -76.4199999, FakeDiis()))
        assert staged.stage == 2
    assert fake_install == []
    assert [h['gpu'] for h in staged.history] == [False, False]


def test_gpu_stages_install_in_s0_s1_and_drop_for_s2(cosx_mf, fake_install):
    staged = StagedSCF(cosx_mf, gpu_stages=(0, 1), gpu_kwargs={'vv10_tile_tol': 0.0})
    with staged:
        assert fake_install == [('install', {'vv10_tile_tol': 0.0})]
        staged._on_cycle_end(_envs(0, 1e-4, -76.4, -76.3999, FakeDiis()))
        assert staged.stage == 1
        assert len(fake_install) == 1                 # kept, not reinstalled
        staged._on_cycle_end(_envs(1, 1e-5, -76.42, -76.4199999, FakeDiis()))
        assert staged.stage == 2
        assert fake_install[-1] == 'restore'          # S2 runs on the CPU FP64
        assert [h['gpu'] for h in staged.history] == [True, True]
    assert fake_install.count('restore') == 1         # restore() adds nothing


def test_gpu_hooks_removed_when_restored_early(cosx_mf, fake_install):
    with StagedSCF(cosx_mf, gpu_stages=(0, 1)):
        assert len(fake_install) == 1
    assert fake_install[-1] == 'restore' and fake_install.count('restore') == 1


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_staged_matches_cpu_staged():
    """spec section 8 layer 2 for the precision switch: S0/S1 through the GPU
    FP32 flows, S2 on the CPU FP64 numint, against the same stages on the CPU
    only -- |dE| <= 1e-6 Ha and cycle counts within 1."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    mol = build_mol('water_dimer', 'def2-svp')
    cpu = run_staged(mol, conv_tol=1e-9)
    gpu = run_staged(mol, conv_tol=1e-9, gpu_stages=(0, 1))
    assert cpu['converged'] and gpu['converged']
    assert any(h['gpu'] for h in gpu['history'])
    assert not any(h['gpu'] for h in gpu['history'] if h['stage'] == 2)
    assert abs(gpu['e_tot'] - cpu['e_tot']) <= 1e-6
    assert abs(gpu['cycles'] - cpu['cycles']) <= 1, (gpu['cycles'], cpu['cycles'])


# -- FP32 until near convergence, FP64 for the last cycles (the design point:
#    precision switch inside S2, not tied to the grid stages) -------------------

def test_fp64_final_needs_s2_on_gpu(cosx_mf):
    with pytest.raises(ValueError):
        StagedSCF(cosx_mf, gpu_stages=(0, 1), fp64_final=True)


def _drive_to_fp64(staged, mf):
    """S0 -> S1 -> S2 on the GPU, then the FP32 -> FP64 switch inside S2."""
    staged._on_cycle_end(_envs(0, 1e-4, -76.4, -76.3999, FakeDiis()))       # -> S1
    staged._on_cycle_end(_envs(1, 1e-5, -76.42, -76.4199999, FakeDiis()))   # -> S2
    assert staged.stage == 2 and not staged.fp64_phase
    diis = FakeDiis()
    mf._nsteps_direct = 5
    staged._on_cycle_end(_envs(2, 1e-5, -76.43, -76.4299999, diis))         # |dE| < 1e-6
    assert staged.fp64_phase
    assert diis._buffer == {} and mf._nsteps_direct == 0                    # reset + rebuild
    assert staged.switches[-1]['precision'] == 'fp64'


def test_fp64_final_switches_precision_inside_s2(cosx_mf, fake_install):
    """Default fp64_final_vv10='gpu': the semilocal XC goes back to the CPU
    FP64 numint, VV10 stays on the GPU (its FP32 error is 1.1e-7 Ha on
    water27)."""
    mf = cosx_mf
    staged = StagedSCF(mf, gpu_stages=(0, 1, 2), fp64_final=True)
    with staged:
        conv = mf.check_convergence
        mf.conv_tol, mf.conv_tol_grad = 1e-9, None
        staged._on_cycle_end(_envs(0, 1e-4, -76.4, -76.3999, FakeDiis()))
        staged._on_cycle_end(_envs(1, 1e-5, -76.42, -76.4199999, FakeDiis()))
        # stock criterion met, but still FP32: must not converge
        assert conv(_envs(2, 1e-6, -76.4, -76.4 + 1e-10, None)) is False
        mf._nsteps_direct = 5
        staged._on_cycle_end(_envs(2, 1e-5, -76.43, -76.4299999, FakeDiis()))
        assert staged.fp64_phase
        assert fake_install[-2:] == ['restore',
                                     ('install', {'semilocal': False,
                                                  'k': False})]
        assert conv(_envs(3, 1e-6, -76.4, -76.4 + 1e-10, None)) is True
        staged._on_cycle_end(_envs(3, 1e-6, -76.4, -76.4 + 1e-10, FakeDiis()))
    assert [h['fp64'] for h in staged.history] == [False, False, False, True]
    assert fake_install.count('restore') == 2        # FP32 set, then the VV10-only set


def test_fp64_final_all_cpu_tail(cosx_mf, fake_install):
    mf = cosx_mf
    staged = StagedSCF(mf, gpu_stages=(0, 1, 2), fp64_final=True, fp64_final_vv10='cpu')
    with staged:
        _drive_to_fp64(staged, mf)
        assert fake_install[-1] == 'restore'           # nothing reinstalled
    assert fake_install.count('restore') == 1


def test_fp64_final_vv10_validated(cosx_mf):
    with pytest.raises(ValueError):
        StagedSCF(cosx_mf, gpu_stages=(0, 1, 2), fp64_final=True, fp64_final_vv10='tpu')


@pytest.mark.gpu
@pytest.mark.slow
def test_fp64_final_staged_matches_cpu_staged():
    """GPU FP32 through S0-S2, FP64 for the last cycles: same energy as the
    CPU-only staged SCF (|dE| <= 1e-6 Ha), final cycles on the CPU."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    mol = build_mol('water_dimer', 'def2-svp')
    cpu = run_staged(mol, conv_tol=1e-9)
    gpu = run_staged(mol, conv_tol=1e-9, gpu_stages=(0, 1, 2), fp64_final=True)
    assert cpu['converged'] and gpu['converged']
    assert any(h['gpu'] for h in gpu['history'] if h['stage'] == 2)
    assert gpu['history'][-1]['fp64']             # converged in the FP64 tail
    assert abs(gpu['e_tot'] - cpu['e_tot']) <= 1e-6


def test_fp64_switch_waits_for_small_gradient(cosx_mf, fake_install):
    """Switch FP32 -> FP64 only when |dE| < tau_fp64 AND |g| < tau_fp64_g, so
    the FP64 tail needs ~2 cycles (water27 with the |dE|-only switch at
    |g| = 2e-4 took 3 cycles)."""
    mf = cosx_mf
    staged = StagedSCF(mf, gpu_stages=(0, 1, 2), fp64_final=True, tau_fp64_g=5e-5)
    with staged:
        staged._on_cycle_end(_envs(0, 1e-4, -76.4, -76.3999, FakeDiis()))       # -> S1
        staged._on_cycle_end(_envs(1, 1e-5, -76.42, -76.4199999, FakeDiis()))   # -> S2
        staged._on_cycle_end(_envs(2, 2e-4, -76.43, -76.4299999, FakeDiis()))   # |g| too big
        assert not staged.fp64_phase
        staged._on_cycle_end(_envs(3, 4e-5, -76.43, -76.4299999, FakeDiis()))   # both met
        assert staged.fp64_phase


def test_fp64_switch_forced_after_max_fp32_cycles(cosx_mf, fake_install):
    """Safety: if FP32 noise keeps |g| above tau_fp64_g, switch anyway after
    max_fp32_s2_cycles S2 cycles in FP32."""
    mf = cosx_mf
    staged = StagedSCF(mf, gpu_stages=(0, 1, 2), fp64_final=True, tau_fp64_g=1e-9,
                       max_fp32_s2_cycles=3)
    with staged:
        staged._on_cycle_end(_envs(0, 1e-4, -76.4, -76.3999, FakeDiis()))
        staged._on_cycle_end(_envs(1, 1e-5, -76.42, -76.4199999, FakeDiis()))   # -> S2
        for c in (2, 3):
            staged._on_cycle_end(_envs(c, 1e-4, -76.43, -76.4299999, FakeDiis()))
            assert not staged.fp64_phase
        staged._on_cycle_end(_envs(4, 1e-4, -76.43, -76.4299999, FakeDiis()))   # 3rd FP32 S2 cycle
        assert staged.fp64_phase
        assert staged.switches[-1]['forced']


def test_s1_to_s2_waits_for_small_gradient(cosx_mf):
    """S1 -> S2 needs |dE| < tau2 AND |g| < tau2_g: right after the S0 -> S1
    switch a tiny |dE| can coincide with a large |g| (water27: |dE| 8.3e-7 at
    |g| 2.2e-2), which pushed the convergence
    work into the more expensive S2."""
    staged = StagedSCF(cosx_mf, tau2_g=1e-3)
    with staged:
        staged._on_cycle_end(_envs(0, 1e-4, -76.4, -76.3999, FakeDiis()))       # -> S1
        staged._on_cycle_end(_envs(1, 2e-2, -76.42, -76.4199999, FakeDiis()))   # |g| too big
        assert staged.stage == 1
        staged._on_cycle_end(_envs(2, 5e-4, -76.42, -76.4199999, FakeDiis()))   # both met
        assert staged.stage == 2


def test_s1_to_s2_forced_after_max_s1_cycles(cosx_mf):
    staged = StagedSCF(cosx_mf, tau2_g=1e-12, max_s1_cycles=2)
    with staged:
        staged._on_cycle_end(_envs(0, 1e-4, -76.4, -76.3999, FakeDiis()))       # -> S1
        staged._on_cycle_end(_envs(1, 2e-2, -76.42, -76.4199999, FakeDiis()))
        assert staged.stage == 1
        staged._on_cycle_end(_envs(2, 2e-2, -76.42, -76.4199999, FakeDiis()))   # 2nd S1 cycle
        assert staged.stage == 2 and staged.switches[-1]['forced']


# -- S5: the FP32 stages' exchange K on the GPU (gpu_kwargs k=True) ----------

def test_fp64_tail_puts_k_back_on_cpu(cosx_mf, fake_install):
    """S5 plan Task 7: gpu_kwargs k=True arms the GPU K for the FP32 stages;
    the FP32 -> FP64 switch reinstalls with k=False forced (the tail's K
    stays CPU FP64), whatever gpu_kwargs said."""
    staged = StagedSCF(cosx_mf, gpu_stages=(0, 1, 2), fp64_final=True,
                       gpu_kwargs={'k': True})
    with staged:
        _drive_to_fp64(staged, cosx_mf)
        assert fake_install[0] == ('install', {'k': True})
        assert fake_install[-1] == ('install', {'k': False, 'semilocal': False})


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_k_staged_dimer_matches_cpu_staged():
    """Full staged water-dimer SCF with the GPU K in the FP32 stages; the
    final energy matches the CPU-only staged SCF (|dE| <= 1e-6 Ha)."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    mol = build_mol('water_dimer', 'def2-svp')
    cpu = run_staged(mol, conv_tol=1e-9)
    gpu = run_staged(mol, conv_tol=1e-9, gpu_stages=(0, 1, 2), fp64_final=True,
                     gpu_kwargs={'k': True})
    assert cpu['converged'] and gpu['converged']
    assert gpu['history'][-1]['fp64']
    assert abs(gpu['e_tot'] - cpu['e_tot']) <= 1e-6


# -- FP64 tail energy tolerance (2026-10-03 Ruling: the tail's |dE| test is
#    matched to the 1e-6 gate, the gradient test stays at the stock value) ---

def test_fp64_conv_tol_needs_fp64_final(cosx_mf):
    with pytest.raises(ValueError):
        StagedSCF(cosx_mf, fp64_conv_tol=1e-7)


def test_fp64_conv_tol_relaxes_energy_only_in_fp64_tail(cosx_mf, fake_install):
    """water27 GPU K run: cycle 17
    had |dE| 2.98e-9 and |g| 5.19e-6, then 6 more cycles (205 s) inside the
    ~5e-9 incremental-COSX energy noise.  fp64_conv_tol loosens only the |dE|
    test of the FP64 tail; |g| keeps sqrt(conv_tol), NOT sqrt(fp64_conv_tol)."""
    mf = cosx_mf
    staged = StagedSCF(mf, gpu_stages=(0, 1, 2), fp64_final=True, fp64_conv_tol=1e-7)
    with staged:
        conv = mf.check_convergence
        mf.conv_tol, mf.conv_tol_grad = 1e-9, None
        staged._on_cycle_end(_envs(0, 1e-4, -76.4, -76.3999, FakeDiis()))       # -> S1
        staged._on_cycle_end(_envs(1, 1e-5, -76.42, -76.4199999, FakeDiis()))   # -> S2
        # FP32 S2: never converged, whatever the tolerances
        assert conv(_envs(2, 1e-6, -76.4, -76.4 + 1e-10, None)) is False
        staged._on_cycle_end(_envs(2, 1e-5, -76.43, -76.4299999, FakeDiis()))   # -> FP64
        assert staged.fp64_phase
        # |dE| 2.98e-9 > conv_tol but < fp64_conv_tol, |g| 5.19e-6: converged
        assert conv(_envs(3, 5.19e-6, -76.4, -76.4 + 2.98e-9, None)) is True
        # |dE| above fp64_conv_tol: not converged
        assert conv(_envs(4, 5.19e-6, -76.4, -76.4 + 3.19e-6, None)) is False
        # |g| 1e-4: below sqrt(1e-7) = 3.2e-4 but above sqrt(1e-9) = 3.16e-5
        assert conv(_envs(5, 1e-4, -76.4, -76.4 + 1e-10, None)) is False


def test_fp64_conv_tol_extra_cycle_relaxes_from_fp64_conv_tol(cosx_mf, fake_install):
    """The conv_check extra cycle keeps the stock x10 / x3 relaxation, applied
    to fp64_conv_tol and to the unchanged gradient threshold."""
    mf = cosx_mf
    staged = StagedSCF(mf, gpu_stages=(0, 1, 2), fp64_final=True, fp64_conv_tol=1e-7)
    with staged:
        conv = mf.check_convergence
        mf.conv_tol, mf.conv_tol_grad = 1e-9, None
        _drive_to_fp64(staged, mf)
        assert conv(_envs(3, 1e-4, -76.4, -76.4 + 5e-7, None)) is False
        assert conv(_envs(3, 1e-4, -76.4, -76.4 + 5e-7, None)) is True   # 5e-7 < 1e-6
        assert conv(_envs(4, 1e-4, -76.4, -76.4 + 2e-6, None)) is False
        assert conv(_envs(4, 1e-4, -76.4, -76.4 + 2e-6, None)) is False  # 1e-4 > 9.5e-5


def test_fp64_conv_tol_turns_conv_check_off_while_attached(cosx_mf, fake_install):
    """The stock extra cycle after convergence is one more full FP64 cycle
    (~30 s on water27, 4090) that moves E by ~2e-9: skipped with
    fp64_conv_tol, restored on detach; untouched without it."""
    mf = cosx_mf
    assert mf.conv_check is True
    with StagedSCF(mf, gpu_stages=(0, 1, 2), fp64_final=True, fp64_conv_tol=1e-7):
        assert mf.conv_check is False
    assert mf.conv_check is True
    with StagedSCF(mf, gpu_stages=(0, 1, 2), fp64_final=True):
        assert mf.conv_check is True


def test_fp64_conv_tol_default_keeps_stock_criterion(cosx_mf, fake_install):
    mf = cosx_mf
    staged = StagedSCF(mf, gpu_stages=(0, 1, 2), fp64_final=True)
    with staged:
        conv = mf.check_convergence
        mf.conv_tol, mf.conv_tol_grad = 1e-9, None
        _drive_to_fp64(staged, mf)
        assert conv(_envs(3, 5.19e-6, -76.4, -76.4 + 2.98e-9, None)) is False
    assert staged.info()['fp64_conv_tol'] is None


@pytest.mark.gpu
@pytest.mark.slow
def test_fp64_conv_tol_dimer_matches_cpu_staged():
    """Real water-dimer SCF, GPU K in the FP32 stages, fp64_conv_tol=1e-7:
    still within the 1e-6 gate of the CPU-only staged SCF, converged in the
    FP64 tail, and the last |g| below the stock sqrt(conv_tol)."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    mol = build_mol('water_dimer', 'def2-svp')
    cpu = run_staged(mol, conv_tol=1e-9)
    gpu = run_staged(mol, conv_tol=1e-9, gpu_stages=(0, 1, 2), fp64_final=True,
                     gpu_kwargs={'k': True}, fp64_conv_tol=1e-7)
    assert cpu['converged'] and gpu['converged']
    assert gpu['fp64_conv_tol'] == 1e-7
    assert gpu['history'][-1]['fp64']
    assert gpu['history'][-1]['norm_gorb'] < 1e-9 ** 0.5
    assert abs(gpu['e_tot'] - cpu['e_tot']) <= 1e-6
