"""S2: Engine-style staged SCF (spec section 6).

The SCF runs in three stages; every stage switch does the three things the
spec requires -- change grids, reset DIIS, force a full Fock rebuild.  Inside
a stage the grids, precision and tolerances never change.

    stage   enter when              XC grids.level  VV10        SGX level
    S0      from the initial guess  1               off         1
    S1      |g| < tau1 (1e-3)       3               on, level 1 1
    S2      |dE| < tau2 (1e-6 Ha)   3               level 3     2

Precision per stage (spec section 6 precision table + its last rule "S2 ...
otherwise fall back to the CPU FP64 reference"): gpu_stages lists the stages
whose XC/VV10 run through gpu.install.install_gpu (FP32 heavy work on the
GPU); the other stages use the stock CPU FP64 numint.  gpu_stages=(0, 1) is
the design point -- the FP32 error grows with system size (benchmarks/xc/
FINDINGS.md), so the converged energy comes from FP64 S2 cycles.  The hooks
go on/off at the same stage switch that resets DIIS and rebuilds Fock.
Default gpu_stages=() keeps the whole SCF on the CPU.

fp64_final=True (needs 2 in gpu_stages) is the design point "FP32 until near
convergence, FP64 for the last cycle or two": S2 runs on the GPU too, and once
|dE| < tau_fp64 inside S2 the hooks come off for the remaining cycles (DIIS
reset + full Fock rebuild, like every switch -- the energy functional changes
by the FP32 error).  Convergence is never declared before that switch.
fp64_final_vv10='gpu' (default) keeps VV10 on the GPU in FP32 for those last
cycles and moves only the semilocal XC back to the CPU FP64 numint: the VV10
FP32 error is 1.1e-7 Ha on water27 (inside
the 1e-6 budget) and the CPU VV10 is the largest piece of an FP64 cycle (66 s
of ~100 s).  fp64_final_vv10='cpu' puts both back on the CPU.
fp64_conv_tol (default None = mf.conv_tol) is the tail's |dE| threshold
alone; the |g| threshold stays mf.conv_tol_grad or sqrt(mf.conv_tol) (a user
check_convergence, if set, overrides both as before).  The incremental COSX
energy jitters by ~5e-9 between FP64 cycles, so a 1e-9 |dE| test can spin
for cycles that change nothing within the 1e-6 gate (2026-10-03 Ruling).
Setting fp64_conv_tol also turns mf.conv_check off while attached (no stock
extra cycle after convergence), like s2_mode='steps'.  With gpu_stages=(0, 1) instead, all
S2 cycles run on the CPU and do the bulk of the convergence (water27: 7 of 16
cycles, 72% of the wall time).

Two switches on top of the table (both spec section 6):

    vv10_mode='late_scf' (default)  VV10 self-consistent in S1/S2, per table.
    vv10_mode='nonscf'              VV10 off for the whole SCF; after
                                    convergence one VV10 energy evaluation on
                                    the final density (CPU FP64 reference),
                                    added to mf.e_tot and scf_summary['exc'].

    s2_mode='converge' (default)    converge to mf.conv_tol inside S2.
    s2_mode='steps'                 stop after s2_steps cycles in S2 (Engine
                                    parity); mf.conv_check is turned off while
                                    attached so no extra cycle runs.

Mechanics (PySCF 2.14, all verified against the source):

* |g| is envs['norm_gorb'] and |dE| is abs(e_tot - last_hf_e), both read in
  mf.callback -- the same quantities PySCF's convergence check uses.
  (core.hooks.track_residual wraps the same callback mechanism; here the
  values are simply read from the envs dict we already receive.)
* DIIS reset: core.hooks.reset_diis(envs['mf_diis']).
* Full Fock rebuild: sgx increments J/K against dm_last; setting
  mf._nsteps_direct = 0 makes _SGXHF.with_full_dm yield will_reset=True on the
  next get_veff, which nulls dm_last/vhf_last -- one full J+K build.
* SGX has its own staged grid (grids_level_i -> grids_level_f switch on
  |ddm| < grids_switch_thrd).  Staging disables it by pinning
  grids_level_i == grids_level_f to the stage level and rebuilding the SGX
  grid eagerly at each switch with with_df.build(level=...).
* XC/VV10 grids: set .level then .reset(); the next get_veff calls
  initialize_grids, which rebuilds at the new level and re-prunes small-rho
  points with the current dm -- exactly what a stock SCF does on cycle 1.
* VV10 on/off: mf.nlc = '' (auto from xc) or 0 (disabled); do_nlc() is read
  dynamically by get_veff, no caching involved.

Usage:
    from pyscf_wb97mv_fast.staging import schedule
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    sgx_patch.apply()
    with schedule.StagedSCF(mf) as staged:
        e = mf.kernel()
    info = staged.info()        # per-stage cycle counts, wall times, switches
"""
import time

from pyscf import dft
from pyscf.lib import logger

from pyscf_wb97mv_fast.core.hooks import HookSet, reset_diis

_STAGE_NAMES = ('S0', 'S1', 'S2')

# spec section 6 defaults
DEFAULT_TAU1 = 1e-3          # |g| threshold, S0 -> S1
DEFAULT_TAU2 = 1e-6          # |dE| threshold (Ha), S1 -> S2
DEFAULT_XC_LEVELS = (1, 3, 3)
DEFAULT_NLC_LEVELS = (None, 1, 3)   # None = VV10 off in that stage
DEFAULT_SGX_LEVELS = (1, 1, 2)


class StagedSCF:
    """Attach to an RKS/COSX mf to run it in stages.  Context manager.

    Everything installed on mf is undone on restore(); the grid objects stay
    built at the final stage's level, which is also what a plain SCF leaves
    behind.
    """

    def __init__(self, mf, tau1=DEFAULT_TAU1, tau2=DEFAULT_TAU2,
                 xc_levels=DEFAULT_XC_LEVELS, nlc_levels=DEFAULT_NLC_LEVELS,
                 sgx_levels=DEFAULT_SGX_LEVELS,
                 vv10_mode='late_scf', s2_mode='converge', s2_steps=1,
                 nlc_production_level=3, gpu_stages=(), gpu_kwargs=None,
                 fp64_final=False, tau_fp64=None, fp64_final_vv10='gpu',
                 tau_fp64_g=5e-5, max_fp32_s2_cycles=8, tau2_g=float('inf'),
                 max_s1_cycles=8, fp64_conv_tol=None):
        if not hasattr(mf, 'with_df') or not hasattr(mf.with_df, 'grids_level_i'):
            raise TypeError('StagedSCF needs a COSX (SGX) mf, got %s' % type(mf))
        if not set(gpu_stages) <= {0, 1, 2}:
            raise ValueError('gpu_stages entries must be stage indices 0, 1, 2')
        if fp64_final and 2 not in gpu_stages:
            raise ValueError('fp64_final switches precision inside S2: needs 2 in gpu_stages')
        if fp64_final_vv10 not in ('gpu', 'cpu'):
            raise ValueError("fp64_final_vv10 must be 'gpu' or 'cpu'")
        if fp64_conv_tol is not None and not fp64_final:
            raise ValueError('fp64_conv_tol applies to the FP64 tail: needs fp64_final=True')
        if vv10_mode not in ('late_scf', 'nonscf'):
            raise ValueError("vv10_mode must be 'late_scf' or 'nonscf'")
        if s2_mode not in ('converge', 'steps'):
            raise ValueError("s2_mode must be 'converge' or 'steps'")
        for name, levels in (('xc_levels', xc_levels),
                             ('nlc_levels', nlc_levels),
                             ('sgx_levels', sgx_levels)):
            if len(levels) != 3:
                raise ValueError('%s needs 3 entries (S0, S1, S2)' % name)
        self.mf = mf
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)
        self.xc_levels = tuple(xc_levels)
        self.nlc_levels = tuple(nlc_levels)
        self.sgx_levels = tuple(sgx_levels)
        self.vv10_mode = vv10_mode
        self.s2_mode = s2_mode
        self.s2_steps = int(s2_steps)
        self.nlc_production_level = int(nlc_production_level)
        self.gpu_stages = tuple(sorted(set(gpu_stages)))
        self.gpu_kwargs = dict(gpu_kwargs or {})
        self.fp64_final = bool(fp64_final)
        self.tau_fp64 = float(tau2 if tau_fp64 is None else tau_fp64)
        self.fp64_final_vv10 = fp64_final_vv10
        self.fp64_phase = False      # True once the FP32 -> FP64 switch happened
        # the switch also waits for |g| < tau_fp64_g, so the FP64 tail is ~2
        # cycles: the next FP64 |dE| scales ~|g|^2 (water27: |g| 2e-4 at the
        # switch -> 3 extra FP64 cycles); forced after
        # max_fp32_s2_cycles in case FP32 noise keeps |g| above it
        self.tau_fp64_g = float(tau_fp64_g)
        # optional |g| gate for S1 -> S2 (off by default).  Tried at 1e-3 on
        # water27: S1 then stalled at |g| ~2e-2 for several cycles, S2 still
        # needed 5 FP32 + 3 FP64 cycles, total 619 s vs 544 s without it
        # (calibrated on the water27 staging runs)
        self.tau2_g = float(tau2_g)
        self.max_s1_cycles = int(max_s1_cycles)
        self._s1_cycles = 0
        self.max_fp32_s2_cycles = int(max_fp32_s2_cycles)
        # |dE| threshold of the FP64 tail only (None = mf.conv_tol, stock).
        # The |g| threshold is NOT derived from it: it stays mf.conv_tol_grad
        # or sqrt(mf.conv_tol).  2026-10-03 Ruling: water27 GPU K spent 6 tail
        # cycles (205 s) inside the ~5e-9 incremental-COSX energy noise after
        # |dE| 3e-9, |g| 5e-6 (water27 GPU K acceptance run).
        self.fp64_conv_tol = None if fp64_conv_tol is None else float(fp64_conv_tol)
        self._fp32_s2_cycles = 0

        self.stage = 0
        self.s2_cycles = 0
        self.switches = []       # dicts: from, to, cycle, t
        self.history = []        # one dict per finished cycle
        self._hooks = HookSet()
        self._attached = False
        self._t_attach = None
        self._nlc_saved = None
        self._conv_check_saved = None
        self._grids_level_saved = None
        self._nlcgrids_level_saved = None
        self._sgx_levels_saved = None
        self._gpu_hooks = None   # HookSet from install_gpu while a GPU stage runs

    # -- context management -------------------------------------------------
    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.restore()

    def attach(self):
        """Pin stage S0 settings and install the callback / convergence hooks."""
        mf = self.mf
        if self._attached:
            raise RuntimeError('StagedSCF already attached')
        self._nlc_saved = mf.nlc
        self._conv_check_saved = mf.conv_check
        self._grids_level_saved = mf.grids.level
        self._nlcgrids_level_saved = mf.nlcgrids.level
        if self.s2_mode == 'steps':
            # Engine parity: stop after s2_steps S2 cycles, no extra cycle.
            mf.conv_check = False
        if self.fp64_conv_tol is not None:
            # the tail test is already matched to the 1e-6 gate; the stock
            # extra cycle is one more full FP64 cycle (~30 s on water27)
            # that changes E
            # by ~2e-9.  e_tot is then the last regular cycle's energy.
            mf.conv_check = False

        self._hooks.wrap(mf, 'callback', self._callback_factory)
        self._hooks.wrap(mf, 'check_convergence', self._conv_factory)

        self.stage = 0
        self.fp64_phase = False
        self._fp32_s2_cycles = 0
        self._s1_cycles = 0
        self._apply_stage(0)
        self._attached = True
        self._t_attach = time.perf_counter()
        logger.note(mf, 'StagedSCF attached, stage S0 (vv10_mode=%s s2_mode=%s)',
                    self.vv10_mode, self.s2_mode)
        return self

    def restore(self):
        """Undo every attribute change.  Grids stay built at the final level."""
        if not self._attached:
            return
        mf = self.mf
        self._hooks.restore_all()
        self._set_gpu(False)
        mf.nlc = self._nlc_saved
        mf.conv_check = self._conv_check_saved
        mf.grids.level = self._grids_level_saved
        mf.nlcgrids.level = self._nlcgrids_level_saved
        df = mf.with_df
        df.grids_level_i, df.grids_level_f = self._sgx_levels_saved
        self._attached = False

    # -- hooks ---------------------------------------------------------------
    def _callback_factory(self, prev):
        def callback(envs):
            self._on_cycle_end(envs)
            if callable(prev):
                prev(envs)
        return callback

    def _conv_factory(self, orig):
        """Stage counting for s2_mode='steps', plus a faithful replica of the
        stock convergence test.

        Installing check_convergence REPLACES PySCF's own convergence branch
        (scf.hf.kernel: `if callable(mf.check_convergence): ... elif abs(dE)
        < conv_tol and ...`), so with no user check_convergence we have to
        evaluate the stock criterion here.  The extra conv_check cycle runs
        the same wrapper with relaxed thresholds (conv_tol*10, grad*3), like
        the stock `elif` there; it is recognised as the second call with the
        same cycle number.
        """
        mf = self.mf
        state = {'last_cycle': None}

        def check_convergence(envs):
            if self.fp64_final and not self.fp64_phase:
                return False                  # the final energy must be FP64
            if self.stage == 2:
                self.s2_cycles += 1
                if (self.s2_mode == 'steps'
                        and self.s2_cycles >= self.s2_steps):
                    return True
            if callable(orig):
                return orig(envs)
            conv_tol, conv_tol_grad = mf.conv_tol, mf.conv_tol_grad
            if conv_tol_grad is None:
                conv_tol_grad = conv_tol ** 0.5
            if self.fp64_phase and self.fp64_conv_tol is not None:
                conv_tol = self.fp64_conv_tol    # gradient threshold unchanged
            cycle = envs.get('cycle')
            extra_cycle = cycle is not None and cycle == state['last_cycle']
            state['last_cycle'] = cycle
            if extra_cycle:
                # hf.kernel conv_check block: conv_tol *= 10, grad *= 3, OR
                return (abs(envs['e_tot'] - envs['last_hf_e']) < conv_tol * 10
                        or envs['norm_gorb'] < conv_tol_grad * 3)
            return (abs(envs['e_tot'] - envs['last_hf_e']) < conv_tol
                    and envs['norm_gorb'] < conv_tol_grad)
        return check_convergence

    def _on_cycle_end(self, envs):
        """Record the cycle, then decide whether to advance a stage.

        A switch decided here takes effect from the next get_veff on: grids
        are reset now, and _nsteps_direct = 0 forces that next get_veff to a
        full rebuild on the new settings.
        """
        t = time.perf_counter()
        cycle = int(envs.get('cycle', -1))
        norm_gorb = float(envs.get('norm_gorb', float('nan')))
        e_tot = float(envs.get('e_tot', float('nan')))
        last_hf_e = float(envs.get('last_hf_e', float('nan')))
        dE = abs(e_tot - last_hf_e)
        rec = dict(cycle=cycle, stage=self.stage, e_tot=e_tot, delta_e=dE,
                   norm_gorb=norm_gorb, t=t, gpu=self._gpu_hooks is not None,
                   fp64=self.fp64_phase)
        timings = getattr(self._gpu_hooks, 'timings', None)
        if timings:                     # overlap split of this cycle's get_veff
            rec.update(t_jk=timings[-1]['jk'], t_xc=timings[-1]['xc'],
                       t_wait=timings[-1]['wait'])
        self.history.append(rec)
        new_stage, forced = None, False
        if self.stage == 1:
            self._s1_cycles += 1
        if self.stage == 0 and norm_gorb < self.tau1:
            new_stage = 1
        elif self.stage == 1 and dE < self.tau2 and norm_gorb < self.tau2_g:
            new_stage = 2
        elif self.stage == 1 and self._s1_cycles >= self.max_s1_cycles:
            new_stage, forced = 2, True
        if self.stage == 2 and not self.fp64_phase:
            self._fp32_s2_cycles += 1
        fp64_due, fp64_forced = (False, False) if new_stage is not None else \
            self._fp64_switch_due(dE, norm_gorb)
        if fp64_due:
            self.switches.append(dict(_from=self.stage, to=self.stage,
                                      cycle=cycle, t=t, precision='fp64',
                                      forced=fp64_forced))
            self._set_gpu(False)
            self.fp64_phase = True
            if self.fp64_final_vv10 == 'gpu':  # VV10 stays FP32 on the GPU
                from pyscf_wb97mv_fast.gpu import install
                # k=False is forced: the FP64 tail's exchange K stays on the
                # CPU (S5 plan Task 7) -- gpu_kwargs cannot override it
                self._gpu_hooks = install.install_gpu(
                    self.mf, **dict(self.gpu_kwargs, semilocal=False, k=False))
            self.mf._nsteps_direct = 0         # full Fock rebuild in FP64
            reset_diis(envs.get('mf_diis'))
            logger.note(self.mf, 'StagedSCF precision switch FP32 -> FP64 in %s at '
                        'cycle %d (|g|=%.3e, |dE|=%.3e): DIIS reset, full Fock rebuild',
                        self.stage_name, cycle + 1, norm_gorb, dE)
        if new_stage is not None:
            self.switches.append(dict(_from=self.stage, to=new_stage,
                                      cycle=cycle, t=t, forced=forced))
            self.stage = new_stage
            self.s2_cycles = 0
            self._apply_stage(new_stage)
            reset_diis(envs.get('mf_diis'))
            logger.note(self.mf, 'StagedSCF switch %s -> %s at cycle %d '
                        '(|g|=%.3e, |dE|=%.3e): DIIS reset, full Fock rebuild',
                        _STAGE_NAMES[new_stage - 1], _STAGE_NAMES[new_stage],
                        cycle + 1, norm_gorb, dE)

    def _fp64_switch_due(self, dE, norm_gorb):
        """(due, forced) for the FP32 -> FP64 switch inside S2."""
        if not (self.fp64_final and self.stage == 2 and not self.fp64_phase):
            return False, False
        if dE < self.tau_fp64 and norm_gorb < self.tau_fp64_g:
            return True, False
        if self._fp32_s2_cycles >= self.max_fp32_s2_cycles:
            return True, True
        return False, False

    # -- stage settings -------------------------------------------------------
    def _vv10_on(self, stage):
        if self.vv10_mode == 'nonscf':
            return False
        return self.nlc_levels[stage] is not None

    def _apply_stage(self, stage):
        """Grids of all three operators for one stage; no Fock/DIIS action."""
        mf = self.mf
        df = mf.with_df
        sgx_level = self.sgx_levels[stage]

        mf.grids.level = self.xc_levels[stage]
        if self.nlc_levels[stage] is not None:
            mf.nlcgrids.level = self.nlc_levels[stage]
        mf.nlc = self._nlc_saved if self._vv10_on(stage) else 0

        # coords=None -> get_veff::initialize_grids rebuilds at the new level
        # and re-prunes small-rho points with the current dm
        if mf.grids.coords is not None:
            mf.grids.reset()
        if mf.nlcgrids.coords is not None:
            mf.nlcgrids.reset()

        # pin the SGX grid (this also disables SGX's internal i->f switch)
        if self._sgx_levels_saved is None:
            self._sgx_levels_saved = (df.grids_level_i, df.grids_level_f)
        df.grids_level_i = df.grids_level_f = sgx_level
        if df.grids is not None:
            df.build(level=sgx_level)          # also clears _pjs_data, _rsh_df
        # dm_last = None on the next get_veff -> full J+K build
        mf._nsteps_direct = 0
        self._set_gpu(stage in self.gpu_stages)

    def _set_gpu(self, on):
        """Install / remove the GPU XC+VV10 hooks (idempotent)."""
        if on and self._gpu_hooks is None:
            from pyscf_wb97mv_fast.gpu import install
            self._gpu_hooks = install.install_gpu(self.mf, **self.gpu_kwargs)
        elif not on and self._gpu_hooks is not None:
            self._gpu_hooks.restore_all()
            self._gpu_hooks = None

    # -- reporting ------------------------------------------------------------
    @property
    def stage_name(self):
        return _STAGE_NAMES[self.stage]

    def info(self):
        """Per-stage cycles / wall time plus the switch list.

        Wall times need run_staged() (or a kernel call made while attached):
        they are derived from the switch timestamps and restore time.
        """
        t_end = time.perf_counter()
        bounds = ([(s['t'], s['to']) for s in self.switches]
                  + [(t_end, None)])
        per_stage = {}
        t_prev = self._t_attach
        stage_prev = 0
        for t, stage in bounds:
            key = _STAGE_NAMES[stage_prev]
            cur = per_stage.setdefault(key, dict(cycles=0, wall=0.0))
            cur['wall'] += t - t_prev
            cur['cycles'] = sum(1 for h in self.history
                                if h['stage'] == stage_prev)
            t_prev, stage_prev = t, (stage if stage is not None else stage_prev)
        return dict(stages=per_stage, switches=list(self.switches),
                    history=list(self.history), gpu_stages=self.gpu_stages,
                    fp64_final=self.fp64_final, fp64_conv_tol=self.fp64_conv_tol,
                    total_wall=t_end - self._t_attach)

    # -- vv10_mode='nonscf' post-processing ------------------------------------
    def add_nonscf_vv10(self):
        """One VV10 energy evaluation on the final density (CPU FP64).

        Adds it to mf.e_tot and mf.scf_summary['exc'] (the slot get_veff folds
        VV10 into), so energy_tot() reproduces the reported value.  Returns
        the VV10 energy.  No-op (returns 0.0) for vv10_mode='late_scf'.
        """
        if self.vv10_mode != 'nonscf':
            return 0.0
        from pyscf_wb97mv_fast.reference.numint import vv10_reference
        mf = self.mf
        mf.nlcgrids.level = self.nlc_production_level
        nelec, e_nl, vnlc = vv10_reference(mf, mf.make_rdm1())
        mf.e_tot += e_nl
        mf.scf_summary['exc'] = mf.scf_summary.get('exc', 0.0) + e_nl
        logger.note(mf, 'StagedSCF non-SCF VV10 energy = %.10f (grids level %d)',
                    e_nl, self.nlc_production_level)
        return e_nl


def run_staged(mol, xc='wb97m-v', conv_tol=1e-9, dm0=None, sgx_patch=True,
               staged=None, **staged_kwargs):
    """One staged SCF on an RKS/COSX(pjs=True) mf; returns an info dict.

    Applies core.sgx_patch for the run and always reverts it.  The attached
    StagedSCF stays reachable as mf._staged.  staged_kwargs go to StagedSCF
    (tau1, tau2, vv10_mode, s2_mode, s2_steps, ...).
    """
    from pyscf_wb97mv_fast.core.sgx_patch import apply as sgx_apply, revert as sgx_revert
    mf = dft.RKS(mol, xc=xc).COSX(pjs=True)
    mf.conv_tol = conv_tol
    if sgx_patch:
        sgx_apply()
    try:
        if staged is None:
            staged = StagedSCF(mf, **staged_kwargs)
        with staged:
            t0 = time.perf_counter()
            e_tot = mf.kernel(dm0=dm0)
            e_nl = staged.add_nonscf_vv10()
        info = staged.info()
        info.update(e_tot=float(mf.e_tot), converged=bool(mf.converged),
                    cycles=int(mf.cycles), nao=int(mol.nao), xc=xc,
                    vv10_mode=staged.vv10_mode, s2_mode=staged.s2_mode,
                    e_nlc=float(e_nl), kernel_wall=time.perf_counter() - t0,
                    dm=mf.make_rdm1())
        mf._staged = staged
        return info
    finally:
        if sgx_patch:
            sgx_revert()
