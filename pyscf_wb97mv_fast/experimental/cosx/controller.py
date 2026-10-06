"""FROZEN (2026-09-30): gates failed, see benchmarks/experimental/cosx/FINDINGS.md section 9; not imported by mainline.

P1b: error-budget controller (A path) and the fast_path() entry point.

Per SCF cycle the SGX DM-screening budget is derived from PySCF's orbital
gradient norm r_n = envs['norm_gorb'] (the |g| that scf.hf.kernel compares
with conv_tol_grad, read via mf.callback -- see _hooks.track_residual):

    eps_E,n = clamp(C * r_n, eps_min, eps_max)
    eps_V,n = vtol rule ('auto' = sqrt(eps_E,n), PySCF's own default rule)

The budget is only pushed into the SGX objects when it changes by more than a
factor ``rel_change``. The push uses tolerance.update_sgx_tolerance, which
keeps the integral bounds and overlap fit and rebuilds only the DM-screening
part; the cleanup uses set_sgx_tolerance (full cache reset).

The first get_veff (initial guess, before any residual exists) runs at
eps_max.

Modes:
    reference  - sgx bug fix only, stock tight tolerance (|dE| <= 1e-10 Ha)
    production - controller + full rebuild + tight cleanup (|dE| <= 1e-8 Ha,
                 ||dP||_max <= 1e-5)
    fast       - controller, no cleanup (|dE| <= 1e-6 Ha)

Usage:
    from pyscf_wb97mv_fast.experimental.cosx.controller import fast_path
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    mf = fast_path(mf, mode='production')
    mf.kernel()
"""
import numpy as np

from pyscf_wb97mv_fast.core import sgx_patch
from pyscf_wb97mv_fast.core.hooks import HookSet, track_residual
from pyscf_wb97mv_fast.experimental.cosx.tolerance import set_sgx_tolerance, update_sgx_tolerance

DEFAULT_C = 1e-5
DEFAULT_EPS_MIN = 1e-13
DEFAULT_EPS_MAX = 1e-8


class ErrorBudgetController:
    """Attach to an mf; drive sgx_tol_energy from the SCF residual.

    After attach(), the controller is reachable as mf._wb97mv_fast_controller.
    Stats: rn, applied, history (every etol pushed, in order),
    n_budget_updates, cleanup_runs.
    """

    def __init__(self, mf, mode='production', c=DEFAULT_C,
                 eps_min=DEFAULT_EPS_MIN, eps_max=DEFAULT_EPS_MAX,
                 rel_change=2.0, vtol='auto', cleanup_tol='auto'):
        if mode not in ('production', 'fast'):
            raise ValueError("mode must be 'production' or 'fast'")
        self.mf = mf
        self.mode = mode
        self.c = c
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.rel_change = rel_change
        self.vtol = vtol
        self.cleanup_tol = cleanup_tol
        self.rn = np.inf          # PySCF norm_gorb of the previous cycle
        self.applied = None       # etol currently baked into the SGX objects
        self.history = []
        self.n_budget_updates = 0
        self.cleanup_runs = 0
        self._hooks = HookSet()
        self._attached = False
        self._locked = False      # cleanup re-run: veff must not touch tolerance
        self._in_cleanup = False

    def budget(self):
        return float(min(max(self.c * self.rn, self.eps_min), self.eps_max))

    def _should_apply(self, etol):
        if self.applied is None:
            return True
        if etol < self.applied / self.rel_change or etol > self.applied * self.rel_change:
            return True
        return False

    def _push_budget(self):
        etol = self.budget()
        if self._should_apply(etol):
            update_sgx_tolerance(self.mf, etol, self.vtol)
            self.applied = etol
            self.history.append(etol)
            self.n_budget_updates += 1

    def _set_rn(self, rn):
        self.rn = rn

    def attach(self):
        if self._attached:
            return self
        self._attached = True
        self.mf._wb97mv_fast_controller = self
        mf = self.mf
        hooks = self._hooks

        track_residual(hooks, mf, self._set_rn)

        def veff_factory(orig):
            def get_veff(mol=None, dm=None, dm_last=None, vhf_last=None,
                         hermi=1, *a, **k):
                if not self._locked:
                    self._push_budget()
                return orig(mol, dm, dm_last, vhf_last, hermi, *a, **k)
            return get_veff
        hooks.wrap(mf, 'get_veff', veff_factory)

        def kernel_factory(orig):
            def kernel(dm0=None, *a, **k):
                self.rn = np.inf
                e = orig(dm0=dm0, *a, **k)
                if (self.mode == 'production' and not self._in_cleanup
                        and getattr(mf, 'converged', False)):
                    self._locked = True
                    self._in_cleanup = True
                    try:
                        set_sgx_tolerance(mf, self.cleanup_tol)
                        self.applied = None
                        e = orig(dm0=mf.make_rdm1(), *a, **k)
                        self.cleanup_runs += 1
                    finally:
                        self._in_cleanup = False
                        self._locked = False
                return e
            return kernel
        hooks.wrap(mf, 'kernel', kernel_factory)
        return self

    def remove(self):
        if self._attached:
            self._hooks.restore_all()
            self._attached = False

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.remove()


def fast_path(mf, mode='production', **kw):
    """mf = pyscf_wb97mv_fast.fast_path(mf, mode='reference'|'production'|'fast').

    Installs the SGX bug fix and (for production/fast) the error-budget
    controller; the returned mf is driven as usual with mf.kernel().
    """
    sgx_patch.apply()
    if mode == 'reference':
        return mf
    ErrorBudgetController(mf, mode=mode, **kw).attach()
    return mf
