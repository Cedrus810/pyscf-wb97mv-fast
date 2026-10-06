"""FROZEN (2026-09-30): gates failed, see benchmarks/experimental/cosx/FINDINGS.md section 9; not imported by mainline.

P4: staged VV10.

wb97m-v carries a VV10 nonlocal-correlation term, evaluated every SCF cycle on
the (finer) nlc grids. Early cycles do not need it: while the orbital gradient
norm r_n is above ``switch_at`` the nr_nlc_vxc call returns zeros (VV10 off);
below it -- and for every cycle after -- VV10 is back in the Fock build, so
convergence is judged, and the energy reported, with the full functional.

If the SCF converges before VV10 was ever switched on, one cleanup kernel call
(dm0 = converged dm, VV10 forced on) is run so the final state satisfies the
full functional's stationary condition. Gate (P4): saves
>= 3% of SCF wall time at |dE| <= 1e-6 Ha.

Usage:
    from pyscf_wb97mv_fast.experimental.cosx.vv10_staged import StagedVV10
    with StagedVV10(mf).attach():
        mf.kernel()
"""
import numpy as np

from pyscf_wb97mv_fast.core.hooks import HookSet, track_residual


class StagedVV10:
    """switch_at=None (default) derives the switch point from the SCF
    convergence target: 10 * sqrt(mf.conv_tol), compared with PySCF's own |g|
    (envs['norm_gorb']), so VV10 returns for the last few cycles instead of
    after convergence."""

    def __init__(self, mf, switch_at=None):
        self.mf = mf
        self.switch_at = switch_at
        self.rn = np.inf
        self.ever_on = False
        self.n_off_calls = 0
        self.cleanup_runs = 0
        self._hooks = HookSet()
        self._attached = False
        self._in_cleanup = False

    def _set_rn(self, rn):
        self.rn = rn

    def attach(self):
        if self._attached:
            return self
        self._attached = True
        if self.switch_at is None:
            self.switch_at = float(np.sqrt(self.mf.conv_tol)) * 10
        mf = self.mf
        hooks = self._hooks

        track_residual(hooks, mf, self._set_rn)

        def nlc_factory(orig):
            def nr_nlc_vxc(mol, grids, xc, dm, *a, **k):
                if not self.ever_on and not self._in_cleanup:
                    self.n_off_calls += 1
                    return 0.0, 0.0, np.zeros_like(np.asarray(dm))
                return orig(mol, grids, xc, dm, *a, **k)
            return nr_nlc_vxc
        ni = getattr(mf, '_numint', None)
        if ni is not None:
            hooks.wrap(ni, 'nr_nlc_vxc', nlc_factory)

        def veff_factory(orig):
            def get_veff(mol=None, dm=None, dm_last=None, vhf_last=None,
                         hermi=1, *a, **k):
                if not self.ever_on and self.rn <= self.switch_at:
                    self.ever_on = True   # latch: stays on for the rest of the SCF
                return orig(mol, dm, dm_last, vhf_last, hermi, *a, **k)
            return get_veff
        hooks.wrap(mf, 'get_veff', veff_factory)

        def kernel_factory(orig):
            def kernel(dm0=None, *a, **k):
                e = orig(dm0=dm0, *a, **k)
                if (not self.ever_on and not self._in_cleanup
                        and getattr(mf, 'converged', False)):
                    self._in_cleanup = True
                    try:
                        self.ever_on = True
                        e = orig(dm0=mf.make_rdm1(), *a, **k)
                        self.cleanup_runs += 1
                    finally:
                        self._in_cleanup = False
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
