"""FROZEN (2026-09-30): gates failed, see benchmarks/experimental/cosx/FINDINGS.md section 9; not imported by mainline.

P5: gradient fast path.

Upstream status in PySCF 2.14: SGX analytic gradients exist in
pyscf/sgx/grad/{rhf,rks,uhf,uks}.py. _SGXHF.nuc_grad_method() dispatches
there (sgx/sgx.py:241), and the RSH branch of sgx/grad/rhf.py:get_jk handles
omega != 0, so wB97M-V / COSX(pjs=True) has an SGX-consistent gradient.
Known upstream restrictions: with_df.direct_j=True raises ValueError in the
gradient mixin. UPSTREAM BUG (2026-09-30): with sgx_grid_response=True (the
default) the SGX gradient is NaN for every system tried (HF, PBE0, wB97X,
wB97M-V; sto-3g and def2-svp). The C routine libdft.VXCgen_grid_lko_deriv
(grad/rks.py:get_dw_partition_sorted, used for the becke_lko SGX grids)
returns NaN at points where some atomic partition value is tiny (<1e-20),
although its inputs are finite. Minimal repro: water, scf.RHF(mol).COSX(pjs=True),
nuc_grad_method().kernel(). sgx_grid_response=False gives finite gradients,
without the SGX grid-weight response.

What this module does NOT cover: the gradient RSH branch is separate upstream
code. sgx_patch does not touch it, and it reuses the RSH copies left in
mf.with_df._rsh_df by the energy calculation. BPath / DynamicAB have no
gradient counterpart, so the gradient is always the A-path gradient.

- check_gradient_support(mf): (supported, reason).
- attach(mf): context that applies sgx_patch and pins the reference (tight)
  SGX tolerance, so the energy that the gradient belongs to is computed at
  reference settings.
- finite_difference_check(mf, dm0=None, disp=1e-3, grid_response=True): P5
  gate utility. Max componentwise |FD(dE) - analytic grad| in Ha/bohr at the
  mf's current settings. grid_response=True turns on the XC/VV10 grid-weight
  response (and the SGX one, on by default upstream); without it FD and
  analytic gradients differ by the grid error, not by a code bug.
"""
import numpy as np

from pyscf_wb97mv_fast.core import sgx_patch
from pyscf_wb97mv_fast.experimental.cosx.tolerance import set_sgx_tolerance


def check_gradient_support(mf):
    """(supported, reason)."""
    from pyscf.sgx import sgx
    df = getattr(mf, 'with_df', None)
    if isinstance(df, sgx.SGX):
        if df.direct_j:
            return False, 'SGX gradients do not support with_df.direct_j=True'
        return True, 'pyscf.sgx.grad (SGX analytic gradients, RSH supported)'
    return True, 'standard pyscf gradients'


def attach(mf):
    """Context: sgx_patch applied + reference tolerance pinned. The patch is
    intentionally left installed on exit (reverting it globally could break
    other users of sgx_patch); the tolerance stays at reference too.
    """
    return _GradCtx(mf)


class _GradCtx:
    def __init__(self, mf):
        self.mf = mf

    def __enter__(self):
        sgx_patch.apply()
        set_sgx_tolerance(self.mf, 'auto')
        return self.mf

    def __exit__(self, *exc):
        return False


def finite_difference_check(mf, dm0=None, disp=1e-3, grid_response=True,
                            sgx_grid_response=True, verbose=0):
    """Max |FD grad - mf grad| (Ha/bohr) by central differences of mf.kernel().

    Displaces every atom along x/y/z by +/- disp (bohr), re-converging from
    dm0 (pass a converged dm for speed). Restores the original geometry and
    resets mf. Returns (max_err, grad_fd, grad_analytic).
    """
    mol = mf.mol
    base = mol.atom_coords()
    natm = mol.natm

    def energy_at(coords):
        mol.set_geom_(coords, unit='Bohr')
        mf.reset(mol)
        e = mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError('SCF not converged at a displaced geometry')
        return e

    try:
        grad_fd = np.zeros((natm, 3))
        for i in range(natm):
            for c in range(3):
                ep, em = base.copy(), base.copy()
                ep[i, c] += disp
                em[i, c] -= disp
                grad_fd[i, c] = (energy_at(ep) - energy_at(em)) / (2 * disp)
        mol.set_geom_(base, unit='Bohr')
        mf.reset(mol)
        mf.verbose = verbose
        mf.kernel(dm0=dm0)
        g = mf.nuc_grad_method()
        if grid_response and hasattr(g, 'grid_response'):
            g.grid_response = True
        if hasattr(g, 'sgx_grid_response'):
            g.sgx_grid_response = sgx_grid_response
        grad = g.kernel()
        if not np.isfinite(grad).all():
            raise FloatingPointError(
                'analytic gradient contains NaN/inf. With COSX and '
                'sgx_grid_response=True this is the upstream PySCF 2.14 bug in '
                'libdft.VXCgen_grid_lko_deriv (see module docstring); '
                'rerun with sgx_grid_response=False')
        err = float(np.abs(grad - grad_fd).max())
        return err, grad_fd, grad
    finally:
        mol.set_geom_(base, unit='Bohr')
        mf.reset(mol)
