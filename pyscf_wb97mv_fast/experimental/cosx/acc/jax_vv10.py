"""FROZEN (2026-09-30): gates failed, see benchmarks/experimental/cosx/FINDINGS.md section 9; not imported by mainline.

Low-rank VV10 nonlocal correlation on JAX (P6), with numpy reference.

Kernel and energy exactly as pyscf/dft/numint.py:_vv10nlc (Vydrov & Van
Voorhuis, JCTC 6, 1119 (2010)), with one grid serving as both the outer and
the inner grid, q_i = rho_i * w_i, and W0, K the per-point local quantities:

    g_ij  = R2_ij * W0_i + K_i
    gp_ij = R2_ij * W0_j + K_j
    M_ij  = 1 / (g_ij * gp_ij * (g_ij + gp_ij))
    F_i   = -1.5 * sum_j q_j M_ij                     (pyscf: F *= -1.5)
    exc_i = Beta + 0.5 * F_i                          (pyscf: Beta + .5*F)
    E_nl  = sum_i q_i exc_i = Beta * sum(q) - 0.75 * q^T M q
    Beta  = (3 / b^2)^(3/4) / 32

M is symmetric with positive entries but NOT necessarily positive definite, so
the low-rank form keeps the eigenpairs with the largest |lambda|:
q^T M q ~= z^T Lambda_r z, z = U_r^T q -- O(N_g r) instead of O(N_g^2)
(original plan, section 7).

pyscf_energy() calls pyscf's own _vv10nlc and is the validation reference.
numpy is the reference implementation; jax (lazy import) provides the device
path. Points with rho < 1e-8 are dropped, as in _vv10nlc.
"""
import numpy as np

THRESH = 1e-8


def beta(b):
    return (3.0 / (b * b)) ** 0.75 / 32.0


def local_quantities(rho, grad_rho2, b, C):
    """W0, K per point. grad_rho2 = |grad rho|^2. Matches _vv10nlc."""
    Pi43 = 4 * np.pi / 3
    Kvv = b * 1.5 * np.pi * (9 * np.pi) ** (-1 / 6)
    W0 = np.sqrt(C * (grad_rho2 / rho**2) ** 2 + Pi43 * rho)
    K = Kvv * rho ** (1 / 6)
    return W0, K


def build_kernel(coords, rho, grad_rho2, b, C, threshold=THRESH, backend='numpy',
                 chunk=2048):
    """Dense M (N_kept x N_kept) on the rho >= threshold points, plus the mask.

    backend='jax' uses the jax device for the chunked products (falls back to
    numpy if jax is unavailable).
    """
    rho = np.asarray(rho)
    mask = rho >= threshold
    c = np.asarray(coords)[mask]
    W0, K = local_quantities(rho[mask], np.asarray(grad_rho2)[mask], b, C)
    n = len(c)
    M = np.empty((n, n))
    xp = _xp(backend)
    for i0 in range(0, n, chunk):
        R2 = _sqdist(xp, xp.asarray(c[i0:i0 + chunk]), xp.asarray(c))
        g = R2 * xp.asarray(W0[i0:i0 + chunk])[:, None] + xp.asarray(K[i0:i0 + chunk])[:, None]
        gp = R2 * xp.asarray(W0)[None, :] + xp.asarray(K)[None, :]
        M[i0:i0 + chunk] = np.asarray(1.0 / (g * gp * (g + gp)))
    return M, mask


def energy(M, q, b):
    """E_nl = Beta*sum(q) - 0.75 q^T M q  (q restricted to the kernel mask)."""
    return float(beta(b) * q.sum() - 0.75 * q @ M @ q)


def low_rank(M, tol=1e-8, min_rank=1):
    """(U_r, lam_r): eigenpairs with the largest |lambda| covering 1-tol of
    sum(|lambda|)."""
    lam, U = np.linalg.eigh(M)
    order = np.argsort(-np.abs(lam))
    lam, U = lam[order], U[:, order]
    tail = np.cumsum(np.abs(lam)) / np.abs(lam).sum()
    rank = max(min_rank, int(np.searchsorted(tail, 1 - tol)) + 1)
    rank = min(rank, len(lam))
    return U[:, :rank], lam[:rank]


def energy_low_rank(U, lam, q, b):
    z = U.T @ q
    return float(beta(b) * q.sum() - 0.75 * z @ (lam * z))


def pyscf_energy(coords, rho, grad_rho2, weights, b, C):
    """E_nl from pyscf's _vv10nlc on the same grid (validation reference)."""
    from pyscf.dft.numint import _vv10nlc
    rho4 = np.zeros((4, len(rho)))
    rho4[0] = rho
    rho4[1] = np.sqrt(grad_rho2)          # |grad rho| along x; only |.|^2 enters
    coords = np.ascontiguousarray(coords, dtype=float)
    exc, _ = _vv10nlc(rho4, coords, rho4, np.asarray(weights, dtype=float), coords, (b, C))
    return float(np.dot(rho * weights, exc))


def _xp(backend):
    if backend == 'jax':
        try:
            import jax
            import jax.numpy as jnp
            # FP64 throughout (plan section 3); jax defaults to float32
            jax.config.update('jax_enable_x64', True)
            return jnp
        except ImportError:
            pass
    return np


def _sqdist(xp, a, b):
    """|a_i - b_j|^2 as (len(a), len(b))."""
    d2 = (a**2).sum(axis=1)[:, None] + (b**2).sum(axis=1)[None, :] - 2 * a @ b.T
    return xp.maximum(d2, 0.0)
