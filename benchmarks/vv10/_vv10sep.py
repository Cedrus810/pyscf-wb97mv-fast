"""S3b spikes: shared machinery for the separable low-rank VV10 kernel.

Used by spike_a_rank.py and spike_b_energy.py (spec section 7.5).  Pure
NumPy; no SCF, no GPU.

Kernel pieces (original VV10, exactly what dft.numint._vv10nlc evaluates):

    g = kappa * (q R^2 + 1)          (= W0 R^2 + K with q = W0/K in PySCF)
    K(R; p, p') = 1 / (g * g' * (g + g'))          positive
    Phi = -1.5 * K                   (what the C kernel's F/U/W carry)

with p = (q, kappa) the per-point parameters.  The energy on a grid is

    E_nl = Beta * sum_i q_i - 0.75 * q^T K q,   q_i = rho_i * w_i
    Beta = ((3/B^2)^0.75)/32

(exc_i = Beta + 0.5 F_i with F_i = sum_j q_j Phi_ij in _vv10nlc.)

Separable model under test (on the positive kernel K):

    K(R; p, p') ~= sum_s A_s(p) * A_s(p') * f_s(R)

A is shared across R (symmetric CP over the two parameter modes, tabulated
on a (q, kappa) node grid); f_s is tabulated on an R grid and interpolated
log-linearly.
"""
import numpy as np


def beta_of(B):
    return ((3.0 / (B * B)) ** 0.75) / 32.0


def phi_points(R2, q, k, q2, k2):
    """Phi at pair arrays: g = k (q R2 + 1); shapes broadcast."""
    g = k * (q * R2 + 1.0)
    g2 = k2 * (q2 * R2 + 1.0)
    return -1.5 / (g * g2 * (g + g2))


def node_grid(q_range, k_range, mq, mk):
    """Flattened (row-major over (q_i, k_j)) node parameters."""
    q = np.geomspace(*q_range, mq)
    k = np.geomspace(*k_range, mk)
    Q, K = np.meshgrid(q, k, indexing='ij')
    return Q.ravel(), K.ravel()


def kernel_tensor(r_grid, q_nodes, k_nodes):
    """K = 1/(g g' (g+g')) on the node grid for every r in r_grid:
    (len(r_grid), P, P), P = len(q_nodes).  Positive kernel."""
    R2 = np.asarray(r_grid) ** 2
    g = k_nodes[None, :] * (q_nodes[None, :] * R2[:, None, None] + 1.0)
    g2 = k_nodes[None, :] * (q_nodes[None, :] * R2[:, None, None] + 1.0)
    return 1.0 / (g[:, :, None] * g2[:, None, :]
                  * (g[:, :, None] + g2[:, None, :]))


def svd_ranks_per_R(Kh, eps_list=(1e-6, 1e-5, 1e-4)):
    """Per R, the smallest rank whose squared-singular-value tail is below
    eps^2 (relative Frobenius tail), i.e. rank r keeps (1-eps^2) of the
    energy.  Returns {eps: array over R}."""
    out = {eps: np.zeros(Kh.shape[0], dtype=int) for eps in eps_list}
    for r in range(Kh.shape[0]):
        s = np.linalg.svd(Kh[r], compute_uv=False)
        tail = (s ** 2)[::-1].cumsum()[::-1]          # tail[r'] = sum_{s>=r'}
        total = tail[0]
        for eps in eps_list:
            out[eps][r] = int(np.searchsorted(-tail, -(eps ** 2) * total)) + 1
    return out


def cp_als(Kh, rank, n_iter=500, seed=0, tol=1e-13):
    """Alternating LS for  Kh[R,i,j] ~= sum_s F[R,s] A[i,s] A[j,s].

    The two parameter modes are tied by alternation: solve mode-j linearly
    (B given F, A), set A <- B, then solve F given A.  Every step is an
    exact linear least-squares solve, so the objective decreases until the
    alternation fixed point (symmetric because the tensor is symmetric).
    Returns (A (P,rank), F (nR,rank), rel_err_final).
    """
    rng = np.random.default_rng(seed)
    nR, P, _ = Kh.shape
    A = rng.normal(size=(P, rank))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    F = np.ones((nR, rank))
    err = np.inf
    for _ in range(n_iter):
        # mode-j (parameter) solve: B given (F, A)
        M = (F.T @ F) * (A.T @ A)
        rhs = np.einsum('rij,rs,is->js', Kh, F, A)        # (P, rank)
        B = rhs @ np.linalg.pinv(M)
        nrm = np.linalg.norm(B, axis=0, keepdims=True)
        nrm[nrm == 0] = 1.0
        A = B / nrm                                       # tie + normalize
        F *= nrm[0] ** 2                                  # scale into f_s
        # mode-r (radial) solve: F given A
        MF = (A.T @ A) * (A.T @ A)
        rhsF = np.einsum('rij,is,js->rs', Kh, A, A)
        F = rhsF @ np.linalg.pinv(MF)
        err = cp_error(Kh, A, F)
        if err < tol:
            break
    return A, F, err


def cp_error(Kh, A, F):
    recon = np.einsum('rs,is,js->rij', F, A, A)
    return float(np.linalg.norm(Kh - recon) / np.linalg.norm(Kh))


def bilinear_weights(q_grid, k_grid, q_pts, k_pts):
    """Bilinear weights on the (log q, log kappa) node grid.

    Returns (idx (N,4), w (N,4)): A_s(p) ~= sum_c w[:,c] A[node idx[:,c], s],
    nodes flattened row-major over (q_i, k_j)."""
    lq, lk = np.log(q_grid), np.log(k_grid)
    x = np.clip(np.interp(np.log(q_pts), lq, np.arange(len(q_grid))),
                0.0, len(q_grid) - 1.0 - 1e-9)
    y = np.clip(np.interp(np.log(k_pts), lk, np.arange(len(k_grid))),
                0.0, len(k_grid) - 1.0 - 1e-9)
    i0, j0 = np.floor(x).astype(int), np.floor(y).astype(int)
    tx, ty = x - i0, y - j0
    mk = len(k_grid)
    idx = np.stack([i0 * mk + j0, (i0 + 1) * mk + j0,
                    i0 * mk + (j0 + 1), (i0 + 1) * mk + (j0 + 1)], axis=1)
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty),
                  (1 - tx) * ty, tx * ty], axis=1)
    return idx, w


def interp_f(f_tab, r_grid, R2):
    """f_s(sqrt(R2)) by linear interpolation in log R: (..., nR2, rank)."""
    r = np.sqrt(np.maximum(np.asarray(R2, dtype=float).ravel(), 1e-300))
    lr = np.log(r)
    lrg = np.log(np.asarray(r_grid))
    out = np.empty((r.size, f_tab.shape[1]))
    for s in range(f_tab.shape[1]):
        out[:, s] = np.interp(lr, lrg, f_tab[:, s])
    return out.reshape(np.shape(R2) + (f_tab.shape[1],))
