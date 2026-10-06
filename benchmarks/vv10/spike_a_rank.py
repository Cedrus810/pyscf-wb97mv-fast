"""S3b spike A: rank of the (2D-interpolated) original VV10 kernel.

Spec section 7.5.  No SCF, no GPU -- pure NumPy.

1. Tabulate Kh = -Phi on a (q, kappa) node grid for a log-spaced R grid.
2. Per-R SVD: r(eps) -- how many shared radial channels the kernel needs at
   relative Frobenius tail eps in {1e-6, 1e-5, 1e-4}.
3. Symmetric CP-ALS at several ranks: the shared-factor model
   Kh[R] ~= sum_s A_s(p) A_s(p') f_s(R); report the achieved relative error
   per rank (r(eps) for the CP form).
4. 2D interpolation sizes: bilinear interpolation of the kernel parameters
   on (log q, log kappa); the smallest (M_q, M_k) whose max relative kernel
   interpolation error stays below eps.

Usage:  python spike_a_rank.py [--mq 48 --mk 32 --out name.json]
Output: benchmarks/results/spike_a_rank.json (+ stdout summary).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vv10sep as sep  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from benchmarks._paths import results_dir  # noqa: E402

# generous physical ranges (spike B logs the actual water27 ranges):
# kappa = Kvv rho^(1/6), Kvv ~ 15.9, rho in [1e-4, 10] -> kappa ~ [4, 24];
# q = W0/kappa, W0 ~ [0.3, 5] -> q ~ [0.02, 1]
Q_RANGE = (0.02, 2.0)
K_RANGE = (2.0, 40.0)
R_GRID = np.geomspace(0.05, 30.0, 60)
EPS_LIST = (1e-6, 1e-5, 1e-4)
CP_RANKS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


def interp_error_test(mq, mk, n_test=200, seed=3):
    """Max relative error of bilinear (log q, log k) interpolation of Phi
    over random (q, k, q', k', R) probes."""
    rng = np.random.default_rng(seed)
    q_fine, k_fine = sep.node_grid(Q_RANGE, K_RANGE, 64, 48)
    q_n, k_n = sep.node_grid(Q_RANGE, K_RANGE, mq, mk)
    # node values of the kernel at one probe's partner parameters
    errs = []
    for _ in range(n_test):
        q2 = np.exp(rng.uniform(np.log(Q_RANGE[0]), np.log(Q_RANGE[1])))
        k2 = np.exp(rng.uniform(np.log(K_RANGE[0]), np.log(K_RANGE[1])))
        R = np.exp(rng.uniform(np.log(R_GRID[0]), np.log(R_GRID[-1])))
        # exact kernel over the fine grid
        Phi_fine = sep.phi_points(R ** 2, q_fine[:, None], k_fine[:, None],
                                  q2, k2).ravel()
        # bilinear from the coarse nodes
        idx, w = sep.bilinear_weights(q_n, k_n, q_fine, k_fine)
        Phi_coarse_nodes = sep.phi_points(R ** 2, q_n[idx], k_n[idx], q2, k2)
        approx = (Phi_coarse_nodes * w).sum(axis=1)
        scale = np.abs(Phi_fine).max()
        errs.append(np.abs(approx - Phi_fine).max() / scale)
    return float(np.max(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mq', type=int, default=24)
    ap.add_argument('--mk', type=int, default=16)
    args = ap.parse_args()

    q_n, k_n = sep.node_grid(Q_RANGE, K_RANGE, args.mq, args.mk)
    Kh = sep.kernel_tensor(R_GRID, q_n, k_n)
    P = len(q_n)
    print(f'node grid: M_q={args.mq} M_k={args.mk} -> P={P}; '
          f'|R|={len(R_GRID)}; tensor {Kh.size * 8 / 1e6:.1f} MB')

    # 1. per-R SVD ranks
    ranks = sep.svd_ranks_per_R(Kh, EPS_LIST)
    svd_summary = {str(eps): dict(median=int(np.median(ranks[eps])),
                                  max=int(ranks[eps].max()))
                   for eps in EPS_LIST}
    print('per-R SVD rank r(eps):', json.dumps(svd_summary))

    # 2. shared-factor CP: error vs rank
    cp = {}
    for r in CP_RANKS:
        A, F, err = sep.cp_als(Kh, r, seed=0)
        cp[r] = err
        print(f'  CP rank {r:3d}: rel err {err:.3e}')
        if err < min(EPS_LIST):
            break

    # 3. interpolation grid sizes
    interp = {}
    for mq in (4, 6, 8, 12, 16, 24, 32):
        row = {}
        for mk in (3, 4, 6, 8, 12, 16, 24):
            row[mk] = interp_error_test(mq, mk)
        interp[mq] = row
        print(f'  interp M_q={mq:3d}: ' +
              ' '.join(f'M_k={mk}:{row[mk]:.1e}' for mk in row))
    min_sizes = {}
    for eps in EPS_LIST:
        best = None
        for mq, row in interp.items():
            for mk, err in row.items():
                if err <= eps and (best is None or mq * mk < best[0] * best[1]):
                    best = (mq, mk)
        min_sizes[str(eps)] = best

    out = dict(q_range=Q_RANGE, k_range=K_RANGE,
               r_grid=R_GRID.tolist(), node_grid=[args.mq, args.mk],
               svd_rank_per_eps=svd_summary,
               svd_rank_per_R={str(eps): ranks[eps].tolist()
                               for eps in EPS_LIST},
               cp_error_per_rank={str(r): e for r, e in cp.items()},
               interp_max_err={str(mq): {str(mk): e for mk, e in row.items()}
                               for mq, row in interp.items()},
               min_interp_sizes=min_sizes)
    path = os.path.join(results_dir(), 'spike_a_rank.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', path)
    print('summary: SVD', json.dumps(svd_summary),
          '| CP r(1e-6)=', min((r for r, e in cp.items() if e <= 1e-6),
                               default=None),
          '| interp sizes', json.dumps(min_sizes))


if __name__ == '__main__':
    main()
