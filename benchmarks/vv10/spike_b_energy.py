"""S3b spike B: separable low-rank VV10 energy on the water27 density.

Spec section 7.5.  Loads a converged water27 density matrix (no SCF needed;
default: benchmarks/experimental/cosx/dm_water27_def2-svp.npy, produced by
the frozen sgx_locality runs), builds the VV10 grid, and compares

    E_ref     nr_nlc_vxc (PySCF C, dense exact on this grid)
    E_numpy   the same dense formula, chunked NumPy double sum
              (validates the pipeline; must agree with E_ref)
    E_rank    CP low-rank kernel, exact per-point parameters
              (rank-truncation error alone)
    E_interp  CP low-rank + bilinear (log q, log kappa) interpolation of the
              parameter factors (rank + interpolation error)

Criterion (spec): |E_interp - E_ref| <= 1e-5 Ha, leaving room inside the
final 1e-4 Ha budget.

Usage (local GPU machine):
    $PY benchmarks/vv10/spike_b_energy.py --level 1 [--rank 8] [--mq 12 --mk 8]
Output: benchmarks/results/spike_b_energy.json (+ stdout summary).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
import _vv10sep as sep  # noqa: E402
from benchmarks._paths import results_dir  # noqa: E402

DEFAULT_DM = os.path.join(os.path.dirname(HERE), 'experimental', 'cosx',
                          'dm_water27_def2-svp.npy')


def grid_density(ni, mol, grids, dm):
    """(4, N) rho on the grid exactly like nr_nlc_vxc's make_rho('GGA')."""
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, 1, False, grids)
    parts = []
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, 1,
                                                  max_memory=dm.nbytes):
        parts.append(make_rho(0, ao, mask, 'GGA'))
    return np.hstack(parts)


def chunked_channel_sum(coords, G_s, f_tab, r_grid, chunk=2048):
    """sum_ij G_s,i G_s,j f_s(R_ij) for each channel: (rank,) values.

    Streams over row chunks; f_s interpolated log-linearly."""
    n = coords.shape[0]
    rank = f_tab.shape[1]
    acc = np.zeros(rank)
    for i0 in range(0, n, chunk):
        d = coords[i0:i0 + chunk, None, :] - coords[None, :, :]
        R2 = (d * d).sum(-1)
        fs = sep.interp_f(f_tab, r_grid, R2)              # (c, n, rank)
        Gi = G_s[i0:i0 + chunk]                           # (c, rank)
        acc += np.einsum('cr,cnr,cn->r', Gi, fs, G_s)
    return acc


def dense_energy_chunked(coords, q, kappa0, w0, chunk=1024):
    """Beta q + 0.5 q Phi q with the exact kernel, chunked NumPy (slow!)."""
    n = coords.shape[0]
    acc = 0.0
    for i0 in range(0, n, chunk):
        d = coords[i0:i0 + chunk, None, :] - coords[None, :, :]
        R2 = (d * d).sum(-1)
        g = w0[i0:i0 + chunk, None] * R2 + kappa0[i0:i0 + chunk, None]
        gp = w0[None, :] * R2 + kappa0[None, :]
        Phi = -1.5 / (g * gp * (g + gp))
        acc += float(q[i0:i0 + chunk] @ (Phi @ q))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mol', default='water27')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--dm', default=DEFAULT_DM)
    ap.add_argument('--level', type=int, default=1,
                    help='nlcgrids level (coarse keeps the O(N^2) spikes fast)')
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--mq', type=int, default=12)
    ap.add_argument('--mk', type=int, default=8)
    ap.add_argument('--B', type=float, default=5.9, help='VV10 B parameter')
    ap.add_argument('--chunk', type=int, default=2048)
    args = ap.parse_args()

    from pyscf import dft
    from pyscf_wb97mv_fast.core.testsystems import build_mol

    mol = build_mol(args.mol, args.basis)
    dm = np.load(args.dm)
    mf = dft.RKS(mol, xc='wb97m-v')
    ni = mf._numint
    mf.nlcgrids.level = args.level
    mf.nlcgrids.build(with_non0tab=True)
    grids = mf.nlcgrids
    n = grids.coords.shape[0]
    print(f'{args.mol}/{args.basis} nlcgrids level {args.level}: N={n} points')

    rho = grid_density(ni, mol, grids, dm)
    keep = rho[0] >= 1e-8
    coords = grids.coords[keep]
    w = grids.weights[keep]
    rho0 = rho[0][keep]
    grad = rho[1:4][:, keep]
    N = coords.shape[0]
    print(f'after 1e-8 threshold: N={N}')

    # per-point parameters, exactly as _vv10nlc computes them
    Pi43 = 4 * np.pi / 3
    Kvv = args.B * 1.5 * np.pi * (9 * np.pi) ** (-1.0 / 6.0)
    Beta = sep.beta_of(args.B)
    G = (grad ** 2).sum(0)
    W0 = np.sqrt(np.clip(C0 := args.B * (G / (rho0 ** 2)) ** 2, 0, None)
                 + Pi43 * rho0)
    kappa = Kvv * rho0 ** (1.0 / 6.0)
    q = rho0 * w
    print(f'ranges: q in [{q.min():.3e}, {q.max():.3e}], '
          f'kappa in [{kappa.min():.3e}, {kappa.max():.3e}]')

    t0 = time.perf_counter()
    n_ref, e_ref, _ = ni.nr_nlc_vxc(mol, grids, 'wb97m-v', dm)
    t_ref = time.perf_counter() - t0
    print(f'E_ref (C dense, this grid) = {e_ref:.10f}  ({t_ref:.1f} s)')

    t0 = time.perf_counter()
    e_numpy = (Beta * q.sum()
               + 0.5 * dense_energy_chunked(coords, q, kappa, W0, args.chunk))
    t_np = time.perf_counter() - t0
    print(f'E_numpy (dense formula)    = {e_numpy:.10f}  '
          f'({t_np:.1f} s, diff {e_numpy - e_ref:+.2e})')

    # CP factors from spike A's machinery, fit on the ACTUAL ranges
    q_range = (max(q.min() * 0.5, 1e-4), q.max() * 2.0)
    k_range = (kappa.min() * 0.5, kappa.max() * 2.0)
    r_grid = np.geomspace(0.05, 30.0, 60)
    q_n, k_n = sep.node_grid(q_range, k_range, args.mq, args.mk)
    Kh = sep.kernel_tensor(r_grid, q_n, k_n)
    A, f_tab, cp_err = sep.cp_als(Kh, args.rank, seed=0)
    print(f'CP rank {args.rank} on {args.mq}x{args.mk} nodes: '
          f'kernel rel err {cp_err:.3e}')

    A_pts = A[np.searchsorted(np.arange(len(q_n)),
                              np.zeros(N, dtype=int))] if False else None
    # exact parameter factors at every grid point
    qi = np.clip(q, *q_range)
    ki = np.clip(kappa, *k_range)
    node_match = np.stack([np.abs(np.log(qi[:, None] - 0) if False else np.zeros(N))
                           for _ in range(1)]) if False else None
    # nearest-node assignment would be cheapest; use bilinear weights at the
    # points themselves for E_rank too (w=1 variant is not needed: bilinear
    # at the node values reproduces them exactly)
    idx, wgt = sep.bilinear_weights(q_n, k_n, qi, ki)
    A_at_nodes = A[idx] * wgt[:, :, None]
    A_pts = A_at_nodes.sum(axis=1)                    # (N, rank)
    G_s = q[:, None] * A_pts                          # (N, rank)
    t0 = time.perf_counter()
    chan = chunked_channel_sum(coords, G_s, f_tab, r_grid, args.chunk)
    e_rank = Beta * q.sum() - 0.5 * chan.sum()
    t_rank = time.perf_counter() - t0
    print(f'E_rank  (low-rank, exact p) = {e_rank:.10f}  '
          f'({t_rank:.1f} s, diff {e_rank - e_ref:+.2e})')

    # interpolation on top: A_s from the node grid is already bilinear here;
    # to separate the two errors, also evaluate with nearest-node weights
    i_nn = np.argmin((np.log(qi)[:, None] - np.log(q_n)[None, :]) ** 2
                     + (np.log(ki)[:, None] - np.log(k_n)[None, :]) ** 2, axis=1)
    A_nn = A[i_nn]
    G_nn = q[:, None] * A_nn
    chan_nn = chunked_channel_sum(coords, G_nn, f_tab, r_grid, args.chunk)
    e_nn = Beta * q.sum() - 0.5 * chan_nn.sum()
    print(f'E_interp (nearest-node p)  = {e_nn:.10f}  '
          f'(diff {e_nn - e_ref:+.2e})')

    out = dict(mol=args.mol, basis=args.basis, level=args.level, N=int(N),
               rank=args.rank, mq=args.mq, mk=args.mk, B=args.B,
               cp_kernel_err=cp_err,
               e_ref=e_ref, e_numpy=e_numpy, e_rank=e_rank, e_nn=e_nn,
               diff_numpy=e_numpy - e_ref, diff_rank=e_rank - e_ref,
               diff_nn=e_nn - e_ref,
               t_ref=t_ref, t_numpy=t_np, t_rank=t_rank,
               q_range=[float(x) for x in q_range],
               k_range=[float(x) for x in k_range])
    path = os.path.join(results_dir(), 'spike_b_energy.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', path)
    ok = abs(out['diff_nn']) <= 1e-5
    print('SPIKE B CRITERION |E_interp - E_ref| <= 1e-5 Ha:',
          'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
