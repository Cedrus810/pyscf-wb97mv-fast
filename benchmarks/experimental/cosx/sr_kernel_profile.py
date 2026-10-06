#!/usr/bin/env python
"""Single-thread cost per ESP integral (int1e_grids) for full / LR / SR kernels,
by shell angular momentum class and grid-point distance from the pair center.

    python benchmarks/experimental/cosx/sr_kernel_profile.py water27
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))  # benchmarks/, for _paths
import _paths  # noqa: E402
_paths.add_repo_to_syspath()

from pyscf import lib  # noqa: E402

from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402

RBINS = ((0, 2), (2, 5), (5, 10), (10, 20))  # bohr


def _pick_pairs(mol):
    """One representative shell pair per (li, lj, same_atom) class, li <= lj <= 2."""
    coords = mol.atom_coords()
    picks = {}
    for i in range(mol.nbas):
        for j in range(i, mol.nbas):
            li, lj = mol.bas_angular(i), mol.bas_angular(j)
            if lj > 2:
                continue
            ai, aj = mol.bas_atom(i), mol.bas_atom(j)
            same = ai == aj
            if not same and np.linalg.norm(coords[ai] - coords[aj]) > 3.0:
                continue  # keep only bonded-neighbour pairs for the cross-atom class
            picks.setdefault((li, lj, same), (i, j))
    return picks


def _points_in_shell(center, r_lo, r_hi, npts, rng):
    v = rng.normal(size=(npts, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    r = rng.uniform(r_lo, r_hi, size=npts)
    return center + v * r[:, None]


def profile_classes(mol, omega, npts=20000, repeat=3, rbins=RBINS):
    rng = np.random.default_rng(0)
    coords = mol.atom_coords()
    nf = np.diff(mol.ao_loc_nr())
    rows = []
    with lib.with_omp_threads(1):
        for (li, lj, same), (i, j) in sorted(_pick_pairs(mol).items()):
            center = 0.5 * (coords[mol.bas_atom(i)] + coords[mol.bas_atom(j)])
            for r_lo, r_hi in rbins:
                grid = _points_in_shell(center, r_lo, r_hi, npts, rng)
                nint = npts * nf[i] * nf[j]
                for kname, om in (('full', 0.0), ('LR', omega), ('SR', -omega)):
                    with mol.with_range_coulomb(om):
                        mol.intor('int1e_grids', grids=grid, shls_slice=(i, i + 1, j, j + 1))
                        t0 = time.perf_counter()
                        for _ in range(repeat):
                            mol.intor('int1e_grids', grids=grid,
                                      shls_slice=(i, i + 1, j, j + 1))
                        dt = (time.perf_counter() - t0) / repeat
                    rows.append(dict(li=li, lj=lj, same_atom=bool(same), r_lo=r_lo, r_hi=r_hi,
                                     kernel=kname, ns_per_int=dt / nint * 1e9))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--npts', type=int, default=20000)
    args = ap.parse_args()
    mol = build_mol(args.system, args.basis)
    rows = profile_classes(mol, 0.3, npts=args.npts)
    tag = os.path.splitext(os.path.basename(args.system))[0] + '_' + args.basis
    out = os.path.join(_paths.results_dir(), f'sr_kernel_{tag}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(rows, fh, indent=1)
    by = {}
    for r in rows:
        by.setdefault((r['li'], r['lj'], r['same_atom'], r['r_lo'], r['r_hi']), {})[r['kernel']] = r['ns_per_int']
    print(f"{'li lj same r(bohr)':22s} {'full':>7s} {'LR':>7s} {'SR':>7s} {'SR/LR':>6s} {'SR/full':>7s}")
    for (li, lj, same, lo, hi), d in sorted(by.items()):
        print(f"{li:2d} {lj:2d} {str(same):5s} {lo:4.0f}-{hi:<4.0f}      "
              f"{d['full']:7.1f} {d['LR']:7.1f} {d['SR']:7.1f} "
              f"{d['SR']/d['LR']:6.2f} {d['SR']/d['full']:7.2f}")
    print('wrote', out)


if __name__ == '__main__':
    main()
