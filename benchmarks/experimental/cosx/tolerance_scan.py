#!/usr/bin/env python
"""Real-bound SGX tolerance scan for K_full / K_LR / K_SR (one K build each).

For each kernel and each sgx_tol_energy (sgx_tol_potential='auto' = sqrt(etol)):
tasks / ESP integrals / integral time / wall, screening error vs the
DM-screening-off reference, and (with --exact) the COSX grid error vs the
analytic K.

    python tolerance_scan.py water27 --exact
    python tolerance_scan.py water64
"""
import argparse
import json
import os
import time

import numpy as np
from pyscf import dft, lib
from pyscf.scf import hf

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling sgx_locality.py
from sgx_locality import HERE, Shim, build_mol, get_dm, make_sgx, run_k

DIRECT_SCF_TOL = 1e-13
ETOLS = [1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]


def _dE(dm, dk):
    return float(0.25 * abs(np.einsum('ij,ji', dm, dk)))


def scan(mol, dm, etols=ETOLS, exact=False):
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    omega, _, _ = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    base = mf.with_df
    base.build()
    shim = Shim(mol)
    out = {}
    for kname, om in (('full', 0.0), ('LR', omega), ('SR', -omega)):
        ref = make_sgx(base, om, 'sample_pos', dm_screening=False)
        k_ref = run_k(ref, om, dm, DIRECT_SCF_TOL)
        grid = None
        if exact:
            k_exact = hf.get_jk(mol, dm, hermi=1, with_j=False, with_k=True,
                                omega=None if om == 0 else om)[1]
            grid = dict(dE_grid=_dE(dm, k_ref - k_exact),
                        dK_grid=float(abs(k_ref - k_exact).max()))
        rows = []
        for etol in etols:
            r = make_sgx(base, om, 'sample_pos', etol=etol)
            run_k(r, om, dm, DIRECT_SCF_TOL)          # warm-up: builds bounds
            shim.start(record=False)
            t0 = time.perf_counter()
            k = run_k(r, om, dm, DIRECT_SCF_TOL)
            wall = time.perf_counter() - t0
            cnt, _ = shim.stop()
            rows.append(dict(etol=etol, tasks=cnt['calls'], ints=cnt['ints'],
                             t_int_thread_s=cnt['t_int_thread_s'], wall_s=wall,
                             dE_screen=_dE(dm, k - k_ref),
                             dK_screen=float(abs(k - k_ref).max())))
        out[kname] = dict(rows=rows, grid=grid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--exact', action='store_true',
                    help='also compute analytic K for the grid-error reference')
    args = ap.parse_args()
    tag = os.path.splitext(os.path.basename(args.system))[0] + '_' + args.basis
    mol = build_mol(args.system, args.basis)
    res = scan(mol, get_dm(mol, tag), exact=args.exact)
    res['info'] = dict(system=args.system, basis=args.basis, nao=mol.nao,
                       threads=lib.num_threads())
    with open(os.path.join(HERE, f'tolscan_{tag}.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    for kname in ('full', 'LR', 'SR'):
        rows, grid = res[kname]['rows'], res[kname]['grid']
        t0 = rows[0]['t_int_thread_s']
        g = f"  grid dE={grid['dE_grid']:.1e} dKmax={grid['dK_grid']:.1e}" if grid else ''
        print(f'[{kname}]{g}')
        print(f"  {'etol':>7s} {'tasks(M)':>9s} {'ints(G)':>8s} {'t_int/t_int(1e-13)':>18s} "
              f"{'wall(s)':>8s} {'dE_screen':>10s} {'dKmax':>9s}"
              + ('  dE_screen/dE_grid' if grid else ''))
        for r in rows:
            ratio = f"  {r['dE_screen'] / grid['dE_grid']:.1e}" if grid else ''
            print(f"  {r['etol']:7.0e} {r['tasks']/1e6:9.2f} {r['ints']/1e9:8.2f} "
                  f"{r['t_int_thread_s']/t0:18.3f} {r['wall_s']:8.2f} "
                  f"{r['dE_screen']:10.1e} {r['dK_screen']:9.1e}{ratio}")


if __name__ == '__main__':
    main()
