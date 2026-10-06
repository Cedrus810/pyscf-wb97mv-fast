#!/usr/bin/env python
"""P3 benchmark: reference (A path) vs B-path-only vs DynamicAB on full SCF.

    python benchmarks/experimental/cosx/bench_dynamic_ab.py water27
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))  # benchmarks/, for _paths
import _paths  # noqa: E402
_paths.add_repo_to_syspath()

from pyscf import dft, lib  # noqa: E402

from pyscf_wb97mv_fast.core import sgx_patch  # noqa: E402
from pyscf_wb97mv_fast.experimental.cosx.dynamic_ab import DynamicAB  # noqa: E402
from pyscf_wb97mv_fast.experimental.cosx.sr_screening import BPath  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402

DE_MAX = 1e-8   # operator identity: decomposition must not move the energy


def run(mol, conv_tol, mode):
    sgx_patch.apply()
    try:
        mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
        mf.conv_tol = conv_tol
        t0 = time.perf_counter()
        if mode == 'reference':
            e = mf.kernel()
        elif mode == 'B_path':
            with BPath(mf).attach():
                e = mf.kernel()
        else:
            with DynamicAB(mf).attach() as dyn:
                e = mf.kernel()
        wall = time.perf_counter() - t0
        res = dict(mode=mode, e_tot=float(e), converged=bool(mf.converged),
                   cycles=int(mf.cycles), wall_s=wall, dm=mf.make_rdm1())
        if mode == 'dynamic':
            res['switch_cycle'] = dyn.switch_cycle
        return res
    finally:
        sgx_patch.revert()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    args = ap.parse_args()
    mol = build_mol(args.system, args.basis)
    results = {}
    for mode in ('reference', 'B_path', 'dynamic'):
        results[mode] = run(mol, args.conv_tol, mode)
        r = results[mode]
        print(f"{mode:10s} e={r['e_tot']:.10f} cyc={r['cycles']} "
              f"wall={r['wall_s']:.1f}s", flush=True)
    ref = results['reference']
    for mode in ('B_path', 'dynamic'):
        r = results[mode]
        r['dE'] = abs(r['e_tot'] - ref['e_tot'])
        r['dP_max'] = float(abs(r.pop('dm') - ref['dm']).max())
        r['wall_ratio'] = r['wall_s'] / ref['wall_s']
        print(f"{mode:10s} dE={r['dE']:.1e} wall/ref={r['wall_ratio']:.3f}")
    results['reference'].pop('dm')
    tag = os.path.splitext(os.path.basename(args.system))[0] + '_' + args.basis
    out = os.path.join(_paths.results_dir(), f'bench_dynamic_ab_{tag}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(dict(info=dict(system=args.system, basis=args.basis,
                                 nao=mol.nao, threads=lib.num_threads()),
                       results=results), fh, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
