#!/usr/bin/env python
"""Full-SCF G1a gate: tight reference vs loosened SGX tolerance (+ tight cleanup).

    python benchmarks/experimental/cosx/scf_tolerance_gate.py water27
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

from pyscf import dft, lib  # noqa: E402

from pyscf_wb97mv_fast.core import sgx_patch  # noqa: E402
from pyscf_wb97mv_fast.core.profiling import ScfProfiler  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402
from pyscf_wb97mv_fast.experimental.cosx.tolerance import set_sgx_tolerance  # noqa: E402

# name, etol during SCF, etol for cleanup (None = no cleanup)
SETTINGS = [
    ('reference', 'auto', None),
    ('etol1e-10', 1e-10, None),
    ('etol1e-9', 1e-9, None),
    ('etol1e-8', 1e-8, None),
    ('etol1e-8+cleanup', 1e-8, 'auto'),
]
# G1a thresholds (plan §4)
DE_NO_CLEANUP = 1e-6
DE_CLEANUP = 1e-8
MAX_EXTRA_CYCLES = 2
MAX_WALL_RATIO = 0.85


def run_setting(mol, etol, cleanup_etol, conv_tol):
    sgx_patch.apply()
    try:
        mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
        mf.conv_tol = conv_tol
        set_sgx_tolerance(mf, etol)
        prof = ScfProfiler()
        cleanup_cycles = 0
        with prof.attach(mf):
            t0 = time.perf_counter()
            e = mf.kernel()
            cycles = mf.cycles
            if cleanup_etol is not None:
                set_sgx_tolerance(mf, cleanup_etol)
                e = mf.kernel(dm0=mf.make_rdm1())
                cleanup_cycles = mf.cycles
            wall = time.perf_counter() - t0
        return dict(e_tot=float(e), cycles=int(cycles), cleanup_cycles=int(cleanup_cycles),
                    converged=bool(mf.converged), wall_s=wall,
                    timings=dict(prof.timings), dm=mf.make_rdm1())
    finally:
        sgx_patch.revert()


def evaluate(results):
    ref = results['reference']
    table = {}
    for name, r in results.items():
        de = abs(r['e_tot'] - ref['e_tot'])
        cleanup = r['cleanup_cycles'] > 0
        extra = r['cycles'] + r['cleanup_cycles'] - ref['cycles']
        wall_ratio = r['wall_s'] / ref['wall_s']
        k = sum(r['timings'].get(x, 0) for x in ('k_full', 'k_lr', 'k_sr'))
        k_ref = sum(ref['timings'].get(x, 0) for x in ('k_full', 'k_lr', 'k_sr'))
        passed = (de <= (DE_CLEANUP if cleanup else DE_NO_CLEANUP)
                  and extra <= MAX_EXTRA_CYCLES and wall_ratio <= MAX_WALL_RATIO)
        table[name] = dict(dE=de, dP_max=float(abs(r['dm'] - ref['dm']).max()),
                           cycles=r['cycles'], cleanup_cycles=r['cleanup_cycles'],
                           extra_cycles=extra, wall_ratio=wall_ratio,
                           k_ratio=k / k_ref if k_ref else None,
                           passes=bool(passed) if name != 'reference' else None)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    args = ap.parse_args()
    mol = build_mol(args.system, args.basis)
    results = {}
    for name, etol, cleanup in SETTINGS:
        results[name] = run_setting(mol, etol, cleanup, args.conv_tol)
        print(f"{name:18s} e={results[name]['e_tot']:.10f} cycles={results[name]['cycles']}"
              f"+{results[name]['cleanup_cycles']} wall={results[name]['wall_s']:.1f}s", flush=True)
    table = evaluate(results)
    for r in results.values():
        r.pop('dm')
    tag = os.path.splitext(os.path.basename(args.system))[0] + '_' + args.basis
    out = os.path.join(_paths.results_dir(), f'scf_gate_{tag}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(dict(info=dict(system=args.system, basis=args.basis, nao=mol.nao,
                                 threads=lib.num_threads(), conv_tol=args.conv_tol),
                       results=results, gate=table), fh, indent=1)
    print(f"\n{'setting':18s} {'dE':>9s} {'dPmax':>9s} {'+cyc':>5s} {'wall':>6s} {'K':>6s}  G1a")
    for name, t in table.items():
        k = f"{t['k_ratio']:6.3f}" if t['k_ratio'] is not None else '     -'
        print(f"{name:18s} {t['dE']:9.1e} {t['dP_max']:9.1e} {t['extra_cycles']:5d} "
              f"{t['wall_ratio']:6.3f} {k}  {t['passes']}")
    print('wrote', out)


if __name__ == '__main__':
    main()
