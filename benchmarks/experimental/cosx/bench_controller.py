#!/usr/bin/env python
"""G1b gate: error-budget controller vs patched stock (A path).

    python benchmarks/experimental/cosx/bench_controller.py water27
    python benchmarks/experimental/cosx/bench_controller.py /path/system.xyz --basis def2-svp
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

from pyscf_wb97mv_fast.core import sgx_patch  # noqa: E402
from pyscf_wb97mv_fast.experimental.cosx.controller import fast_path  # noqa: E402
from pyscf_wb97mv_fast.core.profiling import ScfProfiler  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402

# G1b thresholds (P1b and plan section 5)
MIN_SPEEDUP = 1.2
DE_MAX = {'production': 1e-8, 'fast': 1e-6}
DP_MAX = {'production': 1e-5, 'fast': None}


def run_mode(mol, mode, conv_tol):
    sgx_patch.revert()
    mf = fast_path(dft_build(mol), mode=mode)
    mf.conv_tol = conv_tol
    prof = ScfProfiler()
    with prof.attach(mf):
        t0 = time.perf_counter()
        e = mf.kernel()
        wall = time.perf_counter() - t0
    ctl = getattr(mf, '_wb97mv_fast_controller', None)
    return dict(mode=mode, e_tot=float(e), converged=bool(mf.converged),
                cycles=int(mf.cycles), wall_s=wall, timings=dict(prof.timings),
                counts=dict(prof.counts), dm=mf.make_rdm1(),
                n_budget_updates=getattr(ctl, 'n_budget_updates', 0),
                budget_history=list(getattr(ctl, 'history', [])),
                cleanup_runs=getattr(ctl, 'cleanup_runs', 0))


def dft_build(mol):
    from pyscf import dft
    return dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    args = ap.parse_args()
    mol = build_mol(args.system, args.basis)
    results = {}
    for mode in ('reference', 'production', 'fast'):
        results[mode] = run_mode(mol, mode, args.conv_tol)
        r = results[mode]
        print(f"{mode:12s} e={r['e_tot']:.10f} cyc={r['cycles']}"
              f"+{r['cleanup_runs']} wall={r['wall_s']:.1f}s "
              f"budgets={r['n_budget_updates']}", flush=True)

    ref = results['reference']
    table = {}
    for mode in ('production', 'fast'):
        r = results[mode]
        speedup = ref['wall_s'] / r['wall_s']
        de = abs(r['e_tot'] - ref['e_tot'])
        dp = float(np.abs(r['dm'] - ref['dm']).max())
        dp_ok = DP_MAX[mode] is None or dp <= DP_MAX[mode]
        table[mode] = dict(speedup=speedup, dE=de, dP_max=dp,
                           passed=bool(speedup >= MIN_SPEEDUP and de <= DE_MAX[mode]
                                       and dp_ok and r['converged']))
    for r in results.values():
        r.pop('dm')
    tag = os.path.splitext(os.path.basename(args.system))[0] + '_' + args.basis
    out = os.path.join(_paths.results_dir(), f'bench_controller_{tag}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(dict(info=dict(system=args.system, basis=args.basis,
                                 nao=mol.nao, threads=lib.num_threads(),
                                 conv_tol=args.conv_tol),
                       results=results, gate=table), fh, indent=1)
    print(f"\n{'mode':12s} {'speedup':>8s} {'dE':>9s} {'dPmax':>9s}  G1b")
    for mode, t in table.items():
        print(f"{mode:12s} {t['speedup']:8.3f} {t['dE']:9.1e} {t['dP_max']:9.1e}  {t['passed']}")
    print('wrote', out)


if __name__ == '__main__':
    main()
