#!/usr/bin/env python
"""P4 gate: staged VV10 vs full-SCF VV10.

    python benchmarks/experimental/cosx/bench_vv10_staged.py water27

Gate (P4): saves >= 3% of SCF wall time at |dE| <= 1e-6 Ha.
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

from pyscf_wb97mv_fast.core.profiling import ScfProfiler  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402
from pyscf_wb97mv_fast.experimental.cosx.vv10_staged import StagedVV10  # noqa: E402

DE_MAX = 1e-6
MIN_SAVING = 0.03


def run(mol, conv_tol, staged):
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    mf.conv_tol = conv_tol
    prof = ScfProfiler()
    extra = {}
    with prof.attach(mf):
        t0 = time.perf_counter()
        if staged:
            with StagedVV10(mf).attach() as st:
                e = mf.kernel()
            extra = dict(n_off_calls=st.n_off_calls,
                         cleanup_runs=st.cleanup_runs)
        else:
            e = mf.kernel()
        wall = time.perf_counter() - t0
    return dict(staged=staged, e_tot=float(e), converged=bool(mf.converged),
                cycles=int(mf.cycles), wall_s=wall,
                vv10_s=prof.timings.get('vv10', 0.0),
                scf_total=prof.timings.get('scf_total', wall), **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    args = ap.parse_args()
    mol = build_mol(args.system, args.basis)
    full = run(mol, args.conv_tol, staged=False)
    staged = run(mol, args.conv_tol, staged=True)
    de = abs(staged['e_tot'] - full['e_tot'])
    saving = 1 - staged['wall_s'] / full['wall_s']
    gate = dict(dE=de, wall_saving=saving,
                passed=bool(de <= DE_MAX and saving >= MIN_SAVING))
    out = dict(info=dict(system=args.system, basis=args.basis, nao=mol.nao,
                         threads=lib.num_threads(), conv_tol=args.conv_tol),
               full=full, staged=staged, gate=gate)
    tag = os.path.splitext(os.path.basename(args.system))[0] + '_' + args.basis
    path = os.path.join(_paths.results_dir(), f'bench_vv10_staged_{tag}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f"full   e={full['e_tot']:.10f} wall={full['wall_s']:.1f}s "
          f"vv10={full['vv10_s']:.1f}s")
    print(f"staged e={staged['e_tot']:.10f} wall={staged['wall_s']:.1f}s "
          f"vv10={staged['vv10_s']:.1f}s off_cycles={staged['n_off_calls']} "
          f"cleanup={staged['cleanup_runs']}")
    print(f"P4: |dE|={de:.1e} (<= {DE_MAX}), saving={100 * saving:.1f}% "
          f"(>= {100 * MIN_SAVING:.0f}%) -> {gate['passed']}")
    print('wrote', path)


if __name__ == '__main__':
    main()
