#!/usr/bin/env python
"""G2b gate evaluator: SR ESP-kernel cost from the P1a microbench output.

Reads the newest benchmarks/results/sr_kernel_*.json, aggregates the SR/LR
per-integral cost ratio by (li, lj, same_atom, distance bin), compares against
the G2b thresholds, and reports whether a custom SR kernel effort is warranted.

    python benchmarks/experimental/cosx/sr_kernel_gate.py [sr_kernel_water27_def2-svp.json]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))  # benchmarks/, for _paths
import _paths  # noqa: E402
_paths.add_repo_to_syspath()
# G2b thresholds (P2b)
RT_PASS = 1.0
RT_TARGET = 0.7


def load_rows(path=None):
    path = path or sorted(glob.glob(os.path.join(_paths.results_dir(),
                                                 'sr_kernel_*.json')))[-1]
    with open(path) as fh:
        return json.load(fh), path


def main():
    rows, path = load_rows(sys.argv[1] if len(sys.argv) > 1 else None)
    by = {}
    for r in rows:
        by.setdefault((r['li'], r['lj'], r['same_atom'], r['r_lo'], r['r_hi']),
                      {})[r['kernel']] = r['ns_per_int']
    print(f"source: {path}")
    print(f"{'li lj same r(bohr)':22s} {'full':>7s} {'LR':>7s} {'SR':>7s} {'SR/LR':>6s}")
    ratios = []
    for key, d in sorted(by.items()):
        rt = d['SR'] / d['LR']
        ratios.append(rt)
        li, lj, same, lo, hi = key
        print(f"{li:2d} {lj:2d} {str(same):5s} {lo:4.0f}-{hi:<4.0f}      "
              f"{d['full']:7.1f} {d['LR']:7.1f} {d['SR']:7.1f} {rt:6.2f}")
    # weight by the FINDINGS methodology: compare to the measured overall R_t
    rt_mean = sum(ratios) / len(ratios)
    passed = rt_mean <= RT_PASS
    print(f"\nG2b: mean SR/LR per-integral cost R_t = {rt_mean:.2f} "
          f"(pass <= {RT_PASS}, target <= {RT_TARGET}) -> {passed}")
    # NOTE: unweighted mean over (class, distance bin); a production decision
    # should weight by the shim's ints_by_l from a real SGX build.
    if passed:
        print("Stock SR kernel is already at or below R_t = 1.0: no custom "
              "kernel needed; spend the effort on P2a screening.")
    else:
        print(f"Stock SR kernel costs {rt_mean:.2f}x LR per integral. A custom "
              f"SR kernel passes G2b only if it reaches R_t <= {RT_PASS} "
              f"(target <= {RT_TARGET}); look at the classes/bins with the "
              "largest SR/LR above to decide where it has to win.")
    try:
        import qcint  # noqa: F401
        print('qcint: available (evaluate WITH_POLYNOMIAL_FIT variants)')
    except ImportError:
        print('qcint: not installed (P2b comparison target unavailable)')


if __name__ == '__main__':
    main()
