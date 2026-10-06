#!/usr/bin/env python
"""P6: hardware profile CLI -- what is worth putting on a device.

    python benchmarks/experimental/cosx/acc_profile.py [--json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))  # benchmarks/, for _paths
import _paths  # noqa: E402
_paths.add_repo_to_syspath()

from pyscf_wb97mv_fast.experimental.cosx.acc import hardware_profile  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--json', action='store_true', help='machine-readable output')
    args = ap.parse_args()
    res = hardware_profile.profile(repeat=args.repeat)
    if args.json:
        print(json.dumps(res, indent=1, default=float))
    else:
        for backend in ('cpu', 'jax'):
            if res.get(backend) is None:
                continue
            label = res.get('jax_backend') if backend == 'jax' else 'cpu (numpy)'
            print(f"[{label}]")
            for name, d in res[backend].items():
                print(f"  {name:12s} {d['gflops']:10.1f} GFLOP/s")
        if res['jax'] is None:
            print('[jax] not installed; install jax to profile the device')
    with open(os.path.join(_paths.results_dir(), 'acc_profile.json'), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
