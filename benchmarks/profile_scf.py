#!/usr/bin/env python
"""Per-component ωB97M-V / RIJCOSX SCF profile, patched or stock.

    python benchmarks/profile_scf.py water27
    python benchmarks/profile_scf.py /path/to/system.xyz --basis def2-tzvp --stock
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # benchmarks/, for _paths
import _paths  # noqa: E402
_paths.add_repo_to_syspath()

from pyscf import lib  # noqa: E402

from pyscf_wb97mv_fast.core.profiling import run_profile  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    ap.add_argument('--stock', action='store_true', help='do not apply sgx_patch')
    ap.add_argument('--sgx-tol-energy', default='auto')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    etol = args.sgx_tol_energy if args.sgx_tol_energy == 'auto' else float(args.sgx_tol_energy)
    mol = build_mol(args.system, args.basis)
    res = run_profile(mol, patched=not args.stock, conv_tol=args.conv_tol, sgx_tol_energy=etol)
    res.pop('dm')
    res.update(system=args.system, basis=args.basis, threads=lib.num_threads())

    tag = os.path.splitext(os.path.basename(args.system))[0] + '_' + args.basis
    out = args.out or os.path.join(_paths.results_dir(),
                                   f'profile_{tag}_{"stock" if args.stock else "patched"}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1)

    total = res['timings']['scf_total']
    print(f"{args.system} nao={res['nao']} e_tot={res['e_tot']:.10f} "
          f"cycles={res['cycles']} converged={res['converged']} total={total:.1f}s")
    for k, v in sorted(res['timings'].items(), key=lambda kv: -kv[1]):
        if k != 'scf_total':
            print(f"  {k:18s} {v:10.2f} s  {100 * v / total:5.1f}%  n={res['counts'][k]}")
    print('wrote', out)


if __name__ == '__main__':
    main()
