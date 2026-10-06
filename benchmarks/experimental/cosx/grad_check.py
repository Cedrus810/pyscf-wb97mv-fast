#!/usr/bin/env python
"""P5 gate: analytic gradient vs central finite differences of the energy.

Default: wb97m-v with COSX(pjs=True), i.e. pyscf.sgx.grad's SGX analytic
gradient with XC/VV10 grid response on. Gate: max error <= 1e-5 Ha/bohr.

    python benchmarks/experimental/cosx/grad_check.py water_dimer --basis sto-3g
    python benchmarks/experimental/cosx/grad_check.py water_dimer --basis def2-svp
    python benchmarks/experimental/cosx/grad_check.py water_dimer --xc pbe --no-cosx   # utility sanity check
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))  # benchmarks/, for _paths
import _paths  # noqa: E402
_paths.add_repo_to_syspath()

from pyscf import dft, lib  # noqa: E402

from pyscf_wb97mv_fast.core import sgx_patch  # noqa: E402
from pyscf_wb97mv_fast.experimental.cosx.fastgrad import (check_gradient_support,  # noqa: E402
                                        finite_difference_check)
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402

GRAD_TOL = 1e-5  # Ha/bohr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--basis', default='sto-3g')
    ap.add_argument('--xc', default='wb97m-v')
    ap.add_argument('--no-cosx', action='store_true')
    ap.add_argument('--disp', type=float, default=1e-3)
    ap.add_argument('--no-grid-response', action='store_true')
    ap.add_argument('--no-sgx-grid-response', action='store_true',
                    help='work around the upstream PySCF 2.14 NaN in the SGX grid response')
    args = ap.parse_args()
    mol = build_mol(args.system, args.basis)
    mf = dft.RKS(mol, xc=args.xc)
    if not args.no_cosx:
        sgx_patch.apply()
        mf = mf.COSX(pjs=True)
    supported, reason = check_gradient_support(mf)
    print('gradient support:', supported, '-', reason)
    if not supported:
        return
    mf.conv_tol = 1e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('reference SCF did not converge')
    err, grad_fd, grad = finite_difference_check(
        mf, dm0=mf.make_rdm1(), disp=args.disp,
        grid_response=not args.no_grid_response,
        sgx_grid_response=not args.no_sgx_grid_response)
    out = dict(system=args.system, basis=args.basis, xc=args.xc,
               cosx=not args.no_cosx, disp=args.disp,
               grid_response=not args.no_grid_response,
               sgx_grid_response=not args.no_sgx_grid_response,
               threads=lib.num_threads(), max_err=err, tol=GRAD_TOL,
               passed=bool(err <= GRAD_TOL),
               grad_fd=grad_fd.tolist(), grad=grad.tolist())
    tag = (os.path.splitext(os.path.basename(args.system))[0] + '_' + args.basis
           + '_' + args.xc + ('' if args.no_cosx else '_cosx')
           + ('_nosgxgr' if args.no_sgx_grid_response else ''))
    path = os.path.join(_paths.results_dir(), f'grad_check_{tag}.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f"P5: max |FD - analytic| = {err:.2e} Ha/bohr "
          f"(<= {GRAD_TOL}) -> {out['passed']}")
    print('wrote', path)


if __name__ == '__main__':
    main()
