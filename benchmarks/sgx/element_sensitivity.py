"""S5 Task 0 Step 2: element sensitivity of the COSX grid (CPU, spec 8.5).

For water27 (elements O, H): keep every element's grid at the base SGX
level except one, which is raised one level (atom_grid[symb] =
(SGX_RAD_GRIDS[hi, period], LEBEDEV_ORDER[SGX_ANG_MAPPING[hi, 3]]) -- the
same mapping get_gridss uses for its optimized grids).  For each element
and for full and LR separately, report the change of the energy form of
the K error, 0.25 * tr(D (K - K_exact)), against the all-base-level run.

Output: a "element x level -> error" table, written to
benchmarks/results/element_sensitivity_<system>.json; the conclusion
(which elements are insensitive enough to drop a level at fixed error
budget) goes into benchmarks/sgx/FINDINGS.md.

Usage (CPU node, OMP_NUM_THREADS=16):
    $PY -B benchmarks/sgx/element_sensitivity.py --system water27 \
        --sgx-level 2 --omega 0.3
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _paths import add_repo_to_syspath, results_dir  # noqa: E402

add_repo_to_syspath()
from pyscf import dft  # noqa: E402
from pyscf.data import elements  # noqa: E402
from pyscf.dft.gen_grid import LEBEDEV_ORDER  # noqa: E402
from pyscf.sgx.sgx_jk import SGX_ANG_MAPPING, SGX_RAD_GRIDS  # noqa: E402

from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402

XC = 'wb97m-v'
DM_DIR = os.path.join(os.path.dirname(HERE), 'experimental', 'cosx')


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    sys.stdout.flush()


def load_dm(system, mol):
    path = os.path.join(DM_DIR, 'dm_%s_def2-svp.npy' % system)
    if os.path.exists(path):
        dm = np.load(path)
    else:
        mf = dft.RKS(mol, xc=XC)
        mf.conv_tol = 1e-8
        mf.kernel()
        dm = mf.make_rdm1()
    return (dm + dm.T) * 0.5


def atom_grid_for(level, symb):
    """(nrad, nang) that get_gridss would pick for `level` and this element."""
    tab = np.array((2, 10, 18, 36, 54, 86, 118))
    chg = elements.charge(symb)
    period = int((chg > tab).sum())
    return (int(SGX_RAD_GRIDS[level, period]),
            int(LEBEDEV_ORDER[SGX_ANG_MAPPING[level, 3]]))


def k_energy_error(mf, dm, omega, k_exact):
    k = np.asarray(mf.get_k(mf.mol, dm, hermi=1, omega=omega or None))
    dE = float(0.25 * np.einsum('ij,ji->', dm, k - k_exact))
    rel = float(np.linalg.norm(k - k_exact) / np.linalg.norm(k_exact))
    return dE, rel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--system', default='water27')
    ap.add_argument('--sgx-level', type=int, default=2)
    ap.add_argument('--omega', type=float, default=0.3,
                    help='LR omega; --omega 0 measures the full-range K only')
    args = ap.parse_args()
    system = args.system
    base = args.sgx_level

    mol = build_mol(system, 'def2-svp')
    dm = load_dm(system, mol)
    syms = sorted({mol.atom_symbol(ia) for ia in range(mol.natm)})
    log('%s: elements %s, base SGX level %d' % (system, syms, base))

    # exact references at the highest level available for the bumped grids
    hi = min(base + 1, len(SGX_RAD_GRIDS) - 1)
    out = {'system': system, 'sgx_level': base, 'bump_level': hi,
           'omega': args.omega, 'rows': []}

    for label, omega in (('full', 0.0), ('lr', args.omega)):
        log('=== %s (omega=%g) ===' % (label, omega))
        from pyscf import scf
        k_exact = np.asarray(
            scf.hf.RHF(mol).get_k(mol, dm, hermi=1, omega=omega or None))

        rows = []
        # baseline: everything at the base level
        mf = dft.RKS(mol, xc=XC).COSX(pjs=True)
        mf.with_df.grids_level_i = mf.with_df.grids_level_f = base
        mf.with_df.build(level=base)
        dE0, rel0 = k_energy_error(mf, dm, omega, k_exact)
        rows.append({'element': '(base)', 'dE_ha': dE0, 'rel_fro': rel0})
        log('%-10s 0.25 tr(D dK)=%+.3e Ha  ||dK||/||K||=%.2e'
            % ('(base)', dE0, rel0))

        # one element at a time raised to `hi`
        for symb in syms:
            mf = dft.RKS(mol, xc=XC).COSX(pjs=True)
            df = mf.with_df
            df.grids_level_i = df.grids_level_f = base
            df.atom_grid = {symb: atom_grid_for(hi, symb)}
            df.build(level=base)
            dE, rel = k_energy_error(mf, dm, omega, k_exact)
            rows.append({'element': symb, 'dE_ha': dE, 'rel_fro': rel,
                         'd_dE_vs_base_ha': dE - dE0})
            log('%-10s 0.25 tr(D dK)=%+.3e Ha  (delta %+.3e)'
                % (symb, dE, dE - dE0))
        out['rows'].append({'kernel': label, 'rows': rows})

    path = os.path.join(results_dir(),
                        'element_sensitivity_%s.json' % system)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    log('-> %s' % path)
    log('CONCLUSION RULE: elements whose bump changes 0.25 tr(D dK) by far '
        'less than the 1e-4 Ha budget are candidates for a coarser grid '
        '(record the table in benchmarks/sgx/FINDINGS.md).')


if __name__ == '__main__':
    main()
