import numpy as np
from pyscf import dft

from pyscf_wb97mv_fast.core.testsystems import build_mol
from tolerance_scan import scan


def test_scan_monotone_and_tight_is_accurate():
    mol = build_mol('water_dimer')
    mf = dft.RKS(mol, xc='pbe').density_fit()
    mf.kernel()
    res = scan(mol, mf.make_rdm1(), etols=[1e-13, 1e-8], exact=True)
    for kname in ('full', 'LR', 'SR'):
        rows = res[kname]['rows']
        assert [r['etol'] for r in rows] == [1e-13, 1e-8]
        assert rows[1]['tasks'] <= rows[0]['tasks']
        assert rows[0]['dE_screen'] < 1e-10
        g = res[kname]['grid']
        assert g['dE_grid'] > 0 and np.isfinite(g['dK_grid'])
