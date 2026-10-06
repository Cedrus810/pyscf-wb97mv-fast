"""S5 r1: sanity of the density matrices the K tests use.
water27 file DM: nelec = tr(D S), idempotency |D S D - 2 D| (RKS), symmetry,
against the current build_mol('water27') geometry/basis.  Dimer: the SAD
init guess the dimer tests use, same numbers for contrast."""
import numpy as np
from pyscf import dft
from pyscf_wb97mv_fast.core.testsystems import build_mol
for name, src in (('water27', 'benchmarks/experimental/cosx/dm_water27_def2-svp.npy'), ('water_dimer', 'init_guess')):
    mol = build_mol(name, 'def2-svp')
    S = mol.intor('int1e_ovlp')
    if src == 'init_guess':
        D = dft.RKS(mol, xc='wb97m-v').get_init_guess()
    else:
        D = np.load(src)
    asym = np.abs(D - D.T).max()
    D = (D + D.T) / 2
    idem = np.abs(D @ S @ D - 2 * D).max()
    print('%-12s %-14s nao=%d  shape=%s  tr(DS)=%.10f (nelec %d)  max|D-D^T|=%.1e  max|DSD-2D|=%.2e'
          % (name, src.split('/')[-1], mol.nao, D.shape, np.einsum('ij,ji->', D, S), mol.nelectron, asym, idem))
