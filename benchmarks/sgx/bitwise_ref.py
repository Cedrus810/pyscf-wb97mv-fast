"""S5 perf: bitwise regression reference for kernel refactors that must not
change the arithmetic.  water dimer (level 2, init guess D) and, with
--system water27, a slice of water27's tiles; tile_tol = 0; full and LR.
--save writes benchmarks/results/s5_bitwise_<system>.npz, --check compares."""
import argparse, numpy as np
from pyscf import dft
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends
from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu
ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['save', 'check'])
ap.add_argument('--system', default='water_dimer'); a = ap.parse_args()
cp = backends.cupy_module()
mol = build_mol(a.system, 'def2-svp')
mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True); sgx = mf.with_df
sgx.grids_level_i = sgx.grids_level_f = 2; sgx.build(level=2)
dm = mf.get_init_guess(); dm = (dm + dm.T) / 2
out = {}
for omega in (0.0, 0.3):
    with mol.with_range_coulomb(omega):
        out['k%.1f' % omega] = get_k_only_gpu(GpuKBuilder(mol, cp, tile_tol=0.0), sgx, dm, hermi=1)
f = 'benchmarks/results/s5_bitwise_%s.npz' % a.system
if a.mode == 'save':
    np.savez(f, **out); print('saved', f)
else:
    ref = np.load(f)
    for k, v in out.items():
        print('%s %s bitwise_equal=%s max|diff|=%.3e' % (a.system, k, np.array_equal(v, ref[k]), np.abs(v - ref[k]).max()))
