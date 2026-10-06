"""S5 r1 (after fix 2): switch off ONE kernel compensation at a time in the
fused GPU K and report dE = 1/4 tr(D dK) vs CPU get_k_only (water dimer,
level 2, tile_tol 0).  Toggles: prefactor lo, grid-coordinate lo, Mc/MT lo
(device arrays zeroed), and the R000 coefficient hi+lo (kernel recompiled
with plain FP32 sqrt(theta), (-2 p theta)^n)."""
import numpy as np
from pyscf import dft
from pyscf.sgx import sgx_jk
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends, sgx_k
cp = backends.cupy_module()
mol = build_mol('water_dimer', 'def2-svp')
mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True); sgx = mf.with_df
sgx.grids_level_i = sgx.grids_level_f = 2; sgx.build(level=2)
dm = mf.get_init_guess(); dm = (dm + dm.T) / 2
NEW = '''                const float chi = (float)cn;
                const float clo = (float)(cn - (double)chi);
                R000[n] = fmaf(chi, Fb[n], clo * Fb[n]);'''
OLD = '''                R000[n] = (float)cn * Fb[n];'''
assert NEW in sgx_k._FUSED_SRC
src_plainR = sgx_k._FUSED_SRC.replace(NEW, OLD)
def run(omega, toggle):
    sgx_k._FUSED_KERNELS.clear()
    if toggle == 'R000':
        sgx_k._FUSED_KERNELS[id(cp)] = cp.RawKernel(src_plainR, 'sgx_k_tile')
    b = sgx_k.GpuKBuilder(mol, cp, tile_tol=0.0); b._build()
    pd = b._pairs_dev
    if toggle == 'prefac': pd.prim_prefac_lo = cp.zeros_like(pd.prim_prefac_lo)
    if toggle == 'McMT':
        pd.sh_Mc_lo = cp.zeros_like(pd.sh_Mc_lo); pd.sh_MT_lo = cp.zeros_like(pd.sh_MT_lo)
    st = b._grid_state(sgx.grids)
    if toggle == 'coords': st.coords32_lo = cp.zeros_like(st.coords32_lo)
    with mol.with_range_coulomb(omega):
        kg = b.get_k(sgx, dm, omega)
    sgx_k._FUSED_KERNELS.clear()
    return kg
for omega in (0.0, 0.3):
    with mol.with_range_coulomb(omega):
        kc = sgx_jk.get_k_only(sgx, dm, hermi=1)
    for t in ('none', 'prefac', 'coords', 'McMT', 'R000'):
        kg = run(omega, t)
        print('omega=%.1f  lo off: %-7s dE % .3e  max|dK| %.2e' % (
            omega, t, 0.25 * np.einsum('ij,ji->', dm, kg - kc), np.abs(kg - kc).max()), flush=True)
