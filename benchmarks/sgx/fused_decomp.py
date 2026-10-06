"""S5 r1: decompose the fused GPU K error (after the FP64-X fix) on the
water dimer.  Replays GpuKBuilder.get_k's tile loop (tile_tol = 0) and per
tile compares the kernel's G with G built from the dense FP32 integrals and
the same F, in FP64.  All K's use the same FP64 post-processing; dE is
against CPU get_k_only (= FP64 reference to 1e-14)."""
import numpy as np, sys
from pyscf import dft
from pyscf.sgx import sgx_jk
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends
from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, _fused_kernel, _nsub
from pyscf_wb97mv_fast.gpu.int3c1e import int3c1e_fp32
cp = backends.cupy_module()
mol = build_mol('water_dimer', 'def2-svp')
mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True); sgx = mf.with_df
sgx.grids_level_i = sgx.grids_level_f = 2; sgx.build(level=2)
dm = mf.get_init_guess(); dm = (dm + dm.T) / 2
for omega in (0.0, 0.3):
    with mol.with_range_coulomb(omega):
        kc = sgx_jk.get_k_only(sgx, dm, hermi=1)
    proj = sgx._pjs_data._overlap_correction_matrix; sym = sgx._pjs_data.sym_ovlp
    def e(K):
        K = cp.asnumpy(K) if not isinstance(K, np.ndarray) else K
        if sym: K = proj.T @ K
        K = (K + K.T) * 0.5
        return 0.25 * np.einsum('ij,ji->', dm, K - kc)
    b = GpuKBuilder(mol, cp, tile_tol=0.0); b._build()
    st = b._grid_state(sgx.grids); kern = _fused_kernel(cp); pdv = b._pairs_dev
    pdm = proj @ dm
    hi = pdm.astype(np.float32); pdm_hi = cp.asarray(hi); pdm_lo = cp.asarray((pdm - hi).astype(np.float32))
    pdm64 = cp.asarray(pdm)
    nao = mol.nao; npairs = b._pairs_host.npairs
    acc = {k: cp.zeros((nao, nao)) for k in ('fused', 'G_dense32F', 'G_dense64F', 'G_dense64F_w64', 'fused_w64')}
    w64all = cp.asarray(np.asarray(sgx.grids.weights, dtype=np.float64))
    for (i0, i1) in st.tiles:
        npts = i1 - i0
        X64 = b._ao.eval(st.coords64[i0:i1], np.arange(mol.nbas), deriv=0)[0]
        X = X64.astype(cp.float32); Xlo = (X64 - X).astype(cp.float32)
        F64 = (X @ pdm_hi).astype(cp.float64) + X_lo_term if False else None
        F64 = (X @ pdm_hi).astype(cp.float64); F64 += Xlo @ pdm_hi; F64 += X @ pdm_lo
        F = F64.astype(cp.float32)
        Fex = X64 @ pdm64
        nsub = _nsub(npts, st.ks)
        ids = cp.asarray(np.tile(np.arange(npairs, dtype=np.int32), nsub))
        offs = cp.asarray(np.arange(nsub + 1, dtype=np.int32) * npairs)
        G = cp.zeros((npts, nao), dtype=cp.float32)
        kern((nsub,), (st.ks,), (pdv.sh_dims, pdv.sh_Mc, pdv.sh_MT, pdv.sh_Mc_lo, pdv.sh_MT_lo, pdv.pair_sh,
              pdv.prim_p, pdv.prim_prefac, pdv.prim_P_hi, pdv.prim_P_lo, pdv.prim_E,
              st.coords32[i0:i1], F, st.w32[i0:i1], offs, ids, np.int32(npts), np.int32(nao),
              np.int32(st.ks), np.float64(omega), G), shared_mem=st.ks * nao * 4)
        ones = cp.ones(npts, dtype=cp.float32)
        G1 = cp.zeros((npts, nao), dtype=cp.float32)            # fused, unit weights
        kern((nsub,), (st.ks,), (pdv.sh_dims, pdv.sh_Mc, pdv.sh_MT, pdv.sh_Mc_lo, pdv.sh_MT_lo, pdv.pair_sh,
              pdv.prim_p, pdv.prim_prefac, pdv.prim_P_hi, pdv.prim_P_lo, pdv.prim_E,
              st.coords32[i0:i1], F, ones, offs, ids, np.int32(npts), np.int32(nao),
              np.int32(st.ks), np.float64(omega), G1), shared_mem=st.ks * nao * 4)
        A = int3c1e_fp32(cp, pdv, st.coords32[i0:i1], omega=omega).astype(cp.float64)
        w32 = st.w32[i0:i1].astype(cp.float64); w64 = w64all[i0:i1]
        acc['fused'] += X64.T @ G.astype(cp.float64)
        acc['fused_w64'] += X64.T @ (w64[:, None] * G1.astype(cp.float64))
        acc['G_dense32F'] += X64.T @ (w32[:, None] * cp.einsum('gmn,gm->gn', A, F.astype(cp.float64)))
        acc['G_dense64F'] += X64.T @ (w32[:, None] * cp.einsum('gmn,gm->gn', A, Fex))
        acc['G_dense64F_w64'] += X64.T @ (w64[:, None] * cp.einsum('gmn,gm->gn', A, Fex))
    with mol.with_range_coulomb(omega):
        kg = cp.asnumpy(b.get_k(sgx, dm, omega)) if False else None
    print('omega=%.1f (dE vs CPU; X64 in K = X^T G for all rows)' % omega)
    for k, v in acc.items():
        print('  %-18s % .3e' % (k, e(v)))
