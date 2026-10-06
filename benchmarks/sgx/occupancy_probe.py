"""S5 perf: is the fused kernel bound by per-thread local memory?  Same P1
kernel, same 20 water27 tiles (screened, tile_tol 1e-11), different
residency: warps per block x a dummy dynamic shared-memory request that caps
the blocks per SM.  If fewer resident threads run faster, the ~2 KB/thread
local arrays (A, R, B, Ac) thrash L1 at high occupancy."""
import time, numpy as np
from pyscf import dft
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends
from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, _fused_kernel, _nsub
import datetime
log = lambda m: print('[%s] %s' % (datetime.datetime.now().strftime('%F %T'), m), flush=True)
cp = backends.cupy_module(); sync = cp.cuda.Device().synchronize
dev = cp.cuda.Device(); nsm = dev.attributes['MultiProcessorCount']
smem_sm = dev.attributes['MaxSharedMemoryPerMultiprocessor']
log('GPU %s  SMs %d  shared/SM %d' % (cp.cuda.runtime.getDeviceProperties(0)['name'], nsm, smem_sm))
mol = build_mol('water27', 'def2-svp')
mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True); sgx = mf.with_df
sgx.grids_level_i = sgx.grids_level_f = 2; sgx.build(level=2)
dm = np.load('benchmarks/experimental/cosx/dm_water27_def2-svp.npy'); dm = (dm + dm.T) / 2
b = GpuKBuilder(mol, cp); b._build(); sgx._build_pjs(1e-13)
proj = sgx._pjs_data._overlap_correction_matrix; pdm = cp.asarray(proj @ dm)
st = b._grid_state(sgx.grids); kern = _fused_kernel(cp); pd = b._pairs_dev
log('kernel %s' % b.kernel_info())
work = []
log('blksize %d, tiles used: 1' % b.blksize)
for (i0, i1) in st.tiles[:1]:
    X = b._ao.eval(st.coords64[i0:i1], np.arange(mol.nbas), deriv=0)[0]
    F = (X @ pdm).astype(cp.float32); nsub = _nsub(i1 - i0, st.ks)
    ids, offs, nk = b._kept_ids(st, i0, i1, F, nsub)
    work.append((i0, i1, F, ids, offs, nsub))
for warps, smem in [(4, 0), (2, 0), (1, 0), (4, smem_sm // 2 + 1024), (2, smem_sm // 4 + 1024), (2, smem_sm // 2 + 1024), (1, smem_sm // 2 + 1024)]:
    t = time.perf_counter()
    for (i0, i1, F, ids, offs, nsub) in work:
        G = cp.zeros((i1 - i0, mol.nao), dtype=cp.float32)
        kern((-(-nsub // warps),), (32 * warps,), (pd.sh_dims, pd.sh_Mc, pd.sh_MT, pd.sh_Mc_lo, pd.sh_MT_lo,
             pd.pair_sh, pd.prim_p, pd.prim_prefac, pd.prim_prefac_lo, pd.prim_P_hi, pd.prim_P_lo, pd.prim_E,
             st.coords32[i0:i1], st.coords32_lo[i0:i1], F, offs, ids, np.int32(i1 - i0), np.int32(mol.nao),
             np.int32(st.ks), np.int32(nsub), np.float64(0.0), G), shared_mem=smem)
    sync()
    bps = max(1, smem_sm // smem) if smem else None
    log('warps/block %d  dummy smem %6d  (<= %s blocks/SM -> <= %s threads/SM)  kernel %.2fs (1 tile)' % (
        warps, smem, bps if bps else 'reg-limited', bps * 32 * warps if bps else '~640', time.perf_counter() - t))
