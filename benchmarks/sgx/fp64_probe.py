"""S5 perf: how much of the fused kernel is FP64 work?  Timing-only kernel
variants (their K is WRONG): Boys replaced by a cheap FP32 placeholder,
and/or the per-primitive theta / T / R000-coefficient math moved from double
to float.  One water27 tile (screened), omega = 0, default residency and
capped residency (2 warps/block, 1 block/SM)."""
import time, datetime, numpy as np
from pyscf import dft
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends, sgx_k
from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, _nsub
log = lambda m: print('[%s] %s' % (datetime.datetime.now().strftime('%F %T'), m), flush=True)
cp = backends.cupy_module(); sync = cp.cuda.Device().synchronize
dev = cp.cuda.Device(); smem_sm = dev.attributes['MaxSharedMemoryPerMultiprocessor']
n = cp.cuda.runtime.getDeviceProperties(0)['name']; log('GPU %s' % (n.decode() if isinstance(n, bytes) else n))
SRC = sgx_k._FUSED_SRC
BOYS_HEAD = '__device__ void boys_fp32(float T, float* F, int nmax)\n{'
assert BOYS_HEAD in SRC
boys_dummy = (BOYS_HEAD, BOYS_HEAD + '\n    for (int n = 0; n <= nmax && n < 5; ++n) F[n] = 1.0f / (2 * n + 1 + T);\n    if (T >= 0.f) return;')
theta = [('const float T = (float)(th * pd * (double)r2);', 'const float T = (float)pd * r2;'),
         ('const double q = -2.0 * pd * th;', 'const float q = -2.0f * (float)pd;'),
         ('double cn = sth;', 'float cn = 1.0f;'),
         ('const float chi = (float)cn;', 'const float chi = cn;'),
         ('const float clo = (float)(cn - (double)chi);', 'const float clo = 0.f;')]
for a, _ in theta: assert a in SRC, a
def variant(reps):
    s = SRC
    for a, b in reps: s = s.replace(a, b)
    return cp.RawKernel(s, 'sgx_k_tile')
kernels = {'current': variant([]), 'boys->fp32 dummy': variant([boys_dummy]),
           'theta->float': variant(theta), 'both': variant([boys_dummy] + theta)}
mol = build_mol('water27', 'def2-svp')
mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True); sgx = mf.with_df
sgx.grids_level_i = sgx.grids_level_f = 2; sgx.build(level=2)
dm = np.load('benchmarks/experimental/cosx/dm_water27_def2-svp.npy'); dm = (dm + dm.T) / 2
b = GpuKBuilder(mol, cp); b._build(); sgx._build_pjs(1e-13)
pdm = cp.asarray(sgx._pjs_data._overlap_correction_matrix @ dm)
st = b._grid_state(sgx.grids); pd = b._pairs_dev
i0, i1 = st.tiles[0]
X = b._ao.eval(st.coords64[i0:i1], np.arange(mol.nbas), deriv=0)[0]
F = (X @ pdm).astype(cp.float32); nsub = _nsub(i1 - i0, st.ks)
ids, offs, nk = b._kept_ids(st, i0, i1, F, nsub)
log('tile %d points, %d kept (pair, sub) tasks' % (i1 - i0, nk))
for name, kern in kernels.items():
    for warps, smem in [(4, 0), (2, smem_sm // 2 + 1024)]:
        G = cp.zeros((i1 - i0, mol.nao), dtype=cp.float32)
        t = time.perf_counter()
        kern((-(-nsub // warps),), (32 * warps,), (pd.sh_dims, pd.sh_Mc, pd.sh_MT, pd.sh_Mc_lo, pd.sh_MT_lo,
             pd.pair_sh, pd.prim_p, pd.prim_prefac, pd.prim_prefac_lo, pd.prim_P_hi, pd.prim_P_lo, pd.prim_E,
             st.coords32[i0:i1], st.coords32_lo[i0:i1], F, offs, ids, np.int32(i1 - i0), np.int32(mol.nao),
             np.int32(st.ks), np.int32(nsub), np.float64(0.0), G), shared_mem=smem)
        sync()
        log('%-18s regs %3d local %4dB  warps %d smem %5d  kernel %6.2fs' % (
            name, kern.num_regs, kern.local_size_bytes, warps, smem, time.perf_counter() - t))
