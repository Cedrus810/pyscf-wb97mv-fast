"""S5 perf: where does one GPU K call spend its time?  Replays
GpuKBuilder.get_k's tile loop with a device synchronize around every phase:
AO eval, F GEMM, host screening (_kept_ids incl. pair_bound), the fused
kernel, the K GEMM.  First call (cold: bounds cache empty) and second call
(warm) for full and LR K.

Usage: python -B benchmarks/sgx/profile_gpu_k.py [--system water27] [--tile-tol 1e-11]"""

import atexit as _atexit, datetime as _dt, socket as _socket, time as _time
_T0 = _time.perf_counter()
def log(msg='', **_):
    """Wall-clock stamped line: [YYYY-mm-dd HH:MM:SS +elapsed]."""
    print('[%s +%7.1fs] %s' % (_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               _time.perf_counter() - _T0, msg), flush=True)
log('START %s  host=%s' % (' '.join(__import__('sys').argv), _socket.gethostname()))
_atexit.register(lambda: log('END  wall %.1fs' % (_time.perf_counter() - _T0)))

import argparse, time
import numpy as np
from pyscf import dft
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends, device
from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, _fused_kernel, _nsub

ap = argparse.ArgumentParser()
ap.add_argument('--system', default='water27')
ap.add_argument('--tile-tol', type=float, default=1e-11)
ap.add_argument('--max-tiles', type=int, default=0, help='0 = all tiles; else extrapolate')
ap.add_argument('--cpu-k', action='store_true', help='also time CPU get_k_only (full + LR) on this machine')
a = ap.parse_args()
cp = backends.cupy_module()
sync = cp.cuda.Device().synchronize
_n = cp.cuda.runtime.getDeviceProperties(0)['name']; log('GPU %s' % (_n.decode() if isinstance(_n, bytes) else _n))
mol = build_mol(a.system, 'def2-svp')
mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True); sgx = mf.with_df
sgx.grids_level_i = sgx.grids_level_f = 2; sgx.build(level=2)
dm = (np.load('benchmarks/experimental/cosx/dm_water27_def2-svp.npy') if a.system == 'water27'
      else mf.get_init_guess()); dm = (dm + dm.T) / 2
b = GpuKBuilder(mol, cp, tile_tol=a.tile_tol)
t = time.perf_counter(); b._build(); sync(); t_build = time.perf_counter() - t
sgx._build_pjs(1e-13)
import os
log('OMP_NUM_THREADS=%s' % os.environ.get('OMP_NUM_THREADS'))
if a.cpu_k:
    from pyscf.sgx import sgx_jk
    for omega in (0.0, 0.3):
        with mol.with_range_coulomb(omega):
            t = time.perf_counter(); sgx_jk.get_k_only(sgx, dm, hermi=1); tc = time.perf_counter() - t
        log('CPU get_k_only omega=%.1f  %.1fs' % (omega, tc))
proj = sgx._pjs_data._overlap_correction_matrix
if hasattr(b, 'kernel_info'):
    log('fused kernel: %s' % b.kernel_info())
log('%s nao=%d ngrids=%d npairs=%d nprims=%d  builder build %.1fs' % (
    a.system, mol.nao, sgx.grids.weights.size, b._pairs_host.npairs, b._pairs_host.pairs.nprims, t_build))
for omega in (0.0, 0.3):
    for call in ('cold', 'warm'):
        t0 = time.perf_counter()
        st = b._grid_state(sgx.grids); sync(); t_state = time.perf_counter() - t0
        pdm = cp.asarray(np.ascontiguousarray(proj @ dm))
        T = dict(ao=0., fgemm=0., screen=0., kernel=0., kgemm=0.)
        K = cp.zeros((mol.nao, mol.nao)); kept = tot = 0
        tiles = st.tiles if not a.max_tiles else st.tiles[:a.max_tiles]
        for (i0, i1) in tiles:
            npts = i1 - i0; nsub = _nsub(npts, st.ks)
            t = time.perf_counter(); X = b._ao.eval(st.coords64[i0:i1], np.arange(mol.nbas), deriv=0)[0]; sync(); T['ao'] += time.perf_counter() - t
            t = time.perf_counter(); F = (X @ pdm).astype(cp.float32); sync(); T['fgemm'] += time.perf_counter() - t
            t = time.perf_counter()
            tasks, nk = b._tasks(st, i0, i1, F, nsub)
            sync(); T['screen'] += time.perf_counter() - t
            kept += nk; tot += b._pairs_host.npairs * nsub
            t = time.perf_counter()
            G = b._launch(st, i0, i1, F, tasks, nsub, omega)
            sync(); T['kernel'] += time.perf_counter() - t
            t = time.perf_counter(); K += X.T @ (st.w64[i0:i1, None] * G); sync(); T['kgemm'] += time.perf_counter() - t
        scale = len(st.tiles) / len(tiles)
        tot_t = sum(T.values()) * scale
        log('omega=%.1f %-4s ks=%d tiles=%d%s  kept %.1f%%  total %.1fs | ' % (
            omega, call, st.ks, len(st.tiles), '' if scale == 1 else ' (x%.1f extrapolated)' % scale,
            100.0 * kept / max(tot, 1), tot_t) +
            '  '.join('%s %.1fs' % (k, v * scale) for k, v in T.items()) + '  | state %.1fs' % t_state, flush=True)
