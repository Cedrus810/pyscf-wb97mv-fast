"""S5 r1: GPU fused K vs CPU get_k_only on the water dimer (test_sgx_k setup):
dE = 1/4 tr(D dK) and max|dK| for omega = 0 and 0.3, tile_tol = 0."""

import atexit as _atexit, datetime as _dt, socket as _socket, time as _time
_T0 = _time.perf_counter()
def log(msg='', **_):
    """Wall-clock stamped line: [YYYY-mm-dd HH:MM:SS +elapsed]."""
    print('[%s +%7.1fs] %s' % (_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                               _time.perf_counter() - _T0, msg), flush=True)
log('START %s  host=%s' % (' '.join(__import__('sys').argv), _socket.gethostname()))
_atexit.register(lambda: log('END  wall %.1fs' % (_time.perf_counter() - _T0)))

import numpy as np, sys, time
from pyscf import dft
from pyscf.sgx import sgx_jk
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends
from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu
cp = backends.cupy_module()
_n = cp.cuda.runtime.getDeviceProperties(0)['name']; log('GPU %s' % (_n.decode() if isinstance(_n, bytes) else _n))
system = sys.argv[1] if len(sys.argv) > 1 else 'water_dimer'
mol = build_mol(system, 'def2-svp')
mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True); sgx = mf.with_df
sgx.grids_level_i = sgx.grids_level_f = 2; sgx.build(level=2)
dm = mf.get_init_guess(); dm = (dm + dm.T) / 2
for omega in (0.0, 0.3):
    with mol.with_range_coulomb(omega):
        t = time.perf_counter(); kc = sgx_jk.get_k_only(sgx, dm, hermi=1); tc = time.perf_counter() - t
        b = GpuKBuilder(mol, cp, tile_tol=0.0)
        t = time.perf_counter(); kg = get_k_only_gpu(b, sgx, dm, hermi=1); t = time.perf_counter() - t
    log('%s omega=%.1f  dE % .3e  max|dK| %.2e (rel %.2e)  E_K %.6f  cpu %.1fs  gpu %.1fs'
          % (system, omega, 0.25 * np.einsum('ij,ji->', dm, kg - kc), np.abs(kg - kc).max(),
             np.abs(kg - kc).max() / np.abs(kc).max(), 0.25 * np.einsum('ij,ji->', dm, kc), tc, t), flush=True)
