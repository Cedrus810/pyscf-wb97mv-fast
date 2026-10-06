"""S5: does the K hook really take the GPU inside a full get_veff (the
staged-SCF path), and how is the 'jk' wall split?  water27, SGX level 2,
dm from file; one get_veff with install_gpu(k=True), hooks.k_stats and
timings printed; then sgx_jk.get_k_only CPU timings for comparison."""
import time, datetime, numpy as np, warnings
from pyscf import dft
from pyscf.sgx import sgx_jk
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.core import sgx_patch
from pyscf_wb97mv_fast.gpu import install
log = lambda m: print('[%s] %s' % (datetime.datetime.now().strftime('%F %T'), m), flush=True)
warnings.simplefilter('always')
sgx_patch.apply()
mol = build_mol('water27', 'def2-svp')
mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
mf.with_df.grids_level_i = mf.with_df.grids_level_f = 2
mf.with_df.build(level=2)
dm = np.load('benchmarks/experimental/cosx/dm_water27_def2-svp.npy'); dm = (dm + dm.T) / 2
mf.initialize_grids(mol, dm)
mf._nsteps_direct = 0; mf._in_scf = False   # state SCF.kernel would set
hooks = install.install_gpu(mf, k=True)
try:
    for it in range(2):
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            v = mf.get_veff(mol, dm)
        log('get_veff #%d wall %.1fs  timings %s' % (it, time.perf_counter() - t, hooks.timings[-1]))
        log('  k_stats so far: %d calls; last %s' % (len(hooks.k_stats), hooks.k_stats[-1] if hooks.k_stats else None))
        for w in wl:
            if 'contraction engine' not in str(w.message):
                log('  WARNING %s: %s' % (w.category.__name__, str(w.message)[:200]))
finally:
    hooks.restore_all()
t = time.perf_counter(); mf.with_df.get_jk(dm, with_j=True, with_k=False); log('CPU J only %.1fs' % (time.perf_counter() - t))
for om in (0.0, 0.3):
    with mol.with_range_coulomb(om):
        t = time.perf_counter(); sgx_jk.get_k_only(mf.with_df, dm, hermi=1); log('CPU K omega=%.1f %.1fs' % (om, time.perf_counter() - t))
