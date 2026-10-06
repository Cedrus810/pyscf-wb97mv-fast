"""S5 async: where does one water27 S2-like get_veff spend its time, per
device, and how do the CPU and GPU parts overlap?  SGX level 2, XC grids as
mf.initialize_grids sets them, dm from file.  Same machine for everything:
  A  GPU XC + VV10 alone (the hooked numint functions)
  B  GPU K alone, full and LR (GpuKBuilder, warm)
  C  CPU K alone, full and LR; CPU J alone
  D  get_veff, overlap mode, K on the CPU   (install_gpu(k=False))
  E  get_veff, overlap mode, K on the GPU   (install_gpu(k=True))
Each get_veff is run twice; the second (warm) one is reported.
--switch-interval S sets sys.setswitchinterval (GIL hand-off, default 5 ms);
--only DE runs just the get_veff cases.  The md5 of the GPU sources is
logged first (NFS views on the compute nodes can lag)."""
import argparse, hashlib, os, sys, time, datetime, numpy as np, warnings
from pyscf import dft
from pyscf.sgx import sgx_jk
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.core import sgx_patch
from pyscf_wb97mv_fast.gpu import install, backends
from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu
T0 = time.perf_counter()
log = lambda m: print('[%s +%6.1fs] %s' % (datetime.datetime.now().strftime('%F %T'), time.perf_counter() - T0, m), flush=True)
warnings.filterwarnings('ignore', message='using cupy as the tensor')
ap = argparse.ArgumentParser()
ap.add_argument('--switch-interval', type=float, default=None)
ap.add_argument('--only', default='ABCDE')
args = ap.parse_args()
here = os.path.dirname(os.path.abspath(install.__file__))
log('md5 ' + '  '.join('%s %s' % (f, hashlib.md5(open(os.path.join(here, f), 'rb').read()).hexdigest()[:8])
                       for f in ('install.py', 'sgx_k.py', 'boys.py', 'xc.py', 'vv10.py')))
if args.switch_interval is not None:
    sys.setswitchinterval(args.switch_interval)
log('sys.getswitchinterval() = %g s' % sys.getswitchinterval())
cp = backends.cupy_module(); sync = cp.cuda.Device().synchronize
sgx_patch.apply()
mol = build_mol('water27', 'def2-svp')
dm = np.load('benchmarks/experimental/cosx/dm_water27_def2-svp.npy'); dm = (dm + dm.T) / 2

def new_mf():
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    mf.with_df.grids_level_i = mf.with_df.grids_level_f = 2
    mf.with_df.build(level=2)
    mf.initialize_grids(mol, dm)
    if mf.do_nlc():
        mf.nlcgrids.build(with_non0tab=True)
    mf._nsteps_direct = 0; mf._in_scf = False      # state SCF.kernel would set
    return mf

mf = new_mf()
log('grids: xc %d  nlc %d  sgx %d points' % (mf.grids.weights.size, mf.nlcgrids.weights.size, mf.with_df.grids.weights.size))
# A: GPU XC + VV10 alone
hooks = install.install_gpu(mf, k=False, overlap=False) if 'A' in args.only else None
try:
  if hooks is not None:
    ni = mf._numint
    nlc = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc
    for it in range(2):
        t = time.perf_counter(); ni.nr_rks(mol, mf.grids, mf.xc, dm); sync(); tx = time.perf_counter() - t
        t = time.perf_counter(); ni.nr_nlc_vxc(mol, mf.nlcgrids, nlc, dm); sync(); tv = time.perf_counter() - t
    log('A  GPU XC %.1fs  VV10 %.1fs  (sum %.1fs)' % (tx, tv, tx + tv))
finally:
    if hooks is not None:
        hooks.restore_all()
# B: GPU K alone
b = GpuKBuilder(mol, cp)
for om in ((0.0, 0.3) if 'B' in args.only else ()):
    with mol.with_range_coulomb(om):
        get_k_only_gpu(b, mf.with_df, dm); sync()
        t = time.perf_counter(); get_k_only_gpu(b, mf.with_df, dm); sync()
    log('B  GPU K omega=%.1f %.1fs' % (om, time.perf_counter() - t))
# C: CPU K and J alone
for om in ((0.0, 0.3) if 'C' in args.only else ()):
    with mol.with_range_coulomb(om):
        t = time.perf_counter(); sgx_jk.get_k_only(mf.with_df, dm, hermi=1)
    log('C  CPU K omega=%.1f %.1fs' % (om, time.perf_counter() - t))
if 'C' in args.only:
    t = time.perf_counter(); mf.with_df.get_jk(dm, with_j=True, with_k=False); log('C  CPU J %.1fs' % (time.perf_counter() - t))
# D / E: full get_veff in overlap mode
for label, k in (('D  get_veff, K on CPU', False), ('E  get_veff, K on GPU', True)):
    if label[0] not in args.only:
        continue
    mf = new_mf()
    hooks = install.install_gpu(mf, k=k)
    try:
        for it in range(2):
            mf._nsteps_direct = 0
            t = time.perf_counter(); mf.get_veff(mol, dm); sync(); wall = time.perf_counter() - t
        tm = hooks.timings[-1]
        ks = [s['wall'] for s in hooks.k_stats[-2:]] if k else []
        log('%s: wall %.1fs | main J/K %.1fs | worker XC+VV10 %.1fs | wait %.1fs | GPU K calls %s'
            % (label, wall, tm['jk'], tm['xc'], tm['wait'], ['%.1fs' % x for x in ks]))
        if k:
            for st in hooks.k_stats[-2:]:
                if 'phases' in st:
                    log('   K phases: ' + '  '.join('%s %.2fs' % kv for kv in st['phases'].items()))
    finally:
        hooks.restore_all()
