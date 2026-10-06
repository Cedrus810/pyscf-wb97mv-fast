"""S3: install_gpu hooks, fallbacks and OOM retry (gpu.install)."""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from pyscf import dft

from pyscf_wb97mv_fast.core.hooks import HookSet
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends, install


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope='module')
def mol():
    return build_mol('water_dimer', 'def2-svp')


@pytest.mark.gpu
def test_functional_probe_keeps_cupy_allocator():
    """Probing gpu4pyscf for libxc-cuda must not replace CuPy's global
    allocator: gpu4pyscf's import installs one that bypasses the memory pool
    above 100 MB (cupy_helper.set_conditional_mempool_malloc), and the VV10
    kernel_sum tiles then pay a raw cudaMalloc/cudaFree each (measured 4.2x
    slower).  Fresh interpreter: the side effect
    fires only on the first import of gpu4pyscf."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    code = textwrap.dedent('''
        import cupy
        from pyscf import dft, gto
        from pyscf_wb97mv_fast.gpu import install
        before = cupy.cuda.get_allocator()
        mol = gto.M(atom='He', basis='sto-3g', verbose=0)
        install._default_functional_factory('wb97m-v', dft.RKS(mol, xc='wb97m-v')._numint)
        print('SAME' if cupy.cuda.get_allocator() == before else 'CHANGED')
    ''')
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''))
    res = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True,
                         text=True, cwd=ROOT, env=env, timeout=300)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().splitlines()[-1] == 'SAME', res.stdout


@pytest.fixture(scope='module')
def mf(mol):
    m = dft.RKS(mol, xc='wb97m-v')
    m.grids.build(with_non0tab=True)
    m.nlcgrids.build(with_non0tab=True)
    return m


def test_no_cupy_installs_nothing(mf, monkeypatch):
    ni = mf._numint
    orig_rks, orig_nlc = ni.nr_rks, ni.nr_nlc_vxc
    monkeypatch.setattr(backends, 'have_cupy', lambda: False)
    with pytest.warns(RuntimeWarning, match='CuPy/CUDA not available'):
        hooks = install.install_gpu(mf, backend=None)
    assert isinstance(hooks, HookSet)
    # == not `is`: a plain bound method is a fresh object on every access
    assert ni.nr_rks == orig_rks and ni.nr_nlc_vxc == orig_nlc


def test_install_replaces_and_restores(mf):
    ni = mf._numint
    orig_rks, orig_nlc = ni.nr_rks, ni.nr_nlc_vxc
    hooks = install.install_gpu(mf, backend='numpy', verify_ao=True,
                                blksize=112)
    try:
        assert ni.nr_rks != orig_rks
        assert ni.nr_nlc_vxc != orig_nlc
    finally:
        hooks.restore_all()
    assert ni.nr_rks == orig_rks
    assert ni.nr_nlc_vxc == orig_nlc


def test_hooked_nr_rks_matches_cpu(mf):
    ni = mf._numint
    dm = (mf.get_init_guess() + mf.get_init_guess().T) * 0.5
    ref = ni.nr_rks(mf.mol, mf.grids, 'wb97m-v', dm)
    hooks = install.install_gpu(mf, backend='numpy', verify_ao=True,
                                blksize=112, gemm_dtype='float64',
                                dtype='float64')
    try:
        got = ni.nr_rks(mf.mol, mf.grids, 'wb97m-v', dm)
    finally:
        hooks.restore_all()
    assert abs(got[1] - ref[1]) < 1e-9
    assert np.max(np.abs(got[2] - ref[2])) < 1e-9


def test_fallback_on_unsupported_case(mf):
    """Multiple density matrices are out of the GPU path's scope: the hook
    must warn and hand the call to the CPU implementation."""
    ni = mf._numint
    dm = mf.get_init_guess()
    dms = (dm, dm)
    ref = ni.nr_rks(mf.mol, mf.grids, 'wb97m-v', dms)
    hooks = install.install_gpu(mf, backend='numpy', verify_ao=False,
                                blksize=112)
    try:
        with pytest.warns(RuntimeWarning, match='nr_rks'):
            got = ni.nr_rks(mf.mol, mf.grids, 'wb97m-v', dms)
    finally:
        hooks.restore_all()
    # not array_equal: PySCF's numint is not bit-reproducible across calls
    # on many threads (upstream behaviour)
    assert np.max(np.abs(got[2] - ref[2])) < 1e-12


def test_oom_retries_with_smaller_blocks(mf, monkeypatch):
    calls = []

    def fake_flow(ctx, *args, **kwargs):
        calls.append(ctx.blksize)
        if len(calls) < 3:
            raise MemoryError('out of memory')
        return (0.0, 0.0, np.zeros((1, 1)))

    monkeypatch.setattr(install.xc, 'gpu_nr_rks', fake_flow)
    ni = mf._numint
    dm = mf.get_init_guess()
    hooks = install.install_gpu(mf, backend='numpy', verify_ao=False,
                                blksize=1024)
    try:
        got = ni.nr_rks(mf.mol, mf.grids, 'wb97m-v', dm)
    finally:
        hooks.restore_all()
    assert calls == [1024, 512, 256]
    assert got[0] == 0.0 and got[1] == 0.0 and got[2].shape == (1, 1)


def test_oom_exhaustion_falls_back(mf, monkeypatch):
    def always_oom(ctx, *args, **kwargs):
        raise MemoryError('out of memory')

    monkeypatch.setattr(install.xc, 'gpu_nr_rks', always_oom)
    ni = mf._numint
    dm = mf.get_init_guess()
    ref = ni.nr_rks(mf.mol, mf.grids, 'wb97m-v', dm)
    hooks = install.install_gpu(mf, backend='numpy', verify_ao=False,
                                blksize=1024)
    try:
        with pytest.warns(RuntimeWarning, match='nr_rks'):
            got = ni.nr_rks(mf.mol, mf.grids, 'wb97m-v', dm)
    finally:
        hooks.restore_all()
    assert np.max(np.abs(got[2] - ref[2])) < 1e-12


def test_context_blksize_halving():
    from pyscf_wb97mv_fast.gpu.install import GpuContext
    ctx = GpuContext(None, None, None, None, None, blksize=1024)
    assert ctx.with_blksize(512).blksize == 512
    assert ctx.blksize == 1024


@pytest.mark.gpu
@pytest.mark.slow
def test_full_scf_with_gpu_hooks(mol):
    """A whole water-dimer SCF through the hooked numint on the device."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.core import sgx_patch
    sgx_patch.apply()
    ref = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    ref.conv_tol = 1e-9
    e_ref = ref.kernel()
    sgx_patch.revert()

    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    mf.conv_tol = 1e-9
    hooks = install.install_gpu(mf, vv10_tile_tol=0.0)
    try:
        e_gpu = mf.kernel()
    finally:
        hooks.restore_all()
    # spec section 8 layer 2 (mixed precision, same grids): the FP32-AO
    # pipeline measures |dE| ~ 2.6e-6 Ha on water dimer -- the FP32 rho
    # error (~1e-7 relative) is amplified by the XC nonlinearity (spec
    # section 10).  That exceeds the spec's 1e-6 budget: per spec section 6
    # the production last step must then lift to FP64, which
    # test_full_scf_with_fp64_hooks verifies.
    assert abs(e_gpu - e_ref) < 1e-5


@pytest.mark.gpu
@pytest.mark.slow
def test_full_scf_with_fp64_hooks(mol):
    """Layer 2 with the FP64 fallback settings (spec sections 6/10):
    a whole water-dimer SCF through the hooked numint, |dE| <= 1e-6."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.core import sgx_patch
    sgx_patch.apply()
    ref = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    ref.conv_tol = 1e-9
    e_ref = ref.kernel()
    sgx_patch.revert()

    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    mf.conv_tol = 1e-9
    hooks = install.install_gpu(mf, vv10_tile_tol=0.0,
                                gemm_dtype='float64', dtype='float64')
    try:
        e_gpu = mf.kernel()
    finally:
        hooks.restore_all()
    assert abs(e_gpu - e_ref) < 1e-6


@pytest.mark.gpu
def test_overlap_get_veff_matches_serial(mol):
    """overlap=True: XC + VV10 run on a worker thread while the main thread
    builds J/K through the stock get_veff (with the XC part zeroed), then
    the two are summed.  Must equal the serial hooked get_veff, and the XC
    must really run off the main thread; restore_all removes the wrapper."""
    import threading
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.core import sgx_patch
    sgx_patch.apply()
    try:
        mf0 = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True).build()   # SGX state
        dm = mf0.get_init_guess()
        serial = install.install_gpu(mf0, overlap=False)
        try:
            v0 = mf0.get_veff(mol, dm)
        finally:
            serial.restore_all()

        mf1 = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True).build()
        hooks = install.install_gpu(mf1, overlap=True)
        seen = []
        hooked = mf1._numint.nr_rks

        def spy(*a, **k):
            seen.append(threading.get_ident())
            return hooked(*a, **k)
        mf1._numint.nr_rks = spy
        try:
            v1 = mf1.get_veff(mol, dm)
        finally:
            hooks.restore_all()
    finally:
        sgx_patch.revert()
    assert seen and all(t != threading.get_ident() for t in seen)
    assert np.max(np.abs(np.asarray(v1) - np.asarray(v0))) < 1e-10
    assert abs(v1.exc - v0.exc) < 1e-10
    assert abs(v1.ecoul - v0.ecoul) < 1e-10
    assert getattr(mf1.get_veff, '__func__', None) is type(mf1).get_veff


def test_semilocal_false_keeps_cpu_xc(mf):
    """install_gpu(semilocal=False): only VV10 goes to the GPU (FP64 tail)."""
    ni = mf._numint
    orig_rks, orig_nlc = ni.nr_rks, ni.nr_nlc_vxc
    hooks = install.install_gpu(mf, backend='numpy', verify_ao=False, blksize=112,
                                semilocal=False)
    try:
        assert ni.nr_rks == orig_rks
        assert ni.nr_nlc_vxc != orig_nlc
    finally:
        hooks.restore_all()
    assert ni.nr_nlc_vxc == orig_nlc


# -- S5: the COSX exchange K hook (gpu.sgx_k via get_k_only dispatch) ---------

def _cosx_k(mol, level=2):
    m = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True).build()
    m.with_df.grids_level_i = m.with_df.grids_level_f = level
    m.with_df.build(level=level)
    return m


@pytest.mark.gpu
def test_k_hook_full_and_lazy_lr_copy_match_cpu(mol):
    """Review Focus 3: the LR copy is created AFTER install; full and
    omega=0.3 K both go through the GPU and match the CPU reference."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.core import sgx_patch
    sgx_patch.apply()
    try:
        mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True).build()
        dm = mf.get_init_guess()
        k_cpu = (mf.get_k(mol, dm), mf.get_k(mol, dm, omega=0.3))
        mf2 = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True).build()
        hooks = install.install_gpu(mf2, k=True)
        try:
            assert not mf2.with_df._rsh_df          # no LR copy yet
            k_gpu = (mf2.get_k(mol, dm), mf2.get_k(mol, dm, omega=0.3))
            assert len(hooks.k_stats) == 2          # both went to the GPU
        finally:
            hooks.restore_all()
    finally:
        sgx_patch.revert()
    for a, b in zip(k_gpu, k_cpu):
        assert np.max(np.abs(a - b)) < 1e-6 * np.abs(b).max()
    from pyscf.sgx import sgx_jk
    assert sgx_jk.get_k_only.__module__ == 'pyscf.sgx.sgx_jk'   # restored


@pytest.mark.gpu
def test_k_hook_follows_sgx_grid_rebuild(mol):
    """Review Focus 4: level 1 -> level 2 rebuild must not reuse cached
    grid data."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    dm = dft.RKS(mol, xc='wb97m-v').get_init_guess()
    ref = {lv: _cosx_k(mol, lv).get_k(mol, dm) for lv in (1, 2)}
    mf = _cosx_k(mol, 1)
    hooks = install.install_gpu(mf, k=True)
    try:
        k1 = mf.get_k(mol, dm)
        mf.with_df.grids_level_i = mf.with_df.grids_level_f = 2
        mf.with_df.build(level=2)
        k2 = mf.get_k(mol, dm)
    finally:
        hooks.restore_all()
    for got, lv in ((k1, 1), (k2, 2)):
        assert np.max(np.abs(got - ref[lv])) < 1e-6 * np.abs(ref[lv]).max(), lv


@pytest.mark.gpu
def test_k_hook_out_of_scope_falls_back(mol):
    """Review Focus 5: two dms and hermi=0 -> CPU result exactly, with a
    RuntimeWarning."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    from pyscf.sgx import sgx_jk
    mf = _cosx_k(mol, 2)
    dm = mf.get_init_guess()
    dms = np.stack([dm, 0.5 * dm])
    ref2 = sgx_jk.get_k_only(mf.with_df, dms, hermi=1)
    ref0 = sgx_jk.get_k_only(mf.with_df, dm, hermi=0)
    hooks = install.install_gpu(mf, k=True)
    try:
        with pytest.warns(RuntimeWarning):
            got2 = sgx_jk.get_k_only(mf.with_df, dms, hermi=1)
        with pytest.warns(RuntimeWarning):
            got0 = sgx_jk.get_k_only(mf.with_df, dm, hermi=0)
    finally:
        hooks.restore_all()
    # CPU multi-thread K is not bitwise reproducible:
    # "matches the CPU" means 1e-12 here
    assert np.max(np.abs(got2 - ref2)) < 1e-12
    assert np.max(np.abs(got0 - ref0)) < 1e-12


@pytest.mark.gpu
def test_k_hook_other_mf_untouched(mol):
    """An unrelated COSX mf in the same process must keep the CPU path while
    another mf has the hook installed."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    mf_hooked, mf_other = _cosx_k(mol, 2), _cosx_k(mol, 2)
    dm = mf_other.get_init_guess()
    ref = mf_other.get_k(mol, dm)
    hooks = install.install_gpu(mf_hooked, k=True)
    try:
        got = mf_other.get_k(mol, dm)
        assert len(hooks.k_stats) == 0      # the other mf never reached the GPU
    finally:
        hooks.restore_all()
    assert np.max(np.abs(got - ref)) < 1e-12


@pytest.mark.gpu
def test_k_hook_builds_shell_pairs_once(mol, monkeypatch):
    """The hook wraps every GPU K call in _run_with_oom_retry, which clones
    the builder (with_blksize) even on the first attempt: the shell-pair /
    AO / screening data must be built once and shared by the clones, not
    rebuilt per call (it cost 3.4 s per K call on the 5080)."""
    cupy = backends.cupy_module()
    if cupy is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu import sgx_k
    built = []
    real = sgx_k.ShellPairs

    def counting(*a, **kw):
        built.append(1)
        return real(*a, **kw)
    monkeypatch.setattr(sgx_k, 'ShellPairs', counting)
    mf = _cosx_k(mol, 2)
    dm = mf.get_init_guess()
    hooks = install.install_gpu(mf, k=True)
    try:
        for _ in range(3):
            mf.get_k(mol, dm)
            mf.get_k(mol, dm, omega=0.3)
        assert len(hooks.k_stats) == 6
    finally:
        hooks.restore_all()
    assert len(built) == 1, len(built)


# -- OpenBLAS thread limit (core.blas_threads, 2026-10-03) -------------------

def test_install_limits_pthreads_openblas_and_restores(mf, monkeypatch):
    from pyscf_wb97mv_fast.core import blas_threads

    class FakeOpenblas:                          # pthreads build, 32 threads
        n, calls = 32, []

        def get_parallel(self):
            return 1

        def get_num_threads(self):
            return self.n

        def set_num_threads(self, n):
            self.n = n
            self.calls.append(n)
    for name in blas_threads.USER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    lib = FakeOpenblas()
    lib.calls = []
    monkeypatch.setattr(blas_threads, 'loaded_openblas', lambda: [lib])
    hooks = install.install_gpu(mf, backend='numpy', verify_ao=False, blksize=112)
    try:
        assert lib.n == 1
    finally:
        hooks.restore_all()
    assert lib.n == 32
    hooks = install.install_gpu(mf, backend='numpy', verify_ao=False, blksize=112,
                                openblas_threads=None)
    hooks.restore_all()
    assert lib.calls == [1, 32]                  # untouched with openblas_threads=None
