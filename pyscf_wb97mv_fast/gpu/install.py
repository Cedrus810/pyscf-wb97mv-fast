"""S3: install the GPU backends into an SCF object (spec sections 3, 7).

GPU takes over exactly two entry points, with PySCF-signature compatibility
(spec section 3):

    mf._numint.nr_rks      -> xc.gpu_nr_rks       (XC, semilocal part)
    mf._numint.nr_nlc_vxc  -> vv10.gpu_nr_nlc_vxc (VV10)

Everything else (SCF driver, COSX K, RI-J, diagonalization, DIIS) stays on
the CPU in FP64.  With overlap=True (default) a third hook on mf.get_veff
runs those two GPU flows on a worker thread while the CPU builds J/K, so the
GPU time hides behind K (see _overlap_factory).  Both hooks keep the original call signature and return
convention; on ANY error inside the GPU path they emit a warning with the
traceback and fall back to the original CPU implementation, so an SCF never
dies because of the fast path (spec: "没有 GPU 或 CuPy：自动退回 CPU 参考
路径，并给出警告").  Out-of-memory first retries with the block size halved
(twice) before falling back.

Usage:
    from pyscf_wb97mv_fast.gpu import install
    hooks = install.install_gpu(mf)          # or install_gpu(mf) context
    try:
        e = mf.kernel()
    finally:
        hooks.restore_all()
"""
import contextlib
import threading
import time
import traceback
import warnings

import numpy

from pyscf_wb97mv_fast.core import blas_threads
from pyscf_wb97mv_fast.core.hooks import HookSet
from pyscf_wb97mv_fast.gpu import backends, device, vv10, xc
from pyscf_wb97mv_fast.gpu.ao_eval import AoEvaluator, ShellPack


class GpuContext:
    """Everything the two flows need, built once per mf."""

    def __init__(self, evaluator, functional_factory, kernel, xp, device_state,
                 blksize=8192, origin=None, dtype='float32',
                 gemm_dtype='float32', tile_tol=0.0):
        self.evaluator = evaluator
        self.functional = functional_factory
        self.kernel = kernel
        self.xp = xp
        self.device = device_state
        self.blksize = int(blksize)
        self.origin = origin
        self.dtype = dtype
        self.gemm_dtype = gemm_dtype     # lift to 'float64' if layer-1 fails
        self.tile_tol = float(tile_tol)

    def with_blksize(self, blksize):
        clone = GpuContext(self.evaluator, self.functional, self.kernel,
                           self.xp, self.device, blksize, self.origin,
                           self.dtype, self.gemm_dtype, self.tile_tol)
        return clone


def _default_functional_factory(xc_code, ni):
    """Prefer libxc-cuda (gpu4pyscf); fall back to PySCF's CPU libxc."""
    try:
        return xc.Gpu4PySCFFunctional(xc_code, ni)
    except xc.Unsupported:
        return xc.PyscfLibxcFunctional(xc_code, ni)


def build_context(mol, backend='cupy', blksize=8192, vv10_tile_tol=0.0,
                  functional_factory=None, mem_fraction=device.DEFAULT_MEM_FRACTION,
                  verify_ao=True, gemm_dtype='float32', dtype='float32'):
    xp = backends.get_xp(backend)
    state = device.DeviceState(backend, mem_fraction)
    origin = mol.atom_coords().mean(axis=0)
    pack = ShellPack(mol, origin=origin)
    evaluator = AoEvaluator(mol, xp, pack=pack, mem_budget=state.mem_budget,
                            dtype=dtype, verify=verify_ao)
    kernel = vv10.Vv10Kernel(xp, mem_budget=state.mem_budget,
                             tile_tol=vv10_tile_tol)
    if functional_factory is None:
        functional_factory = _default_functional_factory
    return GpuContext(evaluator, functional_factory, kernel, xp, state,
                      blksize=blksize, origin=origin, dtype=dtype,
                      gemm_dtype=gemm_dtype, tile_tol=vv10_tile_tol)


def _warn_fallback(where, exc):
    warnings.warn(
        'pyscf_wb97mv_fast GPU %s failed (%s: %s); falling back to the CPU '
        'implementation\n%s' % (where, type(exc).__name__, exc,
                                traceback.format_exc()),
        RuntimeWarning)


def _run_with_oom_retry(flow, ctx, *args, **kwargs):
    """Run a flow; halve the block size twice on OOM, then give up."""
    last = None
    for attempt in range(3):
        try:
            return flow(ctx.with_blksize(ctx.blksize >> attempt), *args, **kwargs)
        except Exception as exc:                       # noqa: BLE001
            if not device.is_oom(exc):
                raise
            last = exc
    raise last


def _zero_xc(nao):
    def zero(*args, **kwargs):
        return 0, 0.0, numpy.zeros((nao, nao))
    return zero


def _gpu_stream(ctx, name):
    """Context manager: run this thread's GPU work on ctx's private
    non-blocking stream `name` (created once per context).  The XC/VV10
    worker and the K hook each get their own stream, so a synchronization
    inside one flow (cupy.nonzero, a device->host copy) waits only for that
    flow's own kernels -- on the shared legacy default stream the K calls
    inside get_veff serialized behind the worker's XC kernels instead of
    overlapping them.  Work queued
    on the legacy default stream before (uploads, cached arrays) is
    finished first: non-blocking streams do not order against it."""
    xp = ctx.xp
    if not hasattr(xp, 'cuda'):
        return contextlib.nullcontext()
    streams = ctx.__dict__.setdefault('_streams', {})
    if name not in streams:
        streams[name] = xp.cuda.Stream(non_blocking=True)
    xp.cuda.Stream.null.synchronize()
    return streams[name]


def _overlap_factory(mf, ni, timings, ctx=None):
    """mf.get_veff that runs XC + VV10 on a worker thread while this thread
    builds J/K.

    rks.get_veff (PySCF 2.14) computes XC first and uses its (exc, vxc) only
    after J/K: `vxc += vj - vk*.5`, `exc -= <dm|vk>/4`, tag.  So the stock
    get_veff -- SGX's with_full_dm / incremental-JK logic included -- runs
    here with ni.nr_rks / nr_nlc_vxc returning zero, and the worker's XC is
    added afterwards: the same sums in a different order.  The J/K C code
    and the BLAS release the GIL, so the two really run concurrently.
    Each call appends {'jk', 'xc', 'wait'} wall seconds to `timings` (main
    thread J/K, worker XC+VV10, main thread blocked in join afterwards).
    """
    def factory(orig):
        def get_veff(mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
            if mol is None:
                mol = mf.mol
            if dm is None:
                dm = mf.make_rdm1()
            if hermi == 2 or not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
                return orig(mol, dm, dm_last, vhf_last, hermi)
            # grids exactly as the stock get_veff / numint would set them up
            if mf.grids.coords is None:
                mf.initialize_grids(mol, dm)
            do_nlc = mf.do_nlc()
            if do_nlc and mf.nlcgrids.coords is None:
                mf.nlcgrids.build(with_non0tab=True)
            nr_rks, nr_nlc_vxc = ni.nr_rks, ni.nr_nlc_vxc
            nlc_code = None
            if do_nlc:
                nlc_code = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc
            result = {}

            def xc_job():
                t0 = time.perf_counter()
                try:
                    with (_gpu_stream(ctx, 'xc') if ctx is not None
                          else contextlib.nullcontext()):
                        _, exc, vxc = nr_rks(mol, mf.grids, mf.xc, dm)
                        if do_nlc:
                            _, enlc, vnlc = nr_nlc_vxc(mol, mf.nlcgrids,
                                                       nlc_code, dm)
                            exc += enlc
                            vxc = vxc + vnlc
                    result['xc'] = (exc, vxc)
                except BaseException as err:          # re-raised on this thread
                    result['error'] = err
                result['t_xc'] = time.perf_counter() - t0
            worker = threading.Thread(target=xc_job, name='gpu-xc')
            worker.start()
            zero = _zero_xc(dm.shape[0])
            ni.nr_rks, ni.nr_nlc_vxc = zero, zero
            t0 = time.perf_counter()
            try:
                veff = orig(mol, dm, dm_last, vhf_last, hermi)
            finally:
                ni.nr_rks, ni.nr_nlc_vxc = nr_rks, nr_nlc_vxc
                t1 = time.perf_counter()
                worker.join()
            timings.append({'jk': t1 - t0, 'xc': result.get('t_xc'),
                            'wait': time.perf_counter() - t1})
            if 'error' in result:
                raise result['error']
            exc, vxc = result['xc']
            from pyscf import lib
            return lib.tag_array(numpy.asarray(veff) + vxc, ecoul=veff.ecoul,
                                 exc=veff.exc + exc, vj=veff.vj, vk=veff.vk)
        return get_veff
    return factory


def install_gpu(mf, backend=None, blksize=8192, vv10_tile_tol=0.0,
                functional_factory=None, verify_ao=True,
                gemm_dtype='float32', dtype='float32',
                mem_fraction=device.DEFAULT_MEM_FRACTION, overlap=True,
                semilocal=True, k=False, k_tile_tol=1e-11, openblas_threads=1):
    """Hook ni.nr_rks / ni.nr_nlc_vxc on mf._numint; returns a HookSet.

    backend=None means: use 'cupy' when importable and a CUDA device exists,
    otherwise install nothing and warn (the mf keeps running on the CPU).
    backend='numpy' builds the CPU FP32 flow (tests / no-GPU machines).
    dtype / gemm_dtype lift the AO evaluation and the density/fock GEMMs to
    'float64' (spec section 6: the FP32 backend is used only while it meets
    the <1e-6 Ha layer-2 budget; otherwise fall back to FP64).

    k=True additionally replaces the module function
    pyscf.sgx.sgx_jk.get_k_only with a dispatcher that routes the COSX
    exchange K of mf (mf.with_df and its lazily created long-range copies
    in mf.with_df._rsh_df) through gpu.sgx_k.get_k_only_gpu (FP32, fused,
    screened with k_tile_tol); every other sgx object keeps the CPU
    original.  Any Unsupported condition or GPU error warns and calls the
    original (the same _warn_fallback contract as the XC hooks); OOM halves
    the grid tile via _run_with_oom_retry first.  Each GPU call's
    GpuKBuilder.last_stats is appended to hooks.k_stats.  The FP64 tail
    must reinstall with k=False (staging.schedule does this): the final K
    stays CPU FP64.

    openblas_threads=1 sets a pthreads OpenBLAS to one thread while the hooks
    are installed and restores it on restore_all() (core.blas_threads: it
    oversubscribes inside PySCF's OpenMP regions; 4090 water27 staged 221 ->
    131 s).  None leaves BLAS alone; so does an OPENBLAS_NUM_THREADS the
    user set.
    """
    ni = getattr(mf, '_numint', None)
    if ni is None:
        raise TypeError('mf has no _numint; not a DFT object?')
    if backend is None:
        if not backends.have_cupy():
            warnings.warn('CuPy/CUDA not available; GPU fast path not '
                          'installed, mf stays on the CPU', RuntimeWarning)
            return HookSet()
        backend = 'cupy'

    ctx = build_context(mf.mol, backend=backend, blksize=blksize,
                        vv10_tile_tol=vv10_tile_tol,
                        functional_factory=functional_factory,
                        mem_fraction=mem_fraction, verify_ao=verify_ao,
                        gemm_dtype=gemm_dtype, dtype=dtype)
    hooks = HookSet()
    if openblas_threads is not None:
        restore_blas = blas_threads.limit_openblas_threads(openblas_threads)
        if restore_blas is not None:
            hooks.on_restore(restore_blas)    # registered first: undone last
    hooks.k_stats = []            # one GpuKBuilder.last_stats per GPU K call

    def rks_factory(orig):
        def nr_rks(mol, grids, xc_code, dms, relativity=0, hermi=1,
                   max_memory=2000, verbose=None):
            try:
                return _run_with_oom_retry(xc.gpu_nr_rks, ctx, ni, mol, grids,
                                           xc_code, dms, relativity, hermi,
                                           max_memory, verbose)
            except Exception as exc:                   # noqa: BLE001
                _warn_fallback('nr_rks', exc)
                return orig(mol, grids, xc_code, dms, relativity, hermi,
                            max_memory, verbose)
        return nr_rks

    def nlc_factory(orig):
        def nr_nlc_vxc(mol, grids, xc_code, dm, relativity=0, hermi=1,
                       max_memory=2000, verbose=None):
            try:
                return _run_with_oom_retry(vv10.gpu_nr_nlc_vxc, ctx, ni, mol,
                                           grids, xc_code, dm, relativity,
                                           hermi, max_memory, verbose)
            except Exception as exc:                   # noqa: BLE001
                _warn_fallback('nr_nlc_vxc', exc)
                return orig(mol, grids, xc_code, dm, relativity, hermi,
                            max_memory, verbose)
        return nr_nlc_vxc

    if semilocal:              # False: semilocal XC stays on the CPU FP64 numint
        hooks.wrap(ni, 'nr_rks', rks_factory)
    hooks.wrap(ni, 'nr_nlc_vxc', nlc_factory)

    if k:
        import pyscf.sgx.sgx_jk as sgx_jk_mod
        from pyscf_wb97mv_fast.gpu import sgx_k as sgx_k_mod
        holder = {}

        def k_factory(orig):
            def get_k_only(sgx, dm, hermi=1, direct_scf_tol=1e-13):
                df = mf.with_df
                rsh = getattr(df, '_rsh_df', None) or {}
                if not (sgx is df or any(sgx is v for v in rsh.values()
                                         if v is not None)):
                    return orig(sgx, dm, hermi, direct_scf_tol)
                b = holder.get('builder')
                if b is None:
                    b = sgx_k_mod.GpuKBuilder(
                        mf.mol, ctx.xp, tile_tol=k_tile_tol,
                        mem_budget=ctx.device.mem_budget)
                    holder['builder'] = b
                stats = {}

                def flow(bb):
                    kk = sgx_k_mod.get_k_only_gpu(
                        bb, sgx, dm, hermi=hermi,
                        direct_scf_tol=direct_scf_tol)
                    stats.update(bb.last_stats)
                    return kk

                try:
                    with _gpu_stream(ctx, 'k'):
                        kk = _run_with_oom_retry(flow, b)
                except Exception as exc:               # noqa: BLE001
                    _warn_fallback('get_k_only', exc)
                    return orig(sgx, dm, hermi, direct_scf_tol)
                hooks.k_stats.append(dict(stats))
                return kk
            return get_k_only

        hooks.wrap(sgx_jk_mod, 'get_k_only', k_factory)

    hooks.timings = []            # overlap=True: one dict per get_veff call
    if overlap and hasattr(mf, 'initialize_grids'):
        hooks.wrap(mf, 'get_veff', _overlap_factory(mf, ni, hooks.timings,
                                                    ctx))
    return hooks
