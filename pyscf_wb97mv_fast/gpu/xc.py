"""S3: XC flow and the XcFunctional interface (spec sections 6-7).

``gpu_nr_rks`` mirrors ``dft.numint.NumInt.nr_rks`` (RKS, ground state, a
single density matrix, LDA/GGA/MGGA without laplacian -- the wB97M-V case)
with the same signature and return convention ``(nelec, excsum, vmat)``:

    per block:  AO FP32 -> density_gemm FP32 SGEMM -> functional FP64
                (XcFunctional, chain rule applied here) -> fock_gemm FP32
                partials scattered into an FP64 vmat.

Accumulation of nelec / excsum is FP64, sequential over blocks, so results
are reproducible.  vmat scatter uses unique indices per call (see
device.scatter_add).

``XcFunctional.eval(rho) -> (exc, vrho, vsigma, vtau)`` is the spec's
interface (libxc convention, untransformed); the chain rule to density
parameters lives in xc_flow._chain_rule (it reproduces
xc_deriv.transform_vxc for spin=0: v[1:4] = 2*vsigma*rho[1:4], v[4] = vtau).
PyscfLibxcFunctional always works (CPU FP64, the reference);
Gpu4PySCFFunctional plugs in libxc-cuda when gpu4pyscf is installed.
"""
import numpy

from pyscf_wb97mv_fast.gpu import backends, device
from pyscf_wb97mv_fast.gpu.ao_eval import plan_blocks

SUPPORTED_XCTYPE = ('LDA', 'GGA', 'MGGA')


class Unsupported(Exception):
    """Raised inside the flows for cases the GPU path does not handle;
    the install hooks translate it into a CPU fallback."""


class PyscfLibxcFunctional:
    """PySCF's own libxc (CPU FP64) behind the XcFunctional interface."""

    def __init__(self, xc_code, ni):
        self.xc_code = xc_code
        self.ni = ni

    def eval(self, rho):
        """rho: (5, npts) (den, gx, gy, gz, tau) on any backend."""
        rho = device.to_host(rho)
        exc, vxc = self.ni.eval_xc(self.xc_code, rho, spin=0, deriv=1)[:2]
        vsigma = vxc[1] if len(vxc) > 1 else None
        vtau = vxc[3] if len(vxc) > 3 else None
        return exc, vxc[0], vsigma, vtau


class Gpu4PySCFFunctional:
    """libxc-cuda adapter (gpu4pyscf), FP64 on the GPU.

    gpu4pyscf mirrors PySCF's eval_xc signature with cupy arrays.  The module
    layout differs between releases (dft.libxc / lib.libxc) and older layouts
    expose no eval_xc at all, so the constructor probes for it and raises
    Unsupported (-> the install falls back to the CPU functional).  The
    call convention itself is verified by the layer-1 tests (XC energy/vmat
    vs PyscfLibxcFunctional, <= 1e-12); any mismatch surfaces there.
    """

    def __init__(self, xc_code, ni):
        self.xc_code = xc_code
        self.ni = ni
        import importlib
        # importing gpu4pyscf replaces CuPy's global allocator with one that
        # bypasses the memory pool above 100 MB (cupy_helper.
        # set_conditional_mempool_malloc); the VV10 kernel_sum tiles then pay
        # a raw cudaMalloc/cudaFree each (4.2x slower).  Keep the caller's.
        cp = backends.cupy_module()
        allocator = cp.cuda.get_allocator() if cp is not None else None
        try:
            for modname in ('gpu4pyscf.dft.libxc', 'gpu4pyscf.lib.libxc'):
                try:
                    mod = importlib.import_module(modname)
                except ImportError:
                    continue
                if hasattr(mod, 'eval_xc'):
                    self._libxc = mod
                    return
            raise Unsupported('gpu4pyscf with an eval_xc entry point not found')
        finally:
            if allocator is not None:
                cp.cuda.set_allocator(allocator)

    def eval(self, rho):
        out = self._libxc.eval_xc(self.xc_code, rho, spin=0, deriv=1)
        exc, vxc = out[0], out[1]
        vsigma = vxc[1] if len(vxc) > 1 else None
        vtau = vxc[3] if len(vxc) > 3 else None
        return exc, vxc[0], vsigma, vtau


def chain_rule(xp, rho, exc, vrho, vsigma, vtau, out=None):
    """libxc convention -> density-parameter derivatives, FP64 on device.

    rho: (5, P); returns (exc, v) with v (5, P) = (vrho, 2 vsigma gx,
    2 vsigma gy, 2 vsigma gz, vtau) -- transform_vxc(spin=0) for the
    no-laplacian rho layout used here.  Zero rows where vsigma/vtau absent
    (LDA / GGA).
    """
    if out is None:
        v = xp.zeros((5, rho.shape[1]), dtype=rho.dtype)
    else:
        v = out
    v[0] = xp.asarray(vrho)
    if vsigma is not None:
        v[1:4] = 2.0 * xp.asarray(vsigma) * rho[1:4]
    if vtau is not None:
        v[4] = xp.asarray(vtau)
    # the functional may be CPU-side (PyscfLibxcFunctional): bring exc onto
    # the device so callers can mix it with device arrays
    return xp.asarray(exc), v


def _prepare_dm(dms, hermi):
    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dm = dms
    elif len(dms) == 1:
        dm = numpy.asarray(dms[0])
    else:
        raise Unsupported('GPU XC path supports a single RKS density matrix')
    if dm.ndim != 2:
        raise Unsupported('GPU XC path needs a ground-state RKS dm')
    if hermi != 1:
        dm = (dm + dm.T) * 0.5        # what _gen_rho_evaluator does first
    return numpy.asarray(dm, dtype=numpy.float64, order='C')


def gpu_nr_rks(ctx, ni, mol, grids, xc_code, dms,
               relativity=0, hermi=1, max_memory=2000, verbose=None):
    """GPU FP32 replacement for NumInt.nr_rks (RKS ground state)."""
    xp = ctx.xp
    xctype = ni._xc_type(xc_code)
    if xctype not in SUPPORTED_XCTYPE:
        raise Unsupported('xctype %r not supported on the GPU path' % xctype)
    if relativity not in (0, None):
        raise Unsupported('relativistic XC not supported')
    dm = _prepare_dm(dms, hermi)
    nao = dm.shape[0]

    functional = ctx.functional(xc_code, ni)
    blocks = plan_blocks(mol, grids, ctx.blksize)
    origin = ctx.origin
    ao_loc = ctx.evaluator.pack.ao_loc

    weights_d = xp.asarray(grids.weights, dtype=xp.float64)
    vmat = xp.zeros((nao, nao), dtype=xp.float64)
    v1mat = xp.zeros((nao, nao), dtype=xp.float64)
    nelec = 0.0
    excsum = 0.0
    coords64 = grids.coords - origin                 # numpy FP64

    for i0, i1, shells in blocks:
        idx = device.ao_indices(ao_loc, shells)
        coords = xp.asarray(coords64[i0:i1], dtype=xp.float64)   # FP64: see AoEvaluator.eval
        ao = ctx.evaluator.eval(coords, shells, deriv=1)         # (4,P,M) FP32
        dm_act = xp.asarray(device.gather_dm(dm, idx), dtype=ctx.dtype)

        # density_gemm (FP32 SGEMM; ctx.gemm_dtype lifts to FP64 if the
        # layer-1 gate demands it): one GEMM for value and all gradients
        gd = ctx.gemm_dtype
        ao_g = ao if ao.dtype == gd else ao.astype(gd)
        dm_g = dm_act if dm_act.dtype == gd else dm_act.astype(gd)
        tmp = ao_g[0] @ dm_g                                     # (P,M)
        w = weights_d[i0:i1]
        rho0 = (tmp * ao_g[0]).sum(axis=1)                       # (P,)
        rho_g = 2.0 * xp.einsum('pm,gpm->gp', tmp, ao_g[1:4])    # (3,P)
        rho = xp.empty((5, i1 - i0), dtype=xp.float64)
        rho[0] = rho0
        rho[1:4] = rho_g
        if xctype == 'MGGA':
            c1 = ao_g[1:4] @ dm_g                                # (3,P,M)
            tau = 0.5 * (c1 * ao_g[1:4]).sum(axis=2).sum(axis=0)  # (P,)
            rho[4] = tau
        else:
            rho[4] = 0.0
        rho64 = rho.astype(xp.float64)

        exc, vrho, vsigma, vtau = functional.eval(rho64)
        exc, v = chain_rule(xp, rho64, exc, vrho, vsigma, vtau)
        nelec += float((rho[0].astype(xp.float64) * w).sum())
        excsum += float((rho[0].astype(xp.float64) * w * exc).sum())

        # fock_gemm: FP32 partials, FP64 vmat
        wv = (v * w[None, :]).astype(gd)
        wv[0] *= 0.5                              # *.5 for vmat + vmat.T
        aow = xp.einsum('dpm,dp->pm', ao_g, wv[:4])               # (P,M)
        blk = ao_g[0].T @ aow                                     # (M,M)
        device.scatter_add(xp, vmat, idx, blk)
        if xctype == 'MGGA':
            # tau part: added once (no hermi doubling); *.5 mirrors PySCF's
            # nr_rks (numint.py: `wv[4] *= .5  # *.5 for 1/2 in tau`)
            wtau = wv[4] * 0.5
            v1 = xp.zeros((ao.shape[2], ao.shape[2]), dtype=gd)
            for ax in range(3):
                v1 += (ao_g[1 + ax] * wtau[:, None]).T @ ao_g[1 + ax]
            device.scatter_add(xp, v1mat, idx, v1)

    vmat = vmat + vmat.T + v1mat                   # lib.hermi_sum + tau dot
    return nelec, excsum, device.to_host(vmat)
