"""S3: VV10 flow and kernel_sum on the backend (spec sections 6-7).

``gpu_nr_nlc_vxc`` mirrors ``dft.numint.NumInt.nr_nlc_vxc`` (same signature,
returns ``(nelec, excsum, vmat)``), porting ``dft.numint._vv10nlc`` exactly:

    thresh 1e-8 on rho0 (outer and inner grids coincide here, so one index
    set) -> per-point W0, kappa, dW0/dR, dW0/dG, dK/dR in FP64 ->
    kernel_sum -> exc = Beta + F/2, v0 = Beta + F + 1.5 (U dKdR + W dW0dR),
    v1 = 1.5 W dW0dG -> transform_vxc('GGA') -> fock_gemm.

``Vv10Kernel.kernel_sum(coords, q, W0, kappa) -> (F, U, W)`` is the spec's
interface: q is the inner density-weight (rho*weight), W0/kappa the per-point
kernel parameters (g = kappa + W0 R^2 = kappa (qR^2 + 1) with q = W0/kappa in
the spec's parametrization -- PySCF's W0, K arrays are passed directly).

Precision (spec section 6): pair products in FP32; within a tile the inner
sums use a segment-pairwise + Kahan reduction (precision.kahan_sum_axis);
per-point F, U, W accumulate across tiles in FP64 in a fixed tile order.
No atomics anywhere -> reproducible run to run.

Tile-pair pruning (spec section 7, vv10_tile_tol): for a block pair (I, J)
the kernel is bounded using the minimum inter-block distance and the tile
extrema of W0 and kappa,

    g    >= Rmin^2 W0min_I + Kmin_I,   gp >= Rmin^2 W0min_J + Kmin_J
    F    <= (sum_J q) / (g gp (g+gp))
    U    <= 2 F / g_min,               W <= U Rmax^2

(all terms are positive).  A pair is skipped only if ALL three bounds fall
below tile_tol; tile_tol = 0 disables pruning entirely (the acceptance
reference).
"""
import numpy

from pyscf_wb97mv_fast.gpu import backends, device
from pyscf_wb97mv_fast.gpu.ao_eval import plan_blocks
from pyscf_wb97mv_fast.gpu.precision import kahan_sum_axis
from pyscf_wb97mv_fast.gpu.xc import Unsupported, _prepare_dm

THRESH = 1e-8

# Fused double sum for the cupy backend (spec section 7: "CuPy + RawKernel").
# One thread per outer point; inner points stream through shared memory in
# 256-point tiles.  Inside a tile the three sums are FP32 with Kahan
# compensation, across tiles they accumulate in FP64 in a fixed order: no
# atomics, bitwise reproducible.  Same formula as the numpy path below
# (libdft.VXC_vv10nlc); F is returned raw, the caller applies the -1.5.
_FUSED_TILE = 256
_FUSED_SRC = r'''
#define TILE %d
extern "C" __global__
void vv10_kernel_sum(const float* __restrict__ xyz, const float* __restrict__ q,
                     const float* __restrict__ w0, const float* __restrict__ kap,
                     const int n, const int i0, const int i1,
                     double* __restrict__ F, double* __restrict__ U,
                     double* __restrict__ W)
{
    __shared__ float sx[TILE], sy[TILE], sz[TILE], sq[TILE], sw[TILE], sk[TILE];
    const int i = i0 + blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < i1;
    float xi = 0.f, yi = 0.f, zi = 0.f, w0i = 0.f, ki = 1.f;
    if (active) {
        xi = xyz[3 * i]; yi = xyz[3 * i + 1]; zi = xyz[3 * i + 2];
        w0i = w0[i]; ki = kap[i];
    }
    double f64 = 0.0, u64 = 0.0, w64 = 0.0;
    for (int j0 = 0; j0 < n; j0 += TILE) {
        const int j = j0 + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = xyz[3 * j]; sy[threadIdx.x] = xyz[3 * j + 1];
            sz[threadIdx.x] = xyz[3 * j + 2];
            sq[threadIdx.x] = q[j]; sw[threadIdx.x] = w0[j]; sk[threadIdx.x] = kap[j];
        }
        __syncthreads();
        if (active) {
            const int m = min(TILE, n - j0);
            float fs = 0.f, fc = 0.f, us = 0.f, uc = 0.f, ws = 0.f, wc = 0.f;
            for (int t = 0; t < m; ++t) {
                const float dx = xi - sx[t], dy = yi - sy[t], dz = zi - sz[t];
                const float r2 = dx * dx + dy * dy + dz * dz;
                const float g = r2 * w0i + ki;
                const float gp = r2 * sw[t] + sk[t];
                const float gt = g + gp;
                const float T = sq[t] / (g * gp * gt);
                const float Tu = T * (1.f / g + 1.f / gt);
                const float Tw = Tu * r2;
                float y, s;
                y = T - fc;  s = fs + y; fc = (s - fs) - y; fs = s;
                y = Tu - uc; s = us + y; uc = (s - us) - y; us = s;
                y = Tw - wc; s = ws + y; wc = (s - ws) - y; ws = s;
            }
            f64 += (double)fs; u64 += (double)us; w64 += (double)ws;
        }
        __syncthreads();
    }
    if (active) { F[i] = f64; U[i] = u64; W[i] = w64; }
}
''' % _FUSED_TILE
_FUSED_LAUNCH = 65536          # outer points per launch
_fused_kernel = None


def _get_fused_kernel(cp):
    global _fused_kernel
    if _fused_kernel is None:
        _fused_kernel = cp.RawKernel(_FUSED_SRC, 'vv10_kernel_sum')
    return _fused_kernel


class Vv10Kernel:
    """The O(N_g^2) double sum, streamed tile by tile."""

    def __init__(self, xp, mem_budget=256 << 20, tile_tol=0.0, n_seg=8,
                 outer_chunk=2048):
        self.xp = xp
        self.mem_budget = int(mem_budget)
        self.tile_tol = float(tile_tol)
        self.n_seg = int(n_seg)
        self.outer_chunk = int(outer_chunk)
        self.last_pairs_kept = None       # diagnostics from the last call

    def _inner_chunk(self):
        n_temp = 8
        per_col = 4 * n_temp
        return max(1024, min(65536, self.mem_budget //
                             (self.outer_chunk * per_col)))

    # -- tile pruning bounds (CPU numpy) -------------------------------------
    def _tile_stats(self, coords_host, q_host, w0_host, k_host, size):
        n = coords_host.shape[0]
        nt = -(-n // size)
        stats = {'q_sum': numpy.zeros(nt), 'w0_min': numpy.zeros(nt),
                 'k_min': numpy.zeros(nt), 'cent': numpy.zeros((nt, 3)),
                 'radius': numpy.zeros(nt)}
        for t in range(nt):
            sl = slice(t * size, min((t + 1) * size, n))
            c = coords_host[sl]
            q = q_host[sl]
            stats['q_sum'][t] = q.sum()
            stats['w0_min'][t] = w0_host[sl].min()
            stats['k_min'][t] = k_host[sl].min()
            centroid = c.mean(axis=0)
            stats['cent'][t] = centroid
            stats['radius'][t] = numpy.linalg.norm(c - centroid, axis=1).max()
        return stats

    def _pair_mask(self, stats_o, stats_i):
        co, ci = len(stats_o['q_sum']), len(stats_i['q_sum'])
        if self.tile_tol <= 0.0:
            self.last_pairs_kept = (co * ci, co * ci)
            return numpy.ones((co, ci), dtype=bool)
        cent_o, cent_i = stats_o['cent'][:, None, :], stats_i['cent'][None, :, :]
        dist = numpy.linalg.norm(cent_o - cent_i, axis=2)
        rmin = numpy.maximum(dist - stats_o['radius'][:, None]
                             - stats_i['radius'][None, :], 0.0)
        rmax = dist + stats_o['radius'][:, None] + stats_i['radius'][None, :]
        g_min = rmin ** 2 * stats_o['w0_min'][:, None] + stats_o['k_min'][:, None]
        gp_min = rmin ** 2 * stats_i['w0_min'][None, :] + stats_i['k_min'][None, :]
        b_f = stats_i['q_sum'][None, :] / (g_min * gp_min * (g_min + gp_min))
        b_u = b_f * 2.0 / g_min
        b_w = b_u * rmax ** 2
        keep = numpy.maximum(numpy.maximum(b_f, b_u), b_w) >= self.tile_tol
        self.last_pairs_kept = (int(keep.sum()), co * ci)
        return keep

    # -- the double sum --------------------------------------------------------
    def kernel_sum(self, coords, q, W0, kappa):
        """coords (P,3) FP32 device; q/W0/kappa (P,) -> F, U, W (P,) FP64."""
        xp = self.xp
        n = coords.shape[0]
        F = xp.zeros(n, dtype=xp.float64)
        U = xp.zeros(n, dtype=xp.float64)
        W = xp.zeros(n, dtype=xp.float64)
        if n == 0:
            return F, U, W
        if self.tile_tol <= 0.0 and xp is backends.cupy_module():
            return self._kernel_sum_fused(coords, q, W0, kappa, F, U, W)

        co = self.outer_chunk
        ci = self._inner_chunk()
        coords_h = device.to_host(coords)
        q_h = device.to_host(xp.asarray(q, dtype=xp.float64))
        w0_h = device.to_host(xp.asarray(W0, dtype=xp.float64))
        k_h = device.to_host(xp.asarray(kappa, dtype=xp.float64))
        # one stats set per axis, on the tile sizes the loop below uses:
        # keep[a, b] is the pair (outer tile a of co points, inner tile b of
        # ci points); co != ci in general
        stats_o = self._tile_stats(coords_h, q_h, w0_h, k_h, co)
        stats_i = self._tile_stats(coords_h, q_h, w0_h, k_h, ci)
        keep = self._pair_mask(stats_o, stats_i)
        coords32 = xp.asarray(coords, dtype=xp.float32)
        q32 = xp.asarray(q, dtype=xp.float32)
        w032 = xp.asarray(W0, dtype=xp.float32)
        k32 = xp.asarray(kappa, dtype=xp.float32)

        for io0 in range(0, n, co):
            io1 = min(io0 + co, n)
            xi = coords32[io0:io1]
            w0o = w032[io0:io1]
            ko = k32[io0:io1]
            for ii0 in range(0, n, ci):
                if not keep[io0 // co, ii0 // ci]:
                    continue
                ii1 = min(ii0 + ci, n)
                xj = coords32[ii0:ii1]
                d = xi[:, None, :] - xj[None, :, :]
                R2 = (d * d).sum(axis=2)                       # (co',ci) FP32
                g = R2 * w0o[:, None] + ko[:, None]
                gp = R2 * w032[None, ii0:ii1] + k32[None, ii0:ii1]
                gt = g + gp
                T = q32[None, ii0:ii1] / (g * gp * gt)
                F[io0:io1] += kahan_sum_axis(xp, T, 1, self.n_seg)
                Tu = T * (1.0 / g + 1.0 / gt)
                U[io0:io1] += kahan_sum_axis(xp, Tu, 1, self.n_seg)
                W[io0:io1] += kahan_sum_axis(xp, Tu * R2, 1, self.n_seg)
        # libdft.VXC_vv10nlc convention (numint.c): F is returned as
        # -1.5 x the raw double sum; U and W stay raw.  nr_nlc_vxc builds
        # exc = Beta + F/2 and v0 = Beta + F + 1.5 (U dKdR + W dW0dR)
        # directly on these.
        F *= -1.5
        return F, U, W

    def _kernel_sum_fused(self, coords, q, W0, kappa, F, U, W):
        """cupy, no pruning: the fused RawKernel (see _FUSED_SRC)."""
        cp = self.xp
        n = coords.shape[0]
        nt = -(-n // self.outer_chunk)
        self.last_pairs_kept = (nt * nt, nt * nt)
        xyz = cp.ascontiguousarray(cp.asarray(coords, dtype=cp.float32))
        q32, w032, k32 = (cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32))
                          for a in (q, W0, kappa))
        kernel = _get_fused_kernel(cp)
        for i0 in range(0, n, _FUSED_LAUNCH):
            i1 = min(i0 + _FUSED_LAUNCH, n)
            nblk = -(-(i1 - i0) // _FUSED_TILE)
            kernel((nblk,), (_FUSED_TILE,),
                   (xyz, q32, w032, k32, numpy.int32(n), numpy.int32(i0),
                    numpy.int32(i1), F, U, W))
        F *= -1.5
        return F, U, W


def vv10_pointwise(rho0, grad, weights, nlc_pars, xp):
    """_vv10nlc's per-point FP64 math on the threshed set.

    rho0/grad: (P,) and (3,P) FP64 device arrays, ALREADY threshed (rho0 >=
    1e-8); weights (P,).  Returns (coords-relative) q, W0, kappa, dW0dR,
    dW0dG, dKdR, Beta.
    """
    Bvv, Cvv = nlc_pars
    pi = numpy.pi
    Pi43 = 4.0 * pi / 3.0
    Kvv = Bvv * 1.5 * pi * (9.0 * pi) ** (-1.0 / 6.0)
    Beta = ((3.0 / (Bvv * Bvv)) ** 0.75) / 32.0

    R = rho0
    G = (grad ** 2).sum(axis=0)
    W0tmp = Cvv * (G / (R * R)) ** 2
    W0 = xp.sqrt(W0tmp + Pi43 * R)
    dW0dR = (0.5 * Pi43 * R - 2.0 * W0tmp) / W0
    dW0dG = W0tmp * R / (G * W0)
    kappa = Kvv * R ** (1.0 / 6.0)
    dKdR = kappa / 6.0
    q = R * weights
    return q, W0, kappa, dW0dR, dW0dG, dKdR, Beta


def gpu_nr_nlc_vxc(ctx, ni, mol, grids, xc_code, dm,
                   relativity=0, hermi=1, max_memory=2000, verbose=None):
    """GPU FP32 replacement for NumInt.nr_nlc_vxc (RKS ground state)."""
    xp = ctx.xp
    if relativity not in (0, None):
        raise Unsupported('relativistic NLC not supported')
    dm = _prepare_dm(dm, hermi)
    nao = dm.shape[0]
    nlc_coefs = ni.nlc_coeff(xc_code)
    if not nlc_coefs:
        raise Unsupported('%r has no NLC parameter set' % xc_code)

    blocks = plan_blocks(mol, grids, ctx.blksize)
    ngrids = grids.coords.shape[0]
    weights_d = xp.asarray(grids.weights, dtype=xp.float64)
    coords64 = grids.coords - ctx.origin
    ao_loc = ctx.evaluator.pack.ao_loc

    # pass 1: GGA density on the VV10 grid (FP32 GEMMs, FP64 storage)
    rho4 = xp.zeros((4, ngrids), dtype=xp.float64)
    for i0, i1, shells in blocks:
        idx = device.ao_indices(ao_loc, shells)
        coords = xp.asarray(coords64[i0:i1], dtype=xp.float64)   # FP64: see AoEvaluator.eval
        ao = ctx.evaluator.eval(coords, shells, deriv=1)
        gd = ctx.gemm_dtype
        ao_g = ao if ao.dtype == gd else ao.astype(gd)
        dm_g = xp.asarray(device.gather_dm(dm, idx), dtype=gd)
        tmp = ao_g[0] @ dm_g
        rho4[0, i0:i1] = (tmp * ao_g[0]).sum(axis=1)
        rho4[1:4, i0:i1] = 2.0 * xp.einsum('pm,gpm->gp', tmp, ao_g[1:4])

    nelec = float((rho4[0] * weights_d).sum())

    exc_full = xp.zeros(ngrids, dtype=xp.float64)
    v0_full = xp.zeros(ngrids, dtype=xp.float64)
    v1_full = xp.zeros(ngrids, dtype=xp.float64)
    coords32_all = xp.asarray(coords64, dtype=xp.float32)

    for nlc_pars, fac in nlc_coefs:
        ind = rho4[0] >= THRESH
        rho0_t = rho4[0][ind]
        grad_t = rho4[1:4][:, ind]
        coords_t = coords32_all[ind]
        w_t = weights_d[ind]
        (q, W0, kappa, dW0dR, dW0dG, dKdR, Beta) = vv10_pointwise(
            rho0_t, grad_t, w_t, nlc_pars, xp)
        F, U, W = ctx.kernel.kernel_sum(coords_t, q, W0, kappa)
        exc_full[ind] += fac * (Beta + 0.5 * F)
        v0_full[ind] += fac * (Beta + F + 1.5 * (U * dKdR + W * dW0dR))
        v1_full[ind] += fac * (1.5 * W * dW0dG)

    excsum = float((rho4[0] * weights_d * exc_full).sum())

    # transform_vxc('GGA', spin=0): [v0, 2 v1 gx, 2 v1 gy, 2 v1 gz]
    vv_vxc = xp.empty((4, ngrids), dtype=xp.float64)
    vv_vxc[0] = v0_full
    vv_vxc[1:4] = 2.0 * v1_full[None, :] * rho4[1:4]

    # pass 2: fock_gemm (FP32 partials, FP64 vmat)
    vmat = xp.zeros((nao, nao), dtype=xp.float64)
    for i0, i1, shells in blocks:
        idx = device.ao_indices(ao_loc, shells)
        coords = xp.asarray(coords64[i0:i1], dtype=xp.float64)   # FP64: see AoEvaluator.eval
        ao = ctx.evaluator.eval(coords, shells, deriv=1)
        gd = ctx.gemm_dtype
        ao_g = ao if ao.dtype == gd else ao.astype(gd)
        w = weights_d[i0:i1]
        wv = (vv_vxc[:, i0:i1] * w[None, :]).astype(gd)
        wv[0] *= 0.5                                  # *.5 for vmat + vmat.T
        aow = xp.einsum('dpm,dp->pm', ao_g, wv[:4])
        blk = ao_g[0].T @ aow
        device.scatter_add(xp, vmat, idx, blk)

    vmat = vmat + vmat.T
    return nelec, excsum, device.to_host(vmat)
