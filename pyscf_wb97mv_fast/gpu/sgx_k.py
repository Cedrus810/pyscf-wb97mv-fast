"""S5 Tasks 4+5: fused GPU COSX exchange K (FP32) with per-block screening.

``GpuKBuilder`` implements the spec section 7 KBuilder for the SGX grid:

    per grid tile:  X = AoEvaluator(..., FP64)
                    F = fl32(X @ (proj @ dm))     FP64 GEMM, rounded once
                    fused RawKernel: for every (sub-block, kept shell pair)
                        A_g,munu = (mu|1/|r-g||nu)   (never materialized;
                        computed and contracted inside the kernel)
                        G[g,nu] += sum_mu A[g,mu,nu] F[g,mu]     (FP32)
                        (+ the transposed contraction for ish != jsh)
                    K += X^T (w G)                FP64 weights and GEMM

Why X is evaluated in FP64 (water dimer): the FP32
AoEvaluator X carried 1.54e-7 of the 1.64e-7 long-range and 1.33e-6 of the
full-K energy error (partly cancelled by other terms).  Two mechanisms, both
systematic: (a) its hi+lo constant compensation adds the lo term to an
already-rounded FP32 sum, where it is mostly rounded away (per-AO bias
1.5e-8 left of 4.1e-8); (b) even a correctly rounded FP32 X is biased on the
atom-centred SGX grid: one-centre AO values are identical on every angular
point of a radial shell and on every atom of an element, so their rounding
errors add coherently (fl32(X64) alone: -1.6e-7; 7e-9 once the grid is
jittered by 1e-6 bohr).  The same coherence makes FP32 GEMM accumulation
biased: F = X proj dm as an FP32 GEMM cost -1.5e-7 and K = X^T G -1.3e-7
(full K, dimer), against 5e-9 for rounding the exact F once; the FP32
weights (identical on every same-element atom near the nuclei) +5e-8.  So
X, both GEMMs and the weights are FP64; only F is rounded (once) to FP32 for
the kernel, whose G comes back unweighted.

then K = proj^T K if the SGX object uses sym_ovlp, and K = (K+K^T)/2 for
hermi=1 -- exactly sgx_jk.get_k_only's post-processing.  Every (sub-block,
pair) task is written by exactly one thread; K accumulates in a fixed tile
order: no atomics anywhere, bitwise reproducible.

Screening (Task 5): for a (pair, sub-block) task,

    est = bound(a, b, B) * max_g w_g * max(sum_t in a |F_gt|, sum_t in b |F_gt|)

with bound(...) = shellpairs.pair_bound (locked definition; valid for
omega = 0 and omega > 0) and the F sums over the whole shells a and b, so
est is a strict upper bound of the task's contribution to any G entry
(the benefit comes from F = X proj P decaying away from the molecule,
spec section 8.5).  Tasks with est < tile_tol are skipped; tile_tol = 0
keeps everything (the acceptance reference).  The bounds are
dm-independent and cached per grid (per 64-sub chunk); the F part is
recomputed per call on the host (deterministic).

Scope (raises gpu.xc.Unsupported -> the install hook falls back to the
CPU): single symmetric RKS density matrix, hermi = 1, spherical l <= 3
basis (shellpairs.LMAX; def2-svp and def2-tzvp), cupy backend.
"""
import os
import time

import numpy
from pyscf.lib import logger

from pyscf_wb97mv_fast.gpu import device
from pyscf_wb97mv_fast.gpu.int3c1e import CORE_SRC
from pyscf_wb97mv_fast.gpu.boys import BOYS_SRC
from pyscf_wb97mv_fast.gpu.shellpairs import LMAX, E_T, ShellPairs, numpy_origin
from pyscf_wb97mv_fast.gpu.xc import Unsupported

KSUB = 32                  # grid points per sub-block = one warp
_WARPS_MAX = 4             # sub-blocks (warps) per CUDA block, upper limit
_WARPS_PER_SM = 32         # target warps per launch and SM (auto blksize)
_BYTES_PER_POINT_NAO = 64  # tile bytes per point and AO: X64, F, G and the
                           # FP64 GEMM temporaries (~32), x2 headroom
_CHUNK = 64                # sub-blocks per cached screening-bound chunk
_STATE_CACHE = 4           # grid states kept (full + LR copies alternate)

_FUSED_SRC = (BOYS_SRC + CORE_SRC + r'''
extern "C" __global__ void sgx_k_tile(
    const int* __restrict__ sh_dims,
    const float* __restrict__ sh_Mc,
    const float* __restrict__ sh_MT,
    const float* __restrict__ sh_Mc_lo,
    const float* __restrict__ sh_MT_lo,
    const int* __restrict__ pair_sh,
    const double* __restrict__ p_arr,
    const float* __restrict__ prefac,
    const float* __restrict__ prefac_lo,
    const float* __restrict__ P_hi,
    const float* __restrict__ P_lo,
    const float* __restrict__ Earr,
    const float* __restrict__ tp_hi,    /* per-omega data, unused here */
    const float* __restrict__ tp_lo,
    const float* __restrict__ cn_hi,
    const float* __restrict__ cn_lo,
    const float* __restrict__ coords,   /* (npts, 3), pairs' frame */
    const float* __restrict__ coords_lo,
    const float* __restrict__ Fc,       /* (npts, nao) */
    const int* __restrict__ sub_off,    /* (nsub+1) CSR offsets */
    const int* __restrict__ sub_ids,    /* kept pair ids, ascending per sub */
    int npts, int nao, int ks, int nsub, double omega,
    float* __restrict__ Gout)           /* (npts, nao), zeroed by the caller */
{
    /* one warp per sub-block of ks = 32 points, one thread per point: the
       warp walks its sub-block's pair list in lockstep.  Each thread owns
       its G row in global memory (no other thread touches it), so there are
       no atomics and the accumulation order is fixed (bitwise
       reproducible).  The row used to live in shared memory, which capped
       a block at 44 KB / (4 nao) threads -- 17 for nao = 648, one block per
       SM. */
    const int sub = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (sub >= nsub) return;
    const int gp = sub * ks + (threadIdx.x & 31);
    if (gp >= npts) return;
    float* Grow = Gout + (size_t)gp * nao;

    const float* Fg = Fc + (size_t)gp * nao;
    const float gx = coords[gp * 3], gy = coords[gp * 3 + 1],
                gz = coords[gp * 3 + 2];
    const float gxl = coords_lo[gp * 3], gyl = coords_lo[gp * 3 + 1],
                gzl = coords_lo[gp * 3 + 2];
    const int i0 = sub_off[sub], i1 = sub_off[sub + 1];
    for (int i = i0; i < i1; ++i) {
        const int kk = sub_ids[i];
        const int* sh6 = pair_sh + kk * 6;
        float A[SP_NIA * SP_NIA];
        int3c1e_pair_block(sh_dims, sh_Mc, sh_MT, sh_Mc_lo, sh_MT_lo, sh6,
                           p_arr, prefac, prefac_lo, P_hi, P_lo, Earr,
                           gx, gy, gz, gxl, gyl, gzl, omega, A);
        const int ia0 = sh6[0], ib0 = sh6[1];
        const int nia = sh_dims[sh6[2] * 4 + 0];
        const int nib = sh_dims[sh6[3] * 4 + 0];
        if (ia0 == ib0) {
            /* diagonal shell pair: A is symmetric, one contraction only */
            for (int nu = 0; nu < nib; ++nu) {
                float s = 0.f;
                for (int mu = 0; mu < nia; ++mu)
                    s += A[mu * SP_NIA + nu] * Fg[ia0 + mu];
                Grow[ib0 + nu] += s;
            }
        } else {
            for (int nu = 0; nu < nib; ++nu) {
                float s = 0.f;
                for (int mu = 0; mu < nia; ++mu)
                    s += A[mu * SP_NIA + nu] * Fg[ia0 + mu];
                Grow[ib0 + nu] += s;
            }
            for (int mu = 0; mu < nia; ++mu) {
                float s = 0.f;
                for (int nu = 0; nu < nib; ++nu)
                    s += A[mu * SP_NIA + nu] * Fg[ib0 + nu];
                Grow[ia0 + mu] += s;
            }
        }
    }
}
''' + r'''
/* ---- P3: (la, lb)-specialized kernels ------------------------------------
   Same math and the same operation order as int3c1e_pair_block for every
   entry that is used, but with compile-time sizes: every per-thread array
   (A, B, Ac, R, R000) is exactly as large as the shell pair needs and every
   loop is fully unrolled, so they live in registers instead of the 2 KB of
   local memory the generic version uses (A[256], R[125], B[96], Ac[36]
   indexed at run time).  That local memory was the kernel's bottleneck:
   at full occupancy it thrashed L1 and ran several times slower than at
   64 resident threads/SM, while the double-precision Boys/theta work
   costs only a few percent of the cycle.  Single-contraction shells
   only (nctr =
   1, all of def2-svp); pairs with a generally contracted shell use the
   generic kernel. */
template<int LA, int LB>
__device__ __forceinline__ void pair_block_t(
    const int* __restrict__ sh_dims,
    const float* __restrict__ sh_Mc,
    const float* __restrict__ sh_MT,
    const float* __restrict__ sh_Mc_lo,
    const float* __restrict__ sh_MT_lo,
    const int* __restrict__ sh6,
    const double* __restrict__ p_arr,
    const float* __restrict__ prefac,
    const float* __restrict__ prefac_lo,
    const float* __restrict__ P_hi,
    const float* __restrict__ P_lo,
    const float* __restrict__ Earr,
    const float* __restrict__ tp_hi,     /* theta p, hi + lo (per omega) */
    const float* __restrict__ tp_lo,
    const float* __restrict__ cn_hi,     /* (nprims, SP_ET): sqrt(theta) */
    const float* __restrict__ cn_lo,     /* (-2 p theta)^n, hi + lo */
    float gx, float gy, float gz,
    float gxl, float gyl, float gzl,
    float* A)                            /* (2LA+1) x (2LB+1), row-major */
{
    constexpr int NA = 2 * LA + 1, NB = 2 * LB + 1;
    constexpr int NCA = (LA + 1) * (LA + 2) / 2, NCB = (LB + 1) * (LB + 2) / 2;
    constexpr int L = LA + LB, D = L + 1;
    const int ish = sh6[2], jsh = sh6[3];
    const int pr0 = sh6[4];
    const int npi = sh_dims[ish * 4 + 1], npj = sh_dims[jsh * 4 + 1];
    const float* Mc = sh_Mc + ish * SP_NIA * SP_KMAX;
    const float* MT = sh_MT + jsh * SP_KMAX * SP_NIA;
    const float* Mc_lo = sh_Mc_lo + ish * SP_NIA * SP_KMAX;
    const float* MT_lo = sh_MT_lo + jsh * SP_KMAX * SP_NIA;

    #pragma unroll
    for (int i = 0; i < NA * NB; ++i) A[i] = 0.f;
    for (int a = 0; a < npi; ++a) {
        float B[NCA * NB];
        #pragma unroll
        for (int i = 0; i < NCA * NB; ++i) B[i] = 0.f;
        for (int b = 0; b < npj; ++b) {
            const int m = pr0 + a * npj + b;
            /* P2: theta = w^2 / (w^2 + p), theta p and the R000
               coefficients sqrt(theta) (-2 p theta)^n come from the host
               (GpuKBuilder._omega_data, double -> hi + lo): no double
               division / sqrt per grid point (FP64 is 1/64 on consumer
               Blackwell; the LR kernel ran 2.2x the full one on the 5080) */
            const float dx = (P_hi[m * 3 + 0] - gx) + (P_lo[m * 3 + 0] - gxl);
            const float dy = (P_hi[m * 3 + 1] - gy) + (P_lo[m * 3 + 1] - gyl);
            const float dz = (P_hi[m * 3 + 2] - gz) + (P_lo[m * 3 + 2] - gzl);
            const float r2 = dx * dx + dy * dy + dz * dz;
            const float T = fmaf(tp_hi[m], r2, tp_lo[m] * r2);
            float Fb[L + 1];
            boys_fp32(T, Fb, L);
            float R000[L + 1];
            #pragma unroll
            for (int n = 0; n <= L; ++n)
                R000[n] = fmaf(cn_hi[m * SP_ET + n], Fb[n],
                               cn_lo[m * SP_ET + n] * Fb[n]);
            float R[D * D * D];
            #pragma unroll
            for (int i = 0; i < D * D * D; ++i) R[i] = 0.f;
            #pragma unroll
            for (int n = L; n >= 0; --n) {
                #pragma unroll
                for (int t = L; t >= 1; --t)
                    #pragma unroll
                    for (int u = 0; u < D; ++u)
                        #pragma unroll
                        for (int v = 0; v < D; ++v) {
                            const int idx = (t * D + u) * D + v;
                            if (t == 1)
                                R[idx] = dx * R[(0 * D + u) * D + v];
                            else
                                R[idx] = (float)(t - 1) * R[((t - 2) * D + u) * D + v]
                                       + dx * R[((t - 1) * D + u) * D + v];
                        }
                #pragma unroll
                for (int u = L; u >= 1; --u)
                    #pragma unroll
                    for (int v = 0; v < D; ++v) {
                        const int idx = (0 * D + u) * D + v;
                        if (u == 1)
                            R[idx] = dy * R[v];
                        else
                            R[idx] = (float)(u - 1) * R[(0 * D + u - 2) * D + v]
                                   + dy * R[(0 * D + u - 1) * D + v];
                    }
                #pragma unroll
                for (int v = L; v >= 1; --v) {
                    if (v == 1)
                        R[v] = dz * R[0];
                    else
                        R[v] = (float)(v - 1) * R[v - 2] + dz * R[v - 1];
                }
                R[0] = R000[n];
            }

            const float* Em = Earr + m * SP_ESTRIDE;
            const float pf = prefac[m], pf_lo = prefac_lo[m];
            float Ac[NCA * NCB];
            #pragma unroll
            for (int ca = 0; ca < NCA; ++ca) {
                int i, j, k;
                sp_cart_comp(LA, ca, &i, &j, &k);
                #pragma unroll
                for (int cb = 0; cb < NCB; ++cb) {
                    int ip, jp, kp;
                    sp_cart_comp(LB, cb, &ip, &jp, &kp);
                    float s = 0.f;
                    #pragma unroll
                    for (int t = 0; t <= L; ++t) {
                        if (t > i + ip) break;
                        const float ex = Em[(i * SP_EL + ip) * SP_ET + t];
                        #pragma unroll
                        for (int u = 0; u <= L; ++u) {
                            if (u > j + jp) break;
                            const float exy = ex * Em[SP_EY + (j * SP_EL + jp) * SP_ET + u];
                            #pragma unroll
                            for (int v = 0; v <= L; ++v) {
                                if (v > k + kp) break;
                                s += exy * Em[SP_EZ + (k * SP_EL + kp) * SP_ET + v]
                                        * R[(t * D + u) * D + v];
                            }
                        }
                    }
                    Ac[ca * NCB + cb] = fmaf(pf, s, pf_lo * s);
                }
            }
            const float* MTb = MT + b * NCB * SP_NIA;
            const float* MTb_lo = MT_lo + b * NCB * SP_NIA;
            #pragma unroll
            for (int ca = 0; ca < NCA; ++ca)
                #pragma unroll
                for (int out = 0; out < NB; ++out) {
                    float s = 0.f;
                    #pragma unroll
                    for (int cb = 0; cb < NCB; ++cb) {
                        const float av = Ac[ca * NCB + cb];
                        s += fmaf(av, MTb[cb * SP_NIA + out],
                                  av * MTb_lo[cb * SP_NIA + out]);
                    }
                    B[ca * NB + out] += s;
                }
        }
        #pragma unroll
        for (int row = 0; row < NA; ++row)
            #pragma unroll
            for (int out = 0; out < NB; ++out) {
                float s = 0.f;
                #pragma unroll
                for (int ca = 0; ca < NCA; ++ca) {
                    const float bv = B[ca * NB + out];
                    s += fmaf(Mc[row * SP_KMAX + a * NCA + ca], bv,
                              Mc_lo[row * SP_KMAX + a * NCA + ca] * bv);
                }
                A[row * NB + out] += s;
            }
    }
    if (LA == LB && ish == jsh) {
        #pragma unroll
        for (int mu = 0; mu < NA; ++mu)
            #pragma unroll
            for (int nu = mu + 1; nu < NB; ++nu) {
                const float v = (A[mu * NB + nu] + A[nu * NB + mu]) * 0.5f;
                A[mu * NB + nu] = v;
                A[nu * NB + mu] = v;
            }
    }
}

template<int LA, int LB>
__global__ void sgx_k_cls(
    const int* __restrict__ sh_dims,
    const float* __restrict__ sh_Mc,
    const float* __restrict__ sh_MT,
    const float* __restrict__ sh_Mc_lo,
    const float* __restrict__ sh_MT_lo,
    const int* __restrict__ pair_sh,
    const double* __restrict__ p_arr,
    const float* __restrict__ prefac,
    const float* __restrict__ prefac_lo,
    const float* __restrict__ P_hi,
    const float* __restrict__ P_lo,
    const float* __restrict__ Earr,
    const float* __restrict__ tp_hi,
    const float* __restrict__ tp_lo,
    const float* __restrict__ cn_hi,
    const float* __restrict__ cn_lo,
    const float* __restrict__ coords,
    const float* __restrict__ coords_lo,
    const float* __restrict__ Fc,
    const int* __restrict__ sub_off,
    const int* __restrict__ sub_ids,
    int npts, int nao, int ks, int nsub, double omega,
    float* __restrict__ Gout)
{
    constexpr int NA = 2 * LA + 1, NB = 2 * LB + 1;
    const int sub = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (sub >= nsub) return;
    const int gp = sub * ks + (threadIdx.x & 31);
    if (gp >= npts) return;
    float* Grow = Gout + (size_t)gp * nao;
    const float* Fg = Fc + (size_t)gp * nao;
    const float gx = coords[gp * 3], gy = coords[gp * 3 + 1],
                gz = coords[gp * 3 + 2];
    const float gxl = coords_lo[gp * 3], gyl = coords_lo[gp * 3 + 1],
                gzl = coords_lo[gp * 3 + 2];
    const int i0 = sub_off[sub], i1 = sub_off[sub + 1];
    for (int i = i0; i < i1; ++i) {
        const int* sh6 = pair_sh + sub_ids[i] * 6;
        float A[NA * NB];
        pair_block_t<LA, LB>(sh_dims, sh_Mc, sh_MT, sh_Mc_lo, sh_MT_lo, sh6,
                             p_arr, prefac, prefac_lo, P_hi, P_lo, Earr,
                             tp_hi, tp_lo, cn_hi, cn_lo,
                             gx, gy, gz, gxl, gyl, gzl, A);
        const int ia0 = sh6[0], ib0 = sh6[1];
        #pragma unroll
        for (int nu = 0; nu < NB; ++nu) {
            float s = 0.f;
            #pragma unroll
            for (int mu = 0; mu < NA; ++mu)
                s += A[mu * NB + nu] * Fg[ia0 + mu];
            Grow[ib0 + nu] += s;
        }
        if (ia0 != ib0) {
            #pragma unroll
            for (int mu = 0; mu < NA; ++mu) {
                float s = 0.f;
                #pragma unroll
                for (int nu = 0; nu < NB; ++nu)
                    s += A[mu * NB + nu] * Fg[ib0 + nu];
                Grow[ia0 + mu] += s;
            }
        }
    }
}
''')

_FUSED_KERNELS = {}

# P4: the screening estimate on the GPU, one thread per (sub-block, pair).
# Same locked bound as shellpairs.pair_bound (block centroid / radius,
# d_min, p' = 0.75 p, F_0) times max(sum_a |F| w, sum_b |F| w) of the
# sub-block, as the host reference GpuKBuilder._kept_host.  The host
# version was far too slow per water27 K call to keep on the host.
# Evaluated in FP32 (FP64 is 1/64 on the target cards) and kept a strict
# upper bound by construction: d_min is shrunk by a conservative margin
# (F_0 decreases with d, so a smaller d only raises the bound) and the
# estimate is inflated by _SCREEN_SAFETY before the comparison, which
# covers the FP32 rounding of the prefactor sum and of the F GEMM (positive
# sums: relative error <= nao * eps ~ 4e-5).  Only tasks with an estimate
# within ~0.1% of tile_tol are kept that the exact bound would skip.
_SCREEN_SRC = r'''
extern "C" __global__ void sgx_screen_mask(
    const float* __restrict__ cen,      /* (nsub, 3), pairs' frame */
    const float* __restrict__ rad,      /* (nsub,) */
    const float* __restrict__ msh,      /* (nsub, nbas) */
    const float* __restrict__ P,        /* (nprims, 3) */
    const float* __restrict__ pref,     /* (nprims,) bound prefactors */
    const float* __restrict__ pp,       /* (nprims,) p' */
    const int* __restrict__ poff,       /* (npairs + 1) prim offsets */
    const float* __restrict__ tau,      /* (npairs,) */
    const int* __restrict__ ish,
    const int* __restrict__ jsh,
    int nsub, int npairs, int nbas, float tol_over_safety,
    unsigned char* __restrict__ keep)   /* (nsub, npairs) */
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nsub * npairs) return;
    const int s = (int)(idx / npairs);
    const int k = (int)(idx - (long long)s * npairs);
    const float f = fmaxf(msh[(size_t)s * nbas + ish[k]],
                          msh[(size_t)s * nbas + jsh[k]]);
    const float cx = cen[s * 3], cy = cen[s * 3 + 1], cz = cen[s * 3 + 2];
    const float r = rad[s];
    float b = 0.f;
    for (int m = poff[k]; m < poff[k + 1]; ++m) {
        const float dx = P[m * 3] - cx, dy = P[m * 3 + 1] - cy,
                    dz = P[m * 3 + 2] - cz;
        const float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        /* conservative: shrink d by more than its FP32 error */
        float d = dist - r - 1e-5f * (dist + r) - 1e-6f;
        if (d < 0.f) d = 0.f;
        const float T = pp[m] * d * d;
        /* F_0(T) = sqrt(pi/T)/2 erf(sqrt(T)); erff/sqrtf errors are
           covered by the safety factor */
        const float f0 = (T > 1e-12f)
            ? 0.88622692545f * erff(sqrtf(T)) * rsqrtf(T) : 1.f;
        b += pref[m] * f0;
    }
    keep[idx] = (tau[k] * b * f >= tol_over_safety) ? 1 : 0;
}
'''
_SCREEN_KERNELS = {}
_SCREEN_SAFETY = 1.001     # estimate inflation covering the FP32 roundings


def _screen_kernel(cp):
    key = id(cp)
    if key not in _SCREEN_KERNELS:
        _SCREEN_KERNELS[key] = cp.RawKernel(_SCREEN_SRC, 'sgx_screen_mask')
    return _SCREEN_KERNELS[key]


# task classes: (la, lb) of single-contraction shell pairs -> specialized
# kernel, class id la * (LMAX + 1) + lb; GENERIC = pairs with a generally
# contracted shell (any l <= LMAX)
_CLASS_NAMES = ['sgx_k_cls<%d,%d>' % (la, lb)
                for la in range(LMAX + 1) for lb in range(LMAX + 1)]
GENERIC = len(_CLASS_NAMES)


def _fused_module(cp):
    key = id(cp)
    if key not in _FUSED_KERNELS:
        _FUSED_KERNELS[key] = cp.RawModule(code=_FUSED_SRC,
                                           options=('-std=c++14',),
                                           name_expressions=_CLASS_NAMES)
    return _FUSED_KERNELS[key]


def _fused_kernel(cp, cls=GENERIC):
    mod = _fused_module(cp)
    return mod.get_function('sgx_k_tile' if cls == GENERIC
                            else _CLASS_NAMES[cls])


def _nsub(npts, ks):
    return -(-npts // ks)


class _GridState:
    """Everything about one (grids object, point count, blksize) that
    survives across SCF cycles: coordinate/weight copies, tile/sub layout
    and the dm-independent screening bounds (per 64-sub chunk, lazily
    built, bounded cache)."""

    def __init__(self, builder, grids, blksize):
        xp = builder.xp
        coords = numpy.asarray(grids.coords, dtype=numpy.float64)
        weights = numpy.asarray(grids.weights, dtype=numpy.float64)
        ngrids = coords.shape[0]
        self.ngrids = ngrids
        self.coords_abs = coords
        rel = coords - builder.origin
        self.coords64 = xp.asarray(numpy.ascontiguousarray(rel))
        self.coords32 = xp.asarray(rel.astype(numpy.float32))
        self.coords32_lo = xp.asarray(
            (rel - rel.astype(numpy.float32)).astype(numpy.float32))
        self.w64 = xp.asarray(weights)
        self.w_host = weights
        self.ks = KSUB
        # tiles aligned to the sub-block grid: i0 // ks must be exact
        self.blksize = max(self.ks, (int(blksize) // self.ks) * self.ks)
        self.tiles = [(i0, min(i0 + self.blksize, ngrids))
                      for i0 in range(0, ngrids, self.blksize)]
        self._bounds = {}

    def bounds_chunk(self, builder, chunk):
        """(npairs, <=CHUNK) screening bounds of the sub-blocks starting at
        chunk*CHUNK.  dm-independent, cached (bounded)."""
        if chunk not in self._bounds:
            s0, s1 = chunk * _CHUNK, min((chunk + 1) * _CHUNK, self.nsub_full)
            ks = self.ks
            cols = []
            for s in range(s0, s1):
                i0, i1 = s * ks, min((s + 1) * ks, self.ngrids)
                cols.append(builder._pairs_host.bound_at(self.coords_abs[i0:i1]))
            b = (numpy.stack(cols, axis=1) if cols else
                 numpy.zeros((builder._pairs_host.npairs, 0)))
            self._bounds[chunk] = b
            if len(self._bounds) > 16:
                self._bounds.pop(next(iter(self._bounds)))
        return self._bounds[chunk]

    @property
    def nsub_full(self):
        return _nsub(self.ngrids, self.ks)


class GpuKBuilder:
    """spec section 7 KBuilder, GPU FP32 edition (S5).

    One per (mol, backend); caches the shell-pair pack (mol-only) and, per
    (grids object, weights size, blksize), a grid state.  A rebuilt SGX
    grid (SGX.build creates a new grids object, and a level change also
    changes the point count) gets a fresh state -- stale cached grid data
    is never reused (plan Review Focus 4).  The LR copies share the mol but
    carry their own grids, so one builder serves the full and the long-
    range K (Review Focus 3); omega comes from sgx.mol.omega, which the
    with_range_coulomb context sets on the shared mol.
    """

    def __init__(self, mol, xp, tile_tol=1e-11, mem_budget=None,
                 origin=None, verify_ao=True, warps_per_block=None,
                 blksize=None):
        self.mol = mol
        self.xp = xp
        self.tile_tol = float(tile_tol)
        self.origin = (numpy_origin(mol.atom_coords().mean(axis=0))
                       if origin is None else numpy_origin(origin))
        self.nao = int(mol.nao)
        if mem_budget is None:
            mem_budget = (device.DeviceState('cupy').mem_budget
                          if hasattr(xp, 'RawKernel') else 256 << 20)
        self.mem_budget = int(mem_budget)
        # grid points per tile = per kernel launch.  One launch must hold
        # enough warps to fill the device: auto = SMs x 32 warps,
        # capped by the memory budget; halved by the OOM retry.
        self.blksize = (int(blksize) if blksize is not None
                        else self._auto_blksize())
        self._built = False
        self._ao = None
        self._pack = None
        self._pairs_dev = None
        self._pairs_host = None
        self._grids_states = {}
        self._omega_cache = {}     # shared by with_blksize clones
        self.last_stats = {}
        self.verify_ao = verify_ao
        # launch shape only: every thread accumulates its own G row in a
        # fixed order, so the result does not depend on it (any GPU)
        self.warps_per_block = warps_per_block
        # True: build the screening mask with the host reference
        # implementation (_kept_host) instead of the device kernel (tests)
        self.host_screening = False

    # -- construction ---------------------------------------------------------
    def _build(self):
        if self._built:
            return
        if not hasattr(self.xp, 'RawKernel'):
            raise Unsupported('GpuKBuilder needs the cupy backend; got %s'
                              % type(self.xp).__name__)
        from pyscf_wb97mv_fast.gpu.ao_eval import AoEvaluator, ShellPack
        from pyscf_wb97mv_fast.gpu.precision import FP64
        self._pack = ShellPack(self.mol, origin=self.origin)
        # FP64 X, split into an FP32 hi/lo pair per tile (module docstring)
        self._ao = AoEvaluator(self.mol, self.xp, pack=self._pack,
                               mem_budget=self.mem_budget, dtype=FP64,
                               verify=self.verify_ao)
        pairs = ShellPairs(self.mol)
        self._pairs_host = _HostPairs(pairs)
        self._pairs_dev = pairs.to_device(self.xp,
                                          mem_budget=self.mem_budget,
                                          origin=self.origin)
        self._scr = self._screen_data(pairs)
        self._built = True

    def _auto_blksize(self):
        nsm = 1
        if hasattr(self.xp, 'RawKernel'):
            nsm = int(self.xp.cuda.Device().attributes['MultiProcessorCount'])
        want = nsm * _WARPS_PER_SM * KSUB
        cap = self.mem_budget // (_BYTES_PER_POINT_NAO * max(self.nao, 1))
        return max(KSUB, (min(want, cap) // KSUB) * KSUB)

    def _screen_data(self, pairs):
        """Device copies of the dm-independent screening-bound data (FP64,
        pairs' frame = relative to self.origin)."""
        xp = self.xp
        mol = self.mol
        ao_loc = numpy.asarray(mol.ao_loc_nr(), dtype=numpy.int64)
        ind = numpy.zeros((self.nao, mol.nbas))
        for sh in range(mol.nbas):
            ind[ao_loc[sh]:ao_loc[sh + 1], sh] = 1.0
        f32 = numpy.float32
        return {
            'P': xp.asarray(numpy.ascontiguousarray(
                (pairs.prim_P - self.origin).astype(f32))),
            # prefactors rounded UP to FP32 so the FP32 bound stays a bound
            'pref': xp.asarray(numpy.nextafter(
                pairs.prim_bound_pref.astype(f32), f32(numpy.inf))),
            'pp': xp.asarray(pairs.prim_pp.astype(f32)),
            'poff': xp.asarray(pairs.prim_offsets.astype(numpy.int32)),
            'tau': xp.asarray(numpy.nextafter(
                pairs.pair_tau.astype(f32), f32(numpy.inf))),
            'ish': xp.asarray(pairs.pair_rows[:, 2].astype(numpy.int32)),
            'jsh': xp.asarray(pairs.pair_rows[:, 3].astype(numpy.int32)),
            'shell_ind': xp.asarray(ind.astype(f32)),
            'class_pairs': [xp.asarray(c) for c in
                            self._pairs_host.class_pairs],
        }

    def with_blksize(self, blksize):
        """Clone sharing every cache; only the tile size changes (the OOM
        lever, halved by install._run_with_oom_retry).  The shell-pair /
        AO / screening data are built BEFORE the shallow copy: the install
        hook clones on every call (attempt 0 included), and a clone of an
        unbuilt builder used to rebuild everything on the host on every K
        call -- and then throw it away (the 'build' phase timing)."""
        self._build()
        clone = GpuKBuilder.__new__(GpuKBuilder)
        clone.__dict__.update(self.__dict__)
        clone.blksize = int(blksize)
        return clone

    # -- K --------------------------------------------------------------------
    def get_k(self, sgx, dm, omega, hermi=1, direct_scf_tol=1e-13):
        """K (nao, nao) FP64 host for one RKS density matrix on sgx's grid."""
        t0 = time.perf_counter()
        xp = self.xp
        # opt-in phase timing (PYSCF_WB97MV_KPROFILE=1): synchronizes the
        # current stream between phases, diagnostics only
        prof = os.environ.get('PYSCF_WB97MV_KPROFILE') == '1'
        phases = {}
        tp = [time.perf_counter()]

        def mark(name):
            if prof:
                if hasattr(xp, 'cuda'):
                    xp.cuda.get_current_stream().synchronize()
                now = time.perf_counter()
                phases[name] = phases.get(name, 0.0) + now - tp[0]
                tp[0] = now
        if hermi != 1:
            raise Unsupported('GPU COSX K supports hermi=1 only')
        dm = numpy.asarray(dm)
        if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
            raise Unsupported('GPU COSX K supports a single square density '
                              'matrix')
        if not numpy.allclose(dm, dm.T, rtol=0, atol=1e-12):
            raise Unsupported('GPU COSX K needs a symmetric density matrix')
        mark('checks')
        self._build()
        mark('build')
        mol = sgx.mol
        grids = sgx.grids
        nao = self.nao
        if nao != dm.shape[0]:
            raise Unsupported('dm does not match the molecule')

        # proj / sym exactly as sgx_jk.get_k_only sets them up
        if (sgx._pjs_data is None
                or sgx._pjs_data.mol is not mol
                or sgx._pjs_data.grids is not grids
                or sgx._pjs_data._itol != direct_scf_tol):
            sgx._build_pjs(direct_scf_tol)
        proj = sgx._pjs_data._overlap_correction_matrix
        sym = sgx._pjs_data.sym_ovlp
        mark('pjs')

        state = self._grid_state(grids)
        mark('grid_state')
        proj_dm = xp.asarray(numpy.ascontiguousarray(proj @ dm))
        mark('proj_dm')
        K = xp.zeros((nao, nao), dtype=xp.float64)
        shells = numpy.arange(self.mol.nbas, dtype=numpy.int64)
        pairs_kept = 0
        pairs_total = 0
        t_kernel = 0.0

        for (i0, i1) in state.tiles:
            npts = i1 - i0
            X = self._ao.eval(state.coords64[i0:i1], shells,
                              deriv=0)[0]                # (P, nao) FP64
            mark('ao')
            F = (X @ proj_dm).astype(xp.float32)         # FP64 GEMM, one rounding
            mark('fgemm')
            nsub = _nsub(npts, state.ks)
            tasks, nkept = self._tasks(state, i0, i1, F, nsub)
            mark('screen')
            pairs_kept += nkept
            pairs_total += self._pairs_host.npairs * nsub

            tk0 = time.perf_counter()
            G = self._launch(state, i0, i1, F, tasks, nsub, omega)
            mark('kernel')
            t_kernel += time.perf_counter() - tk0
            K += X.T @ (state.w64[i0:i1, None] * G)      # FP64 weights, GEMM
            mark('kgemm')
        K = device.to_host(K)
        mark('to_host')
        if sym:
            K = proj.T @ K
        if hermi == 1:
            K = (K + K.T) * 0.5
        mark('post')
        self.last_stats = {'pairs_total': int(pairs_total),
                           'pairs_kept': int(pairs_kept),
                           'wall': time.perf_counter() - t0,
                           'kernel': t_kernel}
        if prof:
            self.last_stats['phases'] = phases
        logger.debug1(self.mol, 'gpu K: %d/%d tasks kept, wall %.2fs',
                      pairs_kept, pairs_total, self.last_stats['wall'])
        return K

    # -- helpers --------------------------------------------------------------
    def _launch(self, state, i0, i1, F, tasks, nsub, omega):
        """Fused kernels on tile [i0, i1): unweighted G (npts, nao) FP32.
        One launch per task class, in the fixed class order; each thread
        accumulates its own G row, so the result is deterministic and does
        not depend on the launch shape."""
        xp = self.xp
        npts = i1 - i0
        pd = self._pairs_dev
        G = xp.zeros((npts, self.nao), dtype=xp.float32)
        tp_hi, tp_lo, cn_hi, cn_lo = self._omega_data(omega)
        for cls, offs, ids in tasks:
            kern = _fused_kernel(xp, cls)
            warps = self.warps_per_block
            if warps is None:
                # per device and class: as many warps as registers allow
                warps = max(1, min(_WARPS_MAX,
                                   kern.max_threads_per_block // 32))
            kern(
                (-(-nsub // warps),), (32 * warps,),
                (pd.sh_dims, pd.sh_Mc, pd.sh_MT, pd.sh_Mc_lo, pd.sh_MT_lo,
                 pd.pair_sh, pd.prim_p, pd.prim_prefac, pd.prim_prefac_lo,
                 pd.prim_P_hi, pd.prim_P_lo, pd.prim_E,
                 tp_hi, tp_lo, cn_hi, cn_lo,
                 state.coords32[i0:i1], state.coords32_lo[i0:i1], F,
                 offs, ids, numpy.int32(npts), numpy.int32(self.nao),
                 numpy.int32(state.ks), numpy.int32(nsub),
                 numpy.float64(omega), G))
        return G

    def _omega_data(self, omega):
        """Per-omega primitive-pair constants for the specialized kernels
        (P2): theta p and c_n = sqrt(theta) (-2 p theta)^n, n = 0..2 LMAX, formed
        in double in exactly the order the kernel used to (cn = sqrt(theta);
        cn *= q), uploaded as FP32 hi + lo.  Cached per omega."""
        cache = self._omega_cache
        key = float(omega)
        if key not in cache:
            p = self._pairs_host.pairs.prim_p
            if key > 0.0:
                w2 = key * key
                th = w2 / (w2 + p)
                sth = numpy.sqrt(th)
            else:
                th = numpy.ones_like(p)
                sth = numpy.ones_like(p)
            tp = th * p
            q = -2.0 * p * th
            cn = numpy.empty((p.size, E_T))
            c = sth.copy()
            for n in range(E_T):
                cn[:, n] = c
                c = c * q
            hi = lambda a: a.astype(numpy.float32)
            lo = lambda a: (a - a.astype(numpy.float32)).astype(numpy.float32)
            xp = self.xp
            cache[key] = tuple(xp.asarray(numpy.ascontiguousarray(v)) for v in
                               (hi(tp), lo(tp), hi(cn), lo(cn)))
        return cache[key]

    def _tasks(self, state, i0, i1, F, nsub):
        """[(class, device CSR offsets, device pair ids)] of one tile, one
        entry per non-empty task class, and the number of kept tasks.
        Without screening every pair is kept.  Pair ids stay ascending
        within each (class, sub-block)."""
        hp = self._pairs_host
        if self.tile_tol > 0.0 and not self.host_screening:
            return self._tasks_device(state, i0, i1, F, nsub)
        if self.tile_tol > 0.0:
            ids_all, counts = self._kept_host(state, i0, i1, F, nsub)
            sub_of = numpy.repeat(numpy.arange(nsub), counts)
        out, nkept = [], 0
        for cls in range(GENERIC + 1):
            pc = hp.class_pairs[cls]
            if pc.size == 0:
                continue
            if self.tile_tol > 0.0:
                m = hp.pair_class[ids_all] == cls
                ids = ids_all[m]
                cnt = numpy.bincount(sub_of[m], minlength=nsub)
            else:
                ids = numpy.tile(pc, nsub)
                cnt = numpy.full(nsub, pc.size)
            if ids.size == 0:
                continue
            offs = numpy.zeros(nsub + 1, dtype=numpy.int32)
            offs[1:] = numpy.cumsum(cnt)
            nkept += int(ids.size)
            out.append((cls, self.xp.asarray(offs),
                        self.xp.asarray(ids.astype(numpy.int32))))
        return out, nkept

    def kernel_info(self):
        """Per task class: registers, local memory bytes, max threads per
        block of the fused kernels on the current device (diagnostics)."""
        info = {}
        for cls in range(GENERIC + 1):
            kern = _fused_kernel(self.xp, cls)
            name = 'generic' if cls == GENERIC else 'l%d%d' % divmod(cls, LMAX + 1)
            info[name] = (kern.num_regs, kern.local_size_bytes,
                          kern.max_threads_per_block)
        return info

    def _grid_state(self, grids):
        key = (id(grids), grids.weights.size, self.blksize)
        state = self._grids_states.get(key)
        if state is None:
            state = _GridState(self, grids, self.blksize)
            self._grids_states[key] = state
            while len(self._grids_states) > _STATE_CACHE:
                self._grids_states.pop(next(iter(self._grids_states)))
        return state

    def _keep_mask_device(self, state, i0, i1, F, nsub):
        """(nsub, npairs) uint8 device keep mask of one tile (P4)."""
        xp = self.xp
        scr = self._scr
        ks = state.ks
        npts = i1 - i0
        w = state.w64[i0:i1].astype(xp.float32)
        fw = xp.abs(F) * w[:, None]
        seg = fw @ scr['shell_ind']                         # (P, nbas) FP32
        pad = nsub * ks - npts
        c = state.coords64[i0:i1]
        valid = xp.ones(npts, dtype=xp.float64)
        if pad:
            seg = xp.concatenate([seg, xp.zeros((pad, seg.shape[1]),
                                                dtype=seg.dtype)])
            c = xp.concatenate([c, xp.zeros((pad, 3))])
            valid = xp.concatenate([valid, xp.zeros(pad)])
        msh = xp.ascontiguousarray(seg.reshape(nsub, ks, -1).max(axis=1))
        c = c.reshape(nsub, ks, 3)
        valid = valid.reshape(nsub, ks)
        cen = (c * valid[:, :, None]).sum(axis=1) / valid.sum(axis=1)[:, None]
        rad = (xp.sqrt(((c - cen[:, None, :]) ** 2).sum(axis=2))
               * valid).max(axis=1)
        npairs = self._pairs_host.npairs
        keep = xp.empty((nsub, npairs), dtype=xp.uint8)
        nthr = nsub * npairs
        # centroid / radius in FP64, rounded once; the radius rounded up
        cen32 = xp.ascontiguousarray(cen.astype(xp.float32))
        rad32 = xp.ascontiguousarray(
            xp.nextafter(rad.astype(xp.float32), xp.float32(numpy.inf)))
        _screen_kernel(xp)(
            (-(-nthr // 256),), (256,),
            (cen32, rad32, msh,
             scr['P'], scr['pref'], scr['pp'], scr['poff'], scr['tau'],
             scr['ish'], scr['jsh'], numpy.int32(nsub), numpy.int32(npairs),
             numpy.int32(self.mol.nbas),
             numpy.float32(self.tile_tol / _SCREEN_SAFETY), keep))
        return keep

    def _tasks_device(self, state, i0, i1, F, nsub):
        """_tasks with the keep mask and the per-class CSR built on the
        device.  cupy.nonzero is row-major: (sub, pair) order, pair ids
        ascending within a sub-block, deterministic."""
        xp = self.xp
        keep = self._keep_mask_device(state, i0, i1, F, nsub)
        out, nkept = [], 0
        for cls in range(GENERIC + 1):
            cols = self._scr['class_pairs'][cls]
            if cols.size == 0:
                continue
            rows, cidx = xp.nonzero(keep[:, cols])
            if rows.size == 0:
                continue
            cnt = xp.bincount(rows, minlength=nsub)
            offs = xp.zeros(nsub + 1, dtype=xp.int32)
            offs[1:] = xp.cumsum(cnt)
            nkept += int(rows.size)
            out.append((cls, offs, cols[cidx].astype(xp.int32)))
        return out, nkept

    def _kept_host(self, state, i0, i1, F, nsub):
        """Host (pair ids, per-sub counts) of the (pair, sub) tasks of one
        tile with est >= tile_tol; ids ascending within each sub-block.
        Host-side, fixed order -> deterministic."""
        ks = state.ks
        sub0 = i0 // ks                      # tiles are ks-aligned
        F_host = device.to_host(F)
        fw = numpy.abs(F_host) * state.w_host[i0:i1, None]
        seg = numpy.add.reduceat(fw, self._pairs_host.shell_starts, axis=1)
        pad = nsub * ks - (i1 - i0)
        if pad:
            seg = numpy.concatenate(
                [seg, numpy.zeros((pad, seg.shape[1]))], axis=0)
        msh = seg.reshape(nsub, ks, seg.shape[1]).max(axis=1)
        pmax = self._pairs_host.pair_shell_max(msh)          # (nsub, npairs)
        ids_cols, counts = [], numpy.zeros(nsub, dtype=numpy.int64)
        for s in range(nsub):
            bcol = state.bounds_chunk(self, (sub0 + s) // _CHUNK)[
                :, (sub0 + s) % _CHUNK]
            keep = numpy.flatnonzero(bcol * pmax[s] >= self.tile_tol)
            ids_cols.append(keep)
            counts[s] = keep.size
        ids = (numpy.concatenate(ids_cols) if counts.sum()
               else numpy.zeros(0, numpy.int64))
        return ids, counts


class _HostPairs:
    """Host-side helpers over a ShellPairs object."""

    def __init__(self, pairs):
        self.pairs = pairs
        self.npairs = pairs.npairs
        self.shell_starts = numpy.asarray(
            pairs.mol.ao_loc_nr()[:-1], dtype=numpy.int64)
        self.ish = pairs.pair_rows[:, 2].copy()
        self.jsh = pairs.pair_rows[:, 3].copy()
        mol = pairs.mol
        l = numpy.asarray([mol.bas_angular(i) for i in range(mol.nbas)])
        nctr = numpy.asarray([mol.bas_nctr(i) for i in range(mol.nbas)])
        cls = l[self.ish] * (LMAX + 1) + l[self.jsh]
        cls[(nctr[self.ish] != 1) | (nctr[self.jsh] != 1)] = GENERIC
        self.pair_class = cls.astype(numpy.int64)
        self.class_pairs = [numpy.flatnonzero(cls == c).astype(numpy.int32)
                            for c in range(GENERIC + 1)]

    def bound_at(self, coords_block):
        """(npairs,) locked screening bound for one sub-block (absolute
        frame, like the host pairs)."""
        from pyscf_wb97mv_fast.gpu.shellpairs import pair_bound
        return pair_bound(self.pairs, coords_block)

    def pair_shell_max(self, msh):
        """(nsub, npairs) max_g w_g max(sum_t in a |F|, sum_t in b |F|)."""
        return numpy.maximum(msh[:, self.ish], msh[:, self.jsh])


def get_k_only_gpu(builder, sgx, dm, hermi=1, direct_scf_tol=1e-13):
    """Same signature/contract as pyscf.sgx.sgx_jk.get_k_only for one RKS
    dm; omega is read from sgx.mol.omega (0 unless a with_range_coulomb
    context is active -- the LR copies share the mol, so the dispatch needs
    no special case).  Raises gpu.xc.Unsupported outside the supported
    scope; the install hook translates that into a CPU fallback."""
    omega = float(getattr(sgx.mol, 'omega', 0.0) or 0.0)
    return builder.get_k(sgx, dm, omega, hermi=hermi,
                         direct_scf_tol=direct_scf_tol)
