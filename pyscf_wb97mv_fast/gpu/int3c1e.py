"""S5 Task 3: FP32 three-center point-charge integrals (validation kernel).

``CORE_SRC`` holds the shared CUDA device code used by both the dense
validation kernel below and the fused K kernel of gpu.sgx_k:

    int3c1e_pair_block(...)  ->  A[nia x nib]  (contracted, spherical)

for one (grid point, shell pair): the locked McMurchie-Davidson math per
primitive pair -- Boys F_0..F_6 (gpu.boys, spliced in front), R^0_{tuv}
filled level-by-level 6 -> 0 in place (descending t/u/v loops only read
slots of the previous level; slots outside n+t+u+v <= 6 stay on zero
chains that never feed a valid entry; l <= 3, so 2 l_max = 6), the
E^{i,i'}_t contraction to the
primitive Cartesian block, then the per-shell contraction+spherical GEMMs
(shellpairs.ShellPairs.to_device layout).

Precision policy: every per-primitive constant (p, prefactor, P, E) is
computed on the host in FP64 and rounded to FP32 once.  That rounding is
NOT automatically unbiased: a basis constant serves every grid point, so
its rounding error is the same everywhere (and identical for every copy of
an element).  Measured on the dimer as K energy errors 1/4 tr(D dK)
(full K / long-range K): Mc/MT 1.4e-7
/ 4.8e-8, prefactor -1.1e-7 / 9e-10, grid coordinates -1.7e-7 / 8e-11,
sqrt(theta) - / -1.7e-8, (-2 p theta) -9e-9 / 7e-10, E -1.6e-8 / -4e-9,
Boys -1.2e-8 / -5e-10.  Mc/MT, the prefactor, the grid coordinates and the
R000 coefficient sqrt(theta) (-2 p theta)^n (formed in double) carry FP32
rounding remainders lo, and every constant product is formed as
fmaf(c_hi, x, c_lo * x): one rounding of the compensated product.  Adding
a separately accumulated lo sum to the rounded hi sum (the first H3 fix)
drops the correction whenever it is below half an ulp of the hi sum, i.e.
most of the time.  E stays plain (small); P keeps its hi+lo.

``int3c1e_fp32`` materializes the dense (ng, nao, nao) tensor -- a
validation path for small systems only; it refuses sizes beyond the
memory budget.  The production K path (sgx_k) never materializes A.
"""
import numpy
from pyscf_wb97mv_fast.gpu.boys import BOYS_SRC
from pyscf_wb97mv_fast.gpu import shellpairs as _sp

CORE_SRC = r'''
#define SP_NIA %(NIA)d       /* shellpairs.NIA_MAX */
#define SP_KMAX %(KMAX)d     /* shellpairs.KMAX */
#define SP_NCMAX %(NCA)d     /* shellpairs.NCA_MAX */
#define SP_EL %(EL)d         /* Hermite E layout [axis, i, i', t] */
#define SP_ET %(ET)d         /* = 2 LMAX + 1: Boys orders, R side, cn cols */
#define SP_ESTRIDE %(ES)d
#define SP_EY (SP_EL * SP_EL * SP_ET)
#define SP_EZ (2 * SP_EL * SP_EL * SP_ET)

__device__ __forceinline__ void sp_cart_comp(int l, int c, int* i, int* j,
                                             int* k)
{
    /* libcint order: lx desc, then ly desc, lz = l - lx - ly.  Block r
       (lx = l - r) holds components c = r(r+1)/2 .. r(r+1)/2 + r; with
       compile-time l and c (the unrolled template loops) this folds away */
    int r = 0;
    while ((r + 1) * (r + 2) / 2 <= c) ++r;
    *i = l - r;
    *j = r - (c - r * (r + 1) / 2);
    *k = l - *i - *j;
}

/* A (SP_NIA x SP_NIA, row-major) = contracted spherical integral block of
   one shell pair at one grid point. */
__device__ void int3c1e_pair_block(
    const int* __restrict__ sh_dims,     /* (nsh, 4): nia, nprim, nca, nsa */
    const float* __restrict__ sh_Mc,     /* (nsh, SP_NIA, SP_KMAX) */
    const float* __restrict__ sh_MT,     /* (nsh, SP_KMAX, SP_NIA) */
    const float* __restrict__ sh_Mc_lo,  /* FP32 rounding remainders of */
    const float* __restrict__ sh_MT_lo,  /* Mc / MT (systematic otherwise) */
    const int* __restrict__ sh6,         /* ia0, ib0, ish, jsh, prim0, npk */
    const double* __restrict__ p_arr,
    const float* __restrict__ prefac,
    const float* __restrict__ prefac_lo,
    const float* __restrict__ P_hi,      /* (nprims, 3) */
    const float* __restrict__ P_lo,
    const float* __restrict__ Earr,      /* (nprims, SP_ESTRIDE) */
    float gx, float gy, float gz,        /* grid point hi ... */
    float gxl, float gyl, float gzl,     /* ... and lo, pairs' frame */
    double omega,
    float* A)
{
    const int ish = sh6[2], jsh = sh6[3];
    const int pr0 = sh6[4], npk = sh6[5];
    const int nia = sh_dims[ish * 4 + 0], npi = sh_dims[ish * 4 + 1];
    const int nca = sh_dims[ish * 4 + 2];
    const int nib = sh_dims[jsh * 4 + 0], npj = sh_dims[jsh * 4 + 1];
    const int ncb = sh_dims[jsh * 4 + 2];
    const int la = (sh_dims[ish * 4 + 3] - 1) / 2;    /* nsa = 2l + 1 */
    const int lb = (sh_dims[jsh * 4 + 3] - 1) / 2;
    const float* Mc = sh_Mc + ish * SP_NIA * SP_KMAX;
    const float* MT = sh_MT + jsh * SP_KMAX * SP_NIA;
    const float* Mc_lo = sh_Mc_lo + ish * SP_NIA * SP_KMAX;
    const float* MT_lo = sh_MT_lo + jsh * SP_KMAX * SP_NIA;

    for (int i = 0; i < SP_NIA * SP_NIA; ++i) A[i] = 0.f;
    float Ac[SP_NCMAX * SP_NCMAX];
    float B[SP_NCMAX * SP_NIA];
    float R[SP_ET * SP_ET * SP_ET];

    for (int a = 0; a < npi; ++a) {
        /* B is addressed B[ca * SP_NIA + out] (row stride SP_NIA, not nib):
           clear exactly those slots, or rows ca >= 1 keep stale values */
        for (int ca = 0; ca < nca; ++ca)
            for (int out = 0; out < nib; ++out) B[ca * SP_NIA + out] = 0.f;
        for (int b = 0; b < npj; ++b) {
            const int m = pr0 + a * npj + b;
            const double pd = p_arr[m];
            double th = 1.0, sth = 1.0;
            if (omega > 0.0) {
                const double w2 = omega * omega;
                th = w2 / (w2 + pd);
                sth = sqrt(th);
            }
            const float dx = (P_hi[m * 3 + 0] - gx) + (P_lo[m * 3 + 0] - gxl);
            const float dy = (P_hi[m * 3 + 1] - gy) + (P_lo[m * 3 + 1] - gyl);
            const float dz = (P_hi[m * 3 + 2] - gz) + (P_lo[m * 3 + 2] - gzl);
            const float r2 = dx * dx + dy * dy + dz * dz;
            const float T = (float)(th * pd * (double)r2);
            float Fb[SP_ET];
            boys_fp32(T, Fb, SP_ET - 1);
            /* R000[n] = c_n F_n with c_n = sqrt(theta) (-2 p theta)^n in
               double, applied as hi + lo in one rounding */
            const double q = -2.0 * pd * th;
            float R000[SP_ET];
            double cn = sth;
            #pragma unroll
            for (int n = 0; n < SP_ET; ++n) {
                const float chi = (float)cn;
                const float clo = (float)(cn - (double)chi);
                R000[n] = fmaf(chi, Fb[n], clo * Fb[n]);
                cn *= q;
            }

            #pragma unroll
            for (int i = 0; i < SP_ET * SP_ET * SP_ET; ++i) R[i] = 0.f;
            for (int n = SP_ET - 1; n >= 0; --n) {
                for (int t = SP_ET - 1; t >= 1; --t)
                    for (int u = 0; u < SP_ET; ++u)
                        for (int v = 0; v < SP_ET; ++v) {
                            const int idx = (t * SP_ET + u) * SP_ET + v;
                            if (t == 1)
                                R[idx] = dx * R[(0 * SP_ET + u) * SP_ET + v];
                            else
                                R[idx] = (float)(t - 1) * R[((t - 2) * SP_ET + u) * SP_ET + v]
                                       + dx * R[((t - 1) * SP_ET + u) * SP_ET + v];
                        }
                for (int u = SP_ET - 1; u >= 1; --u)
                    for (int v = 0; v < SP_ET; ++v) {
                        const int idx = (0 * SP_ET + u) * SP_ET + v;
                        if (u == 1)
                            R[idx] = dy * R[v];
                        else
                            R[idx] = (float)(u - 1) * R[(0 * SP_ET + u - 2) * SP_ET + v]
                                   + dy * R[(0 * SP_ET + u - 1) * SP_ET + v];
                    }
                for (int v = SP_ET - 1; v >= 1; --v) {
                    if (v == 1)
                        R[v] = dz * R[0];
                    else
                        R[v] = (float)(v - 1) * R[v - 2] + dz * R[v - 1];
                }
                R[0] = R000[n];
            }

            const float* Em = Earr + m * SP_ESTRIDE;
            const float pf = prefac[m], pf_lo = prefac_lo[m];
            for (int ca = 0; ca < nca; ++ca) {
                int i, j, k;
                sp_cart_comp(la, ca, &i, &j, &k);
                for (int cb = 0; cb < ncb; ++cb) {
                    int ip, jp, kp;
                    sp_cart_comp(lb, cb, &ip, &jp, &kp);
                    float s = 0.f;
                    for (int t = 0; t <= i + ip; ++t) {
                        const float ex = Em[(i * SP_EL + ip) * SP_ET + t];
                        for (int u = 0; u <= j + jp; ++u) {
                            const float exy = ex * Em[SP_EY + (j * SP_EL + jp) * SP_ET + u];
                            for (int v = 0; v <= k + kp; ++v) {
                                s += exy * Em[SP_EZ + (k * SP_EL + kp) * SP_ET + v]
                                        * R[(t * SP_ET + u) * SP_ET + v];
                            }
                        }
                    }
                    Ac[ca * SP_NCMAX + cb] = fmaf(pf, s, pf_lo * s);
                }
            }
            /* B[ca, out] += sum_cb Ac[ca,cb] * MT[(b,cb), out] */
            const float* MTb = MT + b * ncb * SP_NIA;
            const float* MTb_lo = MT_lo + b * ncb * SP_NIA;
            for (int ca = 0; ca < nca; ++ca)
                for (int out = 0; out < nib; ++out) {
                    float s = 0.f;
                    for (int cb = 0; cb < ncb; ++cb) {
                        const float a = Ac[ca * SP_NCMAX + cb];
                        s += fmaf(a, MTb[cb * SP_NIA + out],
                                  a * MTb_lo[cb * SP_NIA + out]);
                    }
                    B[ca * SP_NIA + out] += s;
                }
        }
        /* A[row, out] += sum_ca Mc[row, (a,ca)] * B[ca, out] */
        for (int row = 0; row < nia; ++row)
            for (int out = 0; out < nib; ++out) {
                float s = 0.f;
                for (int ca = 0; ca < nca; ++ca) {
                    const float b = B[ca * SP_NIA + out];
                    s += fmaf(Mc[row * SP_KMAX + a * nca + ca], b,
                              Mc_lo[row * SP_KMAX + a * nca + ca] * b);
                }
                A[row * SP_NIA + out] += s;
            }
    }
    /* a diagonal shell pair's block is symmetric in exact arithmetic, but
       A[mu,nu] and A[nu,mu] take different FP32 paths (Mc vs MT): average
       them, which is bitwise symmetric (FP32 addition commutes) */
    if (ish == jsh) {
        for (int mu = 0; mu < nia; ++mu)
            for (int nu = mu + 1; nu < nib; ++nu) {
                const float v = (A[mu * SP_NIA + nu] + A[nu * SP_NIA + mu]) * 0.5f;
                A[mu * SP_NIA + nu] = v;
                A[nu * SP_NIA + mu] = v;
            }
    }
}
''' % dict(NIA=_sp.NIA_MAX, KMAX=_sp.KMAX, NCA=_sp.NCA_MAX, EL=_sp.E_L,
           ET=_sp.E_T, ES=_sp.E_SIZE)

DENSE_SRC = (BOYS_SRC + CORE_SRC + r'''
extern "C" __global__ void int3c1e_dense(
    const int* __restrict__ sh_dims,
    const float* __restrict__ sh_Mc,
    const float* __restrict__ sh_MT,
    const float* __restrict__ sh_Mc_lo,
    const float* __restrict__ sh_MT_lo,
    const int* __restrict__ pair_sh,     /* (npairs, 6) */
    const double* __restrict__ p_arr,
    const float* __restrict__ prefac,
    const float* __restrict__ prefac_lo,
    const float* __restrict__ P_hi,
    const float* __restrict__ P_lo,
    const float* __restrict__ Earr,
    const float* __restrict__ coords,    /* (ng, 3), same frame as P */
    const float* __restrict__ coords_lo, /* FP32 rounding remainders */
    int ng, int npairs, int nao,
    double omega,
    float* __restrict__ out)             /* (ng, nao, nao) */
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ng * npairs) return;
    const int g = idx / npairs;
    const int kk = idx - g * npairs;
    const int* sh6 = pair_sh + kk * 6;
    float A[SP_NIA * SP_NIA];
    int3c1e_pair_block(sh_dims, sh_Mc, sh_MT, sh_Mc_lo, sh_MT_lo, sh6, p_arr,
                       prefac, prefac_lo, P_hi, P_lo, Earr,
                       coords[g * 3], coords[g * 3 + 1], coords[g * 3 + 2],
                       coords_lo[g * 3], coords_lo[g * 3 + 1],
                       coords_lo[g * 3 + 2], omega, A);
    const int ia0 = sh6[0], ib0 = sh6[1];
    const int nia = sh_dims[sh6[2] * 4 + 0];
    const int nib = sh_dims[sh6[3] * 4 + 0];
    float* row = out + (size_t)g * nao * nao;
    for (int mu = 0; mu < nia; ++mu)
        for (int nu = 0; nu < nib; ++nu) {
            row[(size_t)(ia0 + mu) * nao + ib0 + nu] = A[mu * SP_NIA + nu];
            if (ia0 != ib0)
                row[(size_t)(ib0 + nu) * nao + ia0 + mu] = A[mu * SP_NIA + nu];
        }
}
''')

_DENSE_KERNELS = {}


def _dense_kernel(cp):
    key = id(cp)
    if key not in _DENSE_KERNELS:
        _DENSE_KERNELS[key] = cp.RawKernel(DENSE_SRC, 'int3c1e_dense')
    return _DENSE_KERNELS[key]


def int3c1e_fp32(cp, pairs_dev, coords, omega=0.0):
    """(ng, nao, nao) float32 on device, spherical, symmetric in (mu, nu).

    coords are in the same frame as the pairs' P (absolute, unless
    ShellPairs.to_device was given an origin -- then relative to it).
    Validation path only: refuses ng * nao**2 * 4 bytes > mem_budget."""
    coords64 = cp.asarray(coords, dtype=cp.float64)
    coords = cp.ascontiguousarray(coords64.astype(cp.float32))
    coords_lo = cp.ascontiguousarray(
        (coords64 - coords.astype(cp.float64)).astype(cp.float32))
    ng = int(coords.shape[0])
    nao = int(pairs_dev.nao)
    npairs = int(pairs_dev.npairs)
    need = ng * nao * nao * 4
    if need > pairs_dev.mem_budget:
        raise MemoryError(
            'dense int3c1e needs %d bytes for (ng=%d, nao=%d), above the '
            '%d byte budget' % (need, ng, nao, pairs_dev.mem_budget))
    out = cp.zeros((ng, nao, nao), dtype=cp.float32)
    if ng == 0 or npairs == 0:
        return out
    kern = _dense_kernel(cp)
    nblk = -(-ng * npairs // 128)
    kern((nblk,), (128,),
         (pairs_dev.sh_dims, pairs_dev.sh_Mc, pairs_dev.sh_MT,
          pairs_dev.sh_Mc_lo, pairs_dev.sh_MT_lo, pairs_dev.pair_sh,
          pairs_dev.prim_p, pairs_dev.prim_prefac, pairs_dev.prim_prefac_lo,
          pairs_dev.prim_P_hi, pairs_dev.prim_P_lo, pairs_dev.prim_E,
          coords, coords_lo, numpy.int32(ng), numpy.int32(npairs), numpy.int32(nao),
          numpy.float64(omega), out))
    return out
