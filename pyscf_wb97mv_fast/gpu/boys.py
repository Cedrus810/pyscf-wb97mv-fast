"""S5: FP32 Boys function F_0..F_6 as a spliceable CUDA device function.

``BOYS_SRC`` defines ``__device__ void boys_fp32(float T, float* F, int nmax)``
(nmax <= 6 = 2 l_max for f shells, F[0..nmax] written) plus its tables, so other kernels (int3c1e,
sgx_k) paste it in front of their own source.  ``boys_eval`` is a batch
driver used only by the tests.

No double precision and no division anywhere (2026-10-02; replaces the
double-internal version of Ruling 1): the target cards are consumer
Ampere / Ada / Blackwell with FP64 at 1/64, where the old ~15 double ops,
double exp and 4 double divisions per (primitive pair, grid point) were a
real cost.

* ``T < T_SWITCH`` (36): centred Taylor expansion around the nearest node
  T_k = k/4 (k = 0..144, h = T - T_k in [-1/8, 1/8]), every order directly
  (dF_n/dT = -F_{n+1}):

      F_n(T) = sum_{j=0..6} c_{n,j} (-h)^j,   c_{n,j} = F_{n+j}(T_k) / j!

  Truncation <= F_n h^7 / 7! ~ 1e-10 relative.  No downward recurrence,
  hence no e^{-T}.  c_{n,0} is stored as an FP32 hi + lo pair; the lo part
  is added to the Horner tail (fmaf) BEFORE the single final addition to
  hi, so the compensation is not rounded away (adding a lo term to an
  already rounded FP32 value loses it whenever it is below half an ulp --
  the ao_eval lesson).  The tail is <= ~1/8 of
  the result, so its own FP32 rounding enters scaled down; the remaining
  c_{n,j} roundings enter with odd powers of h around a centred node and
  cancel on average (the unbiased test).
* ``T >= T_SWITCH``: asymptotic F_n = (2n-1)!!/2^{n+1} sqrt(pi) T^{-(n+1/2)},
  whose relative error is exactly Q(n+1/2, T) (regularized upper incomplete
  gamma): at T = 30 it is 1.3e-9 for n = 4 but 5.3e-8 for n = 6 -- a
  systematic FP32-sized bias -- hence the switch at 36 (n = 6: 3.4e-10,
  n = 5: 5.1e-11).  Built as
  F_0 = sqrt(pi)/2 * s and F_n = F_{n-1} * (n - 1/2) * u with s = 1/sqrt(T)
  from rsqrtf + one Newton step kept as hi + lo, u = s^2 as hi + lo, and
  every product a double-float (FMA error term) product, rounded once per
  output.  The double-float runs scaled by 2^100 (BOYS_SC, exact): unscaled,
  F_5 / F_6 at T > 1e5 are 1e-32 .. 1e-37 and their FMA error terms fall
  below FLT_MIN into subnormals, losing the lo part (4.0e-7 relative for F_6
  at T = 9e5, 2026-10-06).  The final * 2^-100 is
  exact while F_n is normal.
* T = 0 hits node 0 with h = 0: F_n(0) = 1/(2n+1) exactly as stored, so grid
  points sitting on a nucleus give finite values; T > 1e30 gives 0 (no
  inf * 0 NaN in the asymptotic residual).

The tables sit in global memory and are read through the read-only cache
(__ldg): threads of a warp index different nodes, which __constant__ memory
would serialize.
"""
import math

import numpy

NMAX = 6                            # highest order (f-f shell pairs)
T_SWITCH = 36.0
T_STEP = 0.25                       # power of two: nodes and h are exact
N_NODES = int(round(T_SWITCH / T_STEP)) + 1     # 145 nodes, 0 .. 36.0
N_TERMS = 7                         # Taylor terms j = 0..6 per order

# (2n-1)!! / 2^{n+1} * sqrt(pi): F_n(T) ~ this * T^{-(n+0.5)}
_ASY = numpy.array([numpy.prod([(2 * k - 1) for k in range(1, n + 1)])
                    / 2.0 ** (n + 1) * numpy.sqrt(numpy.pi)
                    for n in range(NMAX + 1)])


def boys_ref(n, T):
    """FP64 reference F_n(T) (scipy), also the seed for the table."""
    T = numpy.asarray(T, dtype=numpy.float64)
    out = numpy.empty_like(T)
    small = T < 1e-12
    out[small] = 1.0 / (2 * n + 1)
    Ts = T[~small]
    from scipy.special import gammainc, gamma
    out[~small] = (gamma(n + 0.5) * gammainc(n + 0.5, Ts)
                   / (2.0 * Ts ** (n + 0.5)))
    return out


def _hi_lo(a):
    """Split FP64 values into (hi, lo) float32 pairs: hi = fl32(a),
    lo = fl32(a - hi) (ao_eval._remainder32 pattern)."""
    a = numpy.asarray(a, dtype=numpy.float64)
    hi = a.astype(numpy.float32)
    lo = (a - hi.astype(numpy.float64)).astype(numpy.float32)
    return hi, lo


def _build_table():
    """c[k, n, j] = F_{n+j}(T_k) / j!  (k nodes, n = 0..NMAX, j = 0..6) and the
    FP32 remainders of the j = 0 entries."""
    nodes = numpy.arange(N_NODES) * T_STEP
    fm = numpy.stack([boys_ref(m, nodes) for m in range(NMAX + N_TERMS)])
    c = numpy.empty((N_NODES, NMAX + 1, N_TERMS))
    for n in range(NMAX + 1):
        for j in range(N_TERMS):
            c[:, n, j] = fm[n + j] / math.factorial(j)
    hi = c.astype(numpy.float32)
    lo0 = (c[:, :, 0] - hi[:, :, 0].astype(numpy.float64)).astype(numpy.float32)
    return hi, lo0


def _fmt(x):
    return float(numpy.float64(x)).__repr__()


def _build_source():
    c_hi, c0_lo = _build_table()
    a0_hi, a0_lo = _hi_lo(_ASY[0])
    lines = ['#define BOYS_TS %.1ff' % T_SWITCH,
             '#define BOYS_NT %d' % N_TERMS,
             '#define BOYS_NO %d' % (NMAX + 1),
             '__device__ const float BOYS_C[%d] = {%s};'
             % (c_hi.size, ','.join(_fmt(v) for v in c_hi.ravel())),
             '__device__ const float BOYS_C0LO[%d] = {%s};'
             % (c0_lo.size, ','.join(_fmt(v) for v in c0_lo.ravel())),
             '#define BOYS_A0_HI %sf' % _fmt(a0_hi),
             '#define BOYS_A0_LO %sf' % _fmt(a0_lo),
             '#define BOYS_SC %sf' % _fmt(2.0 ** 100),      # exact powers of two
             '#define BOYS_DESC %sf' % _fmt(2.0 ** -100)]
    lines.append(r'''
__device__ void boys_fp32(float T, float* F, int nmax)
{
    if (!(T > 0.f)) T = 0.f;
    if (nmax > BOYS_NO - 1) nmax = BOYS_NO - 1;
    if (T < BOYS_TS) {
        const int k = __float2int_rn(T * 4.0f);          /* nearest node */
        const float mh = 0.25f * (float)k - T;           /* -h, exact */
        const float* c = BOYS_C + k * (BOYS_NO * BOYS_NT);
        for (int n = 0; n <= nmax; ++n) {
            const float* cn = c + n * BOYS_NT;
            float s = __ldg(cn + BOYS_NT - 1);
            #pragma unroll
            for (int j = BOYS_NT - 2; j >= 1; --j) s = fmaf(mh, s, __ldg(cn + j));
            const float tail = fmaf(mh, s, __ldg(BOYS_C0LO + k * BOYS_NO + n));
            F[n] = __ldg(cn) + tail;
        }
    } else if (T > 1e30f) {
        for (int n = 0; n <= nmax; ++n) F[n] = 0.f;
    } else {
        /* s = 1/sqrt(T) as s_hi + s_lo: rsqrtf, then one Newton step whose
           residual r = 1 - T s0^2 is formed exactly enough with FMAs */
        const float s0 = rsqrtf(T);
        const float a = s0 * s0;
        const float a_lo = fmaf(s0, s0, -a);
        const float r = fmaf(-T, a, 1.0f) - T * a_lo;
        const float s_hi = s0, s_lo = 0.5f * s0 * r;
        /* u = 1/T = s^2 as u_hi + u_lo */
        const float u_hi = s_hi * s_hi;
        const float u_lo = fmaf(s_hi, s_hi, -u_hi) + 2.0f * s_hi * s_lo;
        /* F_0 = sqrt(pi)/2 s, double-float, scaled by 2^100 (no subnormal
           error terms for F_5, F_6 at large T) */
        const float a0_hi = BOYS_A0_HI * BOYS_SC, a0_lo = BOYS_A0_LO * BOYS_SC;
        float f_hi = a0_hi * s_hi;
        float f_lo = fmaf(a0_hi, s_hi, -f_hi)
                   + (a0_hi * s_lo + a0_lo * s_hi);
        F[0] = (f_hi + f_lo) * BOYS_DESC;
        for (int n = 1; n <= nmax; ++n) {
            const float cc = (float)n - 0.5f;                /* exact */
            const float cu_hi = cc * u_hi;
            const float cu_lo = fmaf(cc, u_hi, -cu_hi) + cc * u_lo;
            const float p = f_hi * cu_hi;
            const float e = fmaf(f_hi, cu_hi, -p) + (f_hi * cu_lo + f_lo * cu_hi);
            f_hi = p + e;
            f_lo = e - (f_hi - p);
            F[n] = f_hi * BOYS_DESC;
        }
    }
}
''')
    return '\n'.join(lines)


BOYS_SRC = _build_source()
_BATCH_SRC = None


def _batch_src():
    global _BATCH_SRC
    if _BATCH_SRC is None:
        _BATCH_SRC = BOYS_SRC + r'''
extern "C" __global__ void boys_batch(const float* T, int n, float* out,
                                      int nmax)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float f[BOYS_NO];
    boys_fp32(T[i], f, nmax);
    for (int m = 0; m <= nmax; ++m) out[m * n + i] = f[m];
}
'''
    return _BATCH_SRC


def boys_eval(cp, T, nmax):
    """Test entry: T (N,) array_like -> (nmax+1, N) float32 on device."""
    if not 0 <= nmax <= NMAX:
        raise ValueError('nmax must be 0..%d' % NMAX)
    T = cp.ascontiguousarray(T, dtype=cp.float32)
    n = int(T.size)
    out = cp.empty((nmax + 1, n), dtype=cp.float32)
    kern = _batch_kernel(cp)
    nblk = -(-n // 256)
    kern((nblk,), (256,), (T, cp.int32(n), out, cp.int32(nmax)))
    return out


def _batch_kernel(cp):
    global _BATCH_SRC
    if not hasattr(_batch_kernel, '_kern'):
        _batch_kernel._kern = {}
    key = id(cp)
    if key not in _batch_kernel._kern:
        _batch_kernel._kern[key] = cp.RawKernel(_batch_src(), 'boys_batch')
    return _batch_kernel._kern[key]
