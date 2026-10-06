"""Precision policy and compensated summation for S3 (spec section 6).

Policy (heavy = FP32 on GPU, light = FP64):

    operator                  precision
    AO evaluation             FP32
    density_gemm              FP32 SGEMM
    XC functional (pointwise) FP64 (libxc / libxc-cuda behind XcFunctional)
    fock_gemm                 FP32 per-block partials, FP64 accumulation
    VV10 kernel_sum           FP32 tiles, Kahan within a tile, FP64 across tiles
    energy sums               FP64

TF32 is NOT FP32.  It stays off by default; it may be enabled explicitly for
the S0-stage GEMMs only, and only on sm_80+ hardware (spec section 6).

All functions take `xp`, the array module (numpy or cupy), as an explicit
argument -- the same code runs on every backend.
"""
from pyscf_wb97mv_fast.gpu import backends

FP32, FP64 = 'float32', 'float64'


def set_tf32(enabled):
    """Toggle TF32 in cuBLAS GEMMs.  No-op on the numpy backend.  Returns the
    previous state (None if unknown)."""
    cp = backends.cupy_module()
    if cp is None:
        return None
    prev = None
    try:
        prev = cp.cuda.matmul.allow_tf32
        cp.cuda.matmul.allow_tf32 = bool(enabled)
    except AttributeError:
        try:                                   # older CuPy: cupyx.allow_tf32
            import cupyx
            prev = cupyx.allow_tf32
            cupyx.allow_tf32 = bool(enabled)
        except (ImportError, AttributeError):
            return None
    return prev


def allow_tf32_for_stage(stage, enabled_stages=(0,), tf32_capable=True):
    """Spec rule: TF32 only inside S0 GEMMs and only on sm_80+."""
    set_tf32(bool(tf32_capable and stage in enabled_stages))


def kahan_step(acc, comp, term):
    """One vectorized Kahan step: acc, comp <- updated (same dtype as term).

    y = term - comp; t = acc + y; comp = (t - acc) - y; acc = t
    """
    y = term - comp
    t = acc + y
    comp = (t - acc) - y
    return t, comp


def kahan_sum_axis(xp, arr, axis=1, n_seg=8):
    """Kahan-compensated FP32 sum along `axis`, GPU-friendly, deterministic.

    The axis is split into n_seg segments; each segment is reduced by the
    backend's pairwise tree sum (xp.sum -- far better than sequential FP32
    accumulation), then the n_seg partial sums are combined with sequential
    vectorized Kahan steps (n_seg launches).  Zero padding makes the split
    exact for any axis length.  Returns the sum with `axis` removed.
    """
    n = arr.shape[axis]
    if n == 0:
        keep = tuple(d for i, d in enumerate(arr.shape) if i != axis)
        return xp.zeros(keep, dtype=arr.dtype)
    seg = -(-n // n_seg)
    pad = seg * n_seg - n
    if pad:
        pad_shape = list(arr.shape)
        pad_shape[axis] = pad
        arr = xp.concatenate(
            [arr, xp.zeros(tuple(pad_shape), dtype=arr.dtype)], axis=axis)
    parts = arr.reshape(arr.shape[:axis] + (n_seg, seg) + arr.shape[axis + 1:])
    partials = parts.sum(axis=axis + 1)
    index0 = (slice(None),) * axis + (0,)
    acc = partials[index0].copy()
    comp = acc - acc
    for i in range(1, n_seg):
        idx = (slice(None),) * axis + (i,)
        acc, comp = kahan_step(acc, comp, partials[idx])
    return acc
