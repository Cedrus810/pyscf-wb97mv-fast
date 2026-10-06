"""FROZEN (2026-09-30): gates failed, see benchmarks/experimental/cosx/FINDINGS.md section 9; not imported by mainline.

Hardware profile microbenchmark (P6): what is worth putting on a device.

Times dense FP64/FP32 GEMM and one skinny GEMM with numpy (CPU baseline) and,
if jax is installed, with jax on its default backend (GPU when present).
Returns effective GFLOP/s per case. Pure timing, no quantum chemistry.
"""
import time

import numpy as np

_CASES = {
    'gemm_f64': (np.float64, 4096),
    'gemm_f32': (np.float32, 4096),
    'skinny_f64': (np.float64, (4096, 64, 4096)),
}


def _numpy_cases(repeat=3):
    out = {}
    for name, (dtype, n) in _CASES.items():
        if isinstance(n, tuple):
            a = np.random.rand(n[0], n[1]).astype(dtype)
            b = np.random.rand(n[1], n[2]).astype(dtype)
            flops = 2 * n[0] * n[1] * n[2]
        else:
            a = np.random.rand(n, n).astype(dtype)
            b = np.random.rand(n, n).astype(dtype)
            flops = 2 * n**3
        a @ b  # warm-up
        t0 = time.perf_counter()
        for _ in range(repeat):
            a @ b
        dt = (time.perf_counter() - t0) / repeat
        out[name] = dict(gflops=flops / dt / 1e9, s=dt)
    return out


def _jax_cases(repeat=3):
    jax = __import__('jax')
    jnp = jax.numpy
    out = {}
    for name, (dtype, n) in _CASES.items():
        if isinstance(n, tuple):
            a = jnp.array(np.random.rand(n[0], n[1]), dtype=dtype)
            b = jnp.array(np.random.rand(n[1], n[2]), dtype=dtype)
            flops = 2 * n[0] * n[1] * n[2]
        else:
            a = jnp.array(np.random.rand(n, n), dtype=dtype)
            b = jnp.array(np.random.rand(n, n), dtype=dtype)
            flops = 2 * n**3
        c = a @ b
        jax.block_until_ready(c)
        t0 = time.perf_counter()
        for _ in range(repeat):
            c = a @ b
            jax.block_until_ready(c)
        dt = (time.perf_counter() - t0) / repeat
        out[name] = dict(gflops=flops / dt / 1e9, s=dt)
    return out


def profile(repeat=3):
    """{'cpu': {...}, 'jax': {...}|None, 'jax_backend': str}."""
    res = {'cpu': _numpy_cases(repeat), 'jax': None, 'jax_backend': None}
    try:
        import jax
        res['jax'] = _jax_cases(repeat)
        res['jax_backend'] = jax.default_backend()
    except ImportError:
        pass
    return res


if __name__ == '__main__':
    import json
    print(json.dumps(profile(), indent=1, default=float))
