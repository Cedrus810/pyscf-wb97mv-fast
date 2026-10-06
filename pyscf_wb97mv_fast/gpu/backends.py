"""Backend registry for the S3 GPU modules (spec section 7).

A "backend" is the array module the flows run on:

    'numpy'  always available; used by the unit tests (same code path as the
             GPU, FP32-capable) and as the last-resort CPU fallback
    'cupy'   the production backend (CuPy + cuBLAS)
    'jax'    reserved for later; not implemented yet (spec: CuPy first)

The numeric code in ao_eval / xc / vv10 is written against the array-module
interface only (xp.<op>), so the same implementation runs on every backend.
"""
import numpy

_CUPY = None
_CUPY_LOOKED = False


def cupy_module():
    """Import cupy once; returns the module or None (any failure)."""
    global _CUPY, _CUPY_LOOKED
    if not _CUPY_LOOKED:
        _CUPY_LOOKED = True
        try:
            import cupy as _cp
            _CUPY = _cp
        except Exception:                     # ImportError, broken driver, ...
            _CUPY = None
    return _CUPY


def have_cupy():
    return cupy_module() is not None


def get_xp(name):
    """Array module for a backend name."""
    if name == 'numpy':
        return numpy
    if name == 'cupy':
        cp = cupy_module()
        if cp is None:
            raise ImportError("backend 'cupy' requested but CuPy is not importable")
        return cp
    if name == 'jax':
        raise NotImplementedError("the 'jax' backend is planned but not implemented")
    raise ValueError(f'unknown backend {name!r}')


def default_backend():
    return 'cupy' if have_cupy() else 'numpy'
