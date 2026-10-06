"""Limit a pthreads OpenBLAS to n threads while the fast path runs.

PySCF's C kernels call BLAS from inside OpenMP parallel regions.  With a
pthreads OpenBLAS (openblas_get_parallel() == 1, conda-forge's default build)
those calls oversubscribe the cores on top of the OpenMP threads ("OpenBLAS
Warning : Detect OpenMP Loop ..." spam in the log).  Limiting OpenBLAS to
one thread measurably reduces the wall time of both the stock and the
staged path, mostly in the CPU K + CPU XC overlap.

The environment variable only works if set before OpenBLAS is loaded (the
numpy import), so install_gpu calls openblas_set_num_threads instead and
puts the old value back on restore.  Sequential (0) and OpenMP (2) builds
are left alone, and so is a process whose user set OPENBLAS_NUM_THREADS /
GOTO_NUM_THREADS explicitly.
"""
import ctypes
import os

USER_ENV_VARS = ('OPENBLAS_NUM_THREADS', 'GOTO_NUM_THREADS')

# (prefix, suffix) of the exported names: plain OpenBLAS, then the
# scipy-openblas wheels (ILP64 '64_' suffix, or the LP64 build without it)
_NAMINGS = (('', ''), ('scipy_', '64_'), ('scipy_', ''))


class _Openblas:
    """get_parallel / get_num_threads / set_num_threads of one loaded library."""

    def __init__(self, path, get_parallel, get_num_threads, set_num_threads):
        self.path = path
        self._get_parallel = get_parallel
        self._get_num_threads = get_num_threads
        self._set_num_threads = set_num_threads

    def get_parallel(self):
        return int(self._get_parallel())

    def get_num_threads(self):
        return int(self._get_num_threads())

    def set_num_threads(self, n):
        self._set_num_threads(int(n))


def _bind(path):
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    for pre, suf in _NAMINGS:
        names = [pre + 'openblas_' + f + suf
                 for f in ('get_parallel', 'get_num_threads', 'set_num_threads')]
        try:
            gp, gn, sn = (getattr(lib, name) for name in names)
        except AttributeError:
            continue
        gp.restype = gn.restype = ctypes.c_int
        gp.argtypes = gn.argtypes = []
        sn.restype, sn.argtypes = None, [ctypes.c_int]
        return _Openblas(path, gp, gn, sn)
    return None


def loaded_openblas():
    """One _Openblas per distinct OpenBLAS mapped into this process (Linux).

    Libraries that only re-export another one's symbols (scipy's _fblas,
    libblas.so.3 -> libopenblasp) resolve to the same set_num_threads and
    are listed once.
    """
    try:
        with open('/proc/self/maps') as f:
            paths = sorted({line.split()[-1] for line in f
                            if '.so' in line and 'blas' in line.rsplit('/', 1)[-1].lower()})
    except OSError:
        return []
    out, seen = [], set()
    for path in paths:
        lib = _bind(path)
        if lib is None:
            continue
        addr = ctypes.cast(lib._set_num_threads, ctypes.c_void_p).value
        if addr not in seen:
            seen.add(addr)
            out.append(lib)
    return out


def limit_openblas_threads(n=1, libs=None):
    """Set every pthreads OpenBLAS to n threads; returns restore() or None.

    None means nothing was changed: the user set the thread count in the
    environment, or no pthreads OpenBLAS is loaded.
    """
    if any(os.environ.get(name) for name in USER_ENV_VARS):
        return None
    if libs is None:
        libs = loaded_openblas()
    saved = [(lib, lib.get_num_threads()) for lib in libs if lib.get_parallel() == 1]
    if not saved:
        return None
    for lib, _ in saved:
        lib.set_num_threads(n)

    def restore():
        for lib, old in reversed(saved):
            lib.set_num_threads(old)
    return restore
