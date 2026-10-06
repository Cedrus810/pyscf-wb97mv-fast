"""GPU device management for S3 (spec section 7).

Everything the flows need to know about the device: whether one exists, its
compute capability (TF32 gate: sm_80+), and the streaming memory budget
(default 60% of the free VRAM; block sizes are derived from it and halved on
OOM).  CPU ('numpy' backend) reports a generous pseudo-budget so the same
chunking code runs anywhere.
"""
from pyscf_wb97mv_fast.gpu import backends

DEFAULT_MEM_FRACTION = 0.6


class DeviceState:
    """Snapshot of the device the flows will stream over."""

    def __init__(self, backend='cupy', mem_fraction=DEFAULT_MEM_FRACTION):
        self.backend = backend
        self.mem_fraction = float(mem_fraction)
        self.name = 'cpu'
        self.compute_capability = None
        self.tf32_capable = False
        self.free_bytes = None
        self.total_bytes = None
        if backend == 'cupy':
            cp = backends.cupy_module()
            if cp is None:
                raise ImportError('CuPy is not importable; no CUDA backend')
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('no CUDA device visible to CuPy')
            dev = cp.cuda.Device()
            name = cp.cuda.runtime.getDeviceProperties(0)['name']
            if isinstance(name, bytes):
                name = name.decode()
            self.name = name
            cc = dev.compute_capability
            if isinstance(cc, str):        # CuPy >= 14: '75', '86', '90', '120'
                major, minor = int(cc[:-1]), int(cc[-1])
            else:                          # older CuPy: tuple of ints
                major, minor = int(cc[0]), int(cc[1])
            self.compute_capability = (major, minor)
            self.tf32_capable = major >= 8
            free, total = cp.cuda.runtime.memGetInfo()
            self.free_bytes, self.total_bytes = int(free), int(total)
        else:
            import os
            self.free_bytes = 4 << 30                     # 4 GiB working set
            self.total_bytes = self.free_bytes

    @property
    def mem_budget(self):
        """Bytes the flows may use for streaming buffers (spec: 60% free)."""
        if self.free_bytes is None:
            return None
        return int(self.free_bytes * self.mem_fraction)

    def __repr__(self):
        return (f'DeviceState({self.backend}, name={self.name!r}, '
                f'cc={self.compute_capability}, free={self.free_bytes})')


def is_oom(exc):
    """True when an exception looks like an out-of-memory condition."""
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__
    if name in ('OutOfMemoryError', 'CuPyOutOfMemoryError'):
        return True
    text = str(exc).lower()
    return 'out of memory' in text or 'cublas_status_alloc_failed' in text


# -- runtime helpers shared by the xc / vv10 flows -----------------------------

def to_host(arr):
    """Device array -> numpy; numpy in, numpy out."""
    import numpy
    if type(arr).__module__.startswith('cupy'):
        from pyscf_wb97mv_fast.gpu import backends
        cp = backends.cupy_module()
        if cp is not None and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    return numpy.asarray(arr)


def ao_indices(ao_loc, shells):
    """AO column indices of the given shells, in the evaluator's column order
    (shells ascending, ctr-major within a shell)."""
    import numpy
    shells = numpy.asarray(shells, dtype=numpy.int64)
    if shells.size == 0:
        return numpy.zeros(0, dtype=numpy.int64)
    return numpy.concatenate(
        [numpy.arange(ao_loc[sh], ao_loc[sh + 1]) for sh in shells]
    ).astype(numpy.int64)


def gather_dm(dm, idx):
    """(nao,nao) numpy dm -> (len(idx),len(idx)) submatrix, FP64."""
    import numpy
    return dm[numpy.ix_(idx, idx)]


def scatter_add(xp, mat, idx, blk):
    """mat[idx[:,None], idx[None,:]] += blk.

    `blk` is cast to mat's dtype first.  Indices within one call are unique
    (one AO pair per element), so each element has a single writer per
    launch: the accumulation order across calls is stream-sequential and the
    result is reproducible.
    """
    idx = xp.asarray(idx)
    mat[idx[:, None], idx[None, :]] += xp.asarray(blk, dtype=mat.dtype)
