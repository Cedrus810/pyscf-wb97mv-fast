"""OpenBLAS thread limit while the GPU hooks are installed (2026-10-03).

A pthreads OpenBLAS called from inside PySCF's OpenMP regions oversubscribes
the cores (~1e4 "Detect OpenMP Loop" warnings per water27 run).  Limiting
OpenBLAS to one thread measurably reduces the staged water27 wall time.
"""
import numpy  # noqa: F401  (loads numpy's BLAS for the real-library test)
import pytest

from pyscf_wb97mv_fast.core import blas_threads
from pyscf_wb97mv_fast.core.hooks import HookSet


class FakeOpenblas:
    def __init__(self, parallel, n):
        self.parallel, self.n, self.calls = parallel, n, []

    def get_parallel(self):
        return self.parallel

    def get_num_threads(self):
        return self.n

    def set_num_threads(self, n):
        self.n = n
        self.calls.append(n)


@pytest.fixture
def no_user_env(monkeypatch):
    for name in blas_threads.USER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_limit_touches_pthreads_builds_only_and_restores(no_user_env):
    pthreads, sequential, openmp = FakeOpenblas(1, 32), FakeOpenblas(0, 1), FakeOpenblas(2, 16)
    restore = blas_threads.limit_openblas_threads(1, libs=[pthreads, sequential, openmp])
    assert pthreads.n == 1
    assert sequential.calls == [] and openmp.calls == []
    restore()
    assert pthreads.n == 32


@pytest.mark.parametrize('name', ['OPENBLAS_NUM_THREADS', 'GOTO_NUM_THREADS'])
def test_limit_respects_user_setting(no_user_env, monkeypatch, name):
    monkeypatch.setenv(name, '4')
    lib = FakeOpenblas(1, 4)
    assert blas_threads.limit_openblas_threads(1, libs=[lib]) is None
    assert lib.calls == []


def test_limit_noop_without_pthreads_openblas(no_user_env):
    lib = FakeOpenblas(0, 1)
    assert blas_threads.limit_openblas_threads(1, libs=[lib]) is None
    assert blas_threads.limit_openblas_threads(1, libs=[]) is None


def test_hookset_on_restore_runs_in_reverse_order_with_wraps():
    class Owner:
        def f(self):
            return 'orig'
    owner, log = Owner(), []
    hooks = HookSet()
    hooks.on_restore(lambda: log.append(('cb1', owner.f())))
    hooks.wrap(owner, 'f', lambda orig: lambda: 'wrapped')
    hooks.on_restore(lambda: log.append(('cb2', owner.f())))
    hooks.restore_all()
    # cb2 runs first (wrap still on), then the wrap is undone, then cb1
    assert log == [('cb2', 'wrapped'), ('cb1', 'orig')]
    assert owner.f() == 'orig'


def test_real_loaded_pthreads_openblas(no_user_env):
    libs = [lib for lib in blas_threads.loaded_openblas() if lib.get_parallel() == 1]
    if not libs:
        pytest.skip('no pthreads OpenBLAS loaded in this process')
    before = [lib.get_num_threads() for lib in libs]
    restore = blas_threads.limit_openblas_threads(1)
    try:
        assert [lib.get_num_threads() for lib in libs] == [1] * len(libs)
    finally:
        restore()
    assert [lib.get_num_threads() for lib in libs] == before
