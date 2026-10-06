"""Reversible attribute patching shared by the fast-path modules.

Same restore semantics as ScfProfiler._wrap: if the attribute was defined on
the owner itself (module function, class method) it is written back with
setattr; if it only shadowed a class-level method on an instance, delattr
re-exposes the original.
"""
_MISSING = object()


def _install(owner, name, replacement):
    orig = getattr(owner, name)
    own = vars(owner).get(name, _MISSING) if hasattr(owner, '__dict__') else _MISSING
    setattr(owner, name, replacement)
    return (owner, name, own)


def _restore(token):
    owner, name, own = token
    if own is _MISSING:
        delattr(owner, name)
    else:
        setattr(owner, name, own)


class HookSet:
    """Collect installed patches; restore_all() undoes them in reverse order."""

    def __init__(self):
        self._tokens = []
        self._done = False

    def wrap(self, owner, name, factory):
        """Replace owner.name by factory(orig); factory must call orig."""
        if self._done:
            raise RuntimeError('HookSet already restored')
        token = _install(owner, name, factory(getattr(owner, name)))
        self._tokens.append(token)
        return token

    def on_restore(self, fn):
        """Call fn() during restore_all(), in reverse order with the wraps."""
        if self._done:
            raise RuntimeError('HookSet already restored')
        self._tokens.append(fn)

    def restore_all(self):
        for token in reversed(self._tokens):
            if callable(token):
                token()
            else:
                _restore(token)
        self._tokens.clear()
        self._done = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.restore_all()


def reset_diis(mf_diis):
    """Drop the extrapolation history of a pyscf.lib.diis.DIIS (e.g. CDIIS).

    Returns it to the state right after construction; configuration such as
    space, min_space, Corth, damp and rollback is kept. Needed at stage
    switches (grid / VV10 changes): a history built from Fock matrices of a
    different operator steers the extrapolation wrong and costs extra cycles
    (FINDINGS.md section 9). mf_diis=None (DIIS disabled) is a no-op.

    Typical use inside mf.callback: reset_diis(envs['mf_diis']).
    """
    if mf_diis is None:
        return
    mf_diis._buffer = {}
    mf_diis._bookkeep = []
    mf_diis._head = 0
    mf_diis._H = None
    mf_diis._xprev = None
    mf_diis._err_vec_touched = False
    # vectors that did not fit in core live in an HDF5 scratch file; start a
    # fresh one instead of overwriting datasets of the old history
    mf_diis._diisfile = None


def track_residual(hooks, mf, sink):
    """Feed PySCF's per-cycle orbital-gradient norm to sink(norm_gorb).

    Uses mf.callback, which scf.hf.kernel calls with locals() at the end of
    every cycle, i.e. after get_veff and before the next cycle's get_veff.
    envs['norm_gorb'] is exactly the |g| PySCF compares with conv_tol_grad
    (computed from the post-veff, non-DIIS Fock).

    Do NOT derive the residual inside an eig() hook from the Fock passed to
    eig: the orbitals returned by eig diagonalize that Fock, so its occ-vir
    block -- the orbital gradient -- is zero by construction.

    An existing user callback is kept and still called.
    """
    def factory(prev):
        def callback(envs):
            sink(float(envs['norm_gorb']))
            if callable(prev):
                prev(envs)
        return callback
    return hooks.wrap(mf, 'callback', factory)
