"""S3: FP32 AO evaluation on the compute backend (spec sections 6-7).

Reimplements dft.numint.eval_ao (libcint) for spherical (or Cartesian) GTOs,
vectorized over the array module `xp` (numpy for tests/fallback, cupy for the
GPU).  Output layout matches PySCF's deriv=0/1: (ncomp, npts, nao) with
component 0 the AO value and components 1:4 the x,y,z gradients.

Conventions verified against the PySCF 2.14 source:

* Contraction coefficients stored in mol._env already carry the per-primitive
  radial normalization gto_norm (gto/mole.py:1006) and the contracted
  normalization (NORMALIZE_GTO, mole.py:1007); libcint applies no further
  normalization.  The kernel therefore uses mol.bas_exp / mol.bas_ctr_coeff
  exactly as stored.
* Cartesian component order is libcint's: lx descending, then ly descending,
  lz = l - lx - ly (the ordering of the derivative components documented in
  numint.eval_ao).  Pack-time self-check (verify=True) runs the packed
  parameters through the numpy code path and compares against
  mol.eval_gto; a mismatch raises RuntimeError instead of returning wrong
  numbers.
* Spherical transform: T_l = gto.mole.cart2sph(l) (rows in the same libcint
  cart order); per shell, sph columns = cart columns @ T_l, contractions
  first (ctr-major), matching mol.ao_loc_nr() layout.

Grid blocks come from plan_blocks(): blocks span whole multiples of
gto.eval_gto.BLKSIZE (56) points so grids.non0tab applies as-is; the active
shell set of a block is every shell flagged nonzero in any of its sub-blocks
-- the same set libcint evaluates on the CPU, whose other columns are zero.
"""
import numpy
from pyscf.gto import mole as gto_mole
from pyscf.gto.eval_gto import BLKSIZE

from pyscf_wb97mv_fast.gpu.precision import FP32, FP64

_PAD_EXP = 1e9          # padded primitives: coeff 0, exp(-1e9 r^2) finite


def cart_components(l):
    """[(lx,ly,lz)] in libcint order: lx desc, ly desc, lz = l-lx-ly."""
    return [(lx, ly, l - lx - ly)
            for lx in range(l, -1, -1)
            for ly in range(l - lx, -1, -1)]


class _Group:
    """Shells sharing (l, nctr); primitives padded to a common width."""

    def __init__(self, l, nctr, gid):
        self.l, self.nctr, self.gid = l, nctr, gid
        self.shells = numpy.zeros(0, dtype=numpy.int64)   # ascending ids
        self.centers = numpy.zeros((0, 3))                # FP64, host
        self.exps = numpy.zeros((0, 1))                   # (S,NP) host
        self.coeffs = numpy.zeros((0, 1, 1))              # (S,NCTR,NP) host


class ShellPack:
    """Basis parameters packed once per molecule, uploaded to the device.

    Host FP64 arrays are kept (self-check and re-uploads); device copies get
    a ``_d`` suffix via to_device().  ``origin`` is subtracted from shell
    centers; callers pass grid coordinates relative to the same origin (spec:
    FP32 coordinates relative to the molecular center).
    """

    def __init__(self, mol, origin=None):
        self.mol = mol
        self.cart_basis = bool(mol.cart)
        self.origin = numpy.zeros(3) if origin is None \
            else numpy.asarray(origin, dtype=FP64)
        if not self.cart_basis:
            lmax = int(mol._bas[:, 1].max()) if mol.nbas else 0
            self.c2s = [numpy.asarray(gto_mole.cart2sph(l), dtype=FP64)
                        for l in range(lmax + 1)]
        self.ao_loc = numpy.asarray(mol.ao_loc_nr(), dtype=numpy.int64)
        self.nao = int(self.ao_loc[-1])
        self.groups = []
        self.group_id_of_shell = numpy.zeros(max(mol.nbas, 0), dtype=numpy.int64)

        bas_coords = mol.atom_coords()[mol._bas[:, 0]] - self.origin
        by_key = {}
        for sh in range(mol.nbas):
            l, nctr = int(mol.bas_angular(sh)), int(mol.bas_nctr(sh))
            if int(mol.bas_kappa(sh)) != 0:
                raise ValueError('spinor (kappa) shells are not supported '
                                 '(shell %d)' % sh)
            by_key.setdefault((l, nctr), []).append(sh)
        for gid, (l, nctr) in enumerate(sorted(by_key)):
            shells = by_key[(l, nctr)]
            g = _Group(l, nctr, gid)
            nprim = max(int(mol.bas_nprim(sh)) for sh in shells)
            cs, es, ks = [], [], []
            for sh in shells:
                exps = numpy.asarray(mol.bas_exp(sh), dtype=FP64)
                # env coefficients as libcint sees them: (nprim, nctr),
                # carrying the per-primitive gto_norm (bas_ctr_coeff strips
                # that normalization, so it must not be used here)
                coeff = numpy.asarray(mol._libcint_ctr_coeff(sh), dtype=FP64).T
                if coeff.shape != (nctr, exps.size):
                    raise ValueError('coefficient layout mismatch at shell %d'
                                     % sh)
                cs.append(bas_coords[sh])
                es.append(numpy.concatenate([exps, [_PAD_EXP] * (nprim - exps.size)]))
                ks.append(numpy.concatenate(
                    [coeff, numpy.zeros((nctr, nprim - exps.size))], axis=1))
            g.shells = numpy.asarray(shells, dtype=numpy.int64)
            g.centers = numpy.asarray(cs)
            g.exps = numpy.asarray(es)
            g.coeffs = numpy.asarray(ks)
            self.groups.append(g)
            self.group_id_of_shell[numpy.asarray(shells)] = gid

    def to_device(self, xp, dtype=FP32):
        """Upload the parameters.  Centers stay FP64 (the evaluator forms
        coords - center in FP64 before rounding).  For FP32, every constant
        c also gets its rounding remainder lo = fl32(c - fl32(c)), applied
        by the evaluator: a constant's FP32 rounding is the same at every
        grid point, so uncompensated it biases nelec and E_xc in proportion
        to system size (benchmarks/xc/FINDINGS.md)."""
        lo = dtype == FP32
        for g in self.groups:
            g.centers_d = xp.asarray(g.centers, dtype=FP64)
            g.exps_d = xp.asarray(g.exps, dtype=dtype)
            g.coeffs_d = xp.asarray(g.coeffs, dtype=dtype)
            g.exps_lo_d = g.coeffs_lo_d = None
            if lo:
                exps_lo = _remainder32(g.exps)
                exps_lo[g.exps == _PAD_EXP] = 0.0
                g.exps_lo_d = xp.asarray(exps_lo, dtype=dtype)
                g.coeffs_lo_d = xp.asarray(_remainder32(g.coeffs), dtype=dtype)
        return self


def plan_blocks(mol, grids, blksize=8192):
    """[(i0, i1, shells)] per block: index range into grids.coords plus the
    active shell ids (ascending).  Honors grids.non0tab at BLKSIZE
    granularity exactly like numint.block_loop."""
    coords = grids.coords
    if coords is None:
        raise ValueError('grids not built (grids.coords is None)')
    ngrids = coords.shape[0]
    bs = max(BLKSIZE, (int(blksize) // BLKSIZE) * BLKSIZE)
    non0 = grids.non0tab
    blocks = []
    for i0 in range(0, ngrids, bs):
        i1 = min(i0 + bs, ngrids)
        if non0 is not None and mol is grids.mol:
            rows = non0[i0 // BLKSIZE: -(-i1 // BLKSIZE)]
            shells = numpy.flatnonzero(rows.max(axis=0) > 0)
        else:
            shells = numpy.arange(mol.nbas)
        blocks.append((i0, i1, shells.astype(numpy.int64)))
    return blocks


def bucketize(blocks, shell_bucket=64):
    """Group blocks by (npts, shell-count bucket) for shape-uniform batching
    (spec section 7).  Keys sorted for deterministic processing order."""
    out = {}
    for blk in blocks:
        i0, i1, shells = blk
        out.setdefault((i1 - i0, len(shells) // shell_bucket), []).append(blk)
    return dict(sorted(out.items()))


class AoEvaluator:
    """eval(coords, shells, deriv) -> (ncomp, npts, nao_active).

    coords: (npts,3), RELATIVE to the pack origin, FP64 (host or device): the
    coords - center difference is formed in FP64 and only then rounded.  shells: ascending shell ids (host numpy).  deriv 0
    or 1.  Shell chunks are sized from the streaming memory budget; chunk
    boundaries are fixed by the budget, so results are deterministic.
    """

    def __init__(self, mol, xp, pack=None, mem_budget=256 << 20,
                 dtype=FP32, verify=True):
        self.mol = mol
        self.xp = xp
        self.dtype = dtype
        self.mem_budget = int(mem_budget)
        self.pack = pack if pack is not None else ShellPack(mol)
        if verify and mol.nbas:
            self._self_check()
        self.pack.to_device(xp, dtype)

    def eval(self, coords, shells, deriv=1):
        xp = self.xp
        shells = numpy.asarray(shells, dtype=numpy.int64)
        coords = xp.asarray(coords, dtype=FP64)
        npts = coords.shape[0]
        widths = (self.pack.ao_loc[shells + 1] - self.pack.ao_loc[shells]) \
            if shells.size else numpy.zeros(0, dtype=numpy.int64)
        pos = numpy.zeros(shells.size, dtype=numpy.int64)
        if shells.size:
            pos[1:] = numpy.cumsum(widths)[:-1]
        total = int(widths.sum())
        comps = 1 if deriv == 0 else 4
        out = xp.zeros((comps, npts, total), dtype=self.dtype)

        gid_of = self.pack.group_id_of_shell
        if shells.size == 0:
            return out
        order = numpy.argsort(gid_of[shells], kind='stable')
        starts = numpy.flatnonzero(
            numpy.concatenate([[True], numpy.diff(gid_of[shells[order]]) != 0]))
        bounds = list(starts) + [order.size]
        for a, b in zip(bounds[:-1], bounds[1:]):
            sel = order[a:b]                       # positions inside `shells`
            sub = shells[sel]                      # shell ids, ascending
            g = self.pack.groups[int(gid_of[sub[0]])]
            rows = numpy.searchsorted(g.shells, sub)
            piece = self._eval_group(coords, g, rows, deriv)
            # shells of one (l, nctr) group are NOT adjacent in `shells`
            # when groups interleave, so their output columns are not one
            # contiguous run -- write each shell to its own slot
            c0 = 0
            for k in sel:
                s = shells[k]
                w = int(self.pack.ao_loc[s + 1] - self.pack.ao_loc[s])
                out[:, :, pos[k]:pos[k] + w] = piece[:, :, c0:c0 + w]
                c0 += w
        return out

    # -- kernel ---------------------------------------------------------------
    def _shell_chunk(self, g, npts):
        per_shell = max(1, npts * (g.exps.shape[1] + 1) * 4 * 8)
        return max(1, self.mem_budget // per_shell)

    def _eval_group(self, coords, g, rows, deriv):
        centers = g.centers_d[rows]
        exps = g.exps_d[rows]
        coeffs = g.coeffs_d[rows]
        exps_lo = None if g.exps_lo_d is None else g.exps_lo_d[rows]
        coeffs_lo = None if g.coeffs_lo_d is None else g.coeffs_lo_d[rows]
        T = T_lo = None
        if not self.pack.cart_basis:
            T64 = self.pack.c2s[g.l]
            T = self.xp.asarray(T64, dtype=self.dtype)
            if self.dtype == FP32:
                T_lo = self.xp.asarray(_remainder32(T64), dtype=self.dtype)
        chunk = int(self._shell_chunk(g, coords.shape[0]))
        S = centers.shape[0]

        pieces = []
        for s0 in range(0, S, chunk):
            sl = slice(s0, s0 + chunk)
            pieces.append(self._eval_shell_chunk(
                coords, g.l, g.nctr, centers[sl], exps[sl], coeffs[sl], T, deriv,
                None if exps_lo is None else exps_lo[sl],
                None if coeffs_lo is None else coeffs_lo[sl], T_lo))
        if len(pieces) == 1:
            return pieces[0]
        return self.xp.concatenate(pieces, axis=2)

    def _eval_shell_chunk(self, coords, l, nctr, centers, exps, coeffs, T, deriv,
                          exps_lo=None, coeffs_lo=None, T_lo=None):
        """One chunk of same-(l,nctr) shells: (ncomp, npts, S*nctr*nf).

        *_lo: FP32 rounding remainders of the constants (None for FP64);
        each enters as one extra term so the constant is exact to ~1e-15.
        """
        xp = self.xp
        dtype = self.dtype
        npts = coords.shape[0]
        Sc = centers.shape[0]
        comps = cart_components(l)
        ncart = len(comps)

        d = (coords[:, None, :] - centers[None, :, :]).astype(dtype)  # FP64 diff
        r2 = (d * d).sum(axis=2)                             # (P,Sc)
        arg = -exps[None] * r2[:, :, None]                   # (P,Sc,NP)
        if dtype == FP32:
            # CUDA expf is not correctly rounded and is biased (+5.8e-9 mean,
            # +1.1e-8 on [-1, 0): every primitive,
            # hence nelec and E_xc, drifts the same way.  exp in FP64 of the
            # FP32 argument, rounded once: unbiased.
            E = xp.exp(arg.astype(FP64)).astype(dtype)
        else:
            E = xp.exp(arg)
        if exps_lo is not None:
            # exp(-(a+lo) r2) = E - E lo r2.  Not E * (1 - lo r2): lo r2 ~ 3e-8 is
            # below FP32 resolution at 1, so fl(1 - lo r2) is 1 almost always and
            # the correction is lost systematically; fl(E - E lo r2) rounds a
            # value that sits randomly on the FP32 grid, which is unbiased.
            E = E - (E * exps_lo[None]) * r2[:, :, None]
        # integer powers of the coordinate differences, 0 .. l+1 per axis
        Dp = [[xp.ones_like(r2), d[:, :, 0]],
              [xp.ones_like(r2), d[:, :, 1]],
              [xp.ones_like(r2), d[:, :, 2]]]
        for p in range(2, l + 2):
            for ax in range(3):
                Dp[ax].append(Dp[ax][-1] * d[:, :, ax])
        D = [Dp[0], Dp[1], Dp[2]]                            # D[ax][p]

        val = xp.empty((npts, Sc, nctr, ncart), dtype=dtype)
        grad = (xp.empty((3, npts, Sc, nctr, ncart), dtype=dtype)
                if deriv == 1 else None)
        for c, (lx, ly, lz) in enumerate(comps):
            pw = (lx, ly, lz)
            gv = E * D[0][lx][:, :, None] * D[1][ly][:, :, None] * D[2][lz][:, :, None]
            cv = xp.einsum('psk,sck->psc', gv, coeffs)
            if coeffs_lo is not None:
                cv = cv + xp.einsum('psk,sck->psc', gv, coeffs_lo)
            val[:, :, :, c] = cv.reshape(npts, Sc, nctr)
            if deriv == 1:
                for ax in range(3):
                    p = pw[ax]
                    term = D[ax][p + 1][:, :, None] * exps[None] * (-2.0)
                    if exps_lo is not None:
                        term = term + D[ax][p + 1][:, :, None] * exps_lo[None] * (-2.0)
                    if p > 0:
                        term = term + (p * D[ax][p - 1])[:, :, None]
                    gg = E * term \
                        * D[(ax + 1) % 3][pw[(ax + 1) % 3]][:, :, None] \
                        * D[(ax + 2) % 3][pw[(ax + 2) % 3]][:, :, None]
                    cg = xp.einsum('psk,sck->psc', gg, coeffs)
                    if coeffs_lo is not None:
                        cg = cg + xp.einsum('psk,sck->psc', gg, coeffs_lo)
                    grad[ax, :, :, :, c] = cg.reshape(npts, Sc, nctr)

        if not self.pack.cart_basis:
            val = _cart_to_sph(xp, val, T, T_lo)
            if deriv == 1:
                grad = _cart_to_sph(xp, grad, T, T_lo)
        val_flat = val.reshape(npts, -1)
        if deriv == 0:
            return val_flat[None]
        return xp.stack([val_flat] + [grad[a].reshape(npts, -1)
                                      for a in range(3)], axis=0)

    # -- verification ----------------------------------------------------------
    def _self_check(self):
        """Compare the numpy FP64 path (same code, host arrays) against
        mol.eval_gto at deterministic random points."""
        mol, pack = self.mol, self.pack
        rng = numpy.random.default_rng(20260930)
        span = float(numpy.linalg.norm(pack.origin - mol.atom_coords(),
                                       axis=1).max()) + 2.0
        pts = pack.origin + rng.uniform(-span, span, size=(97, 3))
        name = 'GTOval_cart' if self.pack.cart_basis else 'GTOval_sph'
        ref0 = mol.eval_gto(name, pts, comp=1)          # (npts, nao)
        ref1 = mol.eval_gto(name + '_deriv1', pts)      # (4, npts, nao)
        host_pack = ShellPack(mol, origin=pack.origin).to_device(numpy, FP64)
        check = AoEvaluator.__new__(AoEvaluator)
        check.mol, check.xp, check.dtype = mol, numpy, FP64
        check.pack, check.mem_budget = host_pack, 1 << 30
        mine0 = check.eval(pts - pack.origin, numpy.arange(mol.nbas), deriv=0)
        mine1 = check.eval(pts - pack.origin, numpy.arange(mol.nbas), deriv=1)
        scale = max(float(numpy.abs(ref1[0]).max()), 1e-300)
        ok0 = numpy.allclose(mine0[0], ref0, rtol=0, atol=1e-9 * scale)
        ok1 = numpy.allclose(mine1, ref1, rtol=0, atol=1e-8 * scale)
        if not (ok0 and ok1):
            raise RuntimeError(
                'AO pack self-check failed vs mol.eval_gto (scale %.3e): '
                'check cart ordering / normalization conventions' % scale)


def _cart_to_sph(xp, arr, T, T_lo=None):
    """Transform the last axis (cart components) of arr by T (cart->sph),
    keeping every other axis: arr (..., ncart) -> (..., 2l+1).  T_lo is the
    FP32 rounding remainder of T (the s/p normalization constants alone bias
    every s/p AO by ~5e-8 otherwise)."""
    lead = arr.reshape(-1, arr.shape[-1])
    out = lead @ xp.asarray(T, dtype=arr.dtype)
    if T_lo is not None:
        out = out + lead @ xp.asarray(T_lo, dtype=arr.dtype)
    return out.reshape(arr.shape[:-1] + (T.shape[1],))


def _remainder32(a):
    """fl32(a - fl32(a)) as FP64: the part of a constant FP32 cannot hold."""
    a = numpy.asarray(a, dtype=FP64)
    return (a - a.astype(numpy.float32).astype(FP64)).astype(numpy.float32).astype(FP64)
