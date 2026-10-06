"""S5 Task 2: host-side FP64 shell-pair preprocessing for the GPU COSX K.

Everything the int3c1e / sgx_k kernels need, built once per (mol, cutoff) in
FP64 and uploaded as FP32 by ``ShellPairs.to_device``:

* shell pairs (ish >= jsh, l <= LMAX = 3: f shells, def2-TZVP on H..I) with
  a per-primitive magnitude cutoff,
* per primitive PAIR (alpha in ish, beta in jsh): p, P, K_AB*(2pi/p), the
  McMurchie-Davidson Hermite expansion coefficients E^{i,i'}_t for the
  x/y/z axes (locked definitions in the S5 plan), and the screening-bound
  prefactors used by ``pair_bound``,
* per shell, the combined contraction + cart->sph transform matrices that
  turn primitive Cartesian blocks into contracted spherical AO columns.

Contraction layout (the ao_eval.ShellPack convention): the AO columns of a
shell are contraction-major, (ci, mu_sph).  With

    Mc[(ci,mu), (alpha,ca)] = C[alpha,ci] * cart2sph(l)[mu,ca]     (nia x nprim*nca)
    MT[(beta,cb), (cj,nu)]  = C[beta,cj] * cart2sph(l)[nu,cb]      (nprim*ncb x nib)

a primitive Cartesian block A_{alpha beta}[ca,cb] contracts as

    B_alpha[ca,(cj,nu)]  = sum_{beta,cb} MT[(beta,cb),(cj,nu)] A[ca,cb]
    A[(ci,mu),(cj,nu)]  += sum_ca Mc[(ci,mu),(alpha,ca)] B_alpha[ca,(cj,nu)]

so one pass over the primitive pairs of a shell pair yields the full
contracted spherical block.

``int3c1e_reference`` is the FP64 NumPy validation implementation of the
locked math (small systems only); ``pair_bound`` is the locked screening
upper bound, valid for omega = 0 and omega > 0 (erf(wr)/r <= 1/r).
"""
import numpy as np
from pyscf.gto import mole as gto_mole
from scipy.special import erf, gamma, gammainc

from pyscf_wb97mv_fast.gpu.xc import Unsupported


def numpy_origin(origin):
    return np.asarray(origin, dtype=np.float64)

LMAX = 3             # highest shell angular momentum (f)
PPR_MAX = 16         # max primitives per shell
NIA_MAX = 16         # max contracted spherical functions per shell
NSA_MAX = 2 * LMAX + 1                   # 7
NCA_MAX = (LMAX + 1) * (LMAX + 2) // 2   # 10
KMAX = PPR_MAX * NCA_MAX                 # 160
E_L = LMAX + 1       # Hermite E layout [axis, i, i', t]: i, i' <= LMAX,
E_T = 2 * LMAX + 1   # t <= i + i' <= 2 LMAX
E_SIZE = 3 * E_L * E_L * E_T             # 336 floats per primitive pair
_ETA = 0.25          # eta of the screening bound (p' = (1-eta) p)


def _boys_ref(n, T):
    """FP64 F_n(T), scipy exact form (independent of gpu.boys)."""
    T = np.asarray(T, dtype=np.float64)
    out = np.empty_like(T)
    small = T < 1e-12
    out[small] = 1.0 / (2 * n + 1)
    Ts = T[~small]
    out[~small] = gamma(n + 0.5) * gammainc(n + 0.5, Ts) / (2.0 * Ts ** (n + 0.5))
    return out


def _boys0(T):
    """F_0(T) = sqrt(pi/T)/2 erf(sqrt(T)) (pair_bound only)."""
    T = np.asarray(T, dtype=np.float64)
    out = np.ones_like(T)
    nz = T > 0
    Tz = T[nz]
    out[nz] = 0.5 * np.sqrt(np.pi / Tz) * erf(np.sqrt(Tz))
    return out


def _hermite_row(prev, x, inv2p, tmax_new):
    """E^{i+1,j}_t (or E^{i,j+1}_t) from row prev, vectorized over rows:
    prev (m, tmax_prev+1), x (m,) -> (m, tmax_new+1).  Locked recurrence
    E_t <- E_{t-1}/(2p) + X E_t + (t+1) E_{t+1}."""
    m = prev.shape[0]
    new = np.zeros((m, tmax_new + 1))
    for t in range(tmax_new + 1):
        v = np.zeros(m)
        if t > 0:
            v += prev[:, t - 1] * inv2p
        if t < prev.shape[1]:
            v += x * prev[:, t]
        if t + 1 < prev.shape[1]:
            v += (t + 1) * prev[:, t + 1]
        new[:, t] = v
    return new


def _hermite_axes(la, lb, p, dpa, dpb):
    """E^{i,i'}_t for the three axes, vectorized over prim pairs.
    p (m,), dpa/dpb (m, 3) = P - A / P - B.  Returns (m, 3, E_L, E_L, E_T)
    laid out [axis, i, i', t] (i along A, i' along B, t <= i+i'; padded to
    l <= LMAX)."""
    m = len(p)
    inv2p = 0.5 / p
    out = np.zeros((m, 3, E_L, E_L, E_T))
    for ax in range(3):
        E = np.zeros((m, E_L, E_L, E_T))
        E[:, 0, 0, 0] = 1.0
        for i in range(1, la + 1):
            E[:, i, 0, :i + 1] = _hermite_row(E[:, i - 1, 0, :i],
                                              dpa[:, ax], inv2p, i)
        for j in range(1, lb + 1):
            for i in range(la + 1):
                E[:, i, j, :i + j + 1] = _hermite_row(
                    E[:, i, j - 1, :i + j], dpb[:, ax], inv2p, i + j)
        out[:, ax] = E
    return out


def _cart_comps(l):
    """[(lx,ly,lz)] in libcint order (= ao_eval.cart_components)."""
    return [(lx, ly, l - lx - ly)
            for lx in range(l, -1, -1)
            for ly in range(l - lx, -1, -1)]


def _bound_c(la, lb, p, rpa, rpb):
    """C_ab = sup_{s>=0} (s+|PA|)^la (s+|PB|)^lb exp(-eta p s^2) of the
    locked definition: >= 4096-point grid on [0, s_max] with s_max where the
    envelope drops below 1e-30 of the peak, times 1.01 safety.  Vectorized
    over prim pairs; la = lb = 0 gives exactly 1."""
    la = np.asarray(la)
    lb = np.asarray(lb)
    p = np.asarray(p, dtype=np.float64)
    rpa = np.asarray(rpa, dtype=np.float64)
    rpb = np.asarray(rpb, dtype=np.float64)
    C = np.ones_like(p)
    need = (la > 0) | (lb > 0)
    if not need.any():
        return C
    La, Lb = la[need], lb[need]
    p_n, ra, rb = p[need], rpa[need], rpb[need]
    L = (La + Lb).astype(np.float64)
    rmax = np.maximum(ra, rb)
    s = np.sqrt(np.maximum(0.0, (L * np.log(rmax + 1.0) + 69.0) / (_ETA * p_n)))
    for _ in range(8):
        s = np.sqrt(np.maximum(
            0.0, (L * np.log(np.maximum(s, 1e-300) + rmax) + 69.0)
            / (_ETA * p_n)))
    s_max = 1.05 * s
    ss = np.linspace(0.0, 1.0, 4097)
    Cn = np.empty(len(p_n))
    chunk = max(1, int(2e6 // len(ss)))
    emax = int(max(La.max(), Lb.max()))
    for i0 in range(0, len(p_n), chunk):
        sl = slice(i0, min(i0 + chunk, len(p_n)))
        x = ss[:, None] * s_max[sl]                      # (S, m)
        wa = x + ra[sl]
        wb = x + rb[sl]
        fa = np.ones_like(wa); fb = np.ones_like(wb)
        for e in range(1, emax + 1):
            if e > 1:
                wa = wa * (x + ra[sl])
                wb = wb * (x + rb[sl])
            if (La[sl] == e).any():
                fa = np.where(La[sl] == e, wa, fa)
            if (Lb[sl] == e).any():
                fb = np.where(Lb[sl] == e, wb, fb)
        f = fa * fb * np.exp(-_ETA * p_n[sl] * x * x)
        Cn[sl] = f.max(axis=0)
    C[need] = Cn * 1.01
    return C


def _remainder32(a):
    """fl32(a - fl32(a)) as a contiguous float32 array (ao_eval pattern)."""
    a = np.asarray(a, dtype=np.float64)
    return np.ascontiguousarray((a - a.astype(np.float32).astype(np.float64))
                                .astype(np.float32))


def _pad2(a, shape):
    out = np.zeros(shape, dtype=a.dtype)
    out[:a.shape[0], :a.shape[1]] = a
    return out


class ShellPairs:
    """Host FP64 shell-pair data; build once per (mol, cutoff)."""

    def __init__(self, mol, cutoff=1e-14):
        if mol.cart:
            raise Unsupported('GPU COSX K supports spherical bases only')
        ao_loc = np.asarray(mol.ao_loc_nr(), dtype=np.int64)
        self.mol = mol
        self.nao = int(ao_loc[-1])

        # -- per-shell constants ------------------------------------------------
        sh_l, sh_nctr, sh_nprim, sh_exps, sh_ctr, sh_c2s, sh_tau, sh_ctrds = \
            [], [], [], [], [], [], [], []
        for sh in range(mol.nbas):
            l = int(mol.bas_angular(sh))
            if int(mol.bas_kappa(sh)) != 0:
                raise Unsupported('spinor shells are not supported')
            if l > LMAX:
                raise Unsupported('GPU COSX K supports l <= %d (shell %d has '
                                  'l=%d); def2-svp/tzvp are covered'
                                  % (LMAX, sh, l))
            nctr = int(mol.bas_nctr(sh))
            nprim = int(mol.bas_nprim(sh))
            nia = nctr * (2 * l + 1)
            if nprim > PPR_MAX or nia > NIA_MAX:
                raise Unsupported(
                    'shell %d: nprim=%d nctr=%d exceed the packed caps '
                    '(%d, %d functions per shell)' % (sh, nprim, nctr,
                                                      PPR_MAX, NIA_MAX))
            exps = np.asarray(mol.bas_exp(sh), dtype=np.float64)
            # (nctr, nprim) as libcint sees them (per-primitive normalization
            # included; bas_ctr_coeff must not be used)
            C = np.asarray(mol._libcint_ctr_coeff(sh), dtype=np.float64).T
            sh_l.append(l); sh_nctr.append(nctr); sh_nprim.append(nprim)
            sh_exps.append(exps); sh_ctr.append(C)
            sh_c2s.append(np.asarray(gto_mole.cart2sph(l), dtype=np.float64))
            sh_tau.append(float(np.abs(sh_c2s[-1]).sum(axis=0).max()))
            sh_ctrds.append(mol.atom_coords()[mol._bas[sh, 0]])
        self.sh_l = np.array(sh_l, dtype=np.int64)
        self.sh_tau = np.array(sh_tau, dtype=np.float64)

        # per-shell transform matrices (FP64; padded for the device layout)
        nsh = mol.nbas
        self.sh_dims = np.zeros((nsh, 4), dtype=np.int64)   # nia, nprim, nca, nsa
        self.sh_Mc = np.zeros((nsh, NIA_MAX, KMAX))
        self.sh_MT = np.zeros((nsh, KMAX, NIA_MAX))
        for sh in range(nsh):
            l, nctr, nprim = sh_l[sh], sh_nctr[sh], sh_nprim[sh]
            nca, nsa = (l + 1) * (l + 2) // 2, 2 * l + 1
            ni = nctr * nsa
            self.sh_dims[sh] = (ni, nprim, nca, nsa)
            # Mc[(ci,mu),(alpha,ca)] = C[alpha,ci] * c2s[mu,ca]
            # m[alpha, ci, cart, sph] -> rows (ci, sph), cols (alpha, cart)
            m = (sh_ctr[sh].T[:, :, None, None]        # (nprim,nctr,1,1)
                 * sh_c2s[sh][None, None, :, :])       # (1,1,ncart,nsph)
            mc = m.transpose(1, 3, 0, 2).reshape(ni, nprim * nca)
            self.sh_Mc[sh, :ni, :nprim * nca] = mc
            # MT[(beta,cb),(cj,nu)] = C[beta,cj] * c2s[cb,nu]
            # (rows (beta, cart), cols (cj, sph); c2s rows are cart)
            mt = (sh_c2s[sh][:, None, None, :]         # (ncart,1,1,nsph)
                  * sh_ctr[sh].T[None, :, :, None])    # (1,nprim,nctr,1)
            mt = mt.transpose(1, 0, 2, 3).reshape(nprim * nca, ni)
            self.sh_MT[sh, :nprim * nca, :ni] = mt

        # -- candidate primitive pairs of all shell pairs (ish >= jsh) ---------
        cand_pair, cand_a, cand_b = [], [], []
        pair_meta = []
        for ish in range(nsh):
            Ai, npi = sh_ctrds[ish], sh_nprim[ish]
            for jsh in range(ish + 1):
                npj = sh_nprim[jsh]
                base = len(cand_a)
                cand_pair.extend([len(pair_meta)] * (npi * npj))
                # rows enumerate (a, b) with b fastest
                cand_a.extend([a for a in range(npi) for _ in range(npj)])
                cand_b.extend(list(range(npj)) * npi)
                pair_meta.append((ish, jsh, base, npi * npj))
        cand_pair = np.array(cand_pair, dtype=np.int64)
        cand_a = np.array(cand_a, dtype=np.int64)
        cand_b = np.array(cand_b, dtype=np.int64)
        if len(pair_meta) == 0:
            self._empty()
            return
        m_ish = np.array([pm[0] for pm in pair_meta])
        m_jsh = np.array([pm[1] for pm in pair_meta])
        m_base = np.array([pm[2] for pm in pair_meta])
        m_np = np.array([pm[3] for pm in pair_meta])

        # gather primitive exponents/centers per candidate row
        exps_all = np.concatenate(sh_exps)
        ctr_all = np.concatenate(sh_ctrds).reshape(nsh, 3)
        sh_off = np.cumsum([0] + [len(e) for e in sh_exps])
        a_row = m_ish[cand_pair]                     # shell of alpha
        b_row = m_jsh[cand_pair]
        al = exps_all[sh_off[a_row] + cand_a]
        be = exps_all[sh_off[b_row] + cand_b]
        A = ctr_all[a_row]
        B = ctr_all[b_row]
        p = al + be
        mu = al * be / p
        d2 = ((A - B) ** 2).sum(axis=1)
        P = (al[:, None] * A + be[:, None] * B) / p[:, None]
        K = np.exp(-mu * d2)
        # magnitude filter (max over contraction columns per primitive;
        # the shell-level max of |C| bounds both, exact per-prim values
        # are rebuilt below for the kept rows)
        cmax_prim = [np.abs(C).max(axis=0) for C in sh_ctr]      # per prim
        cmax_sh = np.array([cmax_prim[s].max() for s in range(nsh)])
        mag = K * cmax_sh[a_row] * cmax_sh[b_row]
        pair_mag = np.add.reduceat(mag, m_base)
        keep = pair_mag > cutoff
        if not keep.any():
            self._empty()
            return

        # -- kept rows only -----------------------------------------------------
        pair_id_k = np.flatnonzero(keep)
        ish_k = m_ish[pair_id_k]
        jsh_k = m_jsh[pair_id_k]
        base_k = m_base[pair_id_k]
        npk = m_np[pair_id_k]
        row_sel = np.concatenate([base_k[i] + np.arange(npk[i])
                                  for i in range(len(pair_id_k))])
        # prim-pair offsets per kept pair (this defines the flat prim order)
        offs = np.zeros(len(pair_id_k) + 1, dtype=np.int64)
        offs[1:] = np.cumsum(npk)
        self.pair_rows = np.stack([
            ao_loc[ish_k], ao_loc[jsh_k], ish_k, jsh_k,
            offs[:-1], npk], axis=1).astype(np.int64)
        self.npairs = len(pair_id_k)
        self.prim_offsets = offs

        al = al[row_sel]; be = be[row_sel]
        A = A[row_sel]; B = B[row_sel]
        p = al + be
        P = (al[:, None] * A + be[:, None] * B) / p[:, None]
        mu = al * be / p
        K = K[row_sel]
        self.prim_p = p
        self.prim_P = P
        self.prim_prefac = K * (2.0 * np.pi / p)

        # Hermite coefficients (masked vectorized recurrence),
        # layout [axis, i, i', t] padded to l <= LMAX
        a_row_k = np.repeat(ish_k, npk)
        b_row_k = np.repeat(jsh_k, npk)
        cand_a_k = cand_a[row_sel]
        cand_b_k = cand_b[row_sel]
        la_r = self.sh_l[a_row_k]
        lb_r = self.sh_l[b_row_k]
        dpa = P - A
        dpb = P - B
        E = np.zeros((len(p), 3, E_L, E_L, E_T))
        inv2p = 0.5 / p
        E[:, :, 0, 0, 0] = 1.0
        for ax in range(3):
            for i in range(1, LMAX + 1):
                msk = la_r >= i
                if msk.any():
                    E[msk, ax, i, 0, :i + 1] = _hermite_row(
                        E[msk, ax, i - 1, 0, :i], dpa[msk, ax], inv2p[msk], i)
            for j in range(1, LMAX + 1):
                for i in range(LMAX + 1):
                    msk = (lb_r >= j) & (la_r >= i)
                    if msk.any():
                        E[msk, ax, i, j, :i + j + 1] = _hermite_row(
                            E[msk, ax, i, j - 1, :i + j], dpb[msk, ax],
                            inv2p[msk], i + j)
        self.prim_E = E.reshape(len(p), E_SIZE)

        # screening bound (locked definition)
        rpa = np.linalg.norm(P - A, axis=1)
        rpb = np.linalg.norm(P - B, axis=1)
        Cab = _bound_c(la_r, lb_r, p, rpa, rpb)
        # |c| maxima per contraction column, per primitive of the row
        cmax_i = np.empty(len(p))
        cmax_j = np.empty(len(p))
        for s in np.unique(a_row_k):
            sel = a_row_k == s
            cmax_i[sel] = cmax_prim[s][cand_a_k[sel]]
        for s in np.unique(b_row_k):
            sel = b_row_k == s
            cmax_j[sel] = cmax_prim[s][cand_b_k[sel]]
        pp = (1.0 - _ETA) * p
        self.prim_pp = pp
        self.prim_bound_pref = cmax_i * cmax_j * K * Cab * (2.0 * np.pi / pp)
        self.pair_tau = self.sh_tau[ish_k] * self.sh_tau[jsh_k]

    def _empty(self):
        self.pair_rows = np.zeros((0, 6), dtype=np.int64)
        self.npairs = 0
        self.prim_offsets = np.zeros(1, dtype=np.int64)
        self.prim_p = np.zeros(0)
        self.prim_P = np.zeros((0, 3))
        self.prim_prefac = np.zeros(0)
        self.prim_E = np.zeros((0, E_SIZE))
        self.prim_pp = np.zeros(0)
        self.prim_bound_pref = np.zeros(0)
        self.pair_tau = np.zeros(0)

    @property
    def nprims(self):
        return int(self.prim_offsets[-1])

    def ao_blocks(self):
        """(npairs, 4) int64: (i0, i1, j0, j1) AO column ranges per pair."""
        r = self.pair_rows
        nia = self.sh_dims[r[:, 2], 0]
        nib = self.sh_dims[r[:, 3], 0]
        return np.stack([r[:, 0], r[:, 0] + nia,
                         r[:, 1], r[:, 1] + nib], axis=1)

    def to_device(self, xp, dtype='float32', mem_budget=None, origin=None):
        """Upload as FP32 (p stays FP64; P keeps its FP32 rounding remainder
        hi+lo as in ao_eval).  ``origin`` (3,) subtracts a constant offset
        from every P so the kernels can run in the molecular-center frame of
        the AO evaluator.  Host bound data are not uploaded."""
        if mem_budget is None:
            mem_budget = 256 << 20
        P = self.prim_P
        if origin is not None:
            P = P - numpy_origin(origin)
        hi = P.astype(np.float32)
        lo = (P - hi.astype(np.float64)).astype(np.float32)
        return ShellPairsDevice(
            xp=xp, mem_budget=int(mem_budget),
            nao=int(self.nao),
            sh_dims=xp.asarray(self.sh_dims.astype(np.int32)),
            sh_Mc=xp.asarray(np.ascontiguousarray(self.sh_Mc,
                                                  dtype=np.float32)),
            sh_MT=xp.asarray(np.ascontiguousarray(self.sh_MT,
                                                  dtype=np.float32)),
            # FP32 rounding remainders of Mc / MT (c2s x contraction
            # coefficients): the same constants serve every grid point, so
            # their rounding is a systematic per-function bias -- measured
            # 4.2e-8 mean relative on the dimer; applied per product with
            # fmaf (int3c1e)
            sh_Mc_lo=xp.asarray(_remainder32(self.sh_Mc)),
            sh_MT_lo=xp.asarray(_remainder32(self.sh_MT)),
            pair_sh=xp.asarray(np.ascontiguousarray(self.pair_rows,
                                                    dtype=np.int32)),
            prim_p=xp.asarray(np.ascontiguousarray(self.prim_p,
                                                   dtype=np.float64)),
            prim_prefac=xp.asarray(np.ascontiguousarray(
                self.prim_prefac, dtype=np.float32)),
            # per-prim-pair constant as well: -1.1e-7 full-K energy bias
            # uncompensated
            prim_prefac_lo=xp.asarray(_remainder32(self.prim_prefac)),
            prim_P_hi=xp.asarray(np.ascontiguousarray(hi, dtype=np.float32)),
            prim_P_lo=xp.asarray(np.ascontiguousarray(lo, dtype=np.float32)),
            prim_E=xp.asarray(np.ascontiguousarray(self.prim_E,
                                                   dtype=np.float32)),
            npairs=int(self.npairs), nprims=self.nprims,
        )


class ShellPairsDevice:
    """FP32 device-side copy of ShellPairs (see ShellPairs.to_device)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fill_R(T, pth, sth, d):
    """R^0_{tuv}(p, P-C) for all grid points: levels 2 LMAX -> 0 in place.

    Level-n entries are built from level-(n+1) (locked recursion); the
    descending t/u/v loops only read slots not yet overwritten at the
    current level, and slots outside n+t+u+v <= 2 LMAX stay on zero chains,
    which never feed a valid entry."""
    ng = T.shape[0]
    L = E_T - 1
    F = np.stack([_boys_ref(n, T) for n in range(L + 1)])    # (L+1, ng)
    q = -2.0 * pth
    R000 = np.empty((L + 1, ng))
    qn = np.full(ng, sth)
    for n in range(L + 1):
        R000[n] = qn * F[n]
        qn = qn * q
    R = np.zeros((ng, E_T, E_T, E_T))
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    for n in range(L, -1, -1):
        for t in range(L, 0, -1):
            if t == 1:
                R[:, 1, :, :] = x[:, None, None] * R[:, 0, :, :]
            else:
                R[:, t, :, :] = ((t - 1) * R[:, t - 2, :, :]
                                 + x[:, None, None] * R[:, t - 1, :, :])
        for u in range(L, 0, -1):
            if u == 1:
                R[:, 0, 1, :] = y[:, None] * R[:, 0, 0, :]
            else:
                R[:, 0, u, :] = ((u - 1) * R[:, 0, u - 2, :]
                                 + y[:, None] * R[:, 0, u - 1, :])
        for v in range(L, 0, -1):
            if v == 1:
                R[:, 0, 0, 1] = z * R[:, 0, 0, 0]
            else:
                R[:, 0, 0, v] = ((v - 1) * R[:, 0, 0, v - 2]
                                 + z * R[:, 0, 0, v - 1])
        R[:, 0, 0, 0] = R000[n]
    return R


def _flat(comp, n):
    i, j, k = comp
    return (i * n + j) * n + k


def _scatter(out, Ablock, ia0, ia1, ib0, ib1, diagonal):
    out[:, ia0:ia1, ib0:ib1] += Ablock
    if not diagonal:
        out[:, ib0:ib1, ia0:ia1] += Ablock.transpose(0, 2, 1)


def int3c1e_reference(pairs, coords, omega=0.0):
    """FP64 NumPy (ng, nao, nao) spherical A_g,munu = (mu|1/|r-g||nu), with
    the erf(omega r)/r kernel for omega > 0.  Validation only (small
    systems): the dense tensor is materialized."""
    coords = np.asarray(coords, dtype=np.float64)
    ng = coords.shape[0]
    out = np.zeros((ng, pairs.nao, pairs.nao))
    for k in range(pairs.npairs):
        ia0, ib0, ish, jsh, prim0, npk = pairs.pair_rows[k]
        ni, npi, nca, nsa = pairs.sh_dims[ish]
        nj, npj, ncb, nsb = pairs.sh_dims[jsh]
        la, lb = (nsa - 1) // 2, (nsb - 1) // 2
        comps_a, comps_b = _cart_comps(la), _cart_comps(lb)
        ci = pairs.sh_Mc[ish][:ni, :npi * nca]
        mtj = pairs.sh_MT[jsh][:npj * ncb, :nj].reshape(npj, ncb, nj)
        # the einsum box runs over the padded E_L^3 cart index cube
        fa = [_flat(c, E_L) for c in comps_a]
        fb = [_flat(c, E_L) for c in comps_b]
        Ablock = np.zeros((ng, ni, nj))
        for a in range(npi):
            B = np.zeros((ng, nca, nj))
            for b in range(npj):
                m = prim0 + a * npj + b
                p = pairs.prim_p[m]
                P = pairs.prim_P[m]
                theta, stheta = 1.0, 1.0
                if omega > 0:
                    w2 = omega * omega
                    theta = w2 / (w2 + p)
                    stheta = np.sqrt(theta)
                d = P[None, :] - coords                       # (ng, 3)
                r2 = (d * d).sum(axis=1)
                R = _fill_R(theta * p * r2, p * theta, stheta, d)
                Exyz = pairs.prim_E[m].reshape(3, E_L, E_L, E_T)
                Zc = np.einsum('abv,gtuv->gtuab', Exyz[2], R)   # (g,t,u,k,kp)
                Yc = np.einsum('cdu,gtuab->gtabcd', Exyz[1], Zc)
                Yr = Yc.transpose(0, 4, 5, 2, 3, 1)             # (g,j,jp,k,kp,t)
                full = np.einsum('imt,gjcket->gijkmce', Exyz[0], Yr)
                Ac = pairs.prim_prefac[m] * \
                    full.reshape(ng, E_L ** 3, E_L ** 3)[:, fa][:, :, fb]
                B += np.einsum('gab,bn->gan', Ac, mtj[b])
            Ablock += np.einsum('ma,gan->gmn', ci[:, a * nca:(a + 1) * nca], B)
        _scatter(out, Ablock, ia0, ia0 + ni, ib0, ib0 + nj, ish == jsh)
    return out


def pair_bound(pairs, coords_block):
    """(npairs,) FP64 screening bound of the locked definition: an upper
    bound of |A_g,munu| for every grid point g of the block and every
    (mu, nu) column pair.  Valid for omega = 0 and omega > 0."""
    coords = np.asarray(coords_block, dtype=np.float64)
    c = coords.mean(axis=0)
    rad = np.linalg.norm(coords - c, axis=1).max() if len(coords) else 0.0
    d = np.linalg.norm(pairs.prim_P - c, axis=1) - rad
    np.maximum(d, 0.0, out=d)
    b = pairs.prim_bound_pref * _boys0(pairs.prim_pp * d * d)
    return pairs.pair_tau * np.add.reduceat(b, pairs.prim_offsets[:-1])


def build_shell_pairs(mol, cutoff=1e-14):
    """ShellPairs for mol (l <= LMAX, spherical); raises xc.Unsupported
    outside the supported scope."""
    return ShellPairs(mol, cutoff=cutoff)
