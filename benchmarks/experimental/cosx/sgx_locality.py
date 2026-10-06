#!/usr/bin/env python
"""
Measure how much work PySCF's SGX (COSX, pjs=True) actually does for
K_full / K_LR / K_SR, and how much an ideal screening would need.

Part 1 (stock counts): libcint_shim.so wraps int1e_grids so every
(shell pair, grid range) task that survives SGX's C-level screening is counted.
Part 2 (oracle): on a sample of SGX grid blocks, compute the exact ESP
integrals for all shell pairs, the exact per-task contribution to the exchange
energy and to the intermediate G = A F, and count the minimum number of tasks
needed for a given error budget. The gap between the two is locality that
stock SGX leaves on the table.

Usage:
    python sgx_locality.py chain30 --oracle-blocks 40
    python sgx_locality.py water27
    python sgx_locality.py my.xyz --basis def2-svp
"""
import argparse
import ctypes
import json
import os
import time

import numpy as np
from pyscf import dft, gto, lib
from pyscf.scf import _vhf
from pyscf.sgx import sgx_jk

HERE = os.path.dirname(os.path.abspath(__file__))
SGX_BLKSIZE = sgx_jk.SGX_BLKSIZE

import sys  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', '..'))  # benchmarks/, for _paths
import _paths  # noqa: E402
_paths.add_repo_to_syspath()
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402


def get_dm(mol, tag):
    """PBE/RI-J density: cheap, but with realistic density-matrix locality."""
    path = os.path.join(HERE, f'dm_{tag}.npy')
    if os.path.exists(path):
        dm = np.load(path)
        if dm.shape == (mol.nao, mol.nao):
            return dm
    mf = dft.RKS(mol, xc='pbe').density_fit()
    mf.conv_tol = 1e-8
    mf.kernel()
    dm = mf.make_rdm1()
    np.save(path, dm)
    return dm


# ----------------------------------------------------------------- stock counts via shim
class Shim:
    def __init__(self, mol):
        self.lib = ctypes.CDLL(os.path.join(HERE, 'libcint_shim.so'))
        name = 'int1e_grids_cart' if mol.cart else 'int1e_grids_sph'
        real = ctypes.cast(getattr(_vhf.libcvhf, name), ctypes.c_void_p)
        self.lib.shim_set_target(real, ctypes.c_int(int(mol.cart)))
        self.ptr = ctypes.cast(self.lib.shim_int1e_grids, ctypes.c_void_p)
        self.rec = None
        self._recording = False
        self.dm_mask_stats = []
        self.g_mask_stats = []
        self._patch()

    def _patch(self):
        orig_fp = _vhf._fpointer
        shim_ptr = self.ptr

        def fpointer(name):
            if name in ('int1e_grids_sph', 'int1e_grids_cart'):
                return shim_ptr
            return orig_fp(name)
        _vhf._fpointer = fpointer

        orig_gen = sgx_jk._gen_k_direct
        lib_ = self.lib

        def gen_k_direct(*args, **kwargs):
            k_part = orig_gen(*args, **kwargs)

            def wrapped(mol, coords, fg, weights=None, b0=None, full_f_bi=None):
                lib_.shim_set_offset(ctypes.c_int(b0 * SGX_BLKSIZE))
                return k_part(mol, coords, fg, weights, b0, full_f_bi)
            return wrapped
        sgx_jk._gen_k_direct = gen_k_direct

        orig_dm_thr = sgx_jk.SGXData.get_dm_threshold_matrix
        orig_g_thr = sgx_jk.SGXData.get_g_threshold
        shim = self

        def dm_thr(this, *a, **k):
            m = orig_dm_thr(this, *a, **k)
            if m is not None:
                shim.dm_mask_stats.append((int(m.sum()), m.size))
            return m

        def g_thr(this, *a, **k):
            m = orig_g_thr(this, *a, **k)
            if m is not None:
                shim.g_mask_stats.append((int(np.count_nonzero(m)), m.size))
            return m
        sgx_jk.SGXData.get_dm_threshold_matrix = dm_thr
        sgx_jk.SGXData.get_g_threshold = g_thr

    def start(self, cap=400_000_000, record=True):
        self.lib.shim_reset()
        if record:
            if self.rec is None or self.rec.shape[0] < cap:
                self.rec = np.empty((cap, 3), dtype=np.int32)
            self.lib.shim_set_record(self.rec.ctypes.data_as(ctypes.c_void_p),
                                     ctypes.c_longlong(cap))
        else:
            self.lib.shim_set_record(None, ctypes.c_longlong(0))
        self._recording = record
        self.dm_mask_stats.clear()
        self.g_mask_stats.clear()

    def stop(self):
        out = np.zeros(5 + 64, dtype=np.int64)
        self.lib.shim_get(out.ctypes.data_as(ctypes.c_void_p))
        self.lib.shim_set_record(None, ctypes.c_longlong(0))
        if self._recording:
            n = int(min(out[4], self.rec.shape[0]))
            rec = self.rec[:n].copy()
        else:
            n, rec = 0, np.empty((0, 3), dtype=np.int32)
        return dict(calls=int(out[0]), points=int(out[1]), ints=int(out[2]),
                    t_int_thread_s=out[3] * 1e-9, rec_overflow=bool(out[4] > n) and self._recording,
                    ints_by_l=out[5:].reshape(8, 8).tolist()), rec


def make_sgx(base, omega, bound_algo, dm_screening=True, etol='auto', vtol='auto'):
    r = base.copy()
    r._rsh_df = {}
    r._vjopt = None
    r._overlap_correction_matrix = None
    r.bound_algo = bound_algo
    if dm_screening:
        r.sgx_tol_energy = etol
        r.sgx_tol_potential = vtol
    else:
        r.sgx_tol_energy = None
        r.sgx_tol_potential = None
    with r.mol.with_range_coulomb(omega):
        r.build()
    return r


def run_k(r, omega, dm, direct_scf_tol):
    with r.mol.with_range_coulomb(omega):
        return r.get_jk(dm, 1, None, with_j=False, with_k=True,
                        direct_scf_tol=direct_scf_tol)[1]


def summarize_records(rec, nblk):
    if rec.shape[0] == 0:
        return {}
    blk = rec[:, 2] // SGX_BLKSIZE
    ish = np.minimum(rec[:, 0], rec[:, 1]).astype(np.int64)
    jsh = np.maximum(rec[:, 0], rec[:, 1]).astype(np.int64)
    pair_id = ish * 100000 + jsh
    task_id = blk.astype(np.int64) * 10**10 + pair_id
    per_blk = np.bincount(blk, minlength=nblk)
    return dict(unique_tasks=int(np.unique(task_id).size),
                unique_pairs=int(np.unique(pair_id).size),
                blocks_touched=int(np.count_nonzero(per_blk)),
                tasks_per_block_mean=float(per_blk[per_blk > 0].mean()),
                tasks_per_block_max=int(per_blk.max()))


# ----------------------------------------------------------------- oracle
def shell_reduce(x, ao_loc, axes):
    for ax in axes:
        x = np.add.reduceat(x, ao_loc[:-1], axis=ax)
    return x


def oracle_block_tasks(mol, dm, coords, weights, omega):
    """Exact per-(shell pair) contributions on one grid block.

    Returns upper-triangle arrays (I<=J): |dE| contribution to Tr(P K) and
    the norm of the contribution to G_{nu g} = sum_lam A_{nu lam}(g) F_{lam g}.
    """
    ao_loc = mol.ao_loc_nr()
    ao = mol.eval_gto('GTOval', coords)
    f = ao @ dm                                          # F_{g lam}
    with mol.with_range_coulomb(omega):
        a = mol.intor('int1e_grids', grids=coords)       # (ng, nao, nao)
    t = a * f[:, None, :]                                # A_{g nu lam} F_{g lam}
    # energy: sum_g w F_{g nu} A F_{g lam}
    e = np.einsum('g,gn,gnl->nl', weights, f, t)
    e = shell_reduce(e, ao_loc, (0, 1))
    # potential: || sum_{lam in J} A F ||_w over nu in I and g
    tj = shell_reduce(t, ao_loc, (2,))                   # (ng, nao, nbas)
    v2 = np.einsum('g,gnj->nj', weights, tj * tj)
    v2 = shell_reduce(v2, ao_loc, (0,))                  # (nbas, nbas)
    iu = np.triu_indices(mol.nbas)
    e_sym = np.abs(e + e.T - np.diag(np.diag(e)))[iu]
    v_sym = np.sqrt(v2 + v2.T - np.diag(np.diag(v2)))[iu]
    return e_sym, v_sym


def oracle(mol, dm, grids, blocks, omega):
    ao_loc = mol.ao_loc_nr()
    nf = np.diff(ao_loc)
    iu = np.triu_indices(mol.nbas)
    pair_nint = (nf[:, None] * nf[None, :])[iu]
    es, vs, ng = [], [], []
    for b in blocks:
        g0, g1 = b * SGX_BLKSIZE, min((b + 1) * SGX_BLKSIZE, grids.weights.size)
        e, v = oracle_block_tasks(mol, dm, grids.coords[g0:g1], grids.weights[g0:g1], omega)
        es.append(e)
        vs.append(v)
        ng.append(g1 - g0)
    return dict(e=np.array(es), v=np.array(vs), ng=np.array(ng), pair_nint=pair_nint)


def oracle_keep_energy(orc, budget):
    """Min #tasks such that sum of |dE| over dropped tasks <= budget (sampled blocks)."""
    e = orc['e'].ravel()
    w = (orc['pair_nint'][None, :] * orc['ng'][:, None]).ravel()
    order = np.argsort(e)
    drop = np.searchsorted(np.cumsum(e[order]), budget, side='right')
    keep = order[drop:]
    return int(keep.size), int(w[keep].sum())


def oracle_keep_potential(orc, vtol):
    """Per block: drop smallest-v tasks while their summed v <= vtol."""
    n, nint = 0, 0
    for b in range(orc['v'].shape[0]):
        v = orc['v'][b]
        w = orc['pair_nint'] * orc['ng'][b]
        order = np.argsort(v)
        drop = np.searchsorted(np.cumsum(v[order]), vtol, side='right')
        keep = order[drop:]
        n += keep.size
        nint += int(w[keep].sum())
    return n, nint


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--oracle-blocks', type=int, default=40)
    ap.add_argument('--bound-algos', default='ovlp,sample,sample_pos')
    ap.add_argument('--direct-scf-tol', type=float, default=1e-13)
    ap.add_argument('--no-reference', action='store_true',
                    help='skip the DM-screening-off reference K')
    args = ap.parse_args()

    tag = os.path.splitext(os.path.basename(args.system))[0] + '_' + args.basis
    mol = build_mol(args.system, args.basis)
    dm = get_dm(mol, tag)
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    base = mf.with_df
    base.build()
    ngrids = base.grids.weights.size
    nblk = (ngrids + SGX_BLKSIZE - 1) // SGX_BLKSIZE
    npair = mol.nbas * (mol.nbas + 1) // 2
    info = dict(system=args.system, basis=args.basis, nao=mol.nao, nbas=mol.nbas,
                natm=mol.natm, ngrids=ngrids, sgx_blksize=SGX_BLKSIZE, nblk=nblk,
                npairs=npair, pair_mask_pairs=None, omega=omega, alpha=alpha, hyb=hyb,
                threads=lib.num_threads(), dense_tasks=nblk * npair)
    pm = base.mol.get_overlap_cond() < -np.log(base.grids.cutoff * 1e2)
    info['pair_mask_pairs'] = int(np.triu(pm).sum())
    print(json.dumps(info))

    shim = Shim(mol)
    kernels = {'full': 0.0, 'LR': omega, 'SR': -omega}
    stock = {}
    kmats = {}
    configs = [(k, algo, True) for algo in args.bound_algos.split(',') for k in kernels]
    if not args.no_reference:
        configs += [(k, 'sample_pos', False) for k in kernels]
    for kname, algo, dmscr in configs:
        om = kernels[kname]
        r = make_sgx(base, om, algo, dmscr)
        run_k(r, om, dm, args.direct_scf_tol)            # warm-up, builds bounds
        shim.start()
        t0 = time.perf_counter()
        k = run_k(r, om, dm, args.direct_scf_tol)
        wall = time.perf_counter() - t0
        cnt, rec = shim.stop()
        cnt.update(summarize_records(rec, nblk))
        cnt['wall_s'] = wall
        dmm = shim.dm_mask_stats
        cnt['dm_mask_kept_frac'] = (dmm[0][0] / dmm[0][1]) if dmm else None
        gm = np.array(shim.g_mask_stats) if shim.g_mask_stats else None
        cnt['g_mask_kept_frac'] = float(gm[:, 0].sum() / gm[:, 1].sum()) if gm is not None else None
        key = f'{kname}|{algo}|{"dmscr" if dmscr else "nodmscr"}'
        stock[key] = cnt
        kmats[key] = k
        if kname == 'SR' and algo == 'sample_pos' and dmscr:
            stock_rec_sr = rec
        if kname == 'LR' and algo == 'sample_pos' and dmscr:
            stock_rec_lr = rec
        if kname == 'full' and algo == 'sample_pos' and dmscr:
            stock_rec_full = rec
        print(key, json.dumps({x: cnt[x] for x in cnt if x != 'ints_by_l'}), flush=True)

    # accuracy of stock screening vs DM-screening-off reference
    acc = {}
    if not args.no_reference:
        for kname in kernels:
            ks = kmats[f'{kname}|sample_pos|dmscr']
            kr = kmats[f'{kname}|sample_pos|nodmscr']
            acc[kname] = dict(dE=float(0.25 * abs(np.einsum('ij,ji', dm, ks - kr))),
                              dK_max=float(abs(ks - kr).max()))
        ka = hyb * kmats['full|sample_pos|dmscr'] + (alpha - hyb) * kmats['LR|sample_pos|dmscr']
        kb = alpha * kmats['full|sample_pos|dmscr'] + (hyb - alpha) * kmats['SR|sample_pos|dmscr']
        kref = hyb * kmats['full|sample_pos|nodmscr'] + (alpha - hyb) * kmats['LR|sample_pos|nodmscr']
        acc['decomp_A'] = dict(dE=float(0.25 * abs(np.einsum('ij,ji', dm, ka - kref))),
                               dK_max=float(abs(ka - kref).max()))
        acc['decomp_B'] = dict(dE=float(0.25 * abs(np.einsum('ij,ji', dm, kb - kref))),
                               dK_max=float(abs(kb - kref).max()))
        print('accuracy', json.dumps(acc), flush=True)

    # oracle on sampled blocks
    orc_out = {}
    if args.oracle_blocks > 0:
        nsamp = min(args.oracle_blocks, nblk)
        blocks = np.unique(np.linspace(0, nblk - 1, nsamp).round().astype(int))
        frac = blocks.size / nblk
        stock_rec = {'full': stock_rec_full, 'LR': stock_rec_lr, 'SR': stock_rec_sr}
        e_budgets = [1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
        v_budgets = [1e-4, 1e-5, 1e-6, 1e-7]
        orcs = {}
        for kname, om in kernels.items():
            t0 = time.perf_counter()
            orc = oracle(mol, dm, base.grids, blocks, om)
            orcs[kname] = orc
            rec = stock_rec[kname]
            in_samp = np.isin(rec[:, 2] // SGX_BLKSIZE, blocks)
            ish, jsh = rec[in_samp, 0], rec[in_samp, 1]
            nf = np.diff(mol.ao_loc_nr())
            ng_rec = np.full(ish.size, SGX_BLKSIZE)  # upper bound; last block may be short
            d = dict(sample_blocks=int(blocks.size), sample_frac=frac,
                     stock_tasks=int(in_samp.sum()),
                     stock_ints=int((nf[ish] * nf[jsh] * ng_rec).sum()),
                     dense_tasks=int(orc['e'].size),
                     energy={}, potential={}, t_s=time.perf_counter() - t0)
            for eb in e_budgets:
                # budget is for the whole grid (factor 0.25: E_x = -1/4 Tr PK)
                d['energy'][f'{eb:.0e}'] = oracle_keep_energy(orc, eb / 0.25 * frac)
            for vb in v_budgets:
                d['potential'][f'{vb:.0e}'] = oracle_keep_potential(orc, vb)
            orc_out[kname] = d
            print('oracle', kname, json.dumps(d), flush=True)
        # decomposition-level totals at a total energy budget eps, split 50/50
        dec = {}
        for eps in e_budgets:
            s = eps / 0.25 * frac

            def keep(kn, b):
                return oracle_keep_energy(orcs[kn], b)
            fa, la = keep('full', s / 2 / hyb), keep('LR', s / 2 / (alpha - hyb))
            fb, sb = keep('full', s / 2 / alpha), keep('SR', s / 2 / (alpha - hyb))
            dec[f'{eps:.0e}'] = dict(A_full=fa, A_LR=la, B_full=fb, B_SR=sb,
                                     A_total=[fa[0] + la[0], fa[1] + la[1]],
                                     B_total=[fb[0] + sb[0], fb[1] + sb[1]])
        orc_out['decomposition'] = dec
        print('decomposition', json.dumps(dec), flush=True)

    out = dict(info=info, stock=stock, accuracy=acc, oracle=orc_out)
    with open(os.path.join(HERE, f'result_{tag}.json'), 'w') as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


if __name__ == '__main__':
    main()
