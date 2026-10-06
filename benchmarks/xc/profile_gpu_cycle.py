"""Where does one GPU SCF cycle spend its time?  (performance triage)

The water27 FP32 SCF through install_gpu took ~1.5 h per cycle against
~130 s per cycle for the stock CPU SCF (2026-10-01).
No GPU profiler is installed on the development machine (nsys/ncu/nvprof absent), so this
script times the stages by hand with a device synchronize around each one:

    A  kernel_sum throughput (pairs/s) on real VV10 points, n = 16k / 64k,
       plus a per-operation breakdown of ONE tile
    B  gpu_nr_rks on the first K blocks: AO eval / functional / rest
    C  gpu_nr_nlc_vxc on the first K blocks: AO eval / kernel_sum / rest
    D  the stock CPU nr_rks and nr_nlc_vxc on the full grids (reference)

and extrapolates each GPU stage to one full cycle (blocks scale linearly,
kernel_sum as N^2).  Block extrapolation assumes uniform blocks -- an
estimate, flagged as such in the output.

Usage (GPU idle, nothing else running):
    $PY -B benchmarks/xc/profile_gpu_cycle.py [--system water27] [--blocks 8]
Output: benchmarks/results/profile_gpu_cycle_<system>.json (+ stdout).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _paths import add_repo_to_syspath, results_dir  # noqa: E402

add_repo_to_syspath()
from pyscf import dft  # noqa: E402

from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402
from pyscf_wb97mv_fast.gpu import backends, vv10 as vv10_mod, xc as xc_mod  # noqa: E402
from pyscf_wb97mv_fast.gpu.ao_eval import plan_blocks  # noqa: E402
from pyscf_wb97mv_fast.gpu.install import build_context  # noqa: E402
from pyscf_wb97mv_fast.gpu.precision import kahan_sum_axis  # noqa: E402
from pyscf_wb97mv_fast.gpu.vv10 import THRESH, vv10_pointwise  # noqa: E402

XC = 'wb97m-v'
DM_DIR = os.path.join(os.path.dirname(HERE), 'experimental', 'cosx')
cp = backends.cupy_module()


def sync():
    cp.cuda.Device().synchronize()


class Clock:
    def __init__(self):
        self.t = {}

    def wrap(self, label, fn):
        def timed(*a, **k):
            sync()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            sync()
            self.t[label] = self.t.get(label, 0.0) + time.perf_counter() - t0
            return out
        return timed


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    sys.stdout.flush()


def load_dm(system, mol):
    path = os.path.join(DM_DIR, 'dm_%s_def2-svp.npy' % system)
    if os.path.exists(path):
        dm = np.load(path)
    else:
        mf = dft.RKS(mol, xc=XC)
        mf.conv_tol = 1e-8
        mf.kernel()
        dm = mf.make_rdm1()
    return (dm + dm.T) * 0.5


def cpu_rho_gga(ni, mol, grids, dm):
    rho = np.empty((4, grids.coords.shape[0]))
    p0 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, mol.nao, 1):
        p1 = p0 + weight.size
        rho[:, p0:p1] = ni.eval_rho(mol, ao, dm, mask, 'GGA')
        p0 = p1
    return rho


def tile_breakdown(kern, coords, q, W0, kappa):
    """Per-operation time of ONE (outer_chunk x inner_chunk) tile."""
    xp = cp
    co, ci = kern.outer_chunk, kern._inner_chunk()
    ci = min(ci, coords.shape[0])
    c32 = xp.asarray(coords, dtype=xp.float32)
    q32, w032, k32 = (xp.asarray(a, dtype=xp.float32) for a in (q, W0, kappa))
    xi, w0o, ko = c32[:co], w032[:co], k32[:co]
    xj = c32[:ci]
    steps = {}

    def step(label, fn):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        steps[label] = time.perf_counter() - t0
        return out
    for _ in range(2):                          # second pass = warm timings
        d = step('d = xi - xj', lambda: xi[:, None, :] - xj[None, :, :])
        R2 = step('R2 = sum d*d', lambda: (d * d).sum(axis=2))
        g = step('g', lambda: R2 * w0o[:, None] + ko[:, None])
        gp = step('gp', lambda: R2 * w032[None, :ci] + k32[None, :ci])
        gt = step('gt', lambda: g + gp)
        T = step('T', lambda: q32[None, :ci] / (g * gp * gt))
        step('kahan(T)', lambda: kahan_sum_axis(xp, T, 1, kern.n_seg))
        Tu = step('Tu', lambda: T * (1.0 / g + 1.0 / gt))
        step('kahan(Tu)', lambda: kahan_sum_axis(xp, Tu, 1, kern.n_seg))
        step('kahan(Tu*R2)', lambda: kahan_sum_axis(xp, Tu * R2, 1, kern.n_seg))
    return {'outer_chunk': co, 'inner_chunk': ci, 'pairs': co * ci,
            'dtypes': {'T': str(T.dtype), 'Tu': str(Tu.dtype)},
            'steps_s': steps, 'tile_s': sum(steps.values())}


def run(system, nblk, blksize):
    out = {'system': system, 'blocks_profiled': nblk}
    mol = build_mol(system, 'def2-svp')
    dm = load_dm(system, mol)
    mf = dft.RKS(mol, xc=XC)
    mf.grids.build(with_non0tab=True)
    mf.nlcgrids.build(with_non0tab=True)
    ni, grids, nlcgrids = mf._numint, mf.grids, mf.nlcgrids
    blocks_xc = plan_blocks(mol, grids, blksize)
    blocks_nlc = plan_blocks(mol, nlcgrids, blksize)
    out.update(nao=int(mol.nao), npts_xc=int(grids.coords.shape[0]),
               npts_nlc=int(nlcgrids.coords.shape[0]),
               nblocks_xc=len(blocks_xc), nblocks_nlc=len(blocks_nlc),
               device=str(cp.cuda.runtime.getDeviceProperties(0)['name']))
    log('%s nao=%d xc %d pts / %d blocks, nlc %d pts / %d blocks'
        % (system, mol.nao, out['npts_xc'], len(blocks_xc), out['npts_nlc'], len(blocks_nlc)))

    # D: CPU reference on the full grids
    t0 = time.perf_counter()
    ni.nr_rks(mol, grids, XC, dm)
    t_cpu_xc = time.perf_counter() - t0
    t0 = time.perf_counter()
    ni.nr_nlc_vxc(mol, nlcgrids, XC, dm)
    t_cpu_nlc = time.perf_counter() - t0
    out['cpu_full_s'] = {'nr_rks': t_cpu_xc, 'nr_nlc_vxc': t_cpu_nlc}
    log('D cpu: nr_rks %.1fs  nr_nlc_vxc %.1fs' % (t_cpu_xc, t_cpu_nlc))

    # real VV10 point parameters (CPU FP64), threshed like the GPU flow
    rho4 = cpu_rho_gga(ni, mol, nlcgrids, dm)
    ind = rho4[0] >= THRESH
    nlc_pars = ni.nlc_coeff(XC)[0][0]
    q, W0, kappa = vv10_pointwise(rho4[0][ind], rho4[1:4][:, ind],
                                  nlcgrids.weights[ind], nlc_pars, np)[:3]
    coords_t = (nlcgrids.coords[ind] - mol.atom_coords().mean(axis=0))
    n_t = int(ind.sum())
    out['npts_nlc_threshed'] = n_t

    ctx = build_context(mol, backend='cupy', blksize=blksize, verify_ao=False)
    kern = ctx.kernel

    # A: kernel_sum throughput + one-tile breakdown
    rates = {}
    for n in (16384, 65536):
        args = [cp.asarray(a[:n]) for a in (coords_t, q, W0, kappa)]
        kern.kernel_sum(*args)                       # warm-up
        sync()
        t0 = time.perf_counter()
        kern.kernel_sum(*args)
        sync()
        dt = time.perf_counter() - t0
        rates[n] = {'s': dt, 'pairs_per_s': n * n / dt}
        log('A kernel_sum n=%d: %.2fs  %.3e pairs/s' % (n, dt, n * n / dt))
    rate = rates[65536]['pairs_per_s']
    out['kernel_sum'] = {'by_n': {str(k): v for k, v in rates.items()},
                         'inner_chunk': kern._inner_chunk(), 'outer_chunk': kern.outer_chunk,
                         'mem_budget': kern.mem_budget,
                         'full_cycle_est_s': n_t * n_t / rate}
    tb = tile_breakdown(kern, coords_t, q, W0, kappa)
    out['tile_breakdown'] = tb
    log('A one tile %dx%d: %.3fs  %s' % (tb['outer_chunk'], tb['inner_chunk'], tb['tile_s'],
        '  '.join('%s=%.3f' % kv for kv in tb['steps_s'].items())))

    # B: XC flow on the first nblk blocks
    clk = Clock()
    ctx.evaluator.eval = clk.wrap('ao_eval', ctx.evaluator.eval)
    factory = ctx.functional

    def timed_factory(code, ni_):
        f = factory(code, ni_)
        f.eval = clk.wrap('functional', f.eval)
        return f
    ctx.functional = timed_factory
    orig_xc_plan, orig_nlc_plan = xc_mod.plan_blocks, vv10_mod.plan_blocks
    xc_mod.plan_blocks = lambda m, g, b: orig_xc_plan(m, g, b)[:nblk]
    vv10_mod.plan_blocks = lambda m, g, b: orig_nlc_plan(m, g, b)[:nblk]
    try:
        xc_mod.gpu_nr_rks(ctx, ni, mol, grids, XC, dm)       # warm-up
        clk.t.clear()
        sync()
        t0 = time.perf_counter()
        xc_mod.gpu_nr_rks(ctx, ni, mol, grids, XC, dm)
        sync()
        t_xc = time.perf_counter() - t0
        xc_parts = dict(clk.t)
        scale = len(blocks_xc) / nblk
        out['xc_subset_s'] = {'total': t_xc, **xc_parts,
                              'rest': t_xc - sum(xc_parts.values())}
        out['xc_full_cycle_est_s'] = {k: v * scale for k, v in out['xc_subset_s'].items()}
        log('B xc %d blocks: %s  -> full cycle est %.0fs'
            % (nblk, '  '.join('%s=%.2f' % kv for kv in out['xc_subset_s'].items()),
               out['xc_full_cycle_est_s']['total']))

        # C: VV10 flow on the first nblk blocks (kernel_sum then sees only
        # those blocks' points; its full-grid cost comes from A)
        kern.kernel_sum = clk.wrap('kernel_sum', kern.kernel_sum)
        clk.t.clear()
        sync()
        t0 = time.perf_counter()
        vv10_mod.gpu_nr_nlc_vxc(ctx, ni, mol, nlcgrids, XC, dm)
        sync()
        t_nlc = time.perf_counter() - t0
        nlc_parts = dict(clk.t)
        scale = len(blocks_nlc) / nblk
        ao_rest = t_nlc - nlc_parts.get('kernel_sum', 0.0)
        out['nlc_subset_s'] = {'total': t_nlc, **nlc_parts}
        out['nlc_full_cycle_est_s'] = {'ao_and_rest': ao_rest * scale,
                                       'kernel_sum': out['kernel_sum']['full_cycle_est_s']}
        log('C nlc %d blocks: %s  -> full cycle est: AO+rest %.0fs, kernel_sum %.0fs'
            % (nblk, '  '.join('%s=%.2f' % kv for kv in out['nlc_subset_s'].items()),
               out['nlc_full_cycle_est_s']['ao_and_rest'],
               out['nlc_full_cycle_est_s']['kernel_sum']))
    finally:
        xc_mod.plan_blocks, vv10_mod.plan_blocks = orig_xc_plan, orig_nlc_plan

    gpu_total = (out['xc_full_cycle_est_s']['total']
                 + sum(out['nlc_full_cycle_est_s'].values()))
    out['gpu_full_cycle_est_s'] = gpu_total
    out['note'] = 'block extrapolation assumes uniform blocks; kernel_sum scales as N_threshed^2'
    log('== GPU XC+VV10 per cycle est %.0fs  vs CPU %.0fs'
        % (gpu_total, t_cpu_xc + t_cpu_nlc))
    path = os.path.join(results_dir(), 'profile_gpu_cycle_%s.json' % system)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    log('-> %s' % path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--system', default='water27')
    ap.add_argument('--blocks', type=int, default=8)
    ap.add_argument('--blksize', type=int, default=8192)
    args = ap.parse_args(argv)
    if cp is None:
        raise SystemExit('needs CuPy + a CUDA device')
    run(args.system, args.blocks, args.blksize)


if __name__ == '__main__':
    main()
