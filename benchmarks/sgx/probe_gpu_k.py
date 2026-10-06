"""S5 feasibility probe: COSX exchange K on the GPU (spec sections 4, 8.5).

Question: with the CPU XC/VV10 bottleneck gone (benchmarks/xc/FINDINGS.md
section 9), the SCF cycle is now the CPU COSX K.  Can the GPU build the same
K, and how fast, before any FP32 or screening work?

Method -- no new integral code: gpu4pyscf 1.8.1 already has the SGX-type
3-center 1e integrals on the GPU (gto/int3c1e.py, A_g,uv = (u|1/|r-g||v),
erf-attenuated for omega > 0).  Its public int1e_grids copies the whole
(N_g, nao, nao) tensor to the host -- ~1 TB per K build for water27 -- so
this probe re-runs its slice loop (same VHFOpt, same GINTfill_int3c1e calls)
and contracts every slice on the device right away:

    proj_dm = proj @ dm              (proj: SGX fit_ovlp overlap correction)
    F_gv    = sum_k X_gk proj_dm_kv  (X = AO values on the SGX grid)
    G_gv    = w_g sum_t A_g,tv F_gt
    K       = X^T G,  then sym_ovlp / hermi exactly as sgx_jk.get_k_only

FP64 throughout; reference = sgx_jk.get_k_only (the COSX(pjs=True) path) on
the CPU for the same dm and the same SGX grid.  Full K (omega = 0) and the
long-range K (omega = 0.3 for wB97M-V) are both checked.  Weight placement is
not assumed: K with w applied once is compared against the CPU result, and the
report says whether it matched.

Usage (GPU idle, nothing else running):
    $PY -B benchmarks/sgx/probe_gpu_k.py [--system water_dimer|water27]
        [--sgx-level 2] [--slice 256] [--max-slices 0 (= all)]
Output: benchmarks/results/probe_gpu_k_<system>.json (+ stdout).
"""
import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _paths import add_repo_to_syspath, results_dir  # noqa: E402

add_repo_to_syspath()
import cupy as cp  # noqa: E402
from pyscf import dft, lib  # noqa: E402
from pyscf.sgx import sgx_jk  # noqa: E402

from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402

XC = 'wb97m-v'
DM_DIR = os.path.join(os.path.dirname(HERE), 'experimental', 'cosx')


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    sys.stdout.flush()


def import_gpu4pyscf_int3c1e():
    """Import gpu4pyscf's int3c1e without letting it replace CuPy's allocator
    (see gpu/xc.py Gpu4PySCFFunctional)."""
    allocator = cp.cuda.get_allocator()
    try:
        from gpu4pyscf.gto import int3c1e
        from gpu4pyscf.lib.cupy_helper import cart2sph
    finally:
        cp.cuda.set_allocator(allocator)
    return int3c1e, cart2sph


def load_dm(system, mol):
    path = os.path.join(DM_DIR, 'dm_%s_def2-svp.npy' % system)
    if os.path.exists(path):
        dm, src = np.load(path), path
    else:
        mf = dft.RKS(mol, xc=XC)
        mf.conv_tol = 1e-8
        mf.kernel()
        dm, src = mf.make_rdm1(), 'RKS(%s) conv_tol=1e-8' % XC
    return (dm + dm.T) * 0.5, src


def int3c_slice(int3c1e, cart2sph, mol, intopt, grid_slice, omega):
    """A[g, u, v] on the device for one grid slice (copy of get_int3c1e's
    inner loop, minus the host copy)."""
    nao = mol.nao
    p = grid_slice.shape[0]
    out = cp.zeros([p, nao, nao], order='C')
    stream = cp.cuda.get_current_stream()
    for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
        if len(log_q_ij) == 0:
            continue
        cpi, cpj = intopt.cp_idx[cp_ij_id], intopt.cp_jdx[cp_ij_id]
        li, lj = intopt.angular[cpi], intopt.angular[cpj]
        bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi + 1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj + 1]
        ni, nj = i1 - i0, j1 - j0
        ao_offsets = np.array([i0, j0], dtype=np.int32)
        strides = np.array([ni, ni * nj], dtype=np.int32)
        ang = cp.zeros([p, nj, ni], order='C')
        err = int3c1e.libgint.GINTfill_int3c1e(
            ctypes.cast(stream.ptr, ctypes.c_void_p), intopt.bpcache,
            ctypes.cast(grid_slice.data.ptr, ctypes.c_void_p),
            ctypes.c_void_p(None), ctypes.c_int(p),
            ctypes.cast(ang.data.ptr, ctypes.c_void_p),
            strides.ctypes.data_as(ctypes.c_void_p),
            ao_offsets.ctypes.data_as(ctypes.c_void_p),
            bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(1), ctypes.c_int(cp_ij_id), ctypes.c_double(omega))
        if err != 0:
            raise RuntimeError('GINTfill_int3c1e failed')
        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi + 1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj + 1]
        if not mol.cart:
            ang = cart2sph(ang, axis=1, ang=lj)
            ang = cart2sph(ang, axis=2, ang=li)
        out[:, j0:j1, i0:i1] = ang
    row, col = np.tril_indices(nao)
    out[:, row, col] = out[:, col, row]
    return intopt.unsort_orbitals(out, axis=[1, 2])


def gpu_k(int3c1e, cart2sph, mol, sgx, dm, omega, slice_size, max_slices):
    """K on the device, FP64; returns (K_host or None, timing dict)."""
    grids = sgx.grids
    sgxdat = sgx._pjs_data
    proj = sgxdat._overlap_correction_matrix
    proj_dm = cp.asarray(proj @ dm)
    intopt = int3c1e.VHFOpt(mol)
    intopt.build(1e-13, aosym=True)
    coords = cp.asarray(grids.coords, order='C')
    weights = cp.asarray(grids.weights)
    ngrids = coords.shape[0]
    nslices = -(-ngrids // slice_size)
    todo = nslices if max_slices <= 0 else min(max_slices, nslices)
    K = cp.zeros((mol.nao, mol.nao))
    t_int = t_dot = 0.0
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for s in range(todo):
        p0, p1 = s * slice_size, min((s + 1) * slice_size, ngrids)
        ta = time.perf_counter()
        A = int3c_slice(int3c1e, cart2sph, mol, intopt, coords[p0:p1], omega)
        cp.cuda.Device().synchronize()
        tb = time.perf_counter()
        X = cp.asarray(mol.eval_gto('GTOval', grids.coords[p0:p1]))
        F = X @ proj_dm
        G = weights[p0:p1, None] * cp.einsum('gtv,gt->gv', A, F)
        K += X.T @ G
        del A
        cp.cuda.Device().synchronize()
        t_int += tb - ta
        t_dot += time.perf_counter() - tb
    wall = time.perf_counter() - t0
    if sgxdat.sym_ovlp:
        K = cp.asarray(proj).T @ K
    K = (K + K.T) * 0.5
    timing = {'slices_done': todo, 'slices_total': nslices, 'wall': wall,
              'integrals': t_int, 'contraction': t_dot,
              'full_build_est_s': wall * nslices / todo,
              'peak_pool_gb': cp.get_default_memory_pool().total_bytes() / 2**30}
    return (cp.asnumpy(K) if todo == nslices else None), timing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--system', default='water_dimer')
    ap.add_argument('--sgx-level', type=int, default=2)
    ap.add_argument('--slice', type=int, default=256)
    ap.add_argument('--max-slices', type=int, default=0)
    args = ap.parse_args(argv)

    int3c1e, cart2sph = import_gpu4pyscf_int3c1e()
    mol = build_mol(args.system, 'def2-svp')
    dm, dm_src = load_dm(args.system, mol)
    mf = dft.RKS(mol, xc=XC).COSX(pjs=True)
    sgx = mf.with_df
    sgx.grids_level_i = sgx.grids_level_f = args.sgx_level
    sgx.build(level=args.sgx_level)
    omega = mf._numint.rsh_and_hybrid_coeff(XC)[0]
    out = {'system': args.system, 'nao': int(mol.nao), 'sgx_level': args.sgx_level,
           'ngrids_sgx': int(sgx.grids.weights.size), 'dm_source': dm_src,
           'slice': args.slice, 'omega_lr': float(omega),
           'device': str(cp.cuda.runtime.getDeviceProperties(0)['name'])}
    log('%s nao=%d SGX level %d: %d grid points, omega_lr=%.2f'
        % (args.system, mol.nao, args.sgx_level, out['ngrids_sgx'], omega))

    for label, om in (('full', 0.0), ('lr', float(omega))):
        with mol.with_range_coulomb(om):
            t0 = time.perf_counter()
            k_cpu = sgx_jk.get_k_only(sgx, dm, hermi=1)
            t_cpu = time.perf_counter() - t0
        log('%-4s CPU get_k_only: %.1fs' % (label, t_cpu))
        k_gpu, timing = gpu_k(int3c1e, cart2sph, mol, sgx, dm, om,
                              args.slice, args.max_slices)
        rec = {'cpu_s': t_cpu, 'gpu': timing}
        if k_gpu is not None:
            diff = np.abs(k_gpu - k_cpu)
            rec.update(max_abs_diff=float(diff.max()),
                       rel_fro=float(np.linalg.norm(k_gpu - k_cpu) / np.linalg.norm(k_cpu)),
                       dE_exchange=float(-0.25 * np.einsum('ij,ji->', dm, k_gpu - k_cpu)))
        out[label] = rec
        log('%-4s GPU FP64: %d/%d slices %.1fs (integrals %.1fs, contraction %.1fs) '
            '-> full build est %.1fs vs CPU %.1fs; %s'
            % (label, timing['slices_done'], timing['slices_total'], timing['wall'],
               timing['integrals'], timing['contraction'], timing['full_build_est_s'], t_cpu,
               ('max|dK|=%.2e rel=%.2e dE=%.2e' % (rec['max_abs_diff'], rec['rel_fro'],
                                                   rec['dE_exchange'])
                if k_gpu is not None else 'partial run: no K comparison')))

    path = os.path.join(results_dir(), 'probe_gpu_k_%s.json' % args.system)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    log('-> %s' % path)


if __name__ == '__main__':
    main()
