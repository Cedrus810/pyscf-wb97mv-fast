"""Staged SCF with the precision switch, on a real system (spec sections 6, 8).

Default (the design point): S0-S2 run XC + VV10 through the GPU FP32 flows
(install_gpu, CPU K overlapped), and once |dE| < 1e-6 inside S2 the last
cycles switch to the stock CPU FP64 numint -- StagedSCF(gpu_stages=(0, 1, 2),
fp64_final=True).  --gpu-stages 0 1 without --fp64-final gives the earlier
variant (all S2 cycles on the CPU).
Prints one line per SCF cycle (stage, path, |dE|, |g|, wall time of the
cycle) so the cycle count and the per-cycle cost can be read directly.  The
path is GPU (FP32 hooks on), FP64 (the CPU FP64 tail; VV10 stays FP32 on the
GPU unless fp64_final_vv10='cpu') or CPU (no GPU stage at all).

Reference energies are PASSED IN, never recomputed (known values, e.g. water27
stock CPU FP64: --e-ref -2061.0026652808 from benchmarks/results/
scf_gate_water27_def2-svp.json):

    --e-ref         stock PySCF, all-fine grids, CPU FP64 (spec staged gate 1e-4)
    --e-cpu-staged  the same stages on the CPU only (spec precision gate 1e-6);
                    if absent and --cpu-staged is given, it is computed once
                    and printed so it can be passed in next time

Usage (GPU idle, nothing else running):
    $PY -B benchmarks/xc/staged_gpu.py --system water27 --e-ref -2061.0026652808
Output: --out, default benchmarks/results/
staged_gpu_<system>_<basis>[_gpuk][_ftol<x>]_<host>_omp<N>.json (+ stdout); the
suffixes keep runs with different settings or machines from overwriting each
other.  Exit status 3 when a gate fails (not converged, |dE vs --e-ref| >= 1e-4,
or |dE vs CPU-staged| >= 1e-6 when that energy is known), so a batch can tell.
"""
import argparse
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _paths import add_repo_to_syspath, results_dir  # noqa: E402

add_repo_to_syspath()
from pyscf import lib  # noqa: E402

from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402
from pyscf_wb97mv_fast.staging.schedule import run_staged  # noqa: E402


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    sys.stdout.flush()


def print_history(info, t0):
    t_prev = t0
    for h in info['history']:
        split = ('  [jk %.1fs | gpu-xc %.1fs | wait %.1fs]'
                 % (h['t_jk'], h['t_xc'] or float('nan'), h['t_wait'])
                 if 't_jk' in h else '')
        path = 'FP64' if h.get('fp64') else ('GPU' if h['gpu'] else 'CPU')
        log('  cycle %3d  %s  %-4s  E=%.10f  |dE|=%.2e  |g|=%.2e  dt=%6.1fs%s'
            % (h['cycle'] + 1, 'S%d' % h['stage'], path,
               h['e_tot'], h['delta_e'], h['norm_gorb'], h['t'] - t_prev, split))
        t_prev = h['t']


def staged(mol, conv_tol, gpu_stages, fp64_final=False, gpu_kwargs=None,
           fp64_conv_tol=None):
    t0 = time.perf_counter()
    extra = {} if fp64_conv_tol is None else {'fp64_conv_tol': fp64_conv_tol}
    info = run_staged(mol, conv_tol=conv_tol, gpu_stages=gpu_stages,
                      fp64_final=fp64_final, gpu_kwargs=gpu_kwargs or {}, **extra)
    info.pop('dm', None)
    print_history(info, t0)
    return info


def summary(info):
    return {'e_tot': info['e_tot'], 'converged': info['converged'],
            'cycles': info['cycles'], 'kernel_wall': info['kernel_wall'],
            'stages': info['stages'],
            'history': [{k: (float(v) if isinstance(v, float) else v)
                         for k, v in h.items()} for h in info['history']]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--system', default='water27')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--out', default=None, help='JSON path (default: see module doc)')
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    ap.add_argument('--e-ref', type=float, required=True,
                    help='known stock CPU FP64 reference energy (not recomputed)')
    ap.add_argument('--e-cpu-staged', type=float, default=None)
    ap.add_argument('--cpu-staged', action='store_true',
                    help='compute the CPU-only staged energy if not given')
    ap.add_argument('--gpu-stages', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--no-fp64-final', dest='fp64_final', action='store_false')
    ap.add_argument('--gpu-k', action='store_true',
                    help='the FP32 stages also build the COSX exchange K on '
                         'the GPU (S5); the FP64 tail always reverts to the '
                         'CPU K (schedule forces k=False there)')
    ap.add_argument('--k-tile-tol', type=float, default=1e-11)
    ap.add_argument('--fp64-conv-tol', type=float, default=None,
                    help='|dE| threshold of the FP64 tail only (|g| keeps '
                         'sqrt(conv-tol)); default: conv-tol')
    args = ap.parse_args(argv)

    mol = build_mol(args.system, args.basis)
    host, nthreads = socket.gethostname().split('.')[0], lib.num_threads()
    log('%s/%s nao=%d natm=%d; e_ref=%.10f (given); host=%s threads=%d'
        % (args.system, args.basis, mol.nao, mol.natm, args.e_ref, host, nthreads))
    out = {'system': args.system, 'basis': args.basis, 'conv_tol': args.conv_tol,
           'e_ref': args.e_ref, 'host': host, 'threads': nthreads}

    gpu_stages = tuple(args.gpu_stages)
    fp64_final = args.fp64_final and 2 in gpu_stages
    gpu_kwargs = {}
    if args.gpu_k:
        gpu_kwargs = {'k': True, 'k_tile_tol': args.k_tile_tol}
    fp64_conv_tol = args.fp64_conv_tol if fp64_final else None
    log('GPU-staged run, gpu_stages=%s fp64_final=%s fp64_conv_tol=%s gpu_kwargs=%s'
        % (gpu_stages, fp64_final, fp64_conv_tol, gpu_kwargs))
    gpu = staged(mol, args.conv_tol, gpu_stages, fp64_final, gpu_kwargs, fp64_conv_tol)
    out.update(gpu_stages=list(gpu_stages), fp64_final=fp64_final,
               gpu_k=bool(args.gpu_k), k_tile_tol=args.k_tile_tol,
               fp64_conv_tol=fp64_conv_tol)
    out['gpu_staged'] = summary(gpu)
    out['dE_vs_ref'] = gpu['e_tot'] - args.e_ref
    log('GPU-staged e=%.10f conv=%s cycles=%d wall=%.0fs  dE_vs_ref=% .3e '
        '(vs stock CPU FP64 --e-ref; staged gate 1e-4)'
        % (gpu['e_tot'], gpu['converged'], gpu['cycles'], gpu['kernel_wall'], out['dE_vs_ref']))

    e_cpu = args.e_cpu_staged
    if e_cpu is None and args.cpu_staged:
        log('CPU-staged run (no --e-cpu-staged given)')
        cpu = staged(mol, args.conv_tol, ())
        out['cpu_staged'] = summary(cpu)
        e_cpu = cpu['e_tot']
        log('CPU-staged e=%.10f conv=%s cycles=%d wall=%.0fs  (pass --e-cpu-staged %.10f next time)'
            % (cpu['e_tot'], cpu['converged'], cpu['cycles'], cpu['kernel_wall'], cpu['e_tot']))
    if e_cpu is not None:
        out['e_cpu_staged'] = e_cpu
        out['dE_vs_cpu_staged'] = gpu['e_tot'] - e_cpu
        log('GPU-staged vs CPU-staged dE=% .3e (precision gate 1e-6)'
            % out['dE_vs_cpu_staged'])

    gates = {'converged': bool(gpu['converged']),
             'staged_1e-4': abs(out['dE_vs_ref']) < 1e-4}
    if 'dE_vs_cpu_staged' in out:
        gates['precision_1e-6'] = abs(out['dE_vs_cpu_staged']) < 1e-6
    out['gates'] = gates
    log('gates: %s -> %s' % (gates, 'PASS' if all(gates.values()) else 'FAIL'))

    path = args.out
    if path is None:
        tag = '%s_%s' % (os.path.splitext(os.path.basename(args.system))[0], args.basis)
        tag += '_gpuk' if args.gpu_k else ''
        if fp64_conv_tol is not None:
            tag += '_ftol%g' % fp64_conv_tol
        tag += '_%s_omp%d' % (host, nthreads)
        path = os.path.join(results_dir(), 'staged_gpu_%s.json' % tag)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    log('-> %s' % path)
    return 0 if all(gates.values()) else 3


if __name__ == '__main__':
    sys.exit(main())
