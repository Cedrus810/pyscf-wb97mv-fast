"""Shared pieces of the density-matrix / gradient reference check.

ref_dm_grad.py computes the FP64 CPU reference once per system and stores it
in an .npz file; check_dm_grad.py runs only the GPU-staged SCF on any card
and compares against that file.  Both evaluate the observables with the SAME
code on a fresh stock PySCF RKS/COSX(pjs=True) object that gets the
converged orbitals copied in, so a difference can only come from the
density matrix / orbitals themselves.

Gradient: stock pyscf.sgx.grad with sgx_grid_response=False.  PySCF 2.14
returns NaN with the SGX grid response on (upstream bug, recorded
2026-09-30 in experimental/cosx/fastgrad.py); both sides use the same
setting, so the comparison is unaffected.
"""
import socket
import time

import numpy as np
from pyscf import dft, lib

T0 = time.time()


def log(msg):
    print('[%s +%7.1fs] %s' % (time.strftime('%F %T'), time.time() - T0, msg), flush=True)


def stock_mf(mol, xc='wb97m-v'):
    return dft.RKS(mol, xc=xc).COSX(pjs=True)


def observables(mol, mo_coeff, mo_occ, mo_energy, e_tot, xc='wb97m-v'):
    """Gradient, dipole and Mulliken charges from the given orbitals, all on
    a fresh stock mf (no hooks, no sgx_patch)."""
    mf = stock_mf(mol, xc)
    mf.build()                  # SGX SCF state (_nsteps_direct, level_i grid)
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = mo_coeff, mo_occ, mo_energy
    mf.e_tot, mf.converged = e_tot, True
    # the SGX grid a converged stock SCF ends on (grids_level_i -> _f switch);
    # one get_veff at this density initializes XC grids, SGX and the PJS data
    # the way the SCF does
    mf.with_df.build(level=mf.with_df.grids_level_f)
    dm = mf.make_rdm1()
    mf.get_veff(mol, dm)
    t = time.time()
    g = mf.nuc_grad_method()
    g.sgx_grid_response = False
    g.verbose = 0
    grad = g.kernel()
    t_grad = time.time() - t
    dip = mf.dip_moment(mol, dm, unit='Debye', verbose=0)
    chg = mf.mulliken_pop(mol, dm, verbose=0)[1]
    return dict(dm=dm, grad=np.asarray(grad), dipole=np.asarray(dip),
                charges=np.asarray(chg), t_grad=t_grad,
                grid_response=bool(getattr(g, 'grid_response', False)))


def host_info():
    return dict(host=socket.gethostname().split('.')[0], threads=lib.num_threads())
