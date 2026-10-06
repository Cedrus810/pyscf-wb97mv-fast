"""CPU FP64 reference for the XC and VV10 terms, straight from PySCF's NumInt.

These are the yardsticks for the S3 GPU backends. They always call the
NumInt *class* methods, so hooks installed on the mf._numint instance (for
example a GPU backend) are bypassed. They never mutate mf.

Grids: an explicitly passed grid is used as is. Otherwise mf.grids / mf.nlcgrids
are used; if they are not built yet, a copy is prepared exactly like
dft.rks.KohnShamDFT.initialize_grids does for get_veff -- build, then prune
points with rho < mf.small_rho_cutoff when dm is a single (ground-state)
density matrix -- so reference and SCF see the same points.
"""
import copy


def prepared_grids(mf, dm, kind='xc', grids=None):
    """Grids the reference will integrate on. kind: 'xc' or 'nlc'."""
    if grids is not None:
        return grids
    if kind not in ('xc', 'nlc'):
        raise ValueError("kind must be 'xc' or 'nlc'")
    g = mf.grids if kind == 'xc' else mf.nlcgrids
    if g.coords is not None:
        return g
    g = copy.copy(g)
    g.build(with_non0tab=True)
    if mf.small_rho_cutoff > 1e-20 and getattr(dm, 'ndim', 0) == 2:
        from pyscf.dft.rks import prune_small_rho_grids_
        g = prune_small_rho_grids_(mf, mf.mol, dm, g)
    return g


def xc_reference(mf, dm, grids=None):
    """(nelec, exc, vxc) of the semilocal XC part, as in dft.rks.get_veff."""
    ni = mf._numint
    g = prepared_grids(mf, dm, 'xc', grids)
    return type(ni).nr_rks(ni, mf.mol, g, mf.xc, dm, max_memory=mf.max_memory)


def vv10_reference(mf, dm, grids=None):
    """(nelec, enlc, vnlc) of the nonlocal (VV10) part, as in dft.rks.get_veff.

    The functional string follows get_veff: mf.xc when it carries the NLC
    term itself (e.g. 'wb97m-v'), otherwise mf.nlc.
    """
    ni = mf._numint
    if ni.libxc.is_nlc(mf.xc):
        xc = mf.xc
    elif mf.nlc and ni.libxc.is_nlc(mf.nlc):
        xc = mf.nlc
    else:
        raise ValueError(f'{mf.xc!r} has no nonlocal correlation (VV10) term')
    g = prepared_grids(mf, dm, 'nlc', grids)
    return type(ni).nr_nlc_vxc(ni, mf.mol, g, xc, dm, max_memory=mf.max_memory)
