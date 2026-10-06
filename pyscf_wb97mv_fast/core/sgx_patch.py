"""
Local fix for the range-separated (RSH) branch of pyscf.sgx.sgx.SGX.get_jk
(PySCF 2.14).

Upstream problems fixed here:

1. The RSH branch calls ``rsh_df.get_jk(dm, hermi, with_j, with_k,
   direct_scf_tol)`` positionally. The ``vhfopt`` argument is missing, so the
   arguments shift by one: ``vhfopt=False, with_j=True, with_k=direct_scf_tol``,
   and ``direct_scf_tol`` falls back to its default. As a result:
   - with pjs=True, every LR/SR K build also runs an attenuated RI-J that is
     thrown away (about 10% of the LR K time on a 730-AO chain);
   - with pjs=False, a numerical J is computed as well;
   - the caller's direct_scf_tol never reaches the LR/SR K build.
2. ``rsh_df = self.copy()`` is a shallow copy. If the full-range K has already
   been computed, the copy shares the ``_pjs_data`` screening bounds that were
   built for the full 1/r kernel, so the LR/SR K build is screened with the
   wrong bounds. For SR this wastes the erfc decay (730-AO chain: K_SR 42 s
   with the shared bounds vs 31 s with attenuated bounds). Resetting
   ``_pjs_data`` makes the bounds get rebuilt under ``with_range_coulomb(omega)``.

Usage:
    from pyscf_wb97mv_fast.core import sgx_patch
    sgx_patch.apply()
"""
from pyscf import __config__
from pyscf.lib import logger
from pyscf.sgx import sgx as _sgx

_orig_get_jk = _sgx.SGX.get_jk


def get_jk(self, dm, hermi=1, vhfopt=None, with_j=True, with_k=True,
           direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
           omega=None):
    if omega is None:
        return _orig_get_jk(self, dm, hermi, vhfopt, with_j, with_k, direct_scf_tol)

    key = '%.6f' % omega
    rsh_df = self._rsh_df.get(key)
    if rsh_df is None:
        rsh_df = self.copy()
        rsh_df._rsh_df = None  # to avoid circular reference
        rsh_df._vjopt = None
        rsh_df._overlap_correction_matrix = None
        # Do not share the full-kernel screening bounds (problem 2).
        rsh_df._pjs_data = None
        self._rsh_df[key] = rsh_df
        logger.info(self, 'Create RSH-SGX object %s for omega=%s', rsh_df, omega)

    with rsh_df.mol.with_range_coulomb(omega):
        return _orig_get_jk(rsh_df, dm, hermi, vhfopt, with_j=with_j,
                            with_k=with_k, direct_scf_tol=direct_scf_tol)


def apply():
    """Install the fix. Calling it more than once is safe."""
    _sgx.SGX.get_jk = get_jk


def revert():
    _sgx.SGX.get_jk = _orig_get_jk
