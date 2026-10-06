"""FROZEN (2026-09-30): gates failed, see benchmarks/experimental/cosx/FINDINGS.md section 9; not imported by mainline.

Set SGX DM-screening tolerances so that they actually take effect.

PySCF caches screening data in SGX._pjs_data and only rebuilds it when
direct_scf_tol (_itol) changes, and RSH copies in SGX._rsh_df keep the
tolerances they were created with. Changing sgx_tol_energy on mf.with_df alone
therefore has no effect after the first K build.

set_sgx_tolerance     drops the whole cache (integral bounds, overlap fit, DM
                      screen); everything is rebuilt on the next K build.
update_sgx_tolerance  keeps the integral bounds and overlap fit, which do not
                      depend on etol/vtol, and rebuilds only the DM-screening
                      part in place. Meant for callers that change the budget
                      many times during one SCF (controller).
"""


def _sgx_objects(mf):
    sgx = mf.with_df
    return [sgx, *(sgx._rsh_df or {}).values()]


def set_sgx_tolerance(mf, etol='auto', vtol='auto'):
    for obj in _sgx_objects(mf):
        obj.sgx_tol_energy = etol
        obj.sgx_tol_potential = vtol
        obj._pjs_data = None
    return mf


def _resolve(etol, vtol, itol):
    # same rules as pyscf.sgx.sgx_jk.SGXData.__init__
    if etol == 'auto':
        etol = itol
    if vtol == 'auto':
        vtol = itol ** 0.5 if etol is None else etol ** 0.5
    return etol, vtol


def update_sgx_tolerance(mf, etol='auto', vtol='auto'):
    for obj in _sgx_objects(mf):
        obj.sgx_tol_energy = etol
        obj.sgx_tol_potential = vtol
        d = obj._pjs_data
        if d is None:
            continue  # built with the new tolerances on the next K build
        d._etol, d._vtol = _resolve(etol, vtol, d._itol)
        d._screen_energy = d._etol is not None
        d._screen_potential = d._vtol is not None
        if d.use_dm_screening:
            d._build_dm_screen()
        d._setup_opt()
    return mf
