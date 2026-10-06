"""pyscf-wb97mv-fast: VV10/XC mixed-precision fast path for wB97M-V (RIJCOSX) in PySCF.

Mainline (design: docs/design.md):
    core       SGX bug fix (sgx_patch), reversible hooks and DIIS reset,
               per-component SCF profiler, standard test systems
    reference  CPU FP64 reference wrappers around PySCF numint (S1)
    staging    Engine-style SCF staging: coarse grids, VV10 switch-on, DIIS reset (S2)
    gpu        FP32-on-GPU VV10/XC backends (S3)

Frozen, not imported by the mainline:
    experimental.cosx   K-side strategy layer (tolerance controller, SR screening,
                        dynamic A/B, staged VV10 without DIIS reset, gradients guard,
                        accelerator prototypes). All gates failed; see
                        benchmarks/experimental/cosx/FINDINGS.md section 9.

Importing this package does not import PySCF.
"""
__version__ = '0.0.1'
