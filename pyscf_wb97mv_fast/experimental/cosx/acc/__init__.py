"""FROZEN (2026-09-30): gates failed, see benchmarks/experimental/cosx/FINDINGS.md section 9; not imported by mainline.

P6: accelerator backends (JAX first; CUDA later if at all).

Per the plan, accelerators come last and only target dense
workloads that remain dominant after P1-P3. Zig was dropped (no evidence that
scheduling/bucketing is a bottleneck; see the plan's 'deferred' section).

Modules:
- hardware_profile: startup microbenchmark (FP64/FP32 GEMM, skinny GEMM) to
  decide what belongs on a device.
- jax_vv10: low-rank VV10 nonlocal correlation on JAX, with a numpy reference
  implementation so the math is testable without jax installed.
"""
