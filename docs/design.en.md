# Mixed-Precision CPU+GPU Design for VV10 / XC on Consumer GPUs

> 2026-09-30. This document is the design basis of the mainline (staged SCF: FP32 on the GPU + FP64 tail):
> goals and hardware premises, the mixed-precision error budget, the design of each module and the acceptance gates. The criteria in this document are authoritative.
> (The early K-focused P1–P3 route failed all of its gates; that code is frozen.)

Languages / 语言：[中文（权威版，Chinese, authoritative）](design.md) | **English** | [日本語](design.ja.md)

---

## 0. Design rationale at a glance (why this route)

- **Why staged SCF (S0→S1→S2) instead of fine grids throughout?** In the early cycles \|ΔE\| is orders of magnitude larger than the error introduced by coarse grids and by skipping VV10, so the cheap settings early on do not change the solution the SCF converges to; switch once \|ΔE\| and \|g\| have come down to the same order as the grid error. Every switch changes the energy functional, which is why DIIS is reset and the Fock matrix rebuilt in full. This scheme is already validated in the Engine program (a production MD code).
- **Why FP32 on the GPU for the heavy work, with an FP64 tail on the CPU?** On consumer cards (RTX 30/40/50) FP64 throughput is 1/64 of FP32, so running everything in GPU FP64 loses more than it gains. The SCF energy is stationary with respect to the density matrix at convergence (δE = O(‖δD‖²)), so the FP32 stages only affect the convergence path, not the solution reached; the CPU FP64 K and semilocal XC tail then pins the energy at the FP64 fixed point (measured agreement with stock: about 1e-8 Ha).
- **Why not gpu4pyscf?** It runs everything in FP64 and targets A100-class cards; this project targets COSX K and VV10 on consumer cards, and all kernels are written in-house (CuPy RawKernel). gpu4pyscf is used only as a provider of libxc-cuda for the pointwise functional evaluation — a "light" computation — wrapped behind the `XcFunctional` interface, falling back to the CPU libxc when it is absent.
- **Why is error control the core of the design?** Random FP32 rounding cancels; the real danger is systematic bias, from two sources: rounded constants (every integral shifts the same way → constants are stored as FP32 hi+lo pairs and compensated with `fmaf`) and coherent rounding (on atom-centered grids all angular points of a radial shell see identical AO values, so their errors add up in the same direction and grow linearly with system size → in the K path X, both GEMMs and the grid weights stay in FP64 and only F is rounded, once).
- **Why PySCF-driven with removable hooks?** No fork, no site-packages edits, signatures unchanged; automatic fallback to the CPU reference path when there is no GPU or on error. Upstream upgrades and side-by-side validation stay easy.
- **Why is VV10 tile pruning safe?** The VV10 kernel decays as 1/R⁶ at long range, and a strict upper bound is built from the minimum inter-block distance and the extremes of W0 and κ; with `vv10_tile_tol = 0` it degenerates to the exact computation. Screening only removes FP32 work, it does not change the mathematics.
- **Why is the K-side strategy layer (experimental/cosx) frozen?** The per-cycle controller, SR screening and dynamic A/B all failed their gates: the benefit did not justify the complexity. The mainline — staging + GPU FP32 + FP64 tail — already meets the acceptance targets while staying simple.
- **Why are RI-JK and gradients out of scope?** GPU adaptive precision for RI-JK has already been published (Huang, Shao, Hammond 2026), and this project does not repeat it; gradients are blocked by an upstream PySCF 2.14 bug (NaN with the SGX grid response on), pending an upstream fix.

---

## 1. Background and motivation

- **The bottleneck of stock PySCF is VV10.** Measured on water27, def2-SVP, ωB97M-V, RIJCOSX: VV10 takes 65% of the SCF time, the semilocal XC part 9%, and the two K builds 24% together. The reason is that `nlcgrids` defaults to level 3 and `_vv10nlc` is a dense O(N_g²) computation.
- **A production MD program ("Engine" below) already cut VV10 down to 5%** (remdesivir, 1755 basis functions, 8 cores), with two measures: staging the VV10 switch-on and using coarse grids. In Engine, the remaining bottleneck is then the COSX K, at 64%.
- **The performance gap of consumer GPUs.** gpu4pyscf (JPCA 129, 1459; arXiv:2407.09700) is FP64 throughout and was benchmarked on A100 only. On the RTX 30/40/50 series FP64 is 1/64 of FP32. The work of Huang, Shao and Hammond (JPCA 130, 3483, 2026) shows that adaptive precision plus INT8 on RI-JK can be 204% faster on an RTX 4090 with no loss in the converged energy. RI-JK is done — **this project does not repeat it**.

## 2. Goals and non-goals

**Goal**: run ωB97M-V with RIJCOSX and cut the SCF latency of **a single large system**; a CPU+GPU heterogeneous scheme on consumer GPUs, with the heavy work in FP32 on the GPU and what needs FP64 on the CPU.

**Hardware premises**:
- Development on an RTX 2080 Ti (sm_75, no TF32).
- Production on RTX 3090/4090/5090 (sm_80 and above).
- **The CPU side is designed and accepted for 16 cores**, the common configuration of GPU workstations.
- There is no "FP64 GPU": consumer-card FP64 throughput is far too low, so **a GPU FP64 kernel cannot serve as a transitional plan**.

**Acceptance**: the SCF energy only. The final result, compared against a fine-grid, all-FP64 reference, must differ by < 1e-4 Ha.

**Non-goals**:
- RI-JK: already done by Huang 2026.
- Nuclear gradients (forces): later.
- Batch throughput: the target here is the latency of a single system.
- Modifying site-packages: PySCF is pinned to 2.14.0; only removable hooks are used.
- No git operations (project rule).

## 3. Overall architecture (plan 1')

- **The SCF is still driven by PySCF on the CPU.** The GPU takes over exactly two entry points: `ni.nr_rks` (XC) and `ni.nr_nlc_vxc` (VV10). Signatures and return values match PySCF's.
- **Precision is decided by the cost of the computation**:
  - **Heavy work** — the O(N_g·nao²) GEMMs and the O(N_g²) pairwise sum of VV10 — runs on the GPU in **FP32**.
  - **Light work** — O(N_g) pointwise operations and O(nao²) accumulations — **may run in FP64**, and preferably on the GPU, to keep the load off the 16-core CPU.
- **Kept on the CPU (FP64)**: the SCF main loop, the COSX K (PySCF SGX plus `sgx_patch`), RI-J, diagonalization, DIIS, orthogonalization.
- **K will move later**: on a 16-core machine K becomes the next bottleneck, so K sits behind the `KBuilder` interface, to be replaced by S5 (the GPU FP32 COSX).

## 4. Subprojects

| ID | Content | Depends on |
|---|---|---|
| S1 | Code restructuring (§5): VV10 as the mainline, K side frozen | — |
| S2 | Staging (§6), usable on CPU alone: coarse VV10 grid, aligned switch points, DIIS reset on switch | S1 |
| S3 | GPU mixed-precision VV10 and XC (§6, §7) | S1; may run in parallel with S2 |
| S3b | Separable + low-rank VV10: 2D (q, κ) interpolation of the original kernel, then a low-rank compression with shared factors, r channels, cost about r·N log N (§7.5) | S3; two spikes first |
| S5 | GPU FP32 COSX ESP integrals (later) | S3 |
| (S4) | Forces: out of scope for now. Engine has already derived the analytic form of the VV10 grid forces (including grid-point and partition-weight derivatives), a reference for a future S4 | — |

Each subproject gets its own implementation plan.

## 5. Code structure (S1)

```text
pyscf_wb97mv_fast/
├── __init__.py            entry: fast_path(mf, ...)
├── core/                  sgx_patch.py, hooks.py (former _hooks, adds DIIS reset), profiling.py, testsystems.py
├── staging/schedule.py    S2: per-stage grids, switch points, DIIS reset, full Fock rebuild
├── gpu/                   S3: device.py, precision.py, ao_eval.py, vv10.py, xc.py, backends.py, install.py
├── reference/             CPU FP64 reference implementations, thin wrappers around PySCF numint
└── experimental/cosx/     frozen zone: tolerance, controller, sr_screening, dynamic_ab, fastgrad,
                           acc/jax_vv10, acc/hardware_profile
benchmarks/vv10/  benchmarks/experimental/cosx/ (former sgx_locality and K-related benches)
tests/core/ tests/staging/ tests/gpu/ tests/experimental/ (the last directory is not run by default)
```

- The frozen zone stays in the tree, is not imported by the mainline, and its tests are skipped by default.
- Files are only ever `mv`-moved; all import paths and `sys.path` handling updated in the same change.

## 6. Precision policy and stages (S2, S3)

**Stages** (every switch does three things: change the grids, reset DIIS, rebuild the Fock matrix in full; **within a stage the grids, precision and tolerances never change**):

| Stage | Entry condition (tentative, tuned by experiment) | XC grid (`grids.level`) | VV10 (`nlcgrids.level`) | COSX (SGX grid level) |
|---|---|---|---|---|
| S0 | from the initial guess | 1 | off | 1 |
| S1 | \|g\| < τ₁ = 1e-3 | 3 | on, 1 | 1 |
| S2 | \|ΔE\| between consecutive cycles < 1e-6 Ha | 3 | 3 | 2 |

- **Reference** ("fine grids, all FP64 throughout"): stock PySCF with XC at level 3, VV10 at level 3, SGX at level 2 — the PySCF 2.14 defaults — plus `sgx_patch`, with the same conv_tol as the scheme under test.
- So S2 already runs on the reference grids, and the staging error can only come from the different convergence path left by S0 and S1.
- **Two VV10 modes** (`vv10_mode`). Engine measured both as equivalent:
  - `'late_scf'` (default): VV10 is computed self-consistently in S1 and S2 per the table.
  - `'nonscf'`: no VV10 during the SCF at all; after convergence one VV10 energy is evaluated on the final density — VV10 used as a dispersion correction. This is the cheapest option; since acceptance only looks at energies, it is an official option. Acceptance compares its \|ΔE\| against `'late_scf'`.
- **Two S2 modes** (`s2_mode`):
  - `'converge'` (default): converge to conv_tol within S2. The error is of the same order as the convergence threshold, far below 1e-4 Ha.
  - `'steps'`: run only `s2_steps` steps in S2 (default 1), Engine-style. The error is allowed to reach 1e-4 Ha, in exchange for fewer fine-grid cycles.
  - Both modes are measured at acceptance; report error and wall time for each.

Here \|g\| is PySCF's `envs['norm_gorb']`, read via `track_residual`. DIIS is reset by grabbing `envs['mf_diis']` inside `mf.callback` and clearing its `_buffer`, `_bookkeep`, `_head`, `_H` and `_xprev`.

**Precision per operator**:

| Operator | Where | Precision |
|---|---|---|
| AO evaluation (φ, ∇φ) | GPU | FP32 |
| density_gemm (ρ, ∇ρ, τ) | GPU | FP32 SGEMM |
| XC functional evaluation (pointwise) | GPU | FP64, via gpu4pyscf's libxc-cuda (light work); may move to JAX FP32 later |
| fock_gemm (Vxc, Vnlc) | GPU | partial matrices per block in FP32, accumulated in FP64 on the GPU |
| VV10 kernel_sum, O(N_g²) | GPU | FP32 tiling with Kahan-compensated sums inside a tile; per-grid-point F, U, W accumulated in FP64 |
| Energy sums | GPU or CPU | FP64 |
| K, RI-J, diagonalization, DIIS | CPU | FP64 |
| TF32 | — | **off by default**; may be enabled explicitly only in the S0 GEMMs, and only on sm_80+. TF32 must never be treated as FP32 |

**Error budget**:
- Mixed-precision error: on the same grids, < 1e-6 Ha versus all-FP64.
- Staging/grid error: < 1e-4 Ha versus the fine-grid, all-FP64 reference.
- The last S2 step uses the GPU FP32 backend by default, provided it meets < 1e-6 Ha; otherwise it falls back to the CPU FP64 reference implementation.

## 7. GPU backend interface and data flow (S3)

**On entering each stage**: prepare on the CPU with NumPy the grids (coordinates stored FP32 relative to the molecular center, plus weights), the block list (AO sparsity pattern, bucketed by shape) and the FP32-packed basis parameters, then keep them resident in GPU memory.

**Each cycle**:
- Upload the DM (nao² in size).
- For each bucket, on the GPU: the XC flow (AO → ρ/∇ρ/τ → libxc → fock_gemm → Vxc) and the VV10 flow (on its own grid: AO → ρ/∇ρ → W0, κ → kernel_sum → F/U/W → fock_gemm → Vnlc).
- Download Vxc + Vnlc (nao²) and the energies.

**GPU memory**: streamed per bucket; no full N_g × nao array is ever stored; the block size follows the memory budget, by default 60% of free memory.

**Tile pruning of kernel_sum**: the VV10 kernel decays roughly as 1/R⁶ at long range (g ≈ R²·W0). For each pair of tiles (I, J), an upper bound on Σ q_i·q_j·M_ij is computed from the minimum distance between the tiles and the extremes of W0 and κ inside them; tile pairs below `vv10_tile_tol` are skipped. The bound is evaluated on the CPU with NumPy; the GPU only sees the surviving tile pairs. With `vv10_tile_tol = 0` nothing is pruned and the result equals the exact computation — the acceptance cross-check.

**Interface** (CuPy first, backend name `'cupy'`; `'jax'` later, arrays exchanged zero-copy via DLPack):

```python
class AoEvaluator:   eval(block) -> ao32                      # (ncomp, npts, nao_active)
class XcFunctional:  eval(rho) -> exc, vrho, vsigma, vtau
class Vv10Kernel:    kernel_sum(coords, q, W0, kappa) -> F, U, W
class KBuilder:      get_k(dm, omega)                         # CPU SGX for now; S5 replaces it with GPU
```

**Error handling**:
- No GPU or no CuPy: automatic fallback to the CPU reference path with a warning.
- Out of GPU memory (OOM): halve the block size and retry.
- kernel_sum uses tiled reductions, never atomicAdd, so results are reproducible run to run.

**Implementation language**: CuPy plus RawKernel (CUDA C) first. JAX joins later as a second backend, with three cautions: static shapes (solved by bucketing plus padding), `precision='highest'` (to keep XLA's default TF32 away), and `XLA_PYTHON_CLIENT_PREALLOCATE=false` (so it does not fight CuPy for memory). Zig remains optional and is considered only if profiling shows the scheduling logic is the bottleneck.

## 7.5 S3b: making VV10 separable with low-rank compression (O(N_g²) → about r·N log N)

**Motivation**: VV10 is 65% of stock PySCF time (§1 measurements). S3 with GPU FP32 only speeds up the constant; S3b attacks the complexity itself.

**Literature**:
- rVV10: Sabatini et al., PRB 87, 041108 (2013) — a rewritten kernel amenable to Román-Pérez–Soler interpolation.
- ωB97M-rV: Mardirossian et al., JPCL 8, 35 (2017), recommending b = 6.2; B97M-rV performs on par with or better than B97M-V.
- An RPS + FFT implementation of rVV10 with Gaussian orbitals: Lee et al., JCP 155, 164102 (2021), Q-Chem GPW, O(N_g log N_g).

**Approach** (keeps the **original VV10**; what is computed is strictly ωB97M-V):
1. **2D interpolation.** The original kernel is Φ = −3/2 · 1/[g·g′·(g + g′)] with g = κ(qR² + 1), q = ω₀/κ. Because (g + g′) mixes κ and κ′, interpolating in q alone is not enough; interpolate in (q, κ): θ_{ab}(r) = ρ(r) · p_a(q(r)) · p_b(κ(r)), giving M = M_q × M_κ fields.
2. **Low-rank compression with shared factors.** K_{(ab),(cd)}(R) ≈ Σ_{s=1}^{r} A_{(ab),s} · A_{(cd),s} · f_s(R), the factors A independent of R (a symmetric-constrained CP decomposition). Only r channels remain: ρ_s = Aᵀθ, E ≈ ½ Σ_s ⟨ρ_s, f_s ∗ ρ_s⟩. The potential side goes v_θ = A·(f_s ∗ ρ_s), then back to vrho and vsigma through the derivative chains of p_a and p_b.
3. **Convolutions**: one radial convolution per channel. Two candidates on a molecular grid, decided by spike B:
   - project onto a uniform grid and do an open-boundary FFT with padding;
   - exploit the R⁻⁶ decay of f_s(R) and evaluate directly on the Becke grid with a tree or multipole method.
4. **Use per stage** (together with `vv10_mode` of §6): S1 may use S3b (or an rVV10 approximation) as the cheap VV10; the last S2 step uses S3's exact FP32 pairwise sum, or S3b at the high-accuracy setting within budget. DIIS is reset on switch as usual.

**Two spikes before starting** (no SCF, no GPU, NumPy is enough):
- **Spike A (rank)**: tabulate the original kernel (and the rVV10 kernel as a control) on a (q, κ, q′, κ′, R) grid; run per-R SVD and shared-factor CP decomposition; report r(ε) and the required M_q × M_κ for ε = 1e-6, 1e-5, 1e-4 relative error.
- **Spike B (energy)**: with a converged water27 density, evaluate the separable + low-rank kernel as a **direct double sum** on the VV10 grid (no convolution acceleration yet) and compare against the dense exact E_nl. Criterion: interpolation and truncation error ≤ 1e-5 Ha, leaving headroom in the final 1e-4 Ha budget. Then compare the two convolution schemes for accuracy and speed.

**S3b acceptance**: cross-checked against S3's exact version, |ΔE_nl| ≤ 1e-5 Ha and full-SCF |ΔE| ≤ 1e-4 Ha; on the same grids the VV10 wall time must drop another order of magnitude below S3's exact GPU version.

## 8. Testing and acceptance

pytest markers: `gpu` (auto-skipped without a GPU), `slow`, `experimental` (not run by default).

**Layer 1: per-kernel comparison against CPU FP64** (on the 2080 Ti)

| Kernel | Reference | Criterion |
|---|---|---|
| AO evaluation | `mol.eval_gto` | relative error ≤ 1e-6 (normalized by max\|φ\|) |
| ρ, ∇ρ, τ | numint `eval_rho` | relative error ≤ 1e-6 |
| libxc-cuda | PySCF's libxc | ≤ 1e-12 |
| kernel_sum F, U, W | `_vv10nlc` | relative error ≤ 1e-6 |
| VV10 energy and Vnlc | `nr_nlc_vxc` | \|ΔE\| ≤ 1e-7 Ha |
| XC energy and Vxc | `nr_rks` (same grid) | \|ΔE\| ≤ 1e-7 Ha |

**Layer 2: full SCF** (water27, chain30)
- Fixed grids, mixed precision only: |ΔE| ≤ 1e-6 Ha, cycle count within 1.
- Staged: versus the fine-grid, all-FP64 reference, |ΔE| ≤ 1e-4 Ha, at most 2 extra cycles.
- Robustness: hooks remove cleanly; CPU fallback works without a GPU; OOM halves the block size and continues.

**Layer 3: performance** (on RTX 30/40/50 machines, `OMP_NUM_THREADS=16`; the 2080 Ti verifies functionality only)
- Baseline: stock PySCF on the same machine, 16 threads, same final grids.
- **G_gpu1**: on the same grids, the GPU VV10 + XC takes ≤ 0.1× the CPU time.
- **G_gpu2**: the staged full run is at least 2× faster than the baseline in total SCF time, with |ΔE| ≤ 1e-4 Ha.
- Each item records: per-component timings, cycle counts, peak GPU memory, PCIe traffic.

**Test systems**: water dimer for unit tests, water27 and chain30 for integration; real molecules to be selected later.

## 8.5 Question S5 (COSX) must answer before starting: does the long-range exchange need its own numerics

**Question**: K_LR uses the erf(ωr)/r kernel. Should it share one grid/screening/precision-threshold system with the full COSX, or have its own — even its own COSX grid?

**Existing data** (measurements on water27 and chain30; same attribution data as §1):
- **Screening has almost nothing left to save**: in stock, full and LR have nearly identical task counts (29.3M vs 28.5M); the oracle, at every error budget, keeps nearly identical counts too. The locality comes from the decay of F = P·X itself, independent of the integral kernel.
- **On the same grid LR is less accurate**: on SGX level 2, versus the analytic K, the grid error is 5.1e-7 Ha for full, 4.3e-6 Ha for LR, 4.8e-6 Ha for SR. The suspicion is that `fit_ovlp`'s error cancellation was designed around the full 1/r kernel; unverified.
- **LR really is more expensive**: 1.45–1.7× per integral, 1.3–1.5× per K build. Halving the grid points halves the LR overhead roughly proportionally.

**Decision experiment** (inside PySCF, cheap to run):
- SGX already keeps a separate copy `_rsh_df[key]` per ω, with its own `_pjs_data`; the gradient RSH branch reuses the same copy. It suffices to set the LR copy's grid level to 1 while full stays at 2.
- On water27, measure three things: LR's grid error (vs analytic K), LR's K-build time, and the full-SCF \|ΔE\|.
- **How to decide**:
  - If LR's error stays far below 1e-4 Ha and time drops to about 0.5×: an independent grid is worth it, and worth porting to Engine.
  - If LR's error grows clearly: what is needed is an LR-specific error correction (e.g. LR-specific overlap fitting), not a separate grid.
- **Convergence stability**: an LR grid held fixed through the whole SCF does not disturb DIIS; what must be avoided is changing it mid-SCF.

**Element sensitivity** (Engine experience: the COSX grid is sensitive to some elements and not others):
- **PySCF status quo**: SGX's opt grids vary only the radial point count with the period (`SGX_RAD_GRIDS[level, period]`), with angular points pruned by Bragg radius (`sgx_prune`). Elements of the same period share one grid — C, N, O, F are literally identical; there is no per-element sensitivity setting.
- **Ready-made interface**: `grids.atom_grid[symb] = (nrad, nang)` sets a grid per element.
- **Sensitivity experiment**:
  - Keep every other element's grid fixed; raise one element's grid by one level at a time and record the change in the K energy (vs analytic K, or vs the highest grid level).
  - Do it for full and LR separately, since their element sensitivities may differ.
  - The result is an "element × grid level → error contribution" table.
- **Use**: set grids per element — finer where sensitive, one level coarser where not — minimizing the grid-point count at a fixed total error budget. The same idea applies to the XC and VV10 grids.

## 9. What was carried over from the original design document

| Source (original doc) | Practice | Where in this spec |
|---|---|---|
| §2.1 | Care only about wall time; do not chase "everything on the GPU" | §3 |
| §2.2 | Precision is a property of an operator; never treat TF32 as FP32 | §6 |
| §2.3 | The grids of the operators are independent | §6, VV10 gets its own coarse grid |
| §6.1 C | Stage VV10, finish exactly | §6, plus DIIS reset on switch |
| §8 | Keep the irregular and the regular parts apart | §7: CPU schedules with NumPy, GPU does the regular dense work |
| §9 | Bucket by shape; blocks too small stay off the GPU | §7 |
| §10 | HardwareProfile microbench at startup | S3 extension: it decides the TF32 switch and the CPU/GPU placement per operator |
| §11 | XC in direct streaming mode | §7 |
| §17 | Correctness metrics and the fast / production / reference modes | §8 |

**Not adopted**: SR-COSX and the B route, the per-cycle controller (all failed their gates); PS-J and FMM.

**The VV10 low-rank approximation (original doc §7) becomes S3b** and is no longer listed as rejected; the design is §7.5.

## 10. Risks

| Risk | Mitigation |
|---|---|
| FP32 AO or ρ errors amplified by the nonlinear XC | quantify item by item in layer 1; S2 can fall back to CPU FP64 |
| Bad switch points inflate the cycle count | DIIS reset on switch; τ₁ tuned by experiment; cycle count is part of the acceptance criteria |
| libxc-cuda (from gpu4pyscf) interface instability | wrap it behind `XcFunctional`; PySCF's libxc always remains the reference |
| K becomes the bottleneck on a 16-core machine and G_gpu2 is missed | G_gpu2 is set at 2× to leave room; K is deferred to S5 |
| Many block shapes inflate kernel launch overhead | bucket by shape; blocks too small stay on the CPU |
