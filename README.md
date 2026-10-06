**English** | [中文](README.zh-CN.md) | [日本語](README.ja.md)

# pyscf-wb97mv-fast

A fast path for **ωB97M-V / RIJCOSX (COSX, pjs=True)** in PySCF. The heavy work runs in FP32 on consumer GPUs (RTX 30/40/50) and the final FP64 cycles run on the CPU, cutting the SCF wall time for **a single large system**.

**Status: early alpha (0.0.1, 2026-10-06).** Energies are validated against stock PySCF on water27 (def2-SVP) and six drug molecules (def2-TZVP). The density-matrix and gradient comparison is in progress (see "Validation").

---

## Scope

| Item | Supported |
|---|---|
| Method | RKS ωB97M-V + COSX (`dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)`) |
| PySCF | pinned to 2.14.0; only removable hooks, site-packages untouched |
| GPU | consumer RTX 30/40/50 (FP64 at 1/64 rate); CuPy compiles the kernels for each card at run time. The 2080 Ti is a development card only. For A100/H100-class cards use gpu4pyscf directly |
| Basis | spherical, angular momentum up to f (def2-SVP, def2-TZVP); g functions and Cartesian bases are not supported (warning, falls back to the CPU) |
| Elements | validated: H, C, N, O, F, S, Cl, Br; elements with ECPs are untested and out of scope for this version |
| No GPU | falls back to the stock CPU path with a warning |

## Theory

### The terms of ωB97M-V

ωB97M-V combines a semilocal meta-GGA exchange-correlation, range-separated exact exchange and VV10 nonlocal correlation. The exact exchange splits the Coulomb operator at ω = 0.3 bohr⁻¹:

$$\frac{1}{r} = \frac{\mathrm{erfc}(\omega r)}{r} + \frac{\mathrm{erf}(\omega r)}{r}$$

with 15% short-range and 100% long-range exact exchange. Since erfc = 1 − erf, this equals

$$E_x^{\mathrm{HF}} = -\tfrac14\,\mathrm{Tr}\!\left[D\left(0.15\,K^{1/r}[D] + 0.85\,K^{\mathrm{erf}}[D]\right)\right]$$

so every SCF cycle builds two K matrices: the full K (1/r) and the long-range K (erf(ωr)/r). Most of the cost of a cycle is in three places: the two K builds, the semilocal XC and VV10.

### COSX (SGX) exchange

COSX integrates one electron coordinate of the exchange integral numerically on a grid g:

$$K_{\mu\nu} \approx \sum_g w_g\, X_{g\mu} \sum_\lambda A^{g}_{\nu\lambda} F_{g\lambda},\qquad F = X\,Q\,D,\qquad A^{g}_{\nu\lambda} = \int \chi_\nu(\mathbf r)\,\chi_\lambda(\mathbf r)\, v(|\mathbf r-\mathbf r_g|)\,d\mathbf r$$

X holds the AO values on the grid, Q is the PJS overlap-fitting matrix and v is 1/r or erf(ωr)/r. On the GPU one fused kernel computes $G_{g\nu}=\sum_\lambda A^{g}_{\nu\lambda}F_{g\lambda}$ without ever storing A, and an FP64 GEMM gives $K = X^{\mathsf T}(w\circ G)$.

**Screening**: for each (grid sub-block, shell pair), an analytic upper bound on |A| is multiplied by the largest weighted F on that sub-block, and the task is skipped below `k_tile_tol` (1e-11). The bound is evaluated in FP32 and deliberately inflated so that it stays a strict upper bound.

### Three-center integrals (McMurchie–Davidson)

For a primitive pair with exponent p and center P, let θ = ω²/(ω² + p) (θ = 1 for the 1/r kernel) and T = θp|P − r_g|². Then

$$A = \frac{2\pi}{p}\sum_{tuv} E^{x}_{t} E^{y}_{u} E^{z}_{v}\, R_{tuv},\qquad R^{(n)}_{000} = \sqrt{\theta}\,(-2p\theta)^n F_n(T)$$

and $R_{tuv}$ follows from the recurrence $R^{(n)}_{t+1,u,v} = t\,R^{(n+1)}_{t-1,u,v} + X_{PC}\,R^{(n+1)}_{t,u,v}$ (likewise in y and z). f shells need the Boys functions $F_0$ to $F_6$. Each common (la, lb) combination has its own kernel specialized at compile time, with all arrays in registers; shell pairs with generally contracted shells use the generic kernel.

### VV10 nonlocal correlation

$$E_c^{\mathrm{nl}} = \int \rho(\mathbf r)\left[\beta + \tfrac12\int \rho(\mathbf r')\,\Phi(\mathbf r,\mathbf r')\,d\mathbf r'\right] d\mathbf r$$

This is a double sum over the grid, $O(N_g^2)$, and the largest single item in stock PySCF (about 50–65% on water27). The GPU evaluates it in tiles and prunes tile pairs by distance.

### Mixed precision and error control

Consumer cards run FP64 at 1/64 of the FP32 rate, so kernel inner loops are FP32 throughout, with no doubles and no divisions. Random rounding errors cancel; the real danger is **systematic bias**, which has two main sources:

1. **Rounded constants**: once a constant (for example the s-function normalization 1/√4π) is rounded to FP32, every integral shifts the same way. Constants are therefore stored as FP32 hi + lo pairs, and the lo part is added with `fmaf` before the final rounding so that the compensation is not rounded away.
2. **Coherent rounding**: on atom-centered grids, all angular points of a radial shell and all atoms of one element see identical AO values, so their rounding errors add up in the same direction and the bias grows linearly with system size. The K path therefore keeps X, both GEMMs and the grid weights in FP64 and rounds only F to FP32, once, for the kernel. The remaining floor is about 2e-6 Ha for the full K on water27 (see "Known issues").

Boys function: a 7-term Taylor table on nodes spaced 1/4 for T < 36 (truncation error about 1e-10) and the asymptotic form for T ≥ 36, evaluated with double-float products and scaled by 2¹⁰⁰ so that the error terms never become subnormal. Over the whole range T ≤ 1e30 every order is within 0.5 ulp and unbiased.

### Why an FP64 tail is enough

At SCF convergence the energy is stationary with respect to the density matrix, so a density error δD changes the energy only at second order: $\delta E = O(\lVert\delta D\rVert^2)$. The FP32 stages only bring the density close to convergence; the FP64 tail keeps iterating with FP64 operators (CPU FP64 K, CPU FP64 semilocal XC) to the FP64 fixed point. Only VV10 stays in FP32 during the tail; its error is about 1e-7 Ha on water27, within the 1e-6 budget. The measured final energies agree with stock PySCF to about 1e-8.

### Staging (Engine style)

In the early cycles the energy changes far exceed the grid errors and the VV10 contribution, so the SCF starts on coarse grids without VV10 and moves to fine grids near convergence. Each switch changes the energy functional, so DIIS is reset and the Fock matrix is rebuilt in full.

## Implementation

The SCF runs in three stages; each switch changes the grids, resets DIIS and rebuilds the Fock matrix in full:

| Stage | Entered when | XC grid | VV10 | SGX grid |
|---|---|---|---|---|
| S0 | from the initial guess | level 1 | off | level 1 |
| S1 | \|g\| < 1e-3 | level 3 | on, level 1 | level 1 |
| S2 | \|dE\| < 1e-6 (or S1 reaches 8 cycles) | level 3 | level 3 | level 2 |

- In S0–S2 the semilocal XC, VV10 and the COSX exchange K all run in FP32 on the GPU, with the CPU and GPU working asynchronously in parallel.
- Once \|dE\| is small enough in S2, the SCF switches to the **FP64 tail**: K goes back to CPU FP64, the semilocal XC back to the CPU FP64 numint, and VV10 stays in FP32 on the GPU. **The final energy comes from the FP64 tail**; errors in the FP32 stages only affect the convergence path.
- While the hooks are installed, a pthreads OpenBLAS is limited to one thread (it oversubscribes inside PySCF's OpenMP regions); the setting is restored when the hooks are removed.

## Usage

```python
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.staging.schedule import run_staged

mol = build_mol('benchmarks/systems/drugs/caffeine.xyz', 'def2-tzvp')   # xyz path or a built-in system name
info = run_staged(mol, conv_tol=1e-9,
                  gpu_stages=(0, 1, 2), fp64_final=True,               # production configuration
                  gpu_kwargs={'k': True, 'k_tile_tol': 1e-11},         # COSX K on the GPU as well
                  fp64_conv_tol=1e-7)                                  # |dE| criterion of the FP64 tail
print(info['e_tot'], info['converged'], info['cycles'])
```

Command line (prints the stage, path, \|dE\|, \|g\| and wall time of every cycle; `--e-ref` is the known stock PySCF energy, used for the gate):

```bash
OMP_NUM_THREADS=16 python -B benchmarks/xc/staged_gpu.py --system <xyz> --basis def2-tzvp \
    --gpu-k --fp64-conv-tol 1e-7 --e-ref <stock energy>
```

- On cluster nodes, load the CUDA module first (`module load cuda/13.3`). Without it the hooks fall back to the CPU with only a warning.
- The first run on a new card spends about 2.5 extra minutes while CuPy compiles the kernels; later runs hit the cache. Calling `pyscf_wb97mv_fast.gpu.warmup.warmup()` compiles them ahead of time.
- Low-level API: `pyscf_wb97mv_fast.gpu.install.install_gpu(mf, k=True, ...)` installs the GPU hooks on any RKS/COSX object; call `restore_all()` on the returned HookSet to remove them.

## Validation (against stock PySCF)

**Energies** (staged SCF with GPU K, RTX 4090):

| System | Basis | nao | Energy difference from stock (Ha) |
|---|---|---:|---:|
| water27 | def2-SVP | 648 | +3.3e-8 |
| caffeine | def2-TZVP | 494 | +6.2e-9 |
| ibuprofen | def2-TZVP | 573 | +7.8e-9 |
| sulfamethoxazole (S) | def2-TZVP | 599 | +5.5e-9 |
| bromazepam (Br) | def2-TZVP | 666 | +5.2e-9 |
| chlorpromazine (Cl, S) | def2-TZVP | 777 | +8.6e-9 |
| fluoxetine (F) | def2-TZVP | 790 | +7.1e-9 |

All runs converged, more than four orders of magnitude below the 1e-4 gate; the differences are of the same size as the cycle-to-cycle jitter of the COSX energy in FP64 (about 5e-9). The CPU K path shows almost the same differences; GPU K adds at most 1.1e-9 on top.

**Density matrix and gradient**: the FP64 reference is computed once on the CPU and stored; each card runs only the GPU part and compares.

- `benchmarks/xc/ref_dm_grad.py`: stock PySCF in CPU FP64, stored as `benchmarks/results/ref_dm_grad/<molecule>_def2-tzvp.npz` (energy, density matrix, gradient, dipole, Mulliken charges, orbitals).
- `benchmarks/xc/check_dm_grad.py --ref <npz>`: runs the production configuration on the card and compares density, gradient, dipole and charges; gates are \|dE\| < 1e-6 and a largest gradient deviation < 1e-5 Ha/bohr.
- Gradients use stock `pyscf.sgx.grad` with the SGX grid response off on both sides (with it on, PySCF 2.14 returns NaN gradients; upstream bug).
- Status: the water-dimer smoke test passes (largest gradient deviation 8.1e-7 Ha/bohr, largest density deviation 4.4e-6). References for five of the drug molecules are stored; the GPU comparison has not been run yet.

## Performance (RTX 4090 + Ryzen 9 7950X3D 16C/32T)

| System | Basis | Stock CPU | Staged GPU K | Speedup |
|---|---|---:|---:|---:|
| water27 | def2-SVP | 1218 s | 130 s | 9.3× |
| caffeine | def2-TZVP | 252 s | 71 s | 3.5× |
| sulfamethoxazole | def2-TZVP | 367 s | 114 s | 3.2× |
| bromazepam | def2-TZVP | 446 s | 134 s | 3.3× |
| ibuprofen | def2-TZVP | 382 s | 109 s | 3.5× |
| fluoxetine | def2-TZVP | 624 s | 172 s | 3.6× |
| chlorpromazine | def2-TZVP | 742 s | 202 s | 3.7× |

Stock uses 32 threads and the staged runs 16 threads, both with `OPENBLAS_NUM_THREADS=1` (each side's faster setting). On the drug molecules about 40% of the time goes to the FP64 tail on the CPU.

## Tests

```bash
python -m pytest                    # default suite: small-system correctness checks, about 95 s on a 2080 Ti
python -m pytest --slow             # adds the slow tests (full SCF, water27, ...)
python -m pytest --experimental     # adds the frozen COSX experimental code
```

pytest only runs small unit checks; correctness conclusions rest on the real runs against FP64 references ("Validation" above).

## Known issues and to-do

1. **FP32 rounding floor of the full K**: one FP32 full-K build carries an energy error of about 1.9e-7 on the water dimer and 2.2e-6 on water27, from coherent rounding on atom-centered grids. The final energy is guaranteed by the FP64 tail and is unaffected. The thresholds of the corresponding tests were changed to the measured floor plus margin (3e-7 for the water dimer, 3e-6 for water27); the new thresholds have not been run yet.
2. **S1 is often ended by force**: the S1→S2 switch only looks at \|dE\| < 1e-6, so S1 often idles until its 8-cycle limit and the staged SCF takes about twice as many cycles as stock.
3. **Performance work**: the full K rebuild when the FP64 tail starts (compute the FP64 K ahead of time in the background) and the VV10 kernel.
4. **Gradients**: this package has no GPU gradient of its own; gradients come from stock PySCF on the converged orbitals and need the SGX grid response off (upstream bug).
5. `experimental/cosx` (the K-side strategy layer) failed all of its gates; it is frozen and not imported by the mainline.

## Layout

```
pyscf_wb97mv_fast/
  core/        SGX fix, removable hooks, OpenBLAS thread limit, timing, test systems
  reference/   CPU FP64 reference numint
  staging/     staged SCF (StagedSCF, run_staged)
  gpu/         FP32 GPU backends: XC, VV10, COSX K (int3c1e, Boys, shell pairs, screening)
  experimental/cosx/   frozen
benchmarks/    benchmark and validation scripts (xc/staged_gpu.py, xc/ref_dm_grad.py, xc/check_dm_grad.py, ...)
benchmarks/systems/drugs/   xyz files of the six drug molecules
tests/         pytest (default / --slow / --experimental)
docs/          design doc of the mainline (docs/design.md + design.en.md / design.ja.md)
```

## Further reading

| File | Content |
|---|---|
| `docs/design.en.md` | design doc of the mainline: design rationale, goals, the staged SCF scheme, the mixed-precision error budget and the acceptance gates (also in [Chinese](docs/design.md) — the authoritative version — and [Japanese](docs/design.ja.md)) |
