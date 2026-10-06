# VV10 / XC 在消费级 GPU 上的混合精度异构设计

> 2026-09-30。本文是主线（分阶段 SCF：FP32 GPU + FP64 收尾）的设计依据：
> 目标与硬件前提、混合精度的误差预算、各模块的设计与验收判据。判据以本文为准。
> （早期以 K 为主线的 P1–P3 路线 gate 全部未通过，代码已冻结。）

语言 / Languages：**中文（权威版）** | [English](design.en.md) | [日本語](design.ja.md)

---

## 0. 设计理由速览（为什么是这条路）

- **为什么分阶段（S0→S1→S2），而不是全程细网格？** SCF 前期的 \|ΔE\| 比粗网格、暂不算 VV10 带来的误差大几个数量级，前期用便宜设置不改变收敛到的解；等 \|ΔE\|、\|g\| 降到与格点误差同量级再切换。每次切换能量泛函都变了，所以重置 DIIS 并完整重建 Fock。这套做法在 Engine（生产级 MD 程序）里已经过验证。
- **为什么重的计算用 FP32 放 GPU、FP64 收尾留在 CPU？** 消费级卡（RTX 30/40/50）的 FP64 吞吐只有 FP32 的 1/64，全程 GPU FP64 得不偿失；而 SCF 能量在收敛点对密度矩阵驻定（δE = O(‖δD‖²)），FP32 阶段只影响收敛路径、不影响收敛到的解，最后由 CPU FP64 的 K 和半局域 XC 收尾，把能量定位到 FP64 不动点（实测与 stock 差约 1e-8 Ha）。
- **为什么不用 gpu4pyscf？** 它全程 FP64、面向 A100 级别的卡；本项目面向消费级卡上的 COSX K 与 VV10，kernel 全部自写（CuPy RawKernel）。gpu4pyscf 只当作 libxc-cuda 的提供者，用在逐点泛函求值这种"轻计算"上，包在 `XcFunctional` 接口后面，没有它就退回 CPU libxc。
- **为什么误差控制是设计的核心？** FP32 的随机舍入会相消，真正危险的是系统性偏差，来源有二：常数舍入（每个积分同向偏移 → 常数存成 FP32 hi+lo 双长并用 fmaf 补偿）和相干舍入（原子中心格点上同一径向壳的角向点看到相同的 AO 值，误差同向叠加、随体系线性增长 → K 路径的 X、两个 GEMM 与格点权重保持 FP64，只把 F 舍入一次）。
- **为什么坚持 PySCF 主导 + 可撤销 hook？** 不 fork、不改 site-packages、签名不变；没有 GPU 或出错时自动退回 CPU 参考路径。上游升级与对照验证都容易。
- **为什么 VV10 的 tile 裁剪是安全的？** VV10 核在远处按 1/R⁶ 衰减，用块间最小距离与 W0、κ 极值构造严格上界；`vv10_tile_tol = 0` 时退化为精确计算。筛选只减少 FP32 求值的任务量，不改变数学。
- **为什么 K 侧策略层（experimental/cosx）冻结？** 逐循环 controller、SR 筛选、dynamic A/B 的 gate 全部未通过，收益不抵复杂度；主线保持"分阶段 + GPU FP32 + FP64 收尾"就达到验收目标。
- **为什么 RI-JK 与梯度不在范围内？** RI-JK 的 GPU 自适应精度已有发表工作（Huang、Shao、Hammond 2026），不重复；梯度受上游 PySCF 2.14 的 SGX 格点响应 NaN bug 影响，待上游修复。

---

## 1. 背景与动机

- **stock PySCF 的瓶颈是 VV10。** 在 water27、def2-SVP、ωB97M-V、RIJCOSX 下实测：VV10 占 SCF 时间的 65%，XC 半局域部分占 9%，两个 K 合计占 24%。原因是 `nlcgrids` 默认是 level 3，而 `_vv10nlc` 是 O(N_g²) 的稠密计算。
- **一个生产级 MD 程序（下称 Engine）已经把 VV10 降到了 5%**（remdesivir，1755 个基函数，8 核），做法有两点：VV10 分阶段打开，而且用粗网格。在 Engine 里，剩下的瓶颈才是 COSX 的 K，占 64%。
- **消费级 GPU 的性能差距。** gpu4pyscf（JPCA 129, 1459，arXiv:2407.09700）全程 FP64，只在 A100 上测过。而 RTX 30、40、50 系列的 FP64 只有 FP32 的 1/64。Huang、Shao、Hammond 的工作（JPCA 130, 3483, 2026）证明，在 RI-JK 上用自适应精度加 INT8，可以在 RTX 4090 上快 204%，收敛能量不受影响。RI-JK 他们已经做过，**本项目不重复**。

## 2. 目标与非目标

**目标**：用 RIJCOSX 算 ωB97M-V，缩短**单个大体系** SCF 的延迟；在消费级 GPU 上做 CPU+GPU 异构，重的计算用 FP32 放在 GPU 上，需要 FP64 的放在 CPU 上。

**硬件前提**：
- 开发用 RTX 2080 Ti（sm_75，没有 TF32）。
- 正式运行在 RTX 3090、4090、5090（sm_80 及以上）上。
- **CPU 按 16 核设计和验收**，这是大多数 GPU 机器的常见配置。
- 不存在"FP64 的 GPU"：消费级卡的 FP64 吞吐太低，**不能拿 GPU FP64 kernel 当过渡方案**。

**验收**：只看 SCF 能量。最终结果和全程细网格、全 FP64 的参考值相比，误差 < 1e-4 Ha。

**非目标**：
- RI-JK：Huang 2026 已经做过。
- 核梯度（力）：以后再说。
- 批量吞吐：这次的目标是单个体系的延迟。
- 修改 site-packages：PySCF 固定在 2.14.0，只用可撤销的 hook。
- git：不做任何 git 操作。

## 3. 总体架构（方案 1'）

- **SCF 仍由 PySCF 在 CPU 上驱动。** GPU 只接管两个入口：`ni.nr_rks`（XC）和 `ni.nr_nlc_vxc`（VV10）。签名和返回值都和 PySCF 保持一致。
- **按计算量决定精度**：
  - **重的计算**，也就是 O(N_g·nao²) 的 GEMM 和 O(N_g²) 的 VV10 两两求和，放在 GPU 上用 **FP32**。
  - **轻的计算**，也就是 O(N_g) 的逐点运算和 O(nao²) 的累加，**可以用 FP64**，而且尽量放在 GPU 上，减少对 16 核 CPU 的依赖。
- **CPU（FP64）上只保留**：SCF 主流程、COSX 的 K（PySCF SGX 加 `sgx_patch`）、RI-J、对角化、DIIS、正交化。
- **K 以后要搬走**：在 16 核机器上，K 会成为下一个瓶颈，所以 K 放在 `KBuilder` 接口后面，留给 S5（GPU FP32 版 COSX）来替换。

## 4. 子项目

| 编号 | 内容 | 依赖 |
|---|---|---|
| S1 | 代码重构（§5）：以 VV10 为主线，K 那一侧冻结 | — |
| S2 | 分阶段（§6），纯 CPU 也能用：VV10 用粗网格、切换点对齐、切换时重置 DIIS | S1 |
| S3 | GPU 混合精度的 VV10 和 XC（§6、§7） | S1，可以和 S2 并行 |
| S3b | VV10 可分离加低秩：原版 VV10 核做 (q, κ) 二维插值，再做共用因子的低秩压缩，得到 r 个通道，计算量约 r·N log N（§7.5） | S3；开工前先做两个 spike |
| S5 | GPU FP32 版 COSX 的 ESP 积分（以后做） | S3 |
| （S4） | 力：不在这次的范围内。Engine 已经推导出 VV10 网格力（包括格点位置和分区权重的导数）的解析形式，以后做 S4 时可以参考 | — |

每个子项目各写一份实施计划。

## 5. 代码结构（S1）

```text
pyscf_wb97mv_fast/
├── __init__.py            入口 fast_path(mf, ...)
├── core/                  sgx_patch.py、hooks.py（原 _hooks，新增 DIIS 重置）、profiling.py、testsystems.py
├── staging/schedule.py    S2：各阶段的网格、切换点、DIIS 重置、完整 Fock 重建
├── gpu/                   S3：device.py、precision.py、ao_eval.py、vv10.py、xc.py、backends.py、install.py
├── reference/             CPU FP64 参考实现，直接包一层 PySCF numint
└── experimental/cosx/     冻结区：tolerance、controller、sr_screening、dynamic_ab、fastgrad、
                           acc/jax_vv10、acc/hardware_profile
benchmarks/vv10/  benchmarks/experimental/cosx/（原 sgx_locality 和 K 相关的 bench）
tests/core/ tests/staging/ tests/gpu/ tests/experimental/（最后一个目录默认不跑）
```

- 冻结区的代码保留，主线不导入，测试也默认不跑。
- 文件只用 `mv` 移动；所有 import 路径和 `sys.path` 的写法都要同步更新。

## 6. 精度策略与阶段（S2、S3）

**阶段**（每次切换都做三件事：换网格、重置 DIIS、完整重建 Fock；**在同一个阶段内部，网格、精度和 tolerance 都不变**）：

| 阶段 | 进入条件（暂定，由实验调整） | XC 网格（`grids.level`） | VV10（`nlcgrids.level`） | COSX（SGX 网格级别） |
|---|---|---|---|---|
| S0 | 从初始猜测开始 | 1 | 关闭 | 1 |
| S1 | \|g\| < τ₁ = 1e-3 | 3 | 打开，1 | 1 |
| S2 | 相邻两个循环的 \|ΔE\| < 1e-6 Ha | 3 | 3 | 2 |

- **参考值**（"全程细网格、全 FP64"）的定义：stock PySCF，XC 用 level 3，VV10 用 level 3，SGX 用 level 2，三者都是 PySCF 2.14 的默认值；打上 `sgx_patch`，conv_tol 和被测方案相同。
- 所以 S2 用的就是参考值那一套网格，分阶段带来的误差只可能来自 S0 和 S1 留下的收敛路径差异。
- **VV10 有两种用法**（参数 `vv10_mode`）。Engine 实测这两种结果相同：
  - `'late_scf'`（默认）：按上表在 S1 和 S2 里自洽地计算 VV10。
  - `'nonscf'`：整个 SCF 都不算 VV10，收敛后用最终密度算一次 VV10 能量，相当于把 VV10 当成色散校正来用。这是最便宜的一档；因为验收只看能量，它是正式选项。验收时要和 `'late_scf'` 对比 \|ΔE\|。
- **S2 有两种模式**（参数 `s2_mode`）：
  - `'converge'`（默认）：在 S2 里收敛到 conv_tol。误差和收敛阈值同一量级，远小于 1e-4 Ha。
  - `'steps'`：在 S2 里只跑 `s2_steps` 步（默认 1），和 Engine 的做法一样。误差允许达到 1e-4 Ha，但换来更少的高网格循环。
  - 验收时两种模式都要测，分别报告误差和耗时。

这里的 \|g\| 是 PySCF 的 `envs['norm_gorb']`，通过 `track_residual` 读取。重置 DIIS 的办法是在 `mf.callback` 里拿到 `envs['mf_diis']`，把它的 `_buffer`、`_bookkeep`、`_head`、`_H`、`_xprev` 清空。

**每个算子的精度**：

| 算子 | 放在哪里 | 精度 |
|---|---|---|
| AO 求值（φ、∇φ） | GPU | FP32 |
| density_gemm（ρ、∇ρ、τ） | GPU | FP32 SGEMM |
| XC 泛函求值（逐点） | GPU | FP64，用 gpu4pyscf 的 libxc-cuda（轻计算）；以后可以换成 JAX FP32 |
| fock_gemm（Vxc、Vnlc） | GPU | 每组块用 FP32 算出部分矩阵，再在 GPU 上用 FP64 累加 |
| VV10 kernel_sum，O(N_g²) | GPU | FP32 分块，块内用 Kahan 补偿求和；每个格点的 F、U、W 用 FP64 累加 |
| 能量求和 | GPU 或 CPU | FP64 |
| K、RI-J、对角化、DIIS | CPU | FP64 |
| TF32 | — | **默认关闭**，只允许在 S0 的 GEMM 里显式打开，仅限 sm_80 以上。不能把 TF32 当成 FP32 |

**误差预算**：
- 混合精度带来的误差：同一套网格下，和全 FP64 相比 < 1e-6 Ha。
- 分阶段和网格带来的误差：和全程细网格、全 FP64 的参考相比 < 1e-4 Ha。
- S2 最后一步默认用 GPU 的 FP32 后端，前提是它满足 < 1e-6 Ha；否则退回 CPU 上的 FP64 参考实现。

## 7. GPU 后端的接口与数据流（S3）

**进入每个阶段时**：在 CPU 上用 NumPy 准备网格（坐标存成相对于分子中心的 FP32，外加权重）、块列表（AO 稀疏模式，按形状分桶）和 FP32 打包的基组参数，然后常驻显存。

**每个循环**：
- 上传 DM（nao² 大小）。
- 对每个桶，在 GPU 上依次做：XC 流程（AO → ρ/∇ρ/τ → libxc → fock_gemm → Vxc），以及 VV10 流程（在它自己的网格上：AO → ρ/∇ρ → W0、κ → kernel_sum → F/U/W → fock_gemm → Vnlc）。
- 下载 Vxc + Vnlc（nao² 大小）和能量。

**显存**：按桶流式使用，不保存整张 N_g × nao 的数组；块的大小由显存预算决定，默认是空闲显存的 60%。

**kernel_sum 的块对裁剪**：VV10 的核函数在远处大约按 1/R⁶ 衰减（g ≈ R²·W0）。对每一对块 (I, J)，用两块之间的最小距离和块内 W0、κ 的极值，给 Σ q_i·q_j·M_ij 算一个上界，低于阈值 `vv10_tile_tol` 的整对块就跳过。这个上界在 CPU 上用 NumPy 算，GPU 只处理剩下的块对。`vv10_tile_tol = 0` 时不裁剪，结果和精确计算一致，用来做验收对照。

**接口**（先实现 CuPy 版本，名字为 `'cupy'`；以后加 `'jax'`，两者通过 DLPack 零拷贝交换数组）：

```python
class AoEvaluator:   eval(block) -> ao32                      # (ncomp, npts, nao_active)
class XcFunctional:  eval(rho) -> exc, vrho, vsigma, vtau
class Vv10Kernel:    kernel_sum(coords, q, W0, kappa) -> F, U, W
class KBuilder:      get_k(dm, omega)                         # 现在是 CPU 上的 SGX；S5 换成 GPU
```

**出错时的处理**：
- 没有 GPU 或 CuPy：自动退回 CPU 参考路径，并给出警告。
- 显存不够（OOM）：块大小减半后重试。
- kernel_sum 用分块归约，不用 atomicAdd，保证结果可以逐次复现。

**实现语言**：先用 CuPy 加 RawKernel（CUDA C）。JAX 以后作为第二个后端接入，要注意三点：静态形状（用分桶加填充来解决）、`precision='highest'`（防止默认走 TF32）、`XLA_PYTHON_CLIENT_PREALLOCATE=false`（防止和 CuPy 抢显存）。Zig 是可选项，只有在调度逻辑被 profile 证明是瓶颈时才考虑。

## 7.5 S3b：把 VV10 可分离化并做低秩压缩（O(N_g²) → 约 r·N log N）

**动机**：stock PySCF 里 VV10 占 65%（§1 的实测）。S3 用 GPU FP32 只能加快常数；S3b 要把复杂度降下来。

**文献依据**：
- rVV10：Sabatini 等，PRB 87, 041108 (2013)。改写了核，使它可以用 Román-Pérez–Soler 插值。
- ωB97M-rV：Mardirossian 等，JPCL 8, 35 (2017)，推荐 b = 6.2。B97M-rV 的表现和 B97M-V 持平或更好。
- Gaussian 轨道体系下 rVV10 的 RPS 加 FFT 实现：Lee 等，JCP 155, 164102 (2021)，Q-Chem GPW，O(N_g log N_g)。

**做法**（保持**原版 VV10**，最终算的严格是 ωB97M-V）：
1. **二维插值**。原版核 Φ = −3/2 · 1/[g·g′·(g + g′)]，其中 g = κ(qR² + 1)，q = ω₀/κ。因为 (g + g′) 同时含有 κ 和 κ′，所以只在 q 上插值不够，要对 (q, κ) 做二维插值：θ_{ab}(r) = ρ(r) · p_a(q(r)) · p_b(κ(r))，场的个数 M = M_q × M_κ。
2. **共用因子的低秩压缩**。K_{(ab),(cd)}(R) ≈ Σ_{s=1}^{r} A_{(ab),s} · A_{(cd),s} · f_s(R)，其中因子 A 与 R 无关（带对称约束的 CP 分解）。于是只剩 r 个通道：ρ_s = Aᵀθ，E ≈ ½ Σ_s ⟨ρ_s, f_s ∗ ρ_s⟩。势能部分用 v_θ = A·(f_s ∗ ρ_s)，再经 p_a、p_b 的导数链式回到 vrho 和 vsigma。
3. **卷积**：r 个通道各做一次径向卷积。在分子网格上有两个候选方案，由 spike B 来决定：
   - 投影到均匀网格上，加 padding 做开边界 FFT；
   - 利用 f_s(R) 在远处按 R⁻⁶ 衰减，用树方法或多极方法直接在 Becke 网格上计算。
4. **按阶段使用**（和 §6 的 `vv10_mode` 配合）：S1 可以用 S3b（或者 rVV10 近似）作为便宜的 VV10；S2 最后一步用 S3 的精确 FP32 两两求和，或者用 S3b 在误差允许范围内的高精度设定。切换时照常重置 DIIS。

**开工前的两个 spike**（不需要 SCF 和 GPU，用 NumPy 就能做）：
- **spike A（秩）**：在 (q, κ, q′, κ′, R) 网格上把原版核（以及 rVV10 核作为对照）制成表，分别做逐个 R 的 SVD 和共用因子的 CP 分解。给出 r(ε) 以及 M_q × M_κ 分别需要多大，ε 取 1e-6、1e-5、1e-4 的相对误差。
- **spike B（能量）**：用 water27 的收敛密度，在 VV10 网格上用可分离加低秩的核**直接做双重求和**（先不做卷积加速），和稠密精确的 E_nl 相比。判据：插值和截断带来的误差 ≤ 1e-5 Ha，给最终 1e-4 Ha 的预算留出余量。之后再比较两种卷积方案的精度和速度。

**S3b 的验收**：用 S3 的精确版本作对照，|ΔE_nl| ≤ 1e-5 Ha，完整 SCF 的 |ΔE| ≤ 1e-4 Ha；在相同网格下，VV10 的耗时相比 S3 的精确 GPU 版本还要再降一个数量级。

## 8. 测试与验收

pytest 标记：`gpu`（没有 GPU 时自动跳过）、`slow`、`experimental`（默认不跑）。

**第 1 层：逐个 kernel 和 CPU FP64 对照**（在 2080 Ti 上跑）

| kernel | 参考 | 判据 |
|---|---|---|
| AO 求值 | `mol.eval_gto` | 相对误差 ≤ 1e-6（以 max\|φ\| 为基准） |
| ρ、∇ρ、τ | numint `eval_rho` | 相对误差 ≤ 1e-6 |
| libxc-cuda | PySCF 的 libxc | ≤ 1e-12 |
| kernel_sum 的 F、U、W | `_vv10nlc` | 相对误差 ≤ 1e-6 |
| VV10 能量和 Vnlc | `nr_nlc_vxc` | \|ΔE\| ≤ 1e-7 Ha |
| XC 能量和 Vxc | `nr_rks`（同一网格） | \|ΔE\| ≤ 1e-7 Ha |

**第 2 层：完整 SCF**（water27、chain30）
- 固定网格，只看混合精度：|ΔE| ≤ 1e-6 Ha，循环数相差不超过 1。
- 分阶段：和全程细网格、全 FP64 的参考相比，|ΔE| ≤ 1e-4 Ha，循环数最多多 2 个。
- 健壮性：hook 装上又卸掉后完全还原；没有 GPU 时能退回 CPU；OOM 时块大小减半后能继续。

**第 3 层：性能**（在 30、40、50 系列的机器上，`OMP_NUM_THREADS=16`；2080 Ti 只验证功能）
- 基线：同一台机器上的 stock PySCF，16 线程，最终网格相同。
- **G_gpu1**：相同网格下，GPU 版 VV10 加 XC 的耗时 ≤ CPU 版的 0.1 倍。
- **G_gpu2**：分阶段完整跑完，总 SCF 时间比基线快至少 2 倍，同时 |ΔE| ≤ 1e-4 Ha。
- 每项都要记录：各分项时间、循环数、显存峰值、PCIe 传输量。

**测试体系**：单元测试用 water dimer，集成测试用 water27 和 chain30；真实分子以后另行挑选。

## 8.5 S5（COSX）开工前要先回答的问题：长程 exchange 要不要有自己的一套数值积分体系

**问题**：K_LR 用的是 erf(ωr)/r 核。它是否应该和 full COSX 共用同一套网格、screening 和精度阈值，还是应该有自己独立的一套，甚至独立的 COSX 网格？

**已有数据**（water27 和 chain30 的实测，§1 同源的归因数据）：
- **screening 这一项几乎没有可省的**：stock 下 full 和 LR 的任务数几乎相同（2930 万对 2854 万）；oracle 在每一档误差预算下需要保留的任务数也几乎相同。locality 来自 F = P·X 自身的衰减，和积分核无关。
- **同一套网格上 LR 反而更不准**：SGX level 2 上，和解析 K 相比的网格误差是 full 5.1e-7 Ha，LR 4.3e-6 Ha，SR 4.8e-6 Ha。推测原因是 `fit_ovlp` 的误差抵消是按 full 1/r 设计的，这一点尚未验证。
- **LR 的代价确实高**：单个积分比 full 贵 1.45 到 1.7 倍，一次 K build 慢 1.3 到 1.5 倍。网格点数减半，LR 的开销也大致减半。

**判断实验**（在 PySCF 里做，工程代价很小）：
- SGX 本来就给每个 ω 建了独立的副本 `_rsh_df[key]`，有它自己的 `_pjs_data`；梯度的 RSH 分支也复用这个副本。只需要把 LR 副本的网格级别调成 1，full 保持 2。
- 在 water27 上测三样：LR 的网格误差（和解析 K 比）、LR 的 K build 时间、完整 SCF 的 \|ΔE\|。
- **结论怎么下**：
  - LR 的误差仍然远小于 1e-4 Ha，时间降到约 0.5 倍：独立网格值得做，也值得搬到 Engine 里。
  - LR 的误差明显变大：需要的是 LR 专用的误差修正（例如针对 LR 的 overlap fitting），而不是一套独立网格。
- **收敛稳定性方面**：一套在整个 SCF 中保持不变的 LR 网格不会扰乱 DIIS。真正要避免的是在 SCF 中途改动它。

**元素敏感度**（Engine 上的经验：COSX 网格对有些元素很敏感，对另一些不敏感）：
- **PySCF 的现状**：SGX 的 opt grids 只有径向点数随周期变化（`SGX_RAD_GRIDS[level, period]`），角向格点按 Bragg 半径修剪（`sgx_prune`）。同一周期的元素用同一套网格，比如 C、N、O、F 完全相同，没有逐元素的敏感度设定。
- **可以直接用的接口**：`grids.atom_grid[symb] = (nrad, nang)`，可以按元素单独指定网格。
- **敏感度实验**：
  - 其他元素的网格都保持不变，每次只把一种元素的网格提高一级，记下 K 的能量变化（和解析 K 比，或者和最高一级网格比）。
  - full 和 LR 分别做，因为两者对元素的敏感度可能不同。
  - 结果是一张"元素 × 网格级别 → 误差贡献"的表。
- **用途**：按元素定网格。敏感的元素给细网格，不敏感的元素降一级，在总误差预算不变的前提下尽量减少格点数。这个思路同样适用于 XC 网格和 VV10 网格。

## 9. 从原设计文档借鉴的内容

| 来源（原文档） | 做法 | 在本 spec 中的位置 |
|---|---|---|
| §2.1 | 只看 wall time，不追求一切上 GPU | §3 |
| §2.2 | 精度是算子的属性；不要把 TF32 当成 FP32 | §6 |
| §2.3 | 各算子的网格互相独立 | §6，VV10 单独用粗网格 |
| §6.1 C | VV10 分阶段，最后严格收尾 | §6，另外补上切换时重置 DIIS |
| §8 | 不规则部分和规则部分分开处理 | §7：CPU 用 NumPy 调度，GPU 做规则的稠密计算 |
| §9 | 按形状分桶，太小的块不上 GPU | §7 |
| §10 | 启动时做 HardwareProfile microbench | S3 的扩展项：由它决定 TF32 开关和各算子放在 CPU 还是 GPU |
| §11 | XC 用 direct 流式模式 | §7 |
| §17 | 正确性指标，以及 fast、production、reference 三种模式 | §8 |

**不采用**：SR-COSX 和 B 路径、逐循环调整的 controller（这几项的 gate 都没通过）；PS-J 和 FMM。

**VV10 低秩近似（原文档 §7）改为 S3b，不再列为不采用**，设计见 §7.5。

## 10. 风险

| 风险 | 缓解办法 |
|---|---|
| FP32 的 AO 或 ρ 误差经过非线性的 XC 被放大 | 第 1 层逐项量化；S2 可以退回 CPU FP64 |
| 分阶段的切换点不合适，导致循环数增加 | 切换时重置 DIIS；τ₁ 按实验调整；验收时把循环数写进判据 |
| libxc-cuda（来自 gpu4pyscf）的接口不稳定 | 包在 `XcFunctional` 后面；PySCF 的 libxc 始终作为参考 |
| 16 核机器上 K 成为瓶颈，G_gpu2 达不到 | G_gpu2 定为 2 倍，给这种情况留了余量；K 的问题交给 S5 |
| 块的形状多样，kernel 启动开销大 | 按形状分桶，太小的块留在 CPU 上 |
