[English](README.md) | **中文** | [日本語](README.ja.md)

# pyscf-wb97mv-fast

在 PySCF 里给 **ωB97M-V / RIJCOSX（COSX, pjs=True）** 做的快速路径：在消费级 GPU（RTX 30/40/50）上用 FP32 做重计算，FP64 收尾放在 CPU，用来缩短**单个大体系** SCF 的耗时。

**状态：早期 alpha（0.0.1，2026-10-06）。** 能量的正确性已在 water27（def2-SVP）和 6 个药物分子（def2-TZVP）上对照 PySCF 原版验过；密度矩阵和梯度的对照还在进行中（见下文“验证”）。

---

## 适用范围

| 项 | 支持 |
|---|---|
| 方法 | RKS ωB97M-V + COSX（`dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)`） |
| PySCF | 固定 2.14.0；只用可撤销的 hook，不改 site-packages |
| GPU | 消费级 RTX 30/40/50（FP64 只有 1/64）；kernel 由 CuPy 按卡现场编译。2080 Ti 只作开发机。A100/H100 一类请直接用 gpu4pyscf |
| 基组 | 球谐基，角动量 ≤ f（def2-SVP、def2-TZVP）；g 函数和笛卡尔基不支持（会退回 CPU 并警告） |
| 元素 | 已验证 H、C、N、O、F、S、Cl、Br；带 ECP 的元素未测，不在本版范围 |
| 无 GPU | 自动退回 CPU 原版路径并警告 |

## 理论

### ωB97M-V 的各项

ωB97M-V 由半局域 meta-GGA 交换相关、range-separated 精确交换和 VV10 非局域相关组成。精确交换把库仑算符按 ω = 0.3 bohr⁻¹ 拆成短程和长程：

$$\frac{1}{r} = \frac{\mathrm{erfc}(\omega r)}{r} + \frac{\mathrm{erf}(\omega r)}{r}$$

短程取 15%、长程取 100%。由于 erfc = 1 − erf，交换部分等价于

$$E_x^{\mathrm{HF}} = -\tfrac14\,\mathrm{Tr}\!\left[D\left(0.15\,K^{1/r}[D] + 0.85\,K^{\mathrm{erf}}[D]\right)\right]$$

所以每个 SCF 循环要建两个 K：full K（1/r）和长程 K（erf(ωr)/r）。每一步的计算量主要在三处：两个 K、半局域 XC、VV10。

### COSX（SGX）交换

COSX 把交换积分里的一个电子坐标放到格点 g 上做数值求积：

$$K_{\mu\nu} \approx \sum_g w_g\, X_{g\mu} \sum_\lambda A^{g}_{\nu\lambda} F_{g\lambda},\qquad F = X\,Q\,D,\qquad A^{g}_{\nu\lambda} = \int \chi_\nu(\mathbf r)\,\chi_\lambda(\mathbf r)\, v(|\mathbf r-\mathbf r_g|)\,d\mathbf r$$

其中 X 是格点上的 AO 值，Q 是 PJS 的重叠拟合矩阵，v 是 1/r 或 erf(ωr)/r。GPU 上用一个融合 kernel 直接算出 $G_{g\nu}=\sum_\lambda A^{g}_{\nu\lambda}F_{g\lambda}$，A 不落地；再用 FP64 GEMM 得到 $K = X^{\mathsf T}(w\circ G)$。

**筛选**：对每个（格点子块，shell pair）用解析上界估计 |A|，乘以该子块上 F 的最大加权值，低于 `k_tile_tol`（1e-11）就跳过。上界在 FP32 下算，并有意放大，保证它仍是严格的上界。

### 三中心积分（McMurchie–Davidson）

对指数为 p、中心为 P 的一对原始高斯，令 θ = ω²/(ω² + p)（1/r 核时 θ = 1）、T = θp|P − r_g|²：

$$A = \frac{2\pi}{p}\sum_{tuv} E^{x}_{t} E^{y}_{u} E^{z}_{v}\, R_{tuv},\qquad R^{(n)}_{000} = \sqrt{\theta}\,(-2p\theta)^n F_n(T)$$

$R_{tuv}$ 由 $R^{(n)}_{t+1,u,v} = t\,R^{(n+1)}_{t-1,u,v} + X_{PC}\,R^{(n+1)}_{t,u,v}$（y、z 同理）递推得到。f 壳层需要 Boys 函数 $F_0$ 到 $F_6$。常见的 (la, lb) 组合各有一个编译期特化的 kernel，数组都在寄存器里；广义收缩的 shell pair 走通用 kernel。

### VV10 非局域相关

$$E_c^{\mathrm{nl}} = \int \rho(\mathbf r)\left[\beta + \tfrac12\int \rho(\mathbf r')\,\Phi(\mathbf r,\mathbf r')\,d\mathbf r'\right] d\mathbf r$$

这是格点上的双重求和，代价 $O(N_g^2)$，是原版 PySCF 里最大的单项（water27 上约占 50–65%）。GPU 上分块计算，按块间距离剪枝。

### 混合精度和误差控制

消费级卡的 FP64 吞吐只有 FP32 的 1/64，所以 kernel 内层一律 FP32，不用 double、不用除法。随机舍入误差会相互抵消，真正危险的是**系统偏差**，主要有两种：

1. **常数舍入**：同一个常数（例如 s 函数的归一化 1/√4π）舍入成 FP32 后，每个积分都朝同一方向偏。解决办法是把常数存成 FP32 的 hi + lo 两部分，lo 用 `fmaf` 在最后一次舍入之前加上，补偿不会被舍掉。
2. **相干舍入**：原子中心格点上，同一径向壳层的所有角度点、同元素的各个原子，AO 值完全相同，它们的舍入误差同向累加，偏差随体系大小线性增长。因此 K 路径里的 X、两次 GEMM 和格点权重都保留 FP64，只把 F 舍入一次成 FP32 交给 kernel。剩下的下限是 full K 在 water27 上约 2e-6 Ha（见“已知问题”）。

Boys 函数：T < 36 时用节点间距 1/4 的 7 项泰勒表（截断误差约 1e-10），T ≥ 36 时用渐近式，按 double-float 乘积求值，并整体乘 2¹⁰⁰ 防止误差项落入非规格化数。在 T ≤ 1e30 的整个范围内，各阶误差都 ≤ 0.5 ulp，并且无偏。

### 为什么 FP64 收尾就够

SCF 收敛时能量对密度矩阵取驻值，密度误差 δD 对能量的影响是二阶的：$\delta E = O(\lVert\delta D\rVert^2)$。FP32 阶段只负责把密度带到收敛点附近；FP64 收尾用 FP64 的算子（CPU FP64 K、CPU FP64 半局域 XC）继续迭代，收敛到 FP64 的不动点。收尾时只有 VV10 仍是 FP32，其误差在 water27 上约 1e-7 Ha，在 1e-6 的预算内。实测最终能量与原版的差在 1e-8 量级。

### 分阶段（Engine 风格）

前几个循环里，能量变化远大于格点误差和 VV10 的贡献，所以先用粗格点、不开 VV10，接近收敛再换细格点。每次切换都改变了能量泛函，所以要重置 DIIS 并全量重建 Fock。

## 实现

SCF 分三个阶段，每次切换都换格点、重置 DIIS、全量重建 Fock：

| 阶段 | 进入条件 | XC 格点 | VV10 | SGX 格点 |
|---|---|---|---|---|
| S0 | 初猜起 | level 1 | 关 | level 1 |
| S1 | \|g\| < 1e-3 | level 3 | 开，level 1 | level 1 |
| S2 | \|dE\| < 1e-6（或 S1 满 8 轮） | level 3 | level 3 | level 2 |

- S0–S2 的半局域 XC、VV10 和 COSX 交换 K 都在 GPU 上用 FP32 算；CPU 和 GPU 异步并行。
- S2 里 \|dE\| 足够小后切到 **FP64 收尾**：K 换回 CPU FP64，半局域 XC 换回 CPU FP64 numint，VV10 仍在 GPU 上用 FP32。**最终能量来自 FP64 收尾**，FP32 阶段的误差只影响收敛路径。
- 装 hook 时把 pthreads 版 OpenBLAS 限为 1 线程（它在 PySCF 的 OpenMP 区里会过度订阅），卸载时恢复。

## 用法

```python
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.staging.schedule import run_staged

mol = build_mol('benchmarks/systems/drugs/caffeine.xyz', 'def2-tzvp')   # xyz 路径或内置体系名
info = run_staged(mol, conv_tol=1e-9,
                  gpu_stages=(0, 1, 2), fp64_final=True,               # 生产配置
                  gpu_kwargs={'k': True, 'k_tile_tol': 1e-11},         # COSX K 也上 GPU
                  fp64_conv_tol=1e-7)                                  # 收尾的 |dE| 判据
print(info['e_tot'], info['converged'], info['cycles'])
```

命令行（逐轮打印阶段、路径、\|dE\|、\|g\| 和耗时，`--e-ref` 是已知的原版能量，用来判 gate）：

```bash
OMP_NUM_THREADS=16 python -B benchmarks/xc/staged_gpu.py --system <xyz> --basis def2-tzvp \
    --gpu-k --fp64-conv-tol 1e-7 --e-ref <stock 能量>
```

- 集群节点先加载 CUDA 模块（`module load cuda/13.3`）。没加载时 hook 会退回 CPU，只给警告。
- 新显卡第一次运行，CuPy 要编译 kernel，约多 2.5 分钟；之后命中缓存。可以先调用 `pyscf_wb97mv_fast.gpu.warmup.warmup()` 预热。
- 底层接口：`pyscf_wb97mv_fast.gpu.install.install_gpu(mf, k=True, ...)` 给任意 RKS/COSX 对象装 GPU hook，返回的 HookSet 用 `restore_all()` 卸载。

## 验证（对照 PySCF 原版）

**能量**（分阶段 GPU K，RTX 4090）：

| 体系 | 基组 | nao | 与原版能量差（Ha） |
|---|---|---:|---:|
| water27 | def2-SVP | 648 | +3.3e-8 |
| 咖啡因 | def2-TZVP | 494 | +6.2e-9 |
| 布洛芬 | def2-TZVP | 573 | +7.8e-9 |
| 磺胺甲噁唑（S） | def2-TZVP | 599 | +5.5e-9 |
| 溴西泮（Br） | def2-TZVP | 666 | +5.2e-9 |
| 氯丙嗪（Cl、S） | def2-TZVP | 777 | +8.6e-9 |
| 氟西汀（F） | def2-TZVP | 790 | +7.1e-9 |

全部收敛，离 1e-4 的 gate 四个数量级以上；偏差和 COSX 能量在 FP64 循环间的抖动（约 5e-9）同量级。CPU K 路径的偏差几乎一样，GPU K 在其上最多多 1.1e-9。

**密度矩阵和梯度**：FP64 参考由 CPU 算一次存盘，各卡只跑 GPU 部分对照。

- `benchmarks/xc/ref_dm_grad.py`：原版 PySCF、CPU FP64，存 `benchmarks/results/ref_dm_grad/<分子>_def2-tzvp.npz`（能量、密度矩阵、梯度、偶极、Mulliken 电荷、轨道）。
- `benchmarks/xc/check_dm_grad.py --ref <npz>`：本卡跑生产配置，比密度、梯度、偶极、电荷；gate 为 \|dE\| < 1e-6、梯度最大偏差 < 1e-5 Ha/bohr。
- 梯度用原版 `pyscf.sgx.grad`，两边都关掉 SGX 格点响应项（PySCF 2.14 开着它梯度是 NaN，上游 bug）。
- 现状：水二聚体冒烟通过（梯度最大偏差 8.1e-7 Ha/bohr，密度最大偏差 4.4e-6）；5 个药物分子的参考已存盘，GPU 对照未跑。

## 性能（RTX 4090 + Ryzen 9 7950X3D 16C/32T）

| 体系 | 基组 | 原版 CPU | 分阶段 GPU K | 加速 |
|---|---|---:|---:|---:|
| water27 | def2-SVP | 1218 s | 130 s | 9.3× |
| 咖啡因 | def2-TZVP | 252 s | 71 s | 3.5× |
| 磺胺甲噁唑 | def2-TZVP | 367 s | 114 s | 3.2× |
| 溴西泮 | def2-TZVP | 446 s | 134 s | 3.3× |
| 布洛芬 | def2-TZVP | 382 s | 109 s | 3.5× |
| 氟西汀 | def2-TZVP | 624 s | 172 s | 3.6× |
| 氯丙嗪 | def2-TZVP | 742 s | 202 s | 3.7× |

原版用 32 线程，分阶段用 16 线程，两边都是 `OPENBLAS_NUM_THREADS=1`（各自取更快的设置）。药物分子上 40% 左右的时间花在 CPU 上的 FP64 收尾。

## 测试

```bash
python -m pytest                    # 默认套件：小体系对错检查，2080 Ti 上约 95 s
python -m pytest --slow             # 加上完整 SCF、water27 等慢测试
python -m pytest --experimental     # 加上冻结的 COSX 实验代码
```

pytest 只做小体系的单元检查；正确性的结论以对照 FP64 参考的实跑结果为准（上面的“验证”）。

## 已知问题与待办

1. **full K 的 FP32 舍入下限**：单次 FP32 full K 的能量误差，水二聚体约 1.9e-7、water27 约 2.2e-6，来自原子中心格点上的相干舍入。最终能量由 FP64 收尾保证，不受影响；对应测试的门槛已按实测下限加余量调整（水二聚体 3e-7、water27 3e-6），新门槛尚未实跑。
2. **S1 阶段常被强制切换**：S1→S2 只看 \|dE\| < 1e-6，常在 S1 里空转到 8 轮上限，分阶段的总轮数约为原版的两倍。
3. **性能待做**：FP64 收尾时 K 的全量重建（提前在后台算 FP64 K）、VV10 kernel。
4. **VV10 的测试顺序依赖**：一个 VV10 测试在 4090 的 `--experimental` 套件里失败过一次，之后未复现，根因不明（低优先级）。
5. **梯度**：本包不提供自己的 GPU 梯度；梯度用原版 PySCF 在收敛轨道上算，且需关掉 SGX 格点响应项（上游 bug）。
6. `experimental/cosx`（K 侧的策略层）gate 全部未过，已冻结，主线不导入。

## 目录

```
pyscf_wb97mv_fast/
  core/        SGX 修补、可撤销 hook、OpenBLAS 限线程、计时、测试体系
  reference/   CPU FP64 参考 numint
  staging/     分阶段 SCF（StagedSCF、run_staged）
  gpu/         FP32 GPU 后端：XC、VV10、COSX K（int3c1e、Boys、shell pair、筛选）
  experimental/cosx/   已冻结
benchmarks/    基准与验证脚本（xc/staged_gpu.py、xc/ref_dm_grad.py、xc/check_dm_grad.py …）
benchmarks/systems/drugs/   6 个药物分子 xyz
tests/         pytest（默认 / --slow / --experimental）
docs/          主线设计文档（design.md / design.en.md / design.ja.md）
```

## 进一步阅读

| 文件 | 内容 |
|---|---|
| `docs/design.md` | 主线设计文档：设计理由、目标与硬件前提、分阶段 SCF 方案、混合精度误差预算与验收判据（判据以它为准；另有 [English](design.en.md) / [日本語](design.ja.md) 版） |
