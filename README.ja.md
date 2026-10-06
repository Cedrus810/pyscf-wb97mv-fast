[English](README.md) | [中文](README.zh-CN.md) | **日本語**

# pyscf-wb97mv-fast

PySCF における **ωB97M-V / RIJCOSX（COSX, pjs=True）** の高速パスです。重い計算はコンシューマー向け GPU（RTX 30/40/50）上で FP32 で行い、最後の FP64 の仕上げは CPU で行うことで、**単一の大きな系**の SCF の所要時間を短縮します。

**ステータス：初期アルファ版（0.0.1、2026-10-06）。** エネルギーの正しさは water27（def2-SVP）と 6 つの医薬品分子（def2-TZVP）で PySCF 本体と照合済みです。密度行列と勾配の照合は進行中です（下記「検証」を参照）。

---

## 対応範囲

| 項目 | 対応 |
|---|---|
| 手法 | RKS ωB97M-V + COSX（`dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)`） |
| PySCF | 2.14.0 に固定。取り外し可能な hook のみを使い、site-packages は変更しない |
| GPU | コンシューマー向け RTX 30/40/50（FP64 性能は 1/64）。kernel は CuPy がカードごとにその場でコンパイルする。2080 Ti は開発用のみ。A100/H100 クラスでは gpu4pyscf を直接使うこと |
| 基底 | 球面調和基底、角運動量 f まで（def2-SVP、def2-TZVP）。g 関数とデカルト基底は非対応（警告を出して CPU に戻る） |
| 元素 | H、C、N、O、F、S、Cl、Br で検証済み。ECP を使う元素は未検証で、本バージョンの対象外 |
| GPU なし | 警告を出して PySCF 本体の CPU パスに自動で戻る |

## 理論

### ωB97M-V の各項

ωB97M-V は、半局所 meta-GGA の交換相関、range-separated な厳密交換、VV10 非局所相関からなります。厳密交換ではクーロン演算子を ω = 0.3 bohr⁻¹ で短距離と長距離に分割します。

$$\frac{1}{r} = \frac{\mathrm{erfc}(\omega r)}{r} + \frac{\mathrm{erf}(\omega r)}{r}$$

短距離を 15%、長距離を 100% 取ります。erfc = 1 − erf なので、交換部分は次と等価です。

$$E_x^{\mathrm{HF}} = -\tfrac14\,\mathrm{Tr}\!\left[D\left(0.15\,K^{1/r}[D] + 0.85\,K^{\mathrm{erf}}[D]\right)\right]$$

したがって各 SCF サイクルで 2 つの K、すなわち full K（1/r）と長距離 K（erf(ωr)/r）を構築します。各ステップの計算量の大部分は、2 つの K、半局所 XC、VV10 の 3 つです。

### COSX（SGX）交換

COSX は交換積分の片方の電子座標をグリッド点 g 上で数値積分します。

$$K_{\mu\nu} \approx \sum_g w_g\, X_{g\mu} \sum_\lambda A^{g}_{\nu\lambda} F_{g\lambda},\qquad F = X\,Q\,D,\qquad A^{g}_{\nu\lambda} = \int \chi_\nu(\mathbf r)\,\chi_\lambda(\mathbf r)\, v(|\mathbf r-\mathbf r_g|)\,d\mathbf r$$

ここで X はグリッド上の AO の値、Q は PJS の重なりフィッティング行列、v は 1/r または erf(ωr)/r です。GPU では 1 つの融合 kernel で $G_{g\nu}=\sum_\lambda A^{g}_{\nu\lambda}F_{g\lambda}$ を直接計算し、A はメモリに展開しません。その後 FP64 の GEMM で $K = X^{\mathsf T}(w\circ G)$ を求めます。

**スクリーニング**：各（グリッドのサブブロック、シェルペア）について |A| の解析的上界を求め、そのサブブロック上の F の重み付き最大値を掛けて、`k_tile_tol`（1e-11）を下回ればスキップします。上界は FP32 で計算し、意図的に大きめに取ることで、厳密な上界であることを保証しています。

### 3 中心積分（McMurchie–Davidson）

指数 p、中心 P の原始ガウス関数のペアについて、θ = ω²/(ω² + p)（1/r 核では θ = 1）、T = θp|P − r_g|² とすると、

$$A = \frac{2\pi}{p}\sum_{tuv} E^{x}_{t} E^{y}_{u} E^{z}_{v}\, R_{tuv},\qquad R^{(n)}_{000} = \sqrt{\theta}\,(-2p\theta)^n F_n(T)$$

$R_{tuv}$ は $R^{(n)}_{t+1,u,v} = t\,R^{(n+1)}_{t-1,u,v} + X_{PC}\,R^{(n+1)}_{t,u,v}$（y、z も同様）の漸化式で求めます。f 殻には Boys 関数 $F_0$ から $F_6$ が必要です。よく現れる (la, lb) の組には、コンパイル時に特殊化した kernel をそれぞれ用意し、配列をすべてレジスタに置きます。一般縮約のシェルペアは汎用 kernel で計算します。

### VV10 非局所相関

$$E_c^{\mathrm{nl}} = \int \rho(\mathbf r)\left[\beta + \tfrac12\int \rho(\mathbf r')\,\Phi(\mathbf r,\mathbf r')\,d\mathbf r'\right] d\mathbf r$$

これはグリッド上の二重和で、コストは $O(N_g^2)$ です。PySCF 本体では最大の単一項目です（water27 で約 50–65%）。GPU ではブロックに分けて計算し、ブロック間の距離で枝刈りします。

### 混合精度と誤差の制御

コンシューマー向けカードの FP64 スループットは FP32 の 1/64 しかないため、kernel の内側はすべて FP32 で、double も除算も使いません。ランダムな丸め誤差は打ち消し合いますが、本当に危険なのは**系統的な偏り**で、主に次の 2 種類があります。

1. **定数の丸め**：同じ定数（たとえば s 関数の規格化定数 1/√4π）を FP32 に丸めると、すべての積分が同じ向きにずれます。対策として、定数を FP32 の hi + lo の 2 つに分けて保存し、lo を最後の丸めの前に `fmaf` で加えることで、補正が丸めで消えないようにしています。
2. **コヒーレントな丸め**：原子中心のグリッドでは、同じ動径シェル上のすべての角度点や、同じ元素の各原子で AO の値がまったく同じになるため、丸め誤差が同じ向きに積み重なり、偏りが系の大きさに比例して増えます。そのため K のパスでは X、2 回の GEMM、グリッドの重みを FP64 のまま保ち、F だけを 1 回 FP32 に丸めて kernel に渡します。残る下限は、water27 の full K で約 2e-6 Ha です（「既知の問題」を参照）。

Boys 関数：T < 36 では節点間隔 1/4 の 7 項テイラー表（打ち切り誤差約 1e-10）、T ≥ 36 では漸近式を使います。漸近式は double-float の積で評価し、誤差項が非正規化数にならないよう全体に 2¹⁰⁰ を掛けています。T ≤ 1e30 の全範囲で、各次数の誤差は 0.5 ulp 以下で、偏りもありません。

### FP64 の仕上げで十分な理由

SCF が収束すると、エネルギーは密度行列に対して停留値をとるため、密度の誤差 δD がエネルギーに与える影響は 2 次になります：$\delta E = O(\lVert\delta D\rVert^2)$。FP32 の段階は密度を収束点の近くまで運ぶだけで、FP64 の仕上げでは FP64 の演算子（CPU FP64 の K、CPU FP64 の半局所 XC）で反復を続け、FP64 の不動点に収束させます。仕上げで FP32 のまま残るのは VV10 だけで、その誤差は water27 で約 1e-7 Ha と 1e-6 の予算内です。実測でも、最終エネルギーと本体との差は 1e-8 程度です。

### 段階的 SCF（Engine 方式）

最初の数サイクルではエネルギーの変化がグリッド誤差や VV10 の寄与よりはるかに大きいため、粗いグリッドで VV10 なしから始め、収束に近づいてから細かいグリッドに切り替えます。切り替えのたびにエネルギー汎関数が変わるので、DIIS をリセットし、Fock 行列を全再構築します。

## 実装

SCF は 3 つの段階に分かれ、段階が切り替わるたびにグリッドを変え、DIIS をリセットし、Fock 行列を全再構築します。

| 段階 | 開始条件 | XC グリッド | VV10 | SGX グリッド |
|---|---|---|---|---|
| S0 | 初期推測から | level 1 | オフ | level 1 |
| S1 | \|g\| < 1e-3 | level 3 | オン、level 1 | level 1 |
| S2 | \|dE\| < 1e-6（または S1 が 8 サイクルに達したとき） | level 3 | level 3 | level 2 |

- S0–S2 では半局所 XC、VV10、COSX 交換 K をすべて GPU 上で FP32 で計算します。CPU と GPU は非同期に並行して動きます。
- S2 で \|dE\| が十分小さくなると **FP64 の仕上げ**に切り替わります。K は CPU の FP64 に、半局所 XC は CPU の FP64 numint に戻り、VV10 は GPU 上の FP32 のままです。**最終エネルギーは FP64 の仕上げから得られ**、FP32 段階の誤差は収束の経路にしか影響しません。
- hook を取り付ける間は pthreads 版 OpenBLAS を 1 スレッドに制限し（PySCF の OpenMP 領域内で過剰にスレッドを立てるため）、取り外すときに元に戻します。

## 使い方

```python
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.staging.schedule import run_staged

mol = build_mol('benchmarks/systems/drugs/caffeine.xyz', 'def2-tzvp')   # xyz のパス、または組み込みの系の名前
info = run_staged(mol, conv_tol=1e-9,
                  gpu_stages=(0, 1, 2), fp64_final=True,               # 本番構成
                  gpu_kwargs={'k': True, 'k_tile_tol': 1e-11},         # COSX K も GPU で計算
                  fp64_conv_tol=1e-7)                                  # 仕上げの |dE| 判定基準
print(info['e_tot'], info['converged'], info['cycles'])
```

コマンドライン（サイクルごとに段階、パス、\|dE\|、\|g\|、所要時間を表示します。`--e-ref` は既知の PySCF 本体のエネルギーで、gate の判定に使います）：

```bash
OMP_NUM_THREADS=16 python -B benchmarks/xc/staged_gpu.py --system <xyz> --basis def2-tzvp \
    --gpu-k --fp64-conv-tol 1e-7 --e-ref <本体のエネルギー>
```

- クラスタのノードでは先に CUDA モジュールを読み込んでください（`module load cuda/13.3`）。読み込んでいないと hook は警告を出すだけで CPU に戻ります。
- 新しい GPU での初回実行では CuPy が kernel をコンパイルするため、約 2.5 分余計にかかります。2 回目以降はキャッシュが効きます。先に `pyscf_wb97mv_fast.gpu.warmup.warmup()` を呼んでおけば事前にコンパイルできます。
- 低レベル API：`pyscf_wb97mv_fast.gpu.install.install_gpu(mf, k=True, ...)` で任意の RKS/COSX オブジェクトに GPU hook を取り付けられます。返り値の HookSet の `restore_all()` で取り外します。

## 検証（PySCF 本体との照合）

**エネルギー**（段階的 SCF + GPU K、RTX 4090）：

| 系 | 基底 | nao | 本体とのエネルギー差（Ha） |
|---|---|---:|---:|
| water27 | def2-SVP | 648 | +3.3e-8 |
| カフェイン | def2-TZVP | 494 | +6.2e-9 |
| イブプロフェン | def2-TZVP | 573 | +7.8e-9 |
| スルファメトキサゾール（S） | def2-TZVP | 599 | +5.5e-9 |
| ブロマゼパム（Br） | def2-TZVP | 666 | +5.2e-9 |
| クロルプロマジン（Cl、S） | def2-TZVP | 777 | +8.6e-9 |
| フルオキセチン（F） | def2-TZVP | 790 | +7.1e-9 |

すべて収束し、1e-4 の gate より 4 桁以上小さい値です。この差は、FP64 サイクル間の COSX エネルギーのゆらぎ（約 5e-9）と同じ程度です。CPU K のパスでも差はほぼ同じで、GPU K による上乗せは最大 1.1e-9 です。

**密度行列と勾配**：FP64 の参照値を CPU で一度だけ計算してファイルに保存し、各カードでは GPU 部分だけを実行して照合します。

- `benchmarks/xc/ref_dm_grad.py`：PySCF 本体を CPU FP64 で実行し、`benchmarks/results/ref_dm_grad/<分子>_def2-tzvp.npz` に保存します（エネルギー、密度行列、勾配、双極子、Mulliken 電荷、軌道）。
- `benchmarks/xc/check_dm_grad.py --ref <npz>`：そのカードで本番構成を実行し、密度・勾配・双極子・電荷を比較します。gate は \|dE\| < 1e-6、勾配の最大偏差 < 1e-5 Ha/bohr です。
- 勾配は PySCF 本体の `pyscf.sgx.grad` を使い、両側とも SGX グリッド応答項をオフにしています（PySCF 2.14 ではオンにすると勾配が NaN になる、上流のバグ）。
- 現状：水二量体のスモークテストは合格（勾配の最大偏差 8.1e-7 Ha/bohr、密度の最大偏差 4.4e-6）。5 つの医薬品分子の参照値は保存済みで、GPU 側の照合は未実行です。

## 性能（RTX 4090 + Ryzen 9 7950X3D 16C/32T）

| 系 | 基底 | 本体（CPU） | 段階的 SCF + GPU K | 高速化 |
|---|---|---:|---:|---:|
| water27 | def2-SVP | 1218 s | 130 s | 9.3× |
| カフェイン | def2-TZVP | 252 s | 71 s | 3.5× |
| スルファメトキサゾール | def2-TZVP | 367 s | 114 s | 3.2× |
| ブロマゼパム | def2-TZVP | 446 s | 134 s | 3.3× |
| イブプロフェン | def2-TZVP | 382 s | 109 s | 3.5× |
| フルオキセチン | def2-TZVP | 624 s | 172 s | 3.6× |
| クロルプロマジン | def2-TZVP | 742 s | 202 s | 3.7× |

本体は 32 スレッド、段階的 SCF は 16 スレッドで、どちらも `OPENBLAS_NUM_THREADS=1` です（それぞれ速い方の設定）。医薬品分子では、時間の 40% 前後が CPU 上の FP64 の仕上げに費やされています。

## テスト

```bash
python -m pytest                    # 既定のスイート：小さな系の正誤チェック、2080 Ti で約 95 秒
python -m pytest --slow             # 完全な SCF や water27 などの遅いテストを追加
python -m pytest --experimental     # 凍結済みの COSX 実験コードを追加
```

pytest は小さな系の単体チェックだけを行います。正しさの結論は、FP64 参照値と照合した実行結果（上記「検証」）に基づきます。

## 既知の問題と今後の作業

1. **full K の FP32 丸め誤差の下限**：FP32 の full K を 1 回計算したときのエネルギー誤差は、水二量体で約 1.9e-7、water27 で約 2.2e-6 です。原子中心のグリッド上で丸め誤差が同じ向きにそろうことが原因です。最終エネルギーは FP64 の仕上げで保証されるため影響はありません。対応するテストのしきい値は実測の下限に余裕を加えた値（水二量体 3e-7、water27 3e-6）に変更しましたが、新しいしきい値ではまだ実行していません。
2. **S1 段階がよく強制的に打ち切られる**：S1→S2 の判定は \|dE\| < 1e-6 だけを見るため、S1 で 8 サイクルの上限まで空回りすることが多く、段階的 SCF の総サイクル数は本体の約 2 倍になります。
3. **今後の性能改善**：FP64 の仕上げで K を全再構築する部分（FP64 の K をバックグラウンドで先に計算する）、VV10 の kernel。
4. **勾配**：本パッケージは独自の GPU 勾配を提供しません。勾配は収束した軌道を使って PySCF 本体で計算し、SGX グリッド応答項をオフにする必要があります（上流のバグ）。
5. `experimental/cosx`（K 側の戦略層）はすべての gate に不合格のため凍結済みで、メインラインからは import されません。

## ディレクトリ構成

```
pyscf_wb97mv_fast/
  core/        SGX の修正、取り外し可能な hook、OpenBLAS のスレッド制限、計時、テスト用の系
  reference/   CPU FP64 の参照 numint
  staging/     段階的 SCF（StagedSCF、run_staged）
  gpu/         FP32 GPU バックエンド：XC、VV10、COSX K（int3c1e、Boys、シェルペア、スクリーニング）
  experimental/cosx/   凍結済み
benchmarks/    ベンチマークと検証スクリプト（xc/staged_gpu.py、xc/ref_dm_grad.py、xc/check_dm_grad.py など）
benchmarks/systems/drugs/   6 つの医薬品分子の xyz
tests/         pytest（既定 / --slow / --experimental）
docs/          メインラインの設計ドキュメント（design.md / design.en.md / design.ja.md）
```

## 参考資料

| ファイル | 内容 |
|---|---|
| `docs/design.ja.md` | メインラインの設計ドキュメント：設計の理由、目標とハードウェア前提、段階的 SCF の方式、混合精度の誤差予算と受け入れ基準（判定基準はこれに従う。権威版は [中文](design.md)、[English](design.en.md) もあり） |
