# StoneAesthetic-Nest

**Aesthetic-Driven Vein-Matching, Defect Avoidance, and Multi-Tier Remnant Nesting for Natural Stone Fabrication**  
**面向天然石材多变定制订单的顺纹对花、裂纹避让与多级余料套裁开源原型**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](requirements.txt)
[![CAD Output](https://img.shields.io/badge/CAM-AutoCAD_R2010_13--Layer_DXF-orange.svg)](#repository-structure--项目结构)

<p align="center">
  <img src="assets/framework_architecture.png" width="96%" alt="StoneAesthetic-Nest Framework Architecture"/>
</p>

---

## Overview | 项目简介

**English**  
In custom natural stone fabrication (marble, landscape onyx, and travertine), layout quality depends not only on raw sheet utilization, but also on **cross-joint vein continuity, natural grain alignment, structural crack avoidance, and interior cutout reuse**. Even full-strength geometric packing engines treat stock sheets as blank isotropic containers—often scattering multi-panel feature walls into mismatched patchworks or rotating elongated strips across natural mineral veins at identical raw area utilization.

`StoneAesthetic-Nest` is an open-source research prototype coupling anisotropic visual perception, polymorphic workshop order compilation, lightweight neural-combinatorial search, and certified industrial CAM toolpath generation:
1. **Anisotropic Slab Field & Topology-Preserving Crack Perception**: Combines automatic 4-corner ArUco (`DICT_4X4_50`) / dual-polarity homography calibration, a `96×60` integral-image ($O(1)$ query) structure-tensor local orientation field $\theta_{\text{flow}}(x,y)$ & Golden Vein Zone $\Omega_{\text{gold}}$, and dual-mode crack buffering (`Black-Hat Adaptive` zero-GPU CPU or `CrackSeg-MV3-clDice`, `3.17M` params, `72.5%` strict FG-IoU / `86.5%` ODS F1 / `18.4 ms` on RTX 3090).
2. **Polymorphic 3-Tier Workshop Order AST & Zero-Distortion Feedback**: Parses colloquial Chinese cutting descriptions (`cm`/`mm` normalization, spoken fractions, negation handling) across 6 real-world architectural archetypes (`3×2` feature walls, `1×4` landscape screens, hollow sink islands with interior *Part-in-Part* recovery, L-shaped bars, circular tables, `2×2` medallions) plus twin-slab mirrored **Bookmatching** (`96.1%` mirrored seam continuity) and zero-distortion **Multi-Slab Spillover** recovery (`100%` recovery without shrinking customer parts).
3. **Bi-Level Neural-Combinatorial Nesting (`STRtree` + Coarse-to-Fine)**: Uses an unfrozen 12-D PyTorch residual ranker (`AestheticVeinRankerMLP`, `3,085` params, `100.0%` OOF rank accuracy on unseen varieties) with Tier-1/Tier-2 coupled backtracking, `48→12 mm` + micro-angle refinement, and Shapely `STRtree` spatial indexing to anchor core assemblies inside $\Omega_{\text{gold}}$, align bridge-saw strips strictly along $\theta_{\text{flow}}$ (`0.0°` error), and tessellate peripheral remnants (`▲▼` honeycombs / hexagons / squares).
4. **Certifying Geometry & Industrial CAM Deliverables**: Verifies every polygon via an independent `RAC-Checker` (`100%` pass rate) and exports a **13-layer AutoCAD R2010 `.dxf`** featuring **4-corner GTSP 2-Opt rapid-air routing (`-73.6%` air travel)**, **collinear bridge-saw common-line merging (`-38.6%` segments, `2.63 m/slab` saved)**, **`100.0%` neighbor- & defect-safe waterjet pierce points**, and a **Client Dry-Lay Preview**.

**简体中文**  
在天然大理石、山水玉与洞石等定制加工场景中，排版的商业价值不仅取决于板材出材率，更取决于**核心件跨刀缝连纹对花、长条件顺纹一致、暗裂安全避让以及挖孔余料二次套裁**。即便是最顶尖的纯几何排样求解器，由于默认将大板视为均匀白板，往往在相同利用率下将多联背景墙打散成“阴阳脸”，或将踏步/飘窗旋转成横纹废品。

`StoneAesthetic-Nest` 是一个面向天然石材排版与数控刀路规划的轻量级开源原型：
1. **石板体检、裂纹拓扑分割与各向异性美学场**：支持 ArUco 四角标定纸（`DICT_4X4_50`）与双极性边缘透视矫正、`96×60` 积分图 $O(1)$ 局部纹向场 $\theta_{\text{flow}}(x,y)$ 及黄金主纹带 $\Omega_{\text{gold}}$，并提供零 GPU `Black-Hat Adaptive`（CPU `17.8 ms`）与 `CrackSeg-MV3-clDice`（RTX 3090 `18.4 ms` / `54.4 FPS`，`DeepCrack-237` 上 `72.5%` 严格前景 IoU / `86.5%` ODS F1）双后端裂纹检测。
2. **多变定制订单解析、镜面对纹与零缩尺跨板溢出**：支持口语化中文描述（如“半米见方”“一米二”“不要马赛克”等）与 6 大类常见定制订单，内置双板镜面对纹（Bookmatching，镜像连续度 `96.1%`）以及拒绝缩尺变型的**跨板溢出恢复闭环（100% 恢复）**。
3. **两级神经-组合套裁（空间索引 + 粗到精微调 + Tier-1/Tier-2 耦合回溯）**：采用未冻结 12 维轻量残差网络（`3,085` 参数，未知石材品种 5 折交叉验证排序准确率 `100.0%`），结合 `48→12 mm` 粗到精搜索与 `STRtree` 空间索引，锁死核心连纹、实现 `0.0°` 顺纹误差长条排布以及孔内/边角余料密铺。
4. **独立几何校核与工业级 CAM 刀路导出**：所有排样均经独立 `RAC-Checker` 100% 验收，一键导出 **13 层 AutoCAD R2010 `.dxf`**（含 **4 角点 GTSP 2-Opt 空程优化 `-73.6%`**、**桥切共线合并 `-38.6%` 刀段**、**100% 零碰撞穿孔安全点**）及 **客户干铺效果图**。

---

## Visual Showcase | 视觉效果示例

### 1. Value Inversion at Identical Raw Utilization (Official C++ `PackingSolver` vs. Ours) | 同利用率下的价值倒挂对比
<p align="center">
  <img src="assets/paradigm_teaser.png" width="96%" alt="Official C++ PackingSolver vs StoneAesthetic-Nest at Identical Raw Utilization"/>
</p>

### 2. Polymorphic Custom Orders: `1×4` Screen + Hexagons, Hollow Sink Island + Sink-Hole PiP Coasters, & L-Bar | 多变定制订单泛化示例
<p align="center">
  <img src="assets/case_study_gallery.png" width="96%" alt="Polymorphic Orders: 1x4 Screen, Hollow Sink Island with PiP Coasters, and L-Bar"/>
</p>

### 3. Sample Client Dry-Lay Sign-Off Preview (`A1` Feature Wall & `A3` Hollow Sink Island PiP) | 客户干铺预览示例
<p align="center">
  <img src="assets/dry_lay_showcase.png" width="88%" alt="Customer Dry-Lay Preview"/>
</p>

### 4. 36-Panel Multi-Archetype Dry-Lay Comparison Wall (`A1`–`A6`) | 6 大类多变订单干铺效果对比墙
<p align="center">
  <img src="assets/sota_dry_lay_wall.png" width="96%" alt="36-Panel Polymorphic Dry-Lay Comparison Matrix"/>
</p>

---

## Quickstart | 快速上手

### 1. Installation | 环境安装
```bash
pip install -r requirements.txt
```

### 2. Run the 4-Stage Demo | 运行端到端示例
```bash
python quickstart_demo.py
```
Running `quickstart_demo.py` processes `examples/sample_slabs/Onyx011_Jade_Onyx.jpg` and writes the deliverables to `outputs_demo/`:
- `demo_jade_onyx_overlay.png` — Slab cutting layout overlay with Golden Vein Zone & numbered parts
- `demo_jade_onyx_dry_lay_preview.png` — Assembled feature panel + directional strips + offcut mosaic strip
- `demo_jade_onyx_cutting.dxf` — 13-layer AutoCAD R2010 `.dxf` with GTSP air-travel optimization, collinear bridge-saw common-line merging, and collision-free waterjet pierce points

### 3. Real Workshop Slab Photos (ArUco 4-Corner Calibration Kit) | 真实工厂大板实拍模式
Print `real_yunfu_slabs/ArUco_4Corners_A4_Print.pdf` (or capture a clean overhead slab photo), drop your photos into `real_yunfu_slabs/photos/`, and run:
```bash
# Quick end-to-end self-check on a synthetic tilted ArUco workshop photo:
python experiments/run_real_yunfu_slab_case_study.py --smoke-test

# Process your real workshop slab photos (auto-uses GPU CrackSeg-MV3-clDice or CPU Black-Hat):
python experiments/run_real_yunfu_slab_case_study.py --crack-backend auto
```

---

## Reference Evaluation Summary | 实验参考指标

Evaluated across `YunfuPoly-24` (`24` uncut `3000×1800 mm` CC0 marble, onyx, and travertine slabs spanning `6` polymorphic architectural order archetypes `A1`–`A6`, `100%` `RAC-Checker` verified, `3.67 s/slab` clean CPU; fold-isolated 5-fold OOF CV: `100.0%` rank accuracy, `87.2%` seam, `96.7%` Golden hit, `CNY 3,750`). External C++ and Rust baselines execute unmodified official compiled binaries (`EURO/ESICUP-14`: `70.4%` `Jagua-rs LBF`, `79.4%` at `2 s` / `84.0%` at `60 s` `Sparrow v0.3`; `Baldacci-6`: `70.4%` `PackingSolver`, `65.9%` our geometric core):

| Method / 方法 | Core Seam Continuity | Golden-Zone Hit | Strip Vein Error | Offcut Pieces | Net Utilization | Net Profit (CNY) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `BL-Heuristic` | 53.3% | 24.0% | 30.5° | 4.5 | 44.2% | 1,011 |
| `PackingSolver` (Official C++) | 43.4% | 22.2% | 44.6° | 35.3 | **57.0%** | 1,631 |
| `Jagua-rs LBF` (Official Rust) | 34.2% | 31.4% | 72.5° | 35.3 | **57.0%** | 1,409 |
| `Sparrow v0.3` (Official Rust) | 33.0% | 18.9% | 55.1° | 35.3 | **57.0%** | 1,272 |
| `RL-PPO-Adapted` | 52.6% | 21.2% | 25.1° | 13.5 | 48.6% | 1,273 |
| `GFPack++-Adapted` | 52.6% | 21.2% | 22.0° | 13.5 | 48.5% | 1,354 |
| **StoneAesthetic-Nest (Ours)** | **87.7%** | **96.4%** | **0.0°** | **35.3** | **57.0%** | **3,759** |

**Dual-Protocol `DeepCrack` Crack Segmentation Audit (`RTX 3090` & `CPU`)**:
- *Protocol 1 — Strict Scene-Disjoint Official Split (`300` train / `237` unseen test, `8.0%` parent-image prefix overlap)*:
  - `Black-Hat Adaptive` (Ours, zero-GPU CPU): `0.00M` params, `17.8 ms` (`56.3 FPS`), `54.8%` FG-IoU, `67.5%` Strict F1.
  - `CrackSeg-MV3-clDice` (Ours, RTX 3090): `3.17M` params, **`5.27 ms` (`189.8 FPS`, `306.5 MiB` VRAM)** in FP16 + `torch.cuda.CUDAGraph`, `72.5%` strict FG-IoU, `82.8%` Strict F1, `85.4%` Global F1.
  - **`CrackSeg-CNX-SDM` (Ours, Zero-BN, RTX 3090)**: `12.59M` params (truncated `ConvNeXt-Tiny` Stages 1–3 + `k=13` `SDM` gated directional fusion + `GroupNorm` `MSF` + `soft-clDice`), **`13.4 ms` (`74.6 FPS`, `346.8 MiB` VRAM)**, **`72.8%` strict FG-IoU**, **`83.1%` Strict F1**, **`87.0%` OIS**.
- *Protocol 2 — Apples-to-Apples 2023–2026 Literature Split (`7:1:2 @ 512×512`, `85.0%` parent-image overlap, exact `SCSegamba` `eval/evaluate.py`)*:
  - `CrackSeg-CNX-SDM` jumps to **`79.3%` strict FG-IoU**, **`95.4%` safety recall**, **`88.0%` ODS**, **`88.8%` OIS**, **`89.3%` per-image mIoU** (**`91.3%` global 2-class mIoU**), and **`90.9%` F1** — outperforming `DTrCNet` (`84.7%` ODS), `SimCrack` (`85.7%`), and `Crackmer` (`87.1%`), and matching CVPR 2025 `SCSegamba` (`91.1%` F1) at **`2.3×` lower latency (`13.4 ms` vs. `31.3 ms`)** with zero custom CUDA Mamba/RWKV kernels.

**Industrial CAM, Language Compiler & Robustness Highlights**:
- **30-Case Adversarial Colloquial Order Stress Test (`YunfuColloquialOrder-30`)**: **`30/30` (`100.0%`)** exact AST match and **`30/30` (`100.0%`)** formal schema validation at `1.05 ms/order` CPU (vs. `0/30` for unconstrained regex) across spoken Chinese dimensions, mixed `m/cm/mm` units, negation clauses, polymorphic assemblies, bookmatching, and edge quantifiers.
- **4-Corner GTSP 2-Opt Routing**: Rapid air-travel reduced from `29.76 m → 7.14 m` (**`-73.6%`**), with **`100.0%`** neighbor- and defect-safe waterjet pierce points.
- **Bridge-Saw Common-Line Merging**: Collinear shared cuts merged from `34.7 → 21.3` segments (**`-38.6%`**, saving `2.63 m/slab`).
- **Fair 6-Channel Context CNN Audit**: Even when given the exact Golden-Zone, saliency, and alignment channels, `MobileNetV3-S` (`53.7%`) and `ResNet-18` (`46.8%`) overfit to training stone RGB on unseen varieties, while our unfrozen 12-D physics MLP reaches **`100.0 ± 0.0%`** (`1.68M` candidates/s on CPU).
- **Genuine Hyperparameter & Pricing Invariance**: Sweeping $d_{\text{safe}} \in [2,12]\text{ mm}$ (`< 0.2%` profit drift), $\gamma_{\text{look}} \in [0,5]$ (`< 1.1%` profit drift), 5 unfrozen seeds (`CV = 0.26%`), scalability up to `160` parts/slab (`2.33–2.58 s` flat CPU), and `2.05–3.49×` profit lead across Mid-Tier Carrara, Standard Yunfu, and Ultra-Luxury Onyx pricing regimes.

---

## Repository Structure | 项目结构

```text
StoneAesthetic-Nest/
├── assets/                     # Architecture, teaser, polymorphic gallery, ablation & dry-lay figures
├── checkpoints/                # Pretrained 12-D AestheticVeinRankerMLP (16 KB) & CrackSeg-MV3-clDice (12.9 MB)
├── examples/
│   ├── sample_slabs/           # CC0 seamless raw onyx & travertine textures (AmbientCG)
│   └── sample_outputs/         # Reference 13-layer .DXF & Dry-Lay outputs (A1 Wall + A3 Sink-Hole PiP)
├── experiments/                # Reproducible benchmark, ablation, RTX 3090 crack & real-slab case-study scripts
│   └── results/                # JSON audit logs for YunfuPoly-24, Exp D, Exp E (DeepCrack), Exp F & ESICUP-14
├── real_yunfu_slabs/           # Printable A4 ArUco 4-corner calibration PDF & workshop photo manifest template
├── src/
│   ├── vision/                 # ArUco/dual-polarity homography, crack buffering, 96x60 integral texture & flow field
│   ├── llm_compiler/           # Colloquial polymorphic 3-tier order AST compiler & zero-distortion spillover loop
│   ├── solver_core/            # STRtree spatial index, coarse-to-fine coupled nesting, bookmatching & RAC-Checker
│   └── ui_dxf/                 # 13-layer GTSP/common-line AutoCAD R2010 DXF exporter, Dry-Lay renderer & Web API
├── quickstart_demo.py          # One-command end-to-end demonstration script
├── requirements.txt            # Python package dependencies
└── LICENSE                     # MIT Open-Source License
```

---

## Acknowledgments & Data Credits | 致谢与数据来源

- Raw uncut marble, onyx, and travertine PBR scans in `examples/sample_slabs/` originate from [AmbientCG](https://ambientcg.com/) under the **CC0 1.0 Universal Public Domain Dedication**.
- External official geometric solvers evaluated for fair reference include [PackingSolver](https://github.com/fontanf/packingsolver), [jagua-rs](https://github.com/JeroenGar/jagua-rs), and [Sparrow](https://github.com/JeroenGar/sparrow), along with the [EURO ESICUP](https://www.euro-online.org/websites/esicup/), Baldacci defective-sheet, and [DeepCrack](https://github.com/yhlleo/DeepCrack) community benchmarks.

---

## Citation | 引用格式

If you find this open-source prototype, polymorphic stone benchmark, or real-slab ArUco calibration kit helpful, please consider citing:

```bibtex
@article{wu2026stoneaesthetic,
  author  = {Wu, Yonghao},
  title   = {StoneAesthetic-Nest: Beyond Area Maximization---Anisotropic Vein-Continuous and Multi-Tier Nesting for Natural Stone CAD/CAM},
  journal = {Preprint submitted to The International Journal of Advanced Manufacturing Technology},
  year    = {2026}
}
```

## License | 开源协议

Released under the [MIT License](LICENSE).
