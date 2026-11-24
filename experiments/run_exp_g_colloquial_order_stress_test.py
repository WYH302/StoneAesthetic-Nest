"""Experiment G: 30-Case Adversarial Colloquial Yunfu Workshop Order Compiler Stress Test.

Evaluates deterministic schema compilation accuracy, field-level exactness, formal
`validate_order_schema()` pass rate, and parsing latency across 30 challenging real-world
Yunfu natural-stone orders spanning 6 adversarial categories (5 cases each):
  1. Spoken Chinese Shorthand ("半米见方", "一米二乘六十公分", "长120宽60", "常规水刀")
  2. Mixed Metric Units & Noisy Punctuation (mm / cm / m / 公分 / 乘以 / * / X)
  3. Negation & Exclusion Clauses ("不要马赛克", "不用水刀只用桥切", "不要顺纹可自由旋转")
  4. Polymorphic Multi-Tier Assemblies (3x2 walls, 1x4 screens, hollow sink islands, L-bars, circles)
  5. Twin-Slab Bookmatching & Dynamic Remnant Capacity ("对开蝴蝶对花", dynamic_remnant_cap)
  6. Quantifiers, Nested Parentheses & Edge Constraints (single-clause qty+side triangles, kerf/margin)

Compares:
  - Naive Regex Baseline (`compile_naive_baseline`)
  - StoneVL-AestheticCompiler-v3 (`compile_chinese_order`)

Saves structured JSON audit to `experiments/results/exp_g_colloquial_order_stress_test.json`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

MASTER_DIR = Path(__file__).resolve().parents[1]
if str(MASTER_DIR) not in sys.path:
    sys.path.insert(0, str(MASTER_DIR))

from src.llm_compiler.order_compiler import compile_chinese_order, compile_naive_baseline

RESULTS_DIR = MASTER_DIR / "experiments" / "results"
OUT_JSON = RESULTS_DIR / "exp_g_colloquial_order_stress_test.json"


def _build_30_adversarial_cases() -> list[dict[str, Any]]:
    """30 curated Yunfu stone workshop colloquial orders with ground-truth AST assertions."""
    return [
        # ---- Category 1: Spoken Chinese Shorthand (5 cases) ----
        {
            "id": "C1_01",
            "category": "spoken_shorthand",
            "text": "切6块半米见方的主拼花，要落在黄金主纹带连纹对花，常规水刀。",
            "expect": {
                "n_items": 1,
                "first_shape": "square",
                "first_w": 500.0,
                "first_h": 500.0,
                "first_qty": 6,
                "first_tier": 1,
                "cut_mode": "waterjet",
                "kerf": 3.0,
            },
        },
        {
            "id": "C1_02",
            "category": "spoken_shorthand",
            "text": "要4块一米二乘六十公分的大板台面，全部顺纹，锯缝留3毫米。",
            "expect": {
                "n_items": 1,
                "first_shape": "rectangle",
                "first_w": 1200.0,
                "first_h": 600.0,
                "first_qty": 4,
                "first_grain_locked": True,
                "kerf": 3.0,
            },
        },
        {
            "id": "C1_03",
            "category": "spoken_shorthand",
            "text": "切八条一米八长二十公分宽的楼梯踏步，必须顺纹连排。",
            "expect": {
                "n_items": 1,
                "first_shape": "rectangle",
                "first_w": 1800.0,
                "first_h": 200.0,
                "first_qty": 8,
                "first_tier": 2,
                "first_grain_locked": True,
            },
        },
        {
            "id": "C1_04",
            "category": "spoken_shorthand",
            "text": "主件切两件长150宽60厘米的双联洗手台，要黄金带对花。",
            "expect": {
                "n_items": 1,
                "first_shape": "rectangle",
                "first_w": 1500.0,
                "first_h": 600.0,
                "first_qty": 2,
                "first_tier": 1,
                "first_grid": [1, 2],
            },
        },
        {
            "id": "C1_05",
            "category": "spoken_shorthand",
            "text": "切四块0.6米见方的田字格地拼，放主纹带对花，常规水刀。",
            "expect": {
                "n_items": 1,
                "first_shape": "square",
                "first_w": 600.0,
                "first_h": 600.0,
                "first_qty": 4,
                "first_tier": 1,
                "first_grid": [2, 2],
            },
        },
        # ---- Category 2: Mixed Metric Units & Noisy Punctuation (5 cases) ----
        {
            "id": "C2_01",
            "category": "mixed_units_noise",
            "text": "切 6 块 0.5m * 500mm 正方形主拼花（黄金主纹带），余料切一些边长18cm等边三角形。",
            "expect": {
                "n_items": 2,
                "first_shape": "square",
                "first_w": 500.0,
                "first_qty": 6,
                "second_shape": "equilateral_triangle",
                "second_w": 180.0,
            },
        },
        {
            "id": "C2_02",
            "category": "mixed_units_noise",
            "text": "主件：8条 800mm 乘以 20公分 飘窗板（顺纹连排）；锯缝留0.4厘米；边距留6mm。",
            "expect": {
                "n_items": 1,
                "first_shape": "rectangle",
                "first_w": 800.0,
                "first_h": 200.0,
                "first_qty": 8,
                "kerf": 4.0,
                "margin": 6.0,
            },
        },
        {
            "id": "C2_03",
            "category": "mixed_units_noise",
            "text": "切1块 1.8米 × 90厘米 带水槽中岛台面（内孔75x45cm），余料切小方砖15公分。",
            "expect": {
                "n_items": 2,
                "first_shape": "hollow_countertop",
                "first_w": 1800.0,
                "first_h": 900.0,
                "first_hole_w": 750.0,
                "first_hole_h": 450.0,
                "second_shape": "square",
                "second_w": 150.0,
            },
        },
        {
            "id": "C2_04",
            "category": "mixed_units_noise",
            "text": "做4片 45CM X 1.2m 四联屏风（核心件对花），留缝3.5毫米，安全边距8mm。",
            "expect": {
                "n_items": 1,
                "first_shape": "rectangle",
                "first_w": 450.0,
                "first_h": 1200.0,
                "first_qty": 4,
                "first_grid": [1, 4],
                "kerf": 3.5,
                "margin": 8.0,
            },
        },
        {
            "id": "C2_05",
            "category": "mixed_units_noise",
            "text": "切2件直径0.8米的圆茶几，再配6条1000mm*18cm门槛石（顺纹）。",
            "expect": {
                "n_items": 2,
                "first_shape": "circle",
                "first_radius": 400.0,
                "first_qty": 2,
                "second_shape": "rectangle",
                "second_w": 1000.0,
                "second_h": 180.0,
            },
        },
        # ---- Category 3: Negation & Exclusion Clauses (5 cases) ----
        {
            "id": "C3_01",
            "category": "negation_exclusion",
            "text": "切8条80x20厘米楼梯踏步（顺纹连排），不要马赛克，不用水刀，只用桥切，锯缝留3mm。",
            "expect": {
                "n_items": 1,
                "first_shape": "rectangle",
                "first_w": 800.0,
                "first_h": 200.0,
                "cut_mode": "guillotine",
            },
        },
        {
            "id": "C3_02",
            "category": "negation_exclusion",
            "text": "核心件切6块50x50cm背景墙主拼花（对花），别切三角形，不需要余料，纯桥切。",
            "expect": {
                "n_items": 1,
                "first_shape": "square",
                "first_w": 500.0,
                "first_qty": 6,
                "cut_mode": "guillotine",
            },
        },
        {
            "id": "C3_03",
            "category": "negation_exclusion",
            "text": "切10块40x30厘米小侧板，不要顺纹，可自由旋转，忽略裂纹，桥切。",
            "expect": {
                "n_items": 1,
                "first_shape": "rectangle",
                "first_w": 400.0,
                "first_h": 300.0,
                "first_grain_locked": False,
                "avoid_cracks": False,
                "cut_mode": "guillotine",
            },
        },
        {
            "id": "C3_04",
            "category": "negation_exclusion",
            "text": "全部顺纹：切4块60x60cm方板，另外切6块30x20cm挡板（不用顺纹，不限方向）。",
            "expect": {
                "n_items": 2,
                "first_shape": "square",
                "first_grain_locked": True,
                "second_shape": "rectangle",
                "second_grain_locked": False,
            },
        },
        {
            "id": "C3_05",
            "category": "negation_exclusion",
            "text": "切1块140x90cmL形转角吧台（核心件），取消六边形，取消马赛克。",
            "expect": {
                "n_items": 1,
                "first_shape": "l_shape",
                "first_w": 1400.0,
                "first_h": 900.0,
                "first_tier": 1,
            },
        },
        # ---- Category 4: Polymorphic Multi-Tier Assemblies A1-A6 (5 cases) ----
        {
            "id": "C4_01",
            "category": "polymorphic_a1_a6",
            "text": "核心件：6个50x50厘米正方形（茶几主拼花，落黄金主纹带连纹对花）；主件：8个20x80厘米楼梯踏步（顺纹连排）；余料套裁：尽量切等边三角形马赛克（边长18厘米），水刀加桥切。",
            "expect": {
                "n_items": 3,
                "first_shape": "square",
                "first_grid": [2, 3],
                "second_shape": "rectangle",
                "second_w": 800.0,
                "second_h": 200.0,
                "third_shape": "equilateral_triangle",
                "third_w": 180.0,
            },
        },
        {
            "id": "C4_02",
            "category": "polymorphic_a1_a6",
            "text": "核心件：4个45x110cm四联屏（黄金带对花）；次件：6条18x75cm护墙板条（顺纹连排）；边角料套裁：正六边形马赛克（半径10厘米）。",
            "expect": {
                "n_items": 3,
                "first_shape": "rectangle",
                "first_grid": [1, 4],
                "second_shape": "rectangle",
                "second_w": 750.0,
                "third_shape": "regular_hexagon",
                "third_radius": 100.0,
            },
        },
        {
            "id": "C4_03",
            "category": "polymorphic_a1_a6",
            "text": "核心件：1个160x90厘米带水槽中岛台面（内孔70x45cm）；次件：6条18x70cm挡水条（顺纹连排）；余料切一些12厘米茶盘垫杯垫。",
            "expect": {
                "n_items": 3,
                "first_shape": "hollow_countertop",
                "first_hole_w": 700.0,
                "second_shape": "rectangle",
                "third_shape": "square",
                "third_w": 120.0,
            },
        },
        {
            "id": "C4_04",
            "category": "polymorphic_a1_a6",
            "text": "核心件：1个150x100cmL型转角吧台（黄金带）；主件：6个20x75cm飘窗板（顺纹连排）；边角料切等边三角形（边长16cm）。",
            "expect": {
                "n_items": 3,
                "first_shape": "l_shape",
                "first_w": 1500.0,
                "first_h": 1000.0,
                "second_shape": "rectangle",
                "third_shape": "equilateral_triangle",
                "third_w": 160.0,
            },
        },
        {
            "id": "C4_05",
            "category": "polymorphic_a1_a6",
            "text": "切1个直径90厘米的圆形茶几，配4个45x45cm正方形田字格地拼（黄金带连纹对花），余料切六角砖（边长9cm）。",
            "expect": {
                "n_items": 3,
                "first_shape": "circle",
                "first_radius": 450.0,
                "second_shape": "square",
                "second_grid": [2, 2],
                "third_shape": "regular_hexagon",
                "third_radius": 90.0,
            },
        },
        # ---- Category 5: Twin-Slab Bookmatching & Dynamic Remnant Cap (5 cases) ----
        {
            "id": "C5_01",
            "category": "bookmatch_dynamic",
            "text": "双板镜像蝴蝶对花：切6块50x50cm背景墙主拼花（黄金带连纹对花），余料切等边三角形18cm。",
            "dynamic_remnant_cap": True,
            "expect": {
                "n_items": 2,
                "bookmatch": True,
                "first_shape": "square",
                "second_shape": "equilateral_triangle",
                "second_fill_to_cap": True,
            },
        },
        {
            "id": "C5_02",
            "category": "bookmatch_dynamic",
            "text": "做一套对开连纹电视墙：4块50x110cm四联屏（核心件对花），边角料尽量切六边形（半径11cm）。",
            "dynamic_remnant_cap": True,
            "expect": {
                "n_items": 2,
                "bookmatch": True,
                "first_shape": "rectangle",
                "first_grid": [1, 4],
                "second_shape": "regular_hexagon",
                "second_fill_to_cap": True,
            },
        },
        {
            "id": "C5_03",
            "category": "bookmatch_dynamic",
            "text": "客户要镜像对花（bookmatch）：2块90x60cm双盆洗手台（核心件），余料切15cm防滑小地砖。",
            "dynamic_remnant_cap": True,
            "expect": {
                "n_items": 2,
                "bookmatch": True,
                "first_shape": "rectangle",
                "first_grid": [1, 2],
                "second_shape": "square",
                "second_fill_to_cap": True,
            },
        },
        {
            "id": "C5_04",
            "category": "bookmatch_dynamic",
            "text": "单板排布（不做镜像）：切6个50x50cm主拼花（黄金带对花），余料切一些边长16cm等边三角形。",
            "dynamic_remnant_cap": True,
            "expect": {
                "n_items": 2,
                "bookmatch": False,
                "first_shape": "square",
                "second_fill_to_cap": True,
            },
        },
        {
            "id": "C5_05",
            "category": "bookmatch_dynamic",
            "text": "高定蝴蝶对花方案：4块55x55cm田字格主拼花（主纹带），6条20x80cm踏步（顺纹连排）。",
            "dynamic_remnant_cap": False,
            "expect": {
                "n_items": 2,
                "bookmatch": True,
                "first_shape": "square",
                "first_w": 550.0,
                "second_shape": "rectangle",
                "second_w": 800.0,
            },
        },
        # ---- Category 6: Quantifiers, Parentheses & Edge Constraints (5 cases) ----
        {
            "id": "C6_01",
            "category": "edge_quantifiers",
            "text": "切24块边长16cm等边三角形马赛克，水刀锯缝2.5mm，留边6毫米。",
            "expect": {
                "n_items": 1,
                "first_shape": "equilateral_triangle",
                "first_qty": 24,
                "first_w": 160.0,
                "kerf": 2.5,
                "margin": 6.0,
            },
        },
        {
            "id": "C6_02",
            "category": "edge_quantifiers",
            "text": "切18块半径12厘米正六边形六角砖，留缝3毫米。",
            "expect": {
                "n_items": 1,
                "first_shape": "regular_hexagon",
                "first_qty": 18,
                "first_radius": 120.0,
                "kerf": 3.0,
            },
        },
        {
            "id": "C6_03",
            "category": "edge_quantifiers",
            "text": "切十二条85x20厘米门槛石（顺纹连排，不可旋转，留缝4mm）。",
            "expect": {
                "n_items": 1,
                "first_shape": "rectangle",
                "first_qty": 12,
                "first_w": 850.0,
                "first_h": 200.0,
                "kerf": 4.0,
            },
        },
        {
            "id": "C6_04",
            "category": "edge_quantifiers",
            "text": "切两块1200x600mm台面，要顺纹；另切1个半径350mm圆盘（水刀）。",
            "expect": {
                "n_items": 2,
                "first_shape": "rectangle",
                "first_qty": 2,
                "first_grain_locked": True,
                "second_shape": "circle",
                "second_radius": 350.0,
            },
        },
        {
            "id": "C6_05",
            "category": "edge_quantifiers",
            "text": "切九块40x40厘米正方形，落在黄金主纹带对花，边距7mm，锯缝留3mm。",
            "expect": {
                "n_items": 1,
                "first_shape": "square",
                "first_qty": 9,
                "first_w": 400.0,
                "first_grid": [3, 3],
                "margin": 7.0,
                "kerf": 3.0,
            },
        },
    ]


def _check_case(parsed: dict[str, Any], expect: dict[str, Any]) -> tuple[bool, list[str]]:
    """Verify every ground-truth assertion against a compiled order schema."""
    errs: list[str] = []
    items = parsed.get("items", [])
    c = parsed.get("constraints", {})

    if "n_items" in expect and len(items) != expect["n_items"]:
        errs.append(f"n_items:{len(items)}!={expect['n_items']}")

    def _chk_item(idx: int, prefix: str) -> None:
        if idx >= len(items):
            if any(k.startswith(prefix) for k in expect):
                errs.append(f"missing_item_{idx}")
            return
        it = items[idx]
        if f"{prefix}shape" in expect and it.get("shape") != expect[f"{prefix}shape"]:
            errs.append(f"{prefix}shape:{it.get('shape')}!={expect[f'{prefix}shape']}")
        if f"{prefix}w" in expect and abs(float(it.get("width", -1.0)) - expect[f"{prefix}w"]) > 1.0:
            errs.append(f"{prefix}w:{it.get('width')}!={expect[f'{prefix}w']}")
        if f"{prefix}h" in expect and abs(float(it.get("height", -1.0)) - expect[f"{prefix}h"]) > 1.0:
            errs.append(f"{prefix}h:{it.get('height')}!={expect[f'{prefix}h']}")
        if f"{prefix}radius" in expect and abs(float(it.get("radius", -1.0)) - expect[f"{prefix}radius"]) > 1.0:
            errs.append(f"{prefix}radius:{it.get('radius')}!={expect[f'{prefix}radius']}")
        if f"{prefix}hole_w" in expect and abs(float(it.get("hole_width", -1.0)) - expect[f"{prefix}hole_w"]) > 1.0:
            errs.append(f"{prefix}hole_w:{it.get('hole_width')}!={expect[f'{prefix}hole_w']}")
        if f"{prefix}hole_h" in expect and abs(float(it.get("hole_height", -1.0)) - expect[f"{prefix}hole_h"]) > 1.0:
            errs.append(f"{prefix}hole_h:{it.get('hole_height')}!={expect[f'{prefix}hole_h']}")
        if f"{prefix}qty" in expect and int(it.get("quantity", -1)) != expect[f"{prefix}qty"]:
            errs.append(f"{prefix}qty:{it.get('quantity')}!={expect[f'{prefix}qty']}")
        if f"{prefix}tier" in expect and int(it.get("tier", -1)) != expect[f"{prefix}tier"]:
            errs.append(f"{prefix}tier:{it.get('tier')}!={expect[f'{prefix}tier']}")
        if f"{prefix}grain_locked" in expect and bool(it.get("grain_locked")) != expect[f"{prefix}grain_locked"]:
            errs.append(f"{prefix}grain_locked:{it.get('grain_locked')}!={expect[f'{prefix}grain_locked']}")
        if f"{prefix}grid" in expect and it.get("grid_layout") != expect[f"{prefix}grid"]:
            errs.append(f"{prefix}grid:{it.get('grid_layout')}!={expect[f'{prefix}grid']}")
        if f"{prefix}fill_to_cap" in expect and bool(it.get("fill_to_capacity", False)) != expect[f"{prefix}fill_to_cap"]:
            errs.append(f"{prefix}fill_to_cap:{it.get('fill_to_capacity')}!={expect[f'{prefix}fill_to_cap']}")

    _chk_item(0, "first_")
    _chk_item(1, "second_")
    _chk_item(2, "third_")

    if "cut_mode" in expect and c.get("cut_mode") != expect["cut_mode"]:
        errs.append(f"cut_mode:{c.get('cut_mode')}!={expect['cut_mode']}")
    if "kerf" in expect and abs(float(c.get("kerf", -1.0)) - expect["kerf"]) > 0.1:
        errs.append(f"kerf:{c.get('kerf')}!={expect['kerf']}")
    if "margin" in expect and abs(float(c.get("min_board_distance", -1.0)) - expect["margin"]) > 0.1:
        errs.append(f"margin:{c.get('min_board_distance')}!={expect['margin']}")
    if "avoid_cracks" in expect and bool(c.get("avoid_cracks", True)) != expect["avoid_cracks"]:
        errs.append(f"avoid_cracks:{c.get('avoid_cracks')}!={expect['avoid_cracks']}")
    if "bookmatch" in expect and bool(parsed.get("bookmatch_requested", False)) != expect["bookmatch"]:
        errs.append(f"bookmatch:{parsed.get('bookmatch_requested')}!={expect['bookmatch']}")

    return len(errs) == 0, errs


def main() -> None:
    cases = _build_30_adversarial_cases()
    print(f"=== Experiment G: 30-Case Adversarial Colloquial Yunfu Order Benchmark ===", flush=True)

    ours_pass = 0
    ours_schema_valid = 0
    naive_pass = 0
    cat_stats: dict[str, dict[str, int]] = {}
    ours_times_ms: list[float] = []
    details: list[dict[str, Any]] = []

    for case in cases:
        cat = case["category"]
        cat_stats.setdefault(cat, {"total": 0, "ours_pass": 0, "naive_pass": 0})
        cat_stats[cat]["total"] += 1

        t0 = time.perf_counter()
        parsed_ours = compile_chinese_order(
            case["text"],
            dynamic_remnant_cap=bool(case.get("dynamic_remnant_cap", False)),
        )
        ours_times_ms.append((time.perf_counter() - t0) * 1000.0)
        ok_ours, errs_ours = _check_case(parsed_ours, case["expect"])
        if parsed_ours.get("schema_valid", False):
            ours_schema_valid += 1
        if ok_ours:
            ours_pass += 1
            cat_stats[cat]["ours_pass"] += 1

        parsed_naive = compile_naive_baseline(case["text"])
        ok_naive, errs_naive = _check_case(parsed_naive, case["expect"])
        if ok_naive:
            naive_pass += 1
            cat_stats[cat]["naive_pass"] += 1

        details.append(
            {
                "id": case["id"],
                "category": cat,
                "ours_exact_match": ok_ours,
                "ours_schema_valid": bool(parsed_ours.get("schema_valid", False)),
                "ours_errors": errs_ours,
                "naive_exact_match": ok_naive,
                "naive_errors": errs_naive[:3],
            }
        )
        if not ok_ours:
            print(f"  [FAIL] {case['id']} ({cat}): {errs_ours}")

    n = len(cases)
    mean_ms = sum(ours_times_ms) / max(1, n)
    print(f"  Total Adversarial Orders     : {n}")
    print(f"  Naive Regex Baseline Exact   : {naive_pass}/{n} ({100.0 * naive_pass / n:.1f}%)")
    print(f"  StoneVL-Compiler-v3 Exact    : {ours_pass}/{n} ({100.0 * ours_pass / n:.1f}%)")
    print(f"  Formal Schema Valid Pass Rate: {ours_schema_valid}/{n} ({100.0 * ours_schema_valid / n:.1f}%)")
    print(f"  Mean Compilation Latency     : {mean_ms:.3f} ms/order")
    for cat, st in cat_stats.items():
        print(f"    - {cat:20s}: Ours {st['ours_pass']}/{st['total']} vs Naive {st['naive_pass']}/{st['total']}")

    payload = {
        "benchmark": "YunfuColloquialOrder-30",
        "total_cases": n,
        "naive_exact_match_count": naive_pass,
        "naive_exact_match_pct": round(100.0 * naive_pass / n, 2),
        "ours_exact_match_count": ours_pass,
        "ours_exact_match_pct": round(100.0 * ours_pass / n, 2),
        "ours_schema_valid_count": ours_schema_valid,
        "ours_schema_valid_pct": round(100.0 * ours_schema_valid / n, 2),
        "mean_compile_latency_ms": round(mean_ms, 4),
        "per_category": cat_stats,
        "cases": details,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Saved 30-case colloquial compiler audit -> {OUT_JSON}")


def test_yunfu_colloquial_order_30_cases() -> None:
    """Automated pytest regression suite verifying 30/30 adversarial Yunfu workshop orders."""
    for case in _build_30_adversarial_cases():
        parsed = compile_chinese_order(
            case["text"],
            dynamic_remnant_cap=bool(case.get("dynamic_remnant_cap", False)),
        )
        ok, errs = _check_case(parsed, case["expect"])
        assert ok, f"Order {case['id']} ({case['category']}) failed: {errs}"
        assert parsed.get("schema_valid", False), f"Order {case['id']} failed schema validation: {parsed.get('schema_errors')}"


if __name__ == "__main__":
    main()
