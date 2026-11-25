"""Experiment H: Stage-0 Pre-Cutting Slab Grading & Order-Aware Warehouse Inventory Selection.

Evaluates a realistic 12-slab Yunfu stone warehouse rack (all 12 real AmbientCG marble, onyx,
and travertine PBR varieties spanning A+/A/B/C structural & aesthetic grades) across all 6
polymorphic custom architectural order archetypes (A1-A6 -> 72 exact full-solver ground-truth
evaluations).

Compares four warehouse slab-selection policies:
  1. FIFO / Random Rack Pick (mean expected profit across all 12 slabs)
  2. Intrinsic-Only Grade Pick (highest order-independent Q_slab)
  3. Stage-0 Order-Aware SlabSelect (Top-1 fast pick & Top-3 shortlist + nest)
  4. Exhaustive 12-Slab Full-Nesting Oracle (theoretical upper bound)

Saves audit JSON -> `experiments/results/exp_h_stage0_slab_selection.json`
and publication figure -> `paper/figures/fig8_stage0_slab_selection.pdf/.png`.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from shapely.geometry import Polygon as ShapelyPolygon

MASTER_DIR = Path(__file__).resolve().parents[1]
if str(MASTER_DIR) not in sys.path:
    sys.path.insert(0, str(MASTER_DIR))

from experiments.run_polymorphic_native_sota_master import (
    EXP_FIG_DIR,
    POLYMORPHIC_ARCHETYPES,
    _imread_unicode,
    _save_fig_all,
)
from experiments.run_yunfu_aesthetic_benchmark_and_figures import (
    PAPER_FIG_DIR,
    RESULTS_DIR,
    build_yunfu_luxury_24_slabs,
)
from src.llm_compiler.order_compiler import compile_chinese_order, compile_to_solver_case
from src.solver_core.aesthetic_engine import (
    evaluate_aesthetic_nesting_metrics,
    solve_aesthetic_stone_nesting,
)
from src.solver_core.validator import validate_nesting_solution
from src.vision.slab_selector import grade_slab_intrinsic, rank_inventory_for_order
from src.vision.vision_parser import parse_slab_image

OUT_JSON = RESULTS_DIR / "exp_h_stage0_slab_selection.json"


def _make_rect_defect(x0: float, y0: float, x1: float, y1: float, did: str) -> dict[str, Any]:
    poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return {
        "id": did,
        "type": "crack",
        "raw_polygon_mm": poly,
        "polygon_mm": poly,
        "safety_margin_mm": 5.0,
        "area_mm2": round((x1 - x0) * (y1 - y0), 1),
        "confidence": 0.96,
    }


def _make_diag_crack(x0: float, y0: float, x1: float, y1: float, half_w: float, did: str) -> dict[str, Any]:
    dx, dy = x1 - x0, y1 - y0
    L = max(1.0, float(np.hypot(dx, dy)))
    nx, ny = -dy / L * half_w, dx / L * half_w
    poly = [
        [round(x0 + nx, 1), round(y0 + ny, 1)],
        [round(x1 + nx, 1), round(y1 + ny, 1)],
        [round(x1 - nx, 1), round(y1 - ny, 1)],
        [round(x0 - nx, 1), round(y0 - ny, 1)],
    ]
    shp = ShapelyPolygon(poly)
    return {
        "id": did,
        "type": "crack",
        "raw_polygon_mm": poly,
        "polygon_mm": poly,
        "safety_margin_mm": 5.0,
        "area_mm2": round(float(shp.area), 1),
        "confidence": 0.97,
    }


def build_12_slab_yunfu_warehouse() -> list[dict[str, Any]]:
    """Construct a realistic 12-slab Yunfu warehouse rack spanning all 12 real PBR varieties
    and realistic A+ / A / B / C structural & aesthetic profiles (3000x1800 mm each)."""
    raw_24 = build_yunfu_luxury_24_slabs()
    seen_var: set[str] = set()
    unique_12: list[dict[str, Any]] = []
    for item in raw_24:
        var_key = str(item.get("source_scan") or item.get("variety_label") or item.get("slab_id"))
        img_p = Path(item["image_path"])
        base_name = str(item.get("source_scan") or img_p.stem)
        if var_key in seen_var:
            continue
        seen_var.add(var_key)
        unique_12.append({"variety_key": var_key, "stone_name": base_name, "image_path": img_p})
        if len(unique_12) == 12:
            break

    # Realistic shop-floor defect scenarios across the 12-slab warehouse rack:
    #   Slabs 0..2: Pristine / tiny edge chip (A+ Flagship candidates)
    #   Slabs 3..6: Moderate off-center side cracks (A Prime candidates)
    #   Slabs 7..9: B-grade slabs (incl. 7 & 8 with compact central flaw absorbable by A3 sink cutout!)
    #   Slabs 10..11: C-grade severe structural cracks bisecting the central Golden Vein Zone
    defect_profiles: list[tuple[str, list[dict[str, Any]]]] = [
        ("Pristine (Edge-Free)", []),
        ("Tiny Far-Corner Chip", [_make_rect_defect(2810, 45, 2945, 130, "def_edge_1")]),
        ("Pristine (Edge-Free)", []),
        ("Upper-Right Hairline", [_make_diag_crack(2280, 1280, 2760, 1680, 26.0, "def_ur_1")]),
        ("Lower-Left Marginal Crack", [_make_diag_crack(110, 120, 560, 420, 28.0, "def_ll_1")]),
        ("Right-Margin Fissure", [_make_diag_crack(2480, 320, 2840, 960, 30.0, "def_rm_1")]),
        ("Top-Border Flaw", [_make_rect_defect(920, 1590, 1480, 1730, "def_tb_1")]),
        ("Central Compact Pit (Sink-Absorbable)", [_make_rect_defect(1340, 790, 1630, 960, "def_sink_1")]),
        ("Mid-Left Compact Fissure (Sink-Absorbable)", [_make_rect_defect(1120, 740, 1420, 930, "def_sink_2")]),
        ("Twin Off-Center Cracks", [
            _make_diag_crack(420, 680, 980, 1260, 32.0, "def_tw_1"),
            _make_diag_crack(2050, 260, 2580, 760, 32.0, "def_tw_2"),
        ]),
        ("Severe Central X-Fracture (C-Grade)", [
            _make_diag_crack(620, 340, 2380, 1480, 48.0, "def_cx_1"),
            _make_diag_crack(780, 1420, 2180, 380, 45.0, "def_cx_2"),
        ]),
        ("Severe Golden-Corridor Split (C-Grade)", [
            _make_diag_crack(480, 780, 2540, 1080, 52.0, "def_gs_1"),
            _make_diag_crack(1350, 220, 1680, 1580, 46.0, "def_gs_2"),
        ]),
    ]

    rack: list[dict[str, Any]] = []
    for idx, (entry, (prof_name, custom_defs)) in enumerate(zip(unique_12, defect_profiles), start=1):
        img_bgr = _imread_unicode(entry["image_path"])
        # Parse texture field cleanly on clean image, then inject exact realistic warehouse defect complex
        zero_mask = np.zeros((900, 1500), dtype=np.uint8)
        vis = parse_slab_image(
            image_input=img_bgr,
            width_mm=3000.0,
            height_mm=1800.0,
            corners_px=[[0.0, 0.0], [float(img_bgr.shape[1]), 0.0], [float(img_bgr.shape[1]), float(img_bgr.shape[0])], [0.0, float(img_bgr.shape[0])]],
            safety_margin_mm=5.0,
            external_crack_mask=zero_mask,
        )
        vis["defects"] = custom_defs
        grade_info = grade_slab_intrinsic(vis)
        sid = f"S{idx:02d}_{entry['stone_name'].split('_')[0]}"
        rack.append(
            {
                "slab_id": sid,
                "stone_name": entry["stone_name"],
                "defect_profile": prof_name,
                "visual_state": vis,
                "intrinsic": grade_info,
            }
        )
    return rack


def render_fig8_stage0_dashboard(
    rack: list[dict[str, Any]],
    exp_summary: dict[str, Any],
) -> None:
    """Render publication-grade 4-panel IEEE Figure 8: Stage-0 Warehouse Slab Selection Dashboard."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(15.5, 9.6), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.22, 1.0], hspace=0.30, wspace=0.22)

    # ---- Panel (a): 12-Slab Yunfu Warehouse Rack Grade Board (3x4 grid) ----
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title("(a) Stage-0 Intrinsic Slab Grading Board (12-Slab Yunfu Warehouse Rack)", fontsize=10.5, fontweight="bold", pad=8)
    ax_a.axis("off")

    sub_gs = gs[0, 0].subgridspec(3, 4, hspace=0.26, wspace=0.10)
    grade_colors = {"A+": "#059669", "A": "#2563eb", "B": "#d97706", "C": "#dc2626"}
    for i, item in enumerate(rack):
        r, c = divmod(i, 4)
        sax = fig.add_subplot(sub_gs[r, c])
        bgr = item["visual_state"]["rectified_bgr"]
        thumb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (240, 144), interpolation=cv2.INTER_AREA)
        sax.imshow(thumb)
        # Overlay buffered defects in bright crimson
        for d in item["visual_state"]["defects"]:
            pts = np.array(
                [[px / 3000.0 * 240.0, (1800.0 - py) / 1800.0 * 144.0] for px, py in d["polygon_mm"]],
                dtype=np.float32,
            )
            poly_patch = patches.Polygon(pts, closed=True, facecolor="#ef4444", edgecolor="#fee2e2", alpha=0.68, linewidth=1.0)
            sax.add_patch(poly_patch)

        g_lbl = item["intrinsic"]["grade"]
        sc_val = item["intrinsic"]["intrinsic_score_100"]
        g_col = grade_colors.get(g_lbl, "#475569")
        sax.text(
            6,
            18,
            f"{g_lbl} ({sc_val:.0f})",
            fontsize=7.2,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=g_col, edgecolor="none", alpha=0.92),
        )
        short_id = item["slab_id"].replace("_", " ")
        sax.set_xlabel(short_id, fontsize=6.8, labelpad=2, color="#1e293b")
        sax.set_xticks([])
        sax.set_yticks([])
        for sp in sax.spines.values():
            sp.set_edgecolor(g_col)
            sp.set_linewidth(1.5)

    # ---- Panel (b): Predicted vs. Exact Oracle Profit Scatter (72 pairs) ----
    ax_b = fig.add_subplot(gs[0, 1])
    pts_pred, pts_true, pts_col = [], [], []
    for rec in exp_summary["all_72_pairs"]:
        pts_pred.append(rec["stage0_pred_profit_cny"])
        pts_true.append(rec["oracle_exact_profit_cny"])
        pts_col.append(grade_colors.get(rec["intrinsic_grade"], "#475569"))

    ax_b.scatter(pts_pred, pts_true, c=pts_col, s=46, alpha=0.82, edgecolors="#0f172a", linewidths=0.5, zorder=3)
    # Trend line
    if len(pts_pred) >= 3:
        m_slope, b_int = np.polyfit(pts_pred, pts_true, 1)
        xs_line = np.linspace(min(pts_pred), max(pts_pred), 100)
        ax_b.plot(xs_line, m_slope * xs_line + b_int, "--", color="#0f172a", linewidth=1.4, alpha=0.85, label="Linear fit")
    # Legend handles for grades
    for g_k in ("A+", "A", "B", "C"):
        ax_b.scatter([], [], c=grade_colors[g_k], s=42, label=f"Grade {g_k}", edgecolors="#0f172a", linewidths=0.5)
    rho_all = exp_summary["overall_metrics"]["global_spearman_rho"]
    p_all = exp_summary["overall_metrics"]["global_spearman_p"]
    p_str = f"p < 10^{{-10}}" if p_all < 1e-10 else (f"p = {p_all:.2e}" if p_all < 1e-3 else f"p = {p_all:.4f}")
    ax_b.text(
        0.04,
        0.94,
        f"Global Spearman $\\rho = {rho_all:.3f}$\n($N = 72$ pairs, ${p_str}$)",
        transform=ax_b.transAxes,
        fontsize=8.6,
        fontweight="bold",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    ax_b.set_title("(b) Stage-0 Fast Predictor vs. Exact Full-Solver Profit", fontsize=10.5, fontweight="bold", pad=8)
    ax_b.set_xlabel("Stage-0 Predicted Net Profit (CNY)", fontsize=9.2)
    ax_b.set_ylabel("Exact Full-Nesting Net Profit (CNY)", fontsize=9.2)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.legend(loc="lower right", fontsize=7.8, framealpha=0.92, ncol=2)

    # ---- Panel (c): Warehouse Selection Policy Profit Across A1..A6 ----
    ax_c = fig.add_subplot(gs[1, 0])
    archs = [a["archetype_short"] for a in exp_summary["per_archetype"]]
    x = np.arange(len(archs))
    w = 0.17
    v_fifo = [a["fifo_random_profit_cny"] for a in exp_summary["per_archetype"]]
    v_intr = [a["intrinsic_only_profit_cny"] for a in exp_summary["per_archetype"]]
    v_top1 = [a["stage0_top1_profit_cny"] for a in exp_summary["per_archetype"]]
    v_top3 = [a["stage0_top3_profit_cny"] for a in exp_summary["per_archetype"]]
    v_orac = [a["oracle_best_profit_cny"] for a in exp_summary["per_archetype"]]

    ax_c.bar(x - 2 * w, v_fifo, w, label="FIFO / Random Rack", color="#94a3b8", edgecolor="#334155", linewidth=0.5)
    ax_c.bar(x - 1 * w, v_intr, w, label="Intrinsic-Only Grade", color="#60a5fa", edgecolor="#1e3a8a", linewidth=0.5)
    ax_c.bar(x + 0 * w, v_top1, w, label="Stage-0 Top-1 (Ours)", color="#10b981", edgecolor="#064e3b", linewidth=0.6)
    ax_c.bar(x + 1 * w, v_top3, w, label="Stage-0 Top-3 (Ours)", color="#059669", edgecolor="#064e3b", linewidth=0.6)
    ax_c.bar(x + 2 * w, v_orac, w, label="Exhaustive 12-Slab Oracle", color="#f59e0b", edgecolor="#78350f", linewidth=0.5)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(archs, fontsize=8.8, fontweight="bold")
    ax_c.set_ylabel("Realized Net Profit per Order (CNY)", fontsize=9.2)
    ax_c.set_title("(c) Net Commercial Profit by Warehouse Selection Policy (A1--A6)", fontsize=10.5, fontweight="bold", pad=8)
    ax_c.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_c.legend(loc="lower right", fontsize=7.6, ncol=2, framealpha=0.92)

    # ---- Panel (d): Screening Speed vs. Profit Capture Frontier ----
    ax_d = fig.add_subplot(gs[1, 1])
    om = exp_summary["overall_metrics"]
    policies = [
        ("FIFO / Random", om["mean_full_nest_sec_per_slab"], om["fifo_random_mean_cny"], "#64748b", "s", (1.8, -105)),
        ("Intrinsic-Only", om["intrinsic_total_sec"], om["intrinsic_only_mean_cny"], "#2563eb", "^", (1.8, -240)),
        ("Stage-0 Top-1 (Ours)", om["stage0_top1_total_sec"], om["stage0_top1_mean_cny"], "#10b981", "o", (3.2, -105)),
        ("Stage-0 Top-3 (Ours)", om["stage0_top3_total_sec"], om["stage0_top3_mean_cny"], "#047857", "D", (1.8, 125)),
        ("12-Slab Exhaustive Oracle", om["oracle_total_sec"], om["oracle_best_mean_cny"], "#d97706", "*", (-8.2, -280)),
    ]
    for lbl, t_sec, prof, col, mk, (dx, dy) in policies:
        ms = 175 if mk == "*" else 95
        ax_d.scatter(t_sec, prof, c=col, marker=mk, s=ms, edgecolors="#0f172a", linewidths=0.8, zorder=4, label=lbl)
        ax_d.annotate(
            f"{lbl}\n({t_sec:.1f}s | CNY {prof:,.0f})",
            (t_sec, prof),
            xytext=(t_sec + dx, prof + dy),
            fontsize=7.6,
            fontweight="bold" if "Ours" in lbl else "normal",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#e2e8f0", alpha=0.88),
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.85, shrinkA=2, shrinkB=4),
        )

    ax_d.set_title("(d) Warehouse Screening Runtime vs. Realized Net Profit", fontsize=10.5, fontweight="bold", pad=8)
    ax_d.set_xlabel("Total Decision + Nesting Wall-Clock Time per Order (s)", fontsize=9.2)
    ax_d.set_ylabel("Mean Realized Net Profit (CNY)", fontsize=9.2)
    ax_d.grid(True, linestyle="--", alpha=0.35)
    ax_d.set_xlim(0, max(31.0, om["oracle_total_sec"] * 1.15))
    ax_d.set_ylim(min(om["fifo_random_mean_cny"] - 350, 2600), om["oracle_best_mean_cny"] + 360)

    _save_fig_all(fig, "fig8_stage0_slab_selection")
    # Also ensure synced directly to top-level paper/figures/
    import shutil
    top_paper_fig = MASTER_DIR.parents[1] / "paper" / "figures"
    top_paper_fig.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        src_f = PAPER_FIG_DIR / f"fig8_stage0_slab_selection{ext}"
        if src_f.exists():
            shutil.copy2(src_f, top_paper_fig / src_f.name)


def main() -> None:
    print("=== Experiment H: Stage-0 Pre-Cutting Slab Grading & 12x6 Warehouse Selection ===", flush=True)
    t_all = time.perf_counter()
    rack = build_12_slab_yunfu_warehouse()
    print(f"Built 12-Slab Yunfu Warehouse Rack ({time.perf_counter() - t_all:.1f}s):")
    for r in rack:
        ig = r["intrinsic"]
        print(
            f"  {r['slab_id']:14s} | Grade={ig['grade']:2s} ({ig['intrinsic_score_100']:5.1f}) | "
            f"Q_str={ig['q_struct']:.2f} Q_vein={ig['q_vein']:.2f} Q_cln={ig['q_clean']:.2f} | "
            f"Rect={ig['rho_largest_clean_rect']:.2f} | {r['defect_profile']}"
        )

    slab_inv = [(r["slab_id"], r["visual_state"]) for r in rack]
    rack_by_id = {r["slab_id"]: r for r in rack}

    # Intrinsic-only top slab (fixed across orders since order-independent)
    intrinsic_ranked = sorted(rack, key=lambda r: r["intrinsic"]["intrinsic_score_100"], reverse=True)
    intrinsic_top1_id = intrinsic_ranked[0]["slab_id"]

    all_72_pairs: list[dict[str, Any]] = []
    per_archetype: list[dict[str, Any]] = []
    full_nest_times: list[float] = []
    stage0_screen_times_ms: list[float] = []
    c_grade_rejected = 0
    c_grade_total = 0
    cutout_absorptions = 0

    for arch_id, arch_title, order_text in POLYMORPHIC_ARCHETYPES:
        arch_short = arch_id.split("_")[0]
        order_state = compile_chinese_order(order_text)

        # 1. Run fast Stage-0 order-aware warehouse screening across all 12 slabs
        s0 = rank_inventory_for_order(slab_inv, order_state, slab_raw_cost_cny=4200.0)
        stage0_screen_times_ms.append(s0["total_screen_ms"])
        s0_by_id = {item["case_id"]: item for item in s0["ranked_slabs"]}

        # 2. Run EXACT full StoneAesthetic-Nest solver on ALL 12 slabs for true Oracle ground truth
        exact_by_id: dict[str, dict[str, Any]] = {}
        for r in rack:
            sid = r["slab_id"]
            vis = r["visual_state"]
            solver_case = compile_to_solver_case(case_id=f"{arch_short}_{sid}", visual_state=vis, order_state=order_state)
            t0 = time.perf_counter()
            res = solve_aesthetic_stone_nesting(solver_case, rectified_bgr=vis["rectified_bgr"])
            dt_full = time.perf_counter() - t0
            full_nest_times.append(dt_full)
            cert = validate_nesting_solution(solver_case, res)
            am = evaluate_aesthetic_nesting_metrics(solver_case, res, rectified_bgr=vis["rectified_bgr"])

            exact_profit = float(am["net_profit_cny"])
            s0_rec = s0_by_id[sid]
            exact_by_id[sid] = {
                "exact_profit_cny": exact_profit,
                "rac_feasible": bool(cert["feasible"]),
                "tier1_placed": am["tier1_placed"],
                "seam_pct": round(float(am["seam_vein_continuity"]) * 100.0, 2),
                "golden_pct": round(float(am["golden_zone_hit_rate"]) * 100.0, 2),
                "full_nest_sec": round(dt_full, 3),
            }

            if r["intrinsic"]["grade"] == "C":
                c_grade_total += 1
                if s0_rec["rank"] > 3 or not s0_rec["order_match"]["tier1_feasible"]:
                    c_grade_rejected += 1
            if s0_rec["order_match"]["defect_in_cutout_absorbed"]:
                cutout_absorptions += 1

            all_72_pairs.append(
                {
                    "archetype": arch_short,
                    "slab_id": sid,
                    "intrinsic_grade": r["intrinsic"]["grade"],
                    "intrinsic_score_100": r["intrinsic"]["intrinsic_score_100"],
                    "stage0_rank": s0_rec["rank"],
                    "stage0_pred_profit_cny": s0_rec["order_match"]["predicted_net_profit_cny"],
                    "stage0_t1_feasible": s0_rec["order_match"]["tier1_feasible"],
                    "defect_in_cutout_absorbed": s0_rec["order_match"]["defect_in_cutout_absorbed"],
                    "oracle_exact_profit_cny": exact_profit,
                    "oracle_rac_feasible": bool(cert["feasible"]),
                    "oracle_tier1_placed": am["tier1_placed"],
                }
            )

        # Compute policy realized profits for this archetype
        exact_profits_12 = [exact_by_id[r["slab_id"]]["exact_profit_cny"] for r in rack]
        pred_profits_12 = [s0_by_id[r["slab_id"]]["order_match"]["predicted_net_profit_cny"] for r in rack]
        rho_arch, _ = stats.spearmanr(pred_profits_12, exact_profits_12)

        fifo_profit = float(np.mean(exact_profits_12))
        intr_profit = exact_by_id[intrinsic_top1_id]["exact_profit_cny"]
        top1_id = s0["top1_slab_id"]
        top3_ids = s0["top3_slab_ids"]
        top1_profit = exact_by_id[top1_id]["exact_profit_cny"]
        top3_profit = max(exact_by_id[sid]["exact_profit_cny"] for sid in top3_ids)
        oracle_best_id = max(exact_by_id.keys(), key=lambda k: exact_by_id[k]["exact_profit_cny"])
        oracle_profit = exact_by_id[oracle_best_id]["exact_profit_cny"]

        per_archetype.append(
            {
                "archetype_short": arch_short,
                "archetype_title": arch_title,
                "stage0_screen_ms": s0["total_screen_ms"],
                "spearman_rho": round(float(rho_arch), 4),
                "fifo_random_profit_cny": round(fifo_profit, 2),
                "intrinsic_only_profit_cny": round(intr_profit, 2),
                "stage0_top1_slab": top1_id,
                "stage0_top1_profit_cny": round(top1_profit, 2),
                "stage0_top3_slabs": top3_ids,
                "stage0_top3_profit_cny": round(top3_profit, 2),
                "oracle_best_slab": oracle_best_id,
                "oracle_best_profit_cny": round(oracle_profit, 2),
                "top1_uplift_vs_fifo_cny": round(top1_profit - fifo_profit, 2),
                "top3_uplift_vs_fifo_cny": round(top3_profit - fifo_profit, 2),
                "top1_regret_vs_oracle_cny": round(oracle_profit - top1_profit, 2),
                "top3_regret_vs_oracle_cny": round(oracle_profit - top3_profit, 2),
            }
        )
        print(
            f"  [{arch_short}] Screen={s0['total_screen_ms']:6.1f}ms | rho={rho_arch:.3f} | "
            f"FIFO=CNY {fifo_profit:7.1f} | Intr=CNY {intr_profit:7.1f} | "
            f"S0-Top1({top1_id})=CNY {top1_profit:7.1f} | S0-Top3=CNY {top3_profit:7.1f} | "
            f"Oracle({oracle_best_id})=CNY {oracle_profit:7.1f}",
            flush=True,
        )

    # Aggregate global metrics across all 72 evaluations
    mean_full_s = float(np.mean(full_nest_times))
    mean_screen_s = float(np.mean(stage0_screen_times_ms)) / 1000.0
    all_pred = [p["stage0_pred_profit_cny"] for p in all_72_pairs]
    all_exact = [p["oracle_exact_profit_cny"] for p in all_72_pairs]
    g_rho, g_p = stats.spearmanr(all_pred, all_exact)

    fifo_mean = float(np.mean([a["fifo_random_profit_cny"] for a in per_archetype]))
    intr_mean = float(np.mean([a["intrinsic_only_profit_cny"] for a in per_archetype]))
    top1_mean = float(np.mean([a["stage0_top1_profit_cny"] for a in per_archetype]))
    top3_mean = float(np.mean([a["stage0_top3_profit_cny"] for a in per_archetype]))
    orac_mean = float(np.mean([a["oracle_best_profit_cny"] for a in per_archetype]))

    oracle_total_s = 12.0 * mean_full_s
    top1_total_s = mean_screen_s + 1.0 * mean_full_s
    top3_total_s = mean_screen_s + 3.0 * mean_full_s
    intr_total_s = 0.08 + 1.0 * mean_full_s

    summary = {
        "experiment": "Exp_H_Stage0_Slab_Grading_and_Warehouse_Selection",
        "warehouse_slabs": len(rack),
        "order_archetypes": len(POLYMORPHIC_ARCHETYPES),
        "total_exact_evaluations": len(all_72_pairs),
        "rack_summary": [
            {
                "slab_id": r["slab_id"],
                "stone_name": r["stone_name"],
                "defect_profile": r["defect_profile"],
                **r["intrinsic"],
            }
            for r in rack
        ],
        "overall_metrics": {
            "global_spearman_rho": round(float(g_rho), 4),
            "global_spearman_p": float(g_p),
            "mean_per_archetype_spearman_rho": round(float(np.mean([a["spearman_rho"] for a in per_archetype])), 4),
            "fifo_random_mean_cny": round(fifo_mean, 2),
            "intrinsic_only_mean_cny": round(intr_mean, 2),
            "stage0_top1_mean_cny": round(top1_mean, 2),
            "stage0_top3_mean_cny": round(top3_mean, 2),
            "oracle_best_mean_cny": round(orac_mean, 2),
            "top1_uplift_vs_fifo_cny": round(top1_mean - fifo_mean, 2),
            "top1_uplift_vs_fifo_pct": round(100.0 * (top1_mean - fifo_mean) / max(1.0, abs(fifo_mean)), 2),
            "top3_uplift_vs_fifo_cny": round(top3_mean - fifo_mean, 2),
            "top3_uplift_vs_fifo_pct": round(100.0 * (top3_mean - fifo_mean) / max(1.0, abs(fifo_mean)), 2),
            "top1_regret_vs_oracle_cny": round(orac_mean - top1_mean, 2),
            "top1_oracle_capture_pct": round(100.0 * top1_mean / max(1.0, orac_mean), 2),
            "top3_regret_vs_oracle_cny": round(orac_mean - top3_mean, 2),
            "top3_oracle_capture_pct": round(100.0 * top3_mean / max(1.0, orac_mean), 2),
            "c_grade_rejection_rate_pct": round(100.0 * c_grade_rejected / max(1, c_grade_total), 1),
            "defect_in_cutout_absorptions": int(cutout_absorptions),
            "mean_stage0_screen_ms_12slabs": round(mean_screen_s * 1000.0, 2),
            "mean_stage0_screen_ms_per_slab": round(mean_screen_s * 1000.0 / 12.0, 2),
            "mean_full_nest_sec_per_slab": round(mean_full_s, 3),
            "intrinsic_total_sec": round(intr_total_s, 2),
            "stage0_top1_total_sec": round(top1_total_s, 2),
            "stage0_top3_total_sec": round(top3_total_s, 2),
            "oracle_total_sec": round(oracle_total_s, 2),
            "top1_speedup_vs_oracle": round(oracle_total_s / max(1e-3, top1_total_s), 2),
            "top3_speedup_vs_oracle": round(oracle_total_s / max(1e-3, top3_total_s), 2),
            "all_72_rac_verified": all(p["oracle_rac_feasible"] for p in all_72_pairs),
        },
        "per_archetype": per_archetype,
        "all_72_pairs": all_72_pairs,
    }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    render_fig8_stage0_dashboard(rack, summary)

    om = summary["overall_metrics"]
    print("\n=== Experiment H Summary (12-Slab Rack x 6 Archetypes = 72 Exact Runs) ===")
    print(f"  Global Spearman rho           : {om['global_spearman_rho']:.4f} (p={om['global_spearman_p']:.2e})")
    print(f"  FIFO / Random Rack Mean Profit: CNY {om['fifo_random_mean_cny']:,.2f}")
    print(f"  Intrinsic-Only Grade Profit   : CNY {om['intrinsic_only_mean_cny']:,.2f}")
    print(f"  Stage-0 Top-1 (Ours) Profit   : CNY {om['stage0_top1_mean_cny']:,.2f} (+CNY {om['top1_uplift_vs_fifo_cny']:,.0f} / +{om['top1_uplift_vs_fifo_pct']:.1f}% vs FIFO, {om['top1_oracle_capture_pct']:.1f}% Oracle, {om['top1_speedup_vs_oracle']:.1f}x faster)")
    print(f"  Stage-0 Top-3 (Ours) Profit   : CNY {om['stage0_top3_mean_cny']:,.2f} (+CNY {om['top3_uplift_vs_fifo_cny']:,.0f} / +{om['top3_uplift_vs_fifo_pct']:.1f}% vs FIFO, {om['top3_oracle_capture_pct']:.1f}% Oracle, {om['top3_speedup_vs_oracle']:.1f}x faster)")
    print(f"  12-Slab Exhaustive Oracle     : CNY {om['oracle_best_mean_cny']:,.2f} ({om['oracle_total_sec']:.1f}s/order)")
    print(f"  C-Grade Rejection Rate        : {om['c_grade_rejection_rate_pct']:.1f}% | Defect-in-Cutout Hits: {om['defect_in_cutout_absorptions']}")
    print(f"  Stage-0 Screen Speed          : {om['mean_stage0_screen_ms_per_slab']:.1f} ms/slab ({om['mean_stage0_screen_ms_12slabs']:.1f} ms for 12 slabs)")
    print(f"  Saved JSON   -> {OUT_JSON}")
    print(f"  Saved Fig 8  -> {PAPER_FIG_DIR / 'fig8_stage0_slab_selection.pdf'}")


if __name__ == "__main__":
    main()
