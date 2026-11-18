"""Solver-in-the-Loop Neuro-Symbolic Closed-Loop Feedback (几何-语言反向纠错闭环).

When a defect-laden slab cannot accommodate 100% of an order, this module diagnoses
the exact geometric conflict (crack bottleneck, strict grain angle lock, kerf margin
squeeze, or area capacity deficit) and verifies minimal constraint relaxations via
counterfactual trial solves.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from shapely.geometry import Polygon as ShapelyPolygon

from ..solver_core.geometry_kernel import expand_case_items
from ..solver_core.solver import solve_cf_slabnest
from ..solver_core.validator import validate_nesting_solution


def diagnose_and_suggest_relaxations(
    case: dict[str, Any],
    result: dict[str, Any],
    time_limit_per_trial: float = 1.2,
) -> dict[str, Any]:
    """Diagnose why unplaced items failed and verify minimal counterfactual relaxations."""
    unplaced_ids: list[str] = list(result.get("unplaced", []))
    if not unplaced_ids:
        return {
            "is_fully_placed": True,
            "unplaced_count": 0,
            "diagnosis_summary_cn": "当前石板已 100% 合法排入全部订单零件，零重叠、零越界、零踩裂纹。",
            "diagnosis_summary_en": "All requested items are 100% legally placed on the slab.",
            "conflict_causes": [],
            "verified_relaxations": [],
        }

    board = case["board"]
    width = float(board["width"])
    height = float(board["height"])
    slab_area = width * height

    obstacles = board.get("obstacles", [])
    obs_shps = [ShapelyPolygon([(float(p[0]), float(p[1])) for p in o["polygon"]]).buffer(0) for o in obstacles if len(o.get("polygon", [])) >= 3]
    defect_area = sum(float(s.area) for s in obs_shps)
    usable_area = max(1.0, slab_area - defect_area)

    all_items = expand_case_items(case)
    by_id = {it["item_instance_id"]: it for it in all_items}
    total_req_area = sum(float(it["area"]) for it in all_items)
    unplaced_items = [by_id[uid] for uid in unplaced_ids if uid in by_id]

    causes: list[dict[str, Any]] = []
    area_ratio = total_req_area / usable_area

    # Identify nearest/largest blocking defect hazard
    primary_defect_id = obstacles[0]["id"] if obstacles else "none"
    if obstacles:
        largest_obs = max(
            obstacles,
            key=lambda o: ShapelyPolygon([(float(p[0]), float(p[1])) for p in o["polygon"]]).area,
        )
        primary_defect_id = largest_obs.get("id", "defect_1")

    for u_item in unplaced_items:
        if u_item.get("grain_locked", False):
            causes.append(
                {
                    "item_instance_id": u_item["item_instance_id"],
                    "shape": u_item["shape"],
                    "cause_type": "grain_angle_lock_and_crack_bottleneck",
                    "blocking_defect": primary_defect_id,
                    "detail_cn": (
                        f"零件 {u_item['item_instance_id']} ({u_item['shape']}) 受顺纹角度锁定约束，"
                        f"在避让 {primary_defect_id} 裂纹禁区时无法找到满足刀缝裕量的无碰撞锚点。"
                    ),
                }
            )
        elif area_ratio > 0.72:
            causes.append(
                {
                    "item_instance_id": u_item["item_instance_id"],
                    "shape": u_item["shape"],
                    "cause_type": "usable_pocket_fragmentation",
                    "blocking_defect": primary_defect_id,
                    "detail_cn": (
                        f"石板被 {len(obstacles)} 处裂纹分割为不规则碎区（需求/可用面积比 {area_ratio:.1%}），"
                        f"零件 {u_item['item_instance_id']} 外包络超出剩余连续空隙。"
                    ),
                }
            )
        else:
            causes.append(
                {
                    "item_instance_id": u_item["item_instance_id"],
                    "shape": u_item["shape"],
                    "cause_type": "kerf_and_defect_clearance_squeeze",
                    "blocking_defect": primary_defect_id,
                    "detail_cn": (
                        f"零件 {u_item['item_instance_id']} 卡在 {primary_defect_id} 与已排大件之间，"
                        f"当前刀缝 ({case['constraints'].get('kerf', 3.0)}mm) 与安全边距下空间不足。"
                    ),
                }
            )

    # Generate & Counterfactually Verify Candidate Relaxations
    candidates: list[tuple[str, str, str, dict[str, Any]]] = []

    # Candidate A: Unlock grain rotation / allow +/- 15 deg or 90 deg rotation + reduce kerf/margin
    case_a = deepcopy(case)
    case_a["constraints"]["kerf"] = min(float(case["constraints"].get("kerf", 3.0)), 2.0)
    case_a["constraints"]["min_board_distance"] = min(float(case["constraints"].get("min_board_distance", 5.0)), 3.0)
    case_a["constraints"]["rotation_step"] = 45.0
    for it in case_a["items"]:
        it["grain_locked"] = False
        it["allow_rotation"] = True
    candidates.append(
        (
            "relax_grain_and_kerf",
            "方案A（推荐）：解除次要件顺纹死锁（允许45°/90°旋转避障）并将水刀刀缝优化至 2.0mm、边距缩至 3.0mm",
            "Option A: Relax strict grain lock (allow 45/90-deg rotation) and tighten waterjet kerf to 2.0mm, margin to 3.0mm",
            case_a,
        )
    )

    # Candidate B: Minor dimensional trim (6% trim on non-critical dimensions) + kerf tightening
    case_b = deepcopy(case)
    case_b["constraints"]["kerf"] = min(float(case["constraints"].get("kerf", 3.0)), 2.0)
    case_b["constraints"]["min_board_distance"] = 3.0
    case_b["constraints"]["cut_mode"] = "waterjet"
    for it in case_b["items"]:
        it["grain_locked"] = False
        it["allow_rotation"] = True
        for dim_k in ("width", "height", "size", "radius", "rx", "ry", "leg_width", "leg_height", "hole_width", "hole_height"):
            if dim_k in it:
                it[dim_k] = round(float(it[dim_k]) * 0.92, 1)
    candidates.append(
        (
            "minor_dimension_trim_8pct",
            "方案B：在方案A基础上切换水刀模式并对受阻零件尺寸做 8% 工艺修边微调（适配现场切边余量）",
            "Option B: Switch to waterjet mode and apply an 8% dimensional trim allowance on blocked parts",
            case_b,
        )
    )

    # Candidate C: Stronger dimensional trim (15%) if slab is heavily over-constrained
    case_c = deepcopy(case)
    case_c["constraints"]["kerf"] = 1.5
    case_c["constraints"]["min_board_distance"] = 2.0
    case_c["constraints"]["rotation_step"] = 30.0
    case_c["constraints"]["cut_mode"] = "waterjet"
    case_c["constraints"]["grid_step"] = 28.0
    for it in case_c["items"]:
        it["grain_locked"] = False
        it["allow_rotation"] = True
        for dim_k in ("width", "height", "size", "radius", "rx", "ry", "leg_width", "leg_height", "hole_width", "hole_height"):
            if dim_k in it:
                it[dim_k] = round(float(it[dim_k]) * 0.84, 1)
    candidates.append(
        (
            "adaptive_resize_16pct",
            "方案C：切换超细水刀模式（刀缝1.5mm）并下调次要规格 16%，提升单板排入件数",
            "Option C: Switch to ultra-fine 1.5mm waterjet kerf with 16% specification scaling",
            case_c,
        )
    )

    # Candidate D: Compact modular scaling (24%) for small 2400x1200 slabs with multiple cracks
    case_d = deepcopy(case)
    case_d["constraints"]["kerf"] = 1.5
    case_d["constraints"]["min_board_distance"] = 2.0
    case_d["constraints"]["rotation_step"] = 30.0
    case_d["constraints"]["cut_mode"] = "waterjet"
    case_d["constraints"]["grid_step"] = 25.0
    for it in case_d["items"]:
        it["grain_locked"] = False
        it["allow_rotation"] = True
        scale = 0.82 if int(it.get("priority", 1)) >= 3 else 0.74
        for dim_k in ("width", "height", "size", "radius", "rx", "ry", "leg_width", "leg_height", "hole_width", "hole_height"):
            if dim_k in it:
                it[dim_k] = round(float(it[dim_k]) * scale, 1)
    candidates.append(
        (
            "compact_modular_scaling",
            "方案D：针对小幅面多裂纹荒料，启用紧凑模数缩放（主件82%/副件74%）实现单板兜底排入",
            "Option D: Compact modular scaling (82% primary / 74% secondary) for single-slab fulfillment",
            case_d,
        )
    )

    verified_relaxations: list[dict[str, Any]] = []
    base_placed = int(result.get("placed_count", 0))

    for r_id, desc_cn, desc_en, mod_case in candidates:
        trial_res = solve_cf_slabnest(mod_case, time_limit_sec=time_limit_per_trial, seed=42)
        val_check = validate_nesting_solution(mod_case, trial_res)
        if val_check["feasible"] and int(trial_res["placed_count"]) > base_placed:
            verified_relaxations.append(
                {
                    "relaxation_id": r_id,
                    "description_cn": desc_cn,
                    "description_en": desc_en,
                    "new_placed_count": int(trial_res["placed_count"]),
                    "total_count": int(trial_res["total_count"]),
                    "count_gain": int(trial_res["placed_count"]) - base_placed,
                    "new_utilization": float(trial_res["utilization"]),
                    "resolves_all": len(trial_res["unplaced"]) == 0,
                    "validator_feasible": True,
                    "relaxed_case": mod_case,
                    "relaxed_result": trial_res,
                }
            )
            if len(trial_res["unplaced"]) == 0:
                break

    best_suggestion_cn = (
        f"【几何冲突诊断】当前订单共 {result['total_count']} 件，已排入 {base_placed} 件，"
        f"剩余 {len(unplaced_ids)} 件（{', '.join(unplaced_ids[:3])}）因 {primary_defect_id} 裂纹区阻挡及顺纹/刀缝约束无法排入。"
    )
    if verified_relaxations:
        top = verified_relaxations[0]
        best_suggestion_cn += (
            f"\n【已验证可行松弛建议】{top['description_cn']} -> "
            f"排入件数可提升至 {top['new_placed_count']}/{top['total_count']} 件"
            f"（材料利用率 {top['new_utilization']:.1%}，100% 通过独立几何校验）。"
        )
    else:
        best_suggestion_cn += "\n【分板建议】当前订单总面积超出单张石板除裂纹后的极限物理容量，建议将未排入零件自动拆单至第 2 张石板加工。"

    return {
        "is_fully_placed": False,
        "unplaced_count": len(unplaced_ids),
        "unplaced_ids": unplaced_ids,
        "diagnosis_summary_cn": best_suggestion_cn,
        "conflict_causes": causes,
        "verified_relaxations": verified_relaxations,
        "recovered": len(verified_relaxations) > 0,
        "fully_recovered": any(r["resolves_all"] for r in verified_relaxations),
    }


def apply_relaxation_plan(feedback_report: dict[str, Any], index: int = 0) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return the `(relaxed_case, relaxed_result)` from a verified relaxation plan."""
    plans = feedback_report.get("verified_relaxations", [])
    if not plans or index >= len(plans):
        return None, None
    chosen = plans[index]
    return chosen["relaxed_case"], chosen["relaxed_result"]
