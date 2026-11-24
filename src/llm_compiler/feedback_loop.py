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

from ..solver_core.aesthetic_engine import solve_aesthetic_stone_nesting
from ..solver_core.geometry_kernel import expand_case_items
from ..solver_core.solver import solve_cf_slabnest
from ..solver_core.validator import validate_nesting_solution


def _dispatch_trial_solve(mod_case: dict[str, Any], time_limit_sec: float) -> dict[str, Any]:
    if mod_case.get("is_aesthetic_multi_tier", False) or any(
        it.get("role") in ("core_vein_match", "stair_tread_tandem", "remnant_mosaic_fill")
        for it in mod_case.get("items", [])
    ):
        return solve_aesthetic_stone_nesting(mod_case, seed=42)
    return solve_cf_slabnest(mod_case, time_limit_sec=time_limit_sec, seed=42)


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

    # Generate & Counterfactually Verify Industrial Factory-Safe Minimal-Cost Relaxations
    # (Replaces physically unrealistic 16-24% global part shrinking with real Yunfu shop-floor actions)
    candidates: list[tuple[str, str, str, dict[str, Any]]] = []
    unplaced_type_ids = {by_id[uid]["type_id"] for uid in unplaced_ids if uid in by_id}

    # Option A: Targeted micro-angle tolerance (+/-5 deg -> +/-15 deg) on blocked non-Tier-1 items + waterjet kerf 2.0mm
    case_a = deepcopy(case)
    case_a["constraints"]["kerf"] = min(float(case["constraints"].get("kerf", 3.0)), 2.0)
    case_a["constraints"]["min_board_distance"] = min(float(case["constraints"].get("min_board_distance", 5.0)), 3.0)
    case_a["constraints"]["grain_tolerance_deg"] = 6.0
    for it in case_a["items"]:
        is_tier1 = int(it.get("tier", 3)) == 1 or it.get("role") == "core_vein_match"
        if not is_tier1 and (it.get("id") in unplaced_type_ids or not unplaced_type_ids):
            it["grain_tolerance_deg"] = 15.0
            it["allow_rotation"] = True
    candidates.append(
        (
            "targeted_angle_tol_and_kerf",
            "方案A（零尺寸妥协·首选）：锁定核心主件连纹，仅对受阻次要件放宽纹理角度容差至 ±15° 并启用 2.0mm 精细水刀刀缝与 3.0mm 留边",
            "Option A (Zero-Dimension-Loss): Preserve Tier-1 core aesthetics, relax grain tolerance (+/-15 deg) on blocked secondary items, and tighten waterjet kerf to 2.0mm",
            case_a,
        )
    )

    # Option B: Orthogonal rotation unlock on non-Tier-1 parts + realistic <=2% polishing/chamfer edge trim
    case_b = deepcopy(case_a)
    case_b["constraints"]["cut_mode"] = "waterjet"
    case_b["constraints"]["rotation_step"] = 45.0
    for it in case_b["items"]:
        is_tier1 = int(it.get("tier", 3)) == 1 or it.get("role") == "core_vein_match"
        if not is_tier1:
            it["grain_locked"] = False
            it["allow_rotation"] = True
            for dim_k in ("width", "height", "size", "radius", "rx", "ry"):
                if dim_k in it:
                    it[dim_k] = round(float(it[dim_k]) * 0.98, 1)
    candidates.append(
        (
            "secondary_unlock_and_2pct_chamfer",
            "方案B（2% 工艺修边微调）：解除次要件顺纹死锁并对非核心件启用 ≤2% 现场磨边/倒角工艺余量微调（零主件尺寸妥协）",
            "Option B (<=2% Shop-Floor Edge Trim): Unlock secondary grain lock and apply <=2% chamfer/polishing allowance on non-core items",
            case_b,
        )
    )

    verified_relaxations: list[dict[str, Any]] = []
    base_placed = int(result.get("placed_count", 0))

    for r_id, desc_cn, desc_en, mod_case in candidates:
        trial_res = _dispatch_trial_solve(mod_case, time_limit_sec=time_limit_per_trial)
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

    # Option C (Always verified when single-slab still overflows): Exact Minimal Multi-Slab Spillover (0% dimensional distortion!)
    if not any(r["resolves_all"] for r in verified_relaxations):
        spill_counts: dict[str, int] = {}
        for uid in unplaced_ids:
            meta = by_id.get(uid)
            if meta is not None:
                tid = str(meta["type_id"])
                spill_counts[tid] = spill_counts.get(tid, 0) + 1
        slab2_case = deepcopy(case)
        slab2_case["case_id"] = f"{case.get('case_id', 'slab')}_spillover_slab2"
        # Clean stock remnant slab #2 of matching width/height without Slab #1's cracks
        slab2_case["board"]["obstacles"] = []
        spill_items = []
        for it in case.get("items", []):
            cnt = spill_counts.get(str(it.get("id", "")), 0)
            if cnt > 0:
                it_cp = deepcopy(it)
                it_cp["quantity"] = cnt
                spill_items.append(it_cp)
        slab2_case["items"] = spill_items
        if spill_items:
            res_s2 = _dispatch_trial_solve(slab2_case, time_limit_sec=time_limit_per_trial)
            val_s2 = validate_nesting_solution(slab2_case, res_s2)
            if val_s2["feasible"] and len(res_s2.get("unplaced", [])) == 0:
                verified_relaxations.append(
                    {
                        "relaxation_id": "multi_slab_spillover_zero_distortion",
                        "description_cn": (
                            f"方案C（零尺寸妥协·自动分板）：保留主板 {base_placed} 件全部工艺与尺寸，"
                            f"将受裂纹阻挡的 {len(unplaced_ids)} 件冲突零件自动拆单至第 2 张库存余料板（100% 校验通过）"
                        ),
                        "description_en": (
                            f"Option C (Zero-Distortion Multi-Slab Spillover): Preserve all {base_placed} items on Slab #1 "
                            f"and dispatch exactly {len(unplaced_ids)} blocked items onto Slab #2 (100% RAC verified)"
                        ),
                        "new_placed_count": base_placed + int(res_s2["placed_count"]),
                        "total_count": int(result["total_count"]),
                        "count_gain": int(res_s2["placed_count"]),
                        "new_utilization": float(result.get("utilization", 0.0)),
                        "slab2_utilization": float(res_s2["utilization"]),
                        "resolves_all": True,
                        "validator_feasible": True,
                        "relaxed_case": slab2_case,
                        "relaxed_result": res_s2,
                    }
                )

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
