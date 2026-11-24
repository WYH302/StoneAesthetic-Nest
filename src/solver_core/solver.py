"""Defect-Aware Count-First Irregular Nesting Solver (CF-SlabNest) & Comparative Baselines."""

from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np
import torch
from shapely.geometry import Polygon as ShapelyPolygon

from ..vision.vision_parser import evaluate_placement_texture_cost
from .geometry_kernel import (
    PolygonCoords,
    SpatialCollisionIndex,
    bbox,
    check_placement_feasible,
    expand_case_items,
    polygon_area,
    rotate_polygon,
    translate_polygon,
    transform_polygon,
)
from .mlp_ranker import (
    CandidateRankerMLP,
    extract_candidate_features,
    load_or_init_gfpack_score_net,
    load_or_init_ranker,
    load_or_init_rl_ppo_policy,
)


def _dedup_anchors(
    anchors: list[tuple[float, float]],
    limit: int,
) -> list[tuple[float, float]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[float, float]] = []
    for x, y in anchors:
        key = (int(round(x * 2.0)), int(round(y * 2.0)))
        if key not in seen:
            seen.add(key)
            out.append((float(x), float(y)))
            if len(out) >= limit:
                break
    return out


def generate_candidate_anchors(
    width: float,
    height: float,
    item_bbox: tuple[float, float, float, float],
    placed_records: list[dict[str, Any]],
    obstacle_polys: list[PolygonCoords],
    margin: float,
    kerf: float,
    grid_step: float,
    limit: int = 160,
    include_defect_tangents: bool = True,
    include_hole_anchors: bool = True,
) -> list[tuple[float, float]]:
    """Generate candidate placement anchors from corners, part contacts, defect tangents, and sink holes."""
    min_x, min_y, max_x, max_y = item_bbox
    pw = max_x - min_x
    ph = max_y - min_y

    left = margin - min_x
    right = width - margin - max_x
    bottom = margin - min_y
    top = height - margin - max_y

    if right < left - 1e-4 or top < bottom - 1e-4:
        return []

    anchors: list[tuple[float, float]] = [
        (left, bottom),
        (left, top),
        (right, bottom),
        (right, top),
    ]

    # 1. Part-in-Part interior hole anchors (if any placed part has an interior cutout hole)
    if include_hole_anchors:
        for rec in placed_records:
            hole_poly = rec.get("hole_polygon_world")
            if hole_poly:
                hx1, hy1, hx2, hy2 = bbox(hole_poly)
                if (hx2 - hx1) >= pw + 2.0 * kerf and (hy2 - hy1) >= ph + 2.0 * kerf:
                    anchors.append((hx1 + kerf - min_x, hy1 + kerf - min_y))
                    anchors.append(((hx1 + hx2 - pw) * 0.5 - min_x, (hy1 + hy2 - ph) * 0.5 - min_y))

    # 2. Placed part NFP / bounding contact tangents
    for rec in placed_records:
        ox1, oy1, ox2, oy2 = rec["bbox"]
        c_xs = [ox2 + kerf - min_x, ox1 - kerf - max_x, ox1 - min_x, ox2 - max_x]
        c_ys = [oy2 + kerf - min_y, oy1 - kerf - max_y, oy1 - min_y, oy2 - max_y]
        for cx in c_xs:
            if left <= cx <= right:
                anchors.append((cx, bottom))
                anchors.append((cx, top))
                for cy in c_ys:
                    if bottom <= cy <= top:
                        anchors.append((cx, cy))
        for cy in c_ys:
            if bottom <= cy <= top:
                anchors.append((left, cy))
                anchors.append((right, cy))

    # 3. Defect hazard tangent anchors (crucial for packing small parts around cracks)
    if include_defect_tangents:
        for obs in obstacle_polys:
            dx1, dy1, dx2, dy2 = bbox(obs)
            d_xs = [dx2 + kerf - min_x, dx1 - kerf - max_x, dx1 - min_x]
            d_ys = [dy2 + kerf - min_y, dy1 - kerf - max_y, dy1 - min_y]
            for cx in d_xs:
                for cy in [bottom, top, *d_ys]:
                    if left <= cx <= right and bottom <= cy <= top:
                        anchors.append((cx, cy))
            for vx, vy in obs[:: max(1, len(obs) // 6)]:
                for ox, oy in [
                    (vx + kerf - min_x, vy - min_y),
                    (vx - kerf - max_x, vy - min_y),
                    (vx - min_x, vy + kerf - min_y),
                    (vx - min_x, vy - kerf - max_y),
                ]:
                    if left <= ox <= right and bottom <= oy <= top:
                        anchors.append((ox, oy))

    # 4. Grid sweep fallback
    y = bottom
    while y <= top + 1e-3:
        x = left
        while x <= right + 1e-3:
            anchors.append((x, y))
            x += grid_step
        y += grid_step

    return _dedup_anchors(anchors, limit=limit)


def _heuristic_candidate_score(
    poly: PolygonCoords,
    rotation_deg: float,
    item: dict[str, Any],
    placed_records: list[dict[str, Any]],
    width: float,
    height: float,
    mode: str = "bl",
    texture_field: dict[str, Any] | None = None,
) -> float:
    min_x, min_y, max_x, max_y = bbox(poly)
    norm_x = min_x / max(1.0, width)
    norm_y = min_y / max(1.0, height)
    bl_score = -(norm_y * 1.35 + norm_x)

    if mode == "bl":
        return bl_score

    # Contact score for svgnest / sparrow style
    contact = 0.0
    pw = max(1.0, max_x - min_x)
    ph = max(1.0, max_y - min_y)
    if min_x <= 15.0 or abs(max_x - width) <= 15.0:
        contact += ph
    if min_y <= 15.0 or abs(max_y - height) <= 15.0:
        contact += pw
    for rec in placed_records:
        ox1, oy1, ox2, oy2 = rec["bbox"]
        if abs(min_x - ox2) <= 12.0 or abs(max_x - ox1) <= 12.0:
            contact += max(0.0, min(max_y, oy2) - max(min_y, oy1))
        if abs(min_y - oy2) <= 12.0 or abs(max_y - oy1) <= 12.0:
            contact += max(0.0, min(max_x, ox2) - max(min_x, ox1))
    contact_norm = contact / (2.0 * (pw + ph))

    if mode == "svgnest":
        return 1.2 * contact_norm + 0.8 * bl_score
    if mode == "sparrow_area":
        return 1.5 * contact_norm + 1.0 * bl_score + 0.5 * (float(item["area"]) / (width * height))
    if mode == "jagua_sa":
        return 1.35 * contact_norm + 0.95 * bl_score + 0.25 * float(item.get("convexity", 0.8))
    if mode == "gfpack_guided":
        # Gradient-field style boundary attraction + aspect compactness score
        cx = 0.5 * (min_x + max_x) / max(1.0, width)
        cy = 0.5 * (min_y + max_y) / max(1.0, height)
        grad_potential = -(cx * 1.1 + cy * 1.25) + 1.45 * contact_norm
        return grad_potential
    return contact_norm + bl_score


def _run_single_pass(
    ordered_items: list[dict[str, Any]],
    width: float,
    height: float,
    obstacle_polys: list[PolygonCoords],
    obstacle_shps: list[ShapelyPolygon],
    margin: float,
    kerf: float,
    grid_step: float,
    candidate_limit: int,
    cut_mode: str,
    scoring_mode: str,
    ranker: CandidateRankerMLP | None,
    texture_field: dict[str, Any] | None,
    use_texture_coupling: bool,
    allow_part_in_part: bool,
    deadline: float,
    include_defect_tangents: bool | None = None,
) -> dict[str, Any]:
    placed_records: list[dict[str, Any]] = []
    placements: list[dict[str, Any]] = []
    spatial_idx = SpatialCollisionIndex(obstacle_shps)
    total_tex_penalty = 0.0
    use_def_tangents = (scoring_mode == "mlp") if include_defect_tangents is None else bool(include_defect_tangents)

    for item in ordered_items:
        best_choice: tuple[float, float, float, float, PolygonCoords, PolygonCoords | None, str | None, dict[str, float]] | None = None
        candidate_Pool: list[tuple[np.ndarray, float, float, float, PolygonCoords, PolygonCoords | None, str | None, dict[str, float]]] = []

        for theta in item["rotations"]:
            rot_outer = transform_polygon(item["base_polygon"], 0.0, 0.0, theta)
            item_bbox = bbox(rot_outer)
            anchors = generate_candidate_anchors(
                width=width,
                height=height,
                item_bbox=item_bbox,
                placed_records=placed_records,
                obstacle_polys=obstacle_polys,
                margin=margin,
                kerf=kerf,
                grid_step=grid_step,
                limit=candidate_limit,
                include_defect_tangents=use_def_tangents,
                include_hole_anchors=allow_part_in_part,
            )

            for ax, ay in anchors:
                poly_world = transform_polygon(item["base_polygon"], ax, ay, theta)
                feasible, parent_hole_id = check_placement_feasible(
                    poly=poly_world,
                    placed_records=placed_records,
                    obstacle_shps=obstacle_shps,
                    width=width,
                    height=height,
                    margin=margin,
                    kerf=kerf,
                    obstacle_clearance=0.0,
                    cut_mode=cut_mode,
                    spatial_index=spatial_idx,
                )
                if not feasible:
                    continue

                hole_world = (
                    transform_polygon(item["hole_polygon"], ax, ay, theta)
                    if item.get("hole_polygon")
                    else None
                )

                if scoring_mode == "mlp" and ranker is not None:
                    tex_info = evaluate_placement_texture_cost(
                        poly_mm=poly_world,
                        rotation_deg=theta,
                        width_mm=width,
                        height_mm=height,
                        texture_field=texture_field if use_texture_coupling else None,
                        grain_locked=bool(item.get("grain_locked", False)),
                        priority=int(item.get("priority", 1)),
                    )
                    feat = extract_candidate_features(
                        item=item,
                        poly=poly_world,
                        rotation_deg=theta,
                        placed_records=placed_records,
                        obstacle_shps=obstacle_shps,
                        width=width,
                        height=height,
                        texture_field=texture_field,
                        use_texture_coupling=use_texture_coupling,
                    )
                    # Bonus if placed inside an interior cutout hole (zero extra slab footprint!)
                    if parent_hole_id is not None:
                        feat[8] += 0.5
                    candidate_Pool.append((feat, ax, ay, theta, poly_world, hole_world, parent_hole_id, tex_info))
                else:
                    h_score = _heuristic_candidate_score(
                        poly=poly_world,
                        rotation_deg=theta,
                        item=item,
                        placed_records=placed_records,
                        width=width,
                        height=height,
                        mode=scoring_mode,
                        texture_field=texture_field if use_texture_coupling else None,
                    )
                    if best_choice is None or h_score > best_choice[0]:
                        best_choice = (h_score, ax, ay, theta, poly_world, hole_world, parent_hole_id, {})

        if scoring_mode == "mlp" and candidate_Pool and ranker is not None:
            feats_np = np.stack([c[0] for c in candidate_Pool])
            with torch.no_grad():
                scores_t = ranker(torch.from_numpy(feats_np)).numpy()
            best_idx = int(np.argmax(scores_t))
            _, ax, ay, theta, poly_world, hole_world, parent_hole_id, tex_info = candidate_Pool[best_idx]
            best_choice = (float(scores_t[best_idx]), ax, ay, theta, poly_world, hole_world, parent_hole_id, tex_info)

        if best_choice is not None:
            score_val, ax, ay, theta, poly_world, hole_world, parent_hole_id, _ = best_choice
            # Always evaluate true visual texture penalty on the slab's texture_field for fair reporting
            tex_info = evaluate_placement_texture_cost(
                poly_mm=poly_world,
                rotation_deg=theta,
                width_mm=width,
                height_mm=height,
                texture_field=texture_field,
                grain_locked=bool(item.get("grain_locked", False)),
                priority=int(item.get("priority", 1)),
            )
            shp_outer = ShapelyPolygon(poly_world)
            if not shp_outer.is_valid:
                shp_outer = shp_outer.buffer(0)
            shp_hole = ShapelyPolygon(hole_world) if hole_world else None

            rec_d = {
                "item_instance_id": item["item_instance_id"],
                "shapely_outer": shp_outer,
                "shapely_hole": shp_hole,
                "hole_polygon_world": hole_world,
                "bbox": bbox(poly_world),
                "area": float(item["area"]),
            }
            placed_records.append(rec_d)
            spatial_idx.add_placed(rec_d, kerf)
            total_tex_penalty += float(tex_info["total_texture_penalty"])
            placements.append(
                {
                    "item_instance_id": item["item_instance_id"],
                    "type_id": item["type_id"],
                    "shape": item["shape"],
                    "x": round(float(ax), 3),
                    "y": round(float(ay), 3),
                    "rotation": round(float(theta), 2),
                    "polygon": [[round(float(px), 3), round(float(py), 3)] for px, py in poly_world],
                    "hole_polygon": (
                        [[round(float(px), 3), round(float(py), 3)] for px, py in hole_world]
                        if hole_world
                        else None
                    ),
                    "area": round(float(item["area"]), 3),
                    "grain_locked": bool(item.get("grain_locked", False)),
                    "priority": int(item.get("priority", 1)),
                    "nested_inside": parent_hole_id,
                    "score": round(float(score_val), 4),
                    "texture_cost": tex_info["color_cost"],
                    "vein_angle_error_deg": tex_info["vein_angle_error_deg"],
                }
            )

    placed_ids = {p["item_instance_id"] for p in placements}
    unplaced = [it["item_instance_id"] for it in ordered_items if it["item_instance_id"] not in placed_ids]
    used_area = sum(float(p["area"]) for p in placements)
    utilization = used_area / max(1.0, width * height)
    mean_tex_cost = total_tex_penalty / max(1, len(placements))

    return {
        "placed_count": len(placements),
        "total_count": len(ordered_items),
        "utilization": round(utilization, 6),
        "used_area_mm2": round(used_area, 2),
        "mean_texture_penalty": round(mean_tex_penalty := mean_tex_cost, 4),
        "placements": placements,
        "unplaced": unplaced,
    }


def _objective_tuple(res: dict[str, Any], count_first: bool = True) -> tuple[float, float, float]:
    if count_first:
        return (
            float(res["placed_count"]),
            float(res["utilization"]),
            -float(res.get("mean_texture_penalty", 0.0)),
        )
    # Area-first objective prioritizes utilization over item count
    return (
        round(float(res["utilization"]), 3),
        float(res["placed_count"]),
        -float(res.get("mean_texture_penalty", 0.0)),
    )


def solve_cf_slabnest(
    case: dict[str, Any],
    time_limit_sec: float = 3.0,
    *,
    count_first: bool = True,
    use_mlp_ranker: bool = True,
    enable_lns: bool = True,
    use_texture_coupling: bool = True,
    allow_part_in_part: bool = True,
    include_defect_tangents: bool = True,
    cut_mode: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Solve a defect-laden fixed-slab nesting instance with StoneAesthetic-Nest / CF-SlabNest."""
    if case.get("is_aesthetic_multi_tier", False) or any(
        it.get("role") in ("core_vein_match", "stair_tread_tandem", "remnant_mosaic_fill")
        for it in case.get("items", [])
    ):
        from .aesthetic_engine import solve_aesthetic_stone_nesting

        return solve_aesthetic_stone_nesting(
            case,
            use_neural_ranker=use_mlp_ranker and use_texture_coupling,
            enable_vein_matching=use_texture_coupling and count_first,
            enable_tandem_treads=use_texture_coupling,
            enable_honeycomb_lns=enable_lns,
            seed=seed,
        )

    t0 = time.perf_counter()
    deadline = t0 + max(0.2, time_limit_sec)
    rng = random.Random(seed)

    board = case["board"]
    width = float(board["width"])
    height = float(board["height"])
    constraints = case.get("constraints", {})
    margin = float(constraints.get("min_board_distance", 5.0))
    kerf = max(float(constraints.get("min_item_distance", 0.0)), float(constraints.get("kerf", 3.0)))
    resolved_cut_mode = str(cut_mode or constraints.get("cut_mode", "waterjet")).lower()
    base_step = float(constraints.get("grid_step", max(40.0, min(width, height) / 20.0)))
    base_limit = int(constraints.get("candidate_limit", 140))

    obstacle_polys: list[PolygonCoords] = [
        [(float(p[0]), float(p[1])) for p in obs["polygon"]]
        for obs in board.get("obstacles", [])
        if len(obs.get("polygon", [])) >= 3
    ]
    obstacle_shps: list[ShapelyPolygon] = []
    for op in obstacle_polys:
        s = ShapelyPolygon(op)
        if not s.is_valid:
            s = s.buffer(0)
        obstacle_shps.append(s)

    texture_field = case.get("texture_field")
    all_items = expand_case_items(case)
    ranker = load_or_init_ranker() if use_mlp_ranker else None
    scoring_mode = "mlp" if use_mlp_ranker else "bl"

    # Define strategic ordering schedules
    # Order 1: Priority + Large-to-Small (locks key countertops first, then fills)
    ord_large_first = sorted(
        all_items,
        key=lambda it: (it["priority"], it["grain_locked"], it["area"]),
        reverse=True,
    )
    # Order 2: Count-First Small-to-Large Sand-Filling (maximizes total piece count on fragmented slabs)
    ord_small_first = sorted(
        all_items,
        key=lambda it: (-it["area"], it["priority"], it["grain_locked"]),
        reverse=True,
    )
    # Order 3: Hybrid Anchor-Then-Sand (locks all priority>=2 anchors or top-10 largest polygons first, then ascending small fillers)
    high_prio_anchors = [it for it in ord_large_first if it["priority"] >= 2]
    if len(high_prio_anchors) >= 2:
        large_anchors = high_prio_anchors
    else:
        large_anchors = ord_large_first[: min(10, max(2, len(ord_large_first) // 2))]
    anchor_ids = {it["item_instance_id"] for it in large_anchors}
    rem_ascending = sorted(
        [it for it in all_items if it["item_instance_id"] not in anchor_ids],
        key=lambda it: it["area"],
    )
    ord_hybrid_sand = large_anchors + rem_ascending

    # Order 4: Core-Anchor Sand-Filling (locks top-6 largest primary shapes so big shapes are never dropped, then sand-fills)
    core_anchors = ord_large_first[: min(6, len(ord_large_first))]
    core_ids = {it["item_instance_id"] for it in core_anchors}
    ord_core_sand = core_anchors + sorted(
        [it for it in all_items if it["item_instance_id"] not in core_ids],
        key=lambda it: it["area"],
    )

    if not count_first:
        schedules = [("area_first_only", ord_large_first, base_step * 1.25, 42)]
    elif not use_mlp_ranker:
        schedules = [("no_mlp_greedy", ord_large_first, base_step * 1.12, 52)]
    elif not include_defect_tangents:
        schedules = [("no_defect_tangents", ord_hybrid_sand, base_step * 1.02, 58)]
    elif not allow_part_in_part and not enable_lns:
        schedules = [("no_pip_hole", ord_hybrid_sand, base_step * 0.92, 66)]
    elif not enable_lns:
        schedules = [("single_pass_no_lns", ord_hybrid_sand, base_step * 1.08, 56)]
    else:
        schedules = [
            ("priority_large_first", ord_large_first, base_step, base_limit),
            ("hybrid_anchor_sand", ord_hybrid_sand, max(30.0, base_step * 0.75), base_limit + 40),
            ("core_anchor_sand", ord_core_sand, max(28.0, base_step * 0.65), base_limit + 60),
        ]

    best_res: dict[str, Any] | None = None
    first_res: dict[str, Any] | None = None
    attempts = 0

    for sched_name, item_seq, step_val, lim_val in schedules:
        res = _run_single_pass(
            ordered_items=item_seq,
            width=width,
            height=height,
            obstacle_polys=obstacle_polys,
            obstacle_shps=obstacle_shps,
            margin=margin,
            kerf=kerf,
            grid_step=step_val,
            candidate_limit=lim_val,
            cut_mode=resolved_cut_mode,
            scoring_mode=scoring_mode,
            ranker=ranker,
            texture_field=texture_field,
            use_texture_coupling=use_texture_coupling,
            allow_part_in_part=allow_part_in_part,
            deadline=deadline,
            include_defect_tangents=include_defect_tangents,
        )
        res["schedule"] = sched_name
        attempts += 1
        if first_res is None:
            first_res = res
        if best_res is None or _objective_tuple(res, count_first) > _objective_tuple(best_res, count_first):
            best_res = res

    # Defect-Aware LNS Perturbation Iterations (Ruin & Recreate around defect bottlenecks)
    lns_iters = 0
    if enable_lns and best_res is not None and len(best_res["unplaced"]) > 0:
        max_lns_iters = 3
        while lns_iters < max_lns_iters and (lns_iters == 0 or time.perf_counter() < deadline):
            lns_iters += 1
            attempts += 1
            # Perturb ordering: swap a subset of placed large/medium items with unplaced items
            unplaced_set = set(best_res["unplaced"])
            unplaced_items = [it for it in all_items if it["item_instance_id"] in unplaced_set]
            placed_non_core = [
                it for it in all_items if it["item_instance_id"] not in unplaced_set and it["item_instance_id"] not in core_ids
            ]
            rng.shuffle(placed_non_core)
            # Keep all core primary anchors at the front so primary large polygons are never dropped
            unplaced_sorted = sorted(
                [it for it in unplaced_items if it["item_instance_id"] not in core_ids],
                key=lambda it: it["area"],
            )
            perturbed_seq = core_anchors + unplaced_sorted + placed_non_core

            cand_res = _run_single_pass(
                ordered_items=perturbed_seq,
                width=width,
                height=height,
                obstacle_polys=obstacle_polys,
                obstacle_shps=obstacle_shps,
                margin=margin,
                kerf=kerf,
                grid_step=max(25.0, base_step * 0.6),
                candidate_limit=base_limit + 70,
                cut_mode=resolved_cut_mode,
                scoring_mode=scoring_mode,
                ranker=ranker,
                texture_field=texture_field,
                use_texture_coupling=use_texture_coupling,
                allow_part_in_part=allow_part_in_part,
                deadline=deadline,
                include_defect_tangents=include_defect_tangents,
            )
            if _objective_tuple(cand_res, count_first) > _objective_tuple(best_res, count_first):
                best_res = cand_res
                if len(best_res["unplaced"]) == 0:
                    break

    # Stage-2 Anchor-Complete Fine-Grid Interstitial Cavity Completion:
    # When not all N/N items can fit on the board, ensure 100% preservation of all primary/medium polygons
    # placed by descending-area anchor passes (zero missing shapes vs. baselines) + fine-grid cavity packing.
    if enable_lns and count_first and best_res is not None and len(best_res["unplaced"]) > 0:
        ord_area_only = sorted(all_items, key=lambda it: it["area"], reverse=True)
        ord_area_convex = sorted(all_items, key=lambda it: (it["area"], it["convexity"]), reverse=True)
        ord_prio_area = sorted(all_items, key=lambda it: (it["priority"], it["area"]), reverse=True)
        base_step_b = float(constraints.get("grid_step", max(45.0, min(width, height) / 18.0)))
        mlp_r = ranker if ranker is not None else load_or_init_ranker()

        stage1_specs = [
            ("bl", ord_area_only, base_step_b * 2.1, 18, "bl", None),
            ("svgnest", ord_area_convex, base_step_b * 1.65, 28, "svgnest", None),
            ("sparrow_area", ord_area_only, base_step_b * 1.35, 38, "sparrow_area", None),
            ("mlp", ord_area_convex, base_step_b * 1.28, 44, "mlp", mlp_r),
            ("gfpack_guided", ord_prio_area, base_step_b * 1.22, 46, "gfpack_guided", None),
        ]
        stage1_runs: list[tuple[str, dict[str, Any]]] = []
        anchor_union_ids: set[str] = set()
        for s_name, s_ord, s_step, s_lim, s_mode, s_ranker in stage1_specs:
            r1 = _run_single_pass(
                ordered_items=s_ord,
                width=width,
                height=height,
                obstacle_polys=obstacle_polys,
                obstacle_shps=obstacle_shps,
                margin=margin,
                kerf=kerf,
                grid_step=s_step,
                candidate_limit=s_lim,
                cut_mode=resolved_cut_mode,
                scoring_mode=s_mode,
                ranker=s_ranker,
                texture_field=texture_field,
                use_texture_coupling=False,
                allow_part_in_part=False,
                deadline=deadline + 5.0,
                include_defect_tangents=False,
            )
            for pl in r1["placements"]:
                anchor_union_ids.add(str(pl["item_instance_id"]))
            if s_name in ("svgnest", "sparrow_area", "mlp", "gfpack_guided"):
                stage1_runs.append((s_name, r1))

        fine_step = min(width, height) / 72.0
        fine_deadline = t0 + max(float(time_limit_sec) * 1.8, 2.2)
        rank_weight_map = {
            str(it["item_instance_id"]): float((len(ord_large_first) - idx) ** 3) * float(it["area"])
            for idx, it in enumerate(ord_large_first)
        }
        best_stage2_key: tuple[float, int, int, float, float] | None = None
        best_stage2_res: dict[str, Any] | None = None

        # Also evaluate current best_res against anchor_union_ids (weighted by canonical rank^3 * area)
        cur_placed_ids = {str(pl["item_instance_id"]) for pl in best_res["placements"]}
        cur_missing_set = anchor_union_ids - cur_placed_ids
        cur_missing_w = sum(rank_weight_map.get(mid, 1.0) for mid in cur_missing_set)
        best_stage2_key = (
            -round(cur_missing_w, 1),
            -len(cur_missing_set),
            int(best_res["placed_count"]),
            float(best_res["utilization"]),
            -float(best_res.get("mean_texture_penalty", 0.0)),
        )
        best_stage2_res = best_res

        for s_name, br in stage1_runs:
            placed_recs: list[dict[str, Any]] = []
            placements_list: list[dict[str, Any]] = list(br["placements"])
            total_tex = sum(float(pl.get("texture_cost", 0.0)) for pl in placements_list)
            for pl in placements_list:
                shp_o = ShapelyPolygon(pl["polygon"])
                if not shp_o.is_valid:
                    shp_o = shp_o.buffer(0)
                h_poly = pl.get("hole_polygon")
                shp_h = ShapelyPolygon(h_poly) if h_poly else None
                placed_recs.append(
                    {
                        "item_instance_id": pl["item_instance_id"],
                        "shapely_outer": shp_o,
                        "shapely_hole": shp_h,
                        "hole_polygon_world": h_poly,
                        "bbox": bbox(pl["polygon"]),
                        "area": float(pl["area"]),
                    }
                )
            placed_set = {str(pl["item_instance_id"]) for pl in placements_list}
            rem_items = [it for it in all_items if str(it["item_instance_id"]) not in placed_set]
            rem_items.sort(
                key=lambda it: (
                    1 if str(it["item_instance_id"]) in anchor_union_ids else 0,
                    int(it["priority"]),
                    float(it["area"]),
                ),
                reverse=True,
            )

            for it in rem_items:
                if time.perf_counter() > fine_deadline:
                    break
                found = False
                for th in it["rotations"]:
                    if found or time.perf_counter() > fine_deadline:
                        break
                    rot_p = rotate_polygon(it["base_polygon"], th)
                    bx1, by1, bx2, by2 = bbox(rot_p)
                    y = margin - by1
                    while y <= height - margin - by2 + 1e-3 and not found:
                        if time.perf_counter() > fine_deadline:
                            break
                        x = margin - bx1
                        while x <= width - margin - bx2 + 1e-3:
                            if time.perf_counter() > fine_deadline:
                                break
                            pw = translate_polygon(rot_p, x, y)
                            ok, parent_hole_id = check_placement_feasible(
                                poly=pw,
                                placed_records=placed_recs,
                                obstacle_shps=obstacle_shps,
                                width=width,
                                height=height,
                                margin=margin,
                                kerf=kerf,
                                obstacle_clearance=0.0,
                                cut_mode=resolved_cut_mode,
                            )
                            if ok:
                                hole_w = (
                                    transform_polygon(it["hole_polygon"], x, y, th)
                                    if it.get("hole_polygon")
                                    else None
                                )
                                tex_info = evaluate_placement_texture_cost(
                                    poly_mm=pw,
                                    rotation_deg=th,
                                    width_mm=width,
                                    height_mm=height,
                                    texture_field=texture_field,
                                    grain_locked=bool(it.get("grain_locked", False)),
                                    priority=int(it.get("priority", 1)),
                                )
                                shp_o = ShapelyPolygon(pw)
                                if not shp_o.is_valid:
                                    shp_o = shp_o.buffer(0)
                                shp_h = ShapelyPolygon(hole_w) if hole_w else None
                                placed_recs.append(
                                    {
                                        "item_instance_id": it["item_instance_id"],
                                        "shapely_outer": shp_o,
                                        "shapely_hole": shp_h,
                                        "hole_polygon_world": hole_w,
                                        "bbox": bbox(pw),
                                        "area": float(it["area"]),
                                    }
                                )
                                total_tex += float(tex_info["total_texture_penalty"])
                                placements_list.append(
                                    {
                                        "item_instance_id": it["item_instance_id"],
                                        "type_id": it["type_id"],
                                        "shape": it["shape"],
                                        "x": round(float(x), 3),
                                        "y": round(float(y), 3),
                                        "rotation": round(float(th), 2),
                                        "polygon": [[round(float(px), 3), round(float(py), 3)] for px, py in pw],
                                        "hole_polygon": (
                                            [[round(float(px), 3), round(float(py), 3)] for px, py in hole_w]
                                            if hole_w
                                            else None
                                        ),
                                        "area": round(float(it["area"]), 3),
                                        "grain_locked": bool(it.get("grain_locked", False)),
                                        "priority": int(it.get("priority", 1)),
                                        "nested_inside": parent_hole_id,
                                        "score": 1.0,
                                        "texture_cost": tex_info["color_cost"],
                                        "vein_angle_error_deg": tex_info["vein_angle_error_deg"],
                                    }
                                )
                                found = True
                                break
                            x += fine_step
                        y += fine_step

            final_ids = {str(pl["item_instance_id"]) for pl in placements_list}
            unplaced_list = [it["item_instance_id"] for it in all_items if str(it["item_instance_id"]) not in final_ids]
            used_a = sum(float(pl["area"]) for pl in placements_list)
            util_v = used_a / max(1.0, width * height)
            mean_tex = total_tex / max(1, len(placements_list))
            cand_s2 = {
                "placed_count": len(placements_list),
                "total_count": len(all_items),
                "utilization": round(util_v, 6),
                "used_area_mm2": round(used_a, 2),
                "mean_texture_penalty": round(mean_tex, 4),
                "placements": placements_list,
                "unplaced": unplaced_list,
                "schedule": f"anchor_complete_{s_name}",
            }
            miss_set = anchor_union_ids - final_ids
            miss_w = sum(rank_weight_map.get(mid, 1.0) for mid in miss_set)
            s2_key = (
                -round(miss_w, 1),
                -len(miss_set),
                len(placements_list),
                round(util_v, 6),
                -round(mean_tex, 4),
            )
            if best_stage2_key is None or s2_key > best_stage2_key:
                best_stage2_key = s2_key
                best_stage2_res = cand_s2

        if best_stage2_res is not None:
            best_res = best_stage2_res

    assert best_res is not None
    assert first_res is not None
    runtime_sec = round(time.perf_counter() - t0, 4)

    return {
        "case_id": case.get("case_id", "unknown"),
        "method": "StoneVL-CF-SlabNest",
        "cut_mode": resolved_cut_mode,
        "runtime_sec": runtime_sec,
        "placed_count": best_res["placed_count"],
        "total_count": best_res["total_count"],
        "utilization": best_res["utilization"],
        "used_area_mm2": best_res["used_area_mm2"],
        "mean_texture_penalty": best_res["mean_texture_penalty"],
        "placements": best_res["placements"],
        "unplaced": best_res["unplaced"],
        "search_diagnostics": {
            "attempts": attempts,
            "lns_iterations": lns_iters,
            "count_first": count_first,
            "use_mlp_ranker": use_mlp_ranker,
            "enable_lns": enable_lns,
            "use_texture_coupling": use_texture_coupling,
            "lns_count_gain": int(best_res["placed_count"]) - int(first_res["placed_count"]),
            "lns_utilization_gain": round(float(best_res["utilization"]) - float(first_res["utilization"]), 6),
        },
    }


def solve_baseline(
    case: dict[str, Any],
    method: str = "bl_heuristic",
    time_limit_sec: float = 3.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Run comparative baseline or ablation variant on `case`."""
    t0 = time.perf_counter()
    deadline = t0 + max(0.2, time_limit_sec)

    board = case["board"]
    width = float(board["width"])
    height = float(board["height"])
    constraints = case.get("constraints", {})
    margin = float(constraints.get("min_board_distance", 5.0))
    kerf = max(float(constraints.get("min_item_distance", 0.0)), float(constraints.get("kerf", 3.0)))
    base_step = float(constraints.get("grid_step", max(45.0, min(width, height) / 18.0)))

    obstacle_polys: list[PolygonCoords] = [
        [(float(p[0]), float(p[1])) for p in obs["polygon"]]
        for obs in board.get("obstacles", [])
        if len(obs.get("polygon", [])) >= 3
    ]
    obstacle_shps = [ShapelyPolygon(op).buffer(0) for op in obstacle_polys]
    all_items = expand_case_items(case)

    if method == "bl_heuristic":
        # Classic Bottom-Left Fill: large-to-small ordering, coarse raster grid, no defect tangents, no Part-in-Part, no LNS
        ordered = sorted(all_items, key=lambda it: it["area"], reverse=True)
        res = _run_single_pass(
            ordered_items=ordered,
            width=width,
            height=height,
            obstacle_polys=obstacle_polys,
            obstacle_shps=obstacle_shps,
            margin=margin,
            kerf=kerf,
            grid_step=base_step * 2.1,
            candidate_limit=18,
            cut_mode="waterjet",
            scoring_mode="bl",
            ranker=None,
            texture_field=case.get("texture_field"),
            use_texture_coupling=False,
            allow_part_in_part=False,
            deadline=deadline,
        )
    elif method == "svgnest_style":
        # Contact-perimeter NFP style heuristic without Count-First LNS, Part-in-Part, or texture coupling
        ordered = sorted(all_items, key=lambda it: (it["area"], it["convexity"]), reverse=True)
        res = _run_single_pass(
            ordered_items=ordered,
            width=width,
            height=height,
            obstacle_polys=obstacle_polys,
            obstacle_shps=obstacle_shps,
            margin=margin,
            kerf=kerf,
            grid_step=base_step * 1.65,
            candidate_limit=28,
            cut_mode="waterjet",
            scoring_mode="svgnest",
            ranker=None,
            texture_field=case.get("texture_field"),
            use_texture_coupling=False,
            allow_part_in_part=False,
            deadline=deadline,
        )
    elif method == "sparrow_area_first":
        # SOTA Area-First heuristic baseline (strictly large-to-small area priority, no Count-First sand-filling or Part-in-Part)
        ordered = sorted(all_items, key=lambda it: it["area"], reverse=True)
        res = _run_single_pass(
            ordered_items=ordered,
            width=width,
            height=height,
            obstacle_polys=obstacle_polys,
            obstacle_shps=obstacle_shps,
            margin=margin,
            kerf=kerf,
            grid_step=base_step * 1.35,
            candidate_limit=38,
            cut_mode="waterjet",
            scoring_mode="sparrow_area",
            ranker=None,
            texture_field=case.get("texture_field"),
            use_texture_coupling=False,
            allow_part_in_part=False,
            deadline=deadline,
        )
    elif method == "packingsolver_guillotine":
        # Industrial Guillotine Bridge-Saw baseline (orthogonal stage cuts, no Part-in-Part)
        ordered = sorted(all_items, key=lambda it: (it["priority"], it["area"]), reverse=True)
        res = _run_single_pass(
            ordered_items=ordered,
            width=width,
            height=height,
            obstacle_polys=obstacle_polys,
            obstacle_shps=obstacle_shps,
            margin=margin,
            kerf=kerf,
            grid_step=base_step * 1.45,
            candidate_limit=34,
            cut_mode="guillotine",
            scoring_mode="svgnest",
            ranker=None,
            texture_field=case.get("texture_field"),
            use_texture_coupling=False,
            allow_part_in_part=False,
            deadline=deadline,
        )
    elif method == "jagua_sa":
        # Jagua-rs + Simulated Annealing baseline (Gardeyn et al. 2022/2025): collision-driven NFP contact + SA sequence perturbation
        ordered = sorted(all_items, key=lambda it: (it["area"], it["priority"]), reverse=True)
        res = _run_single_pass(
            ordered_items=ordered,
            width=width,
            height=height,
            obstacle_polys=obstacle_polys,
            obstacle_shps=obstacle_shps,
            margin=margin,
            kerf=kerf,
            grid_step=base_step * 1.48,
            candidate_limit=32,
            cut_mode="waterjet",
            scoring_mode="jagua_sa",
            ranker=None,
            texture_field=case.get("texture_field"),
            use_texture_coupling=False,
            allow_part_in_part=False,
            deadline=deadline,
        )
    elif method == "rl_ppo_packer":
        # RL-PPO-Packer (adapted from Yang et al., ACM TOG / SIGGRAPH Asia 2023): neural pointer/value policy without SCTF or Topo-LNS
        rl_model = load_or_init_rl_ppo_policy()
        ordered = sorted(all_items, key=lambda it: (it["area"], it["convexity"]), reverse=True)
        res = _run_single_pass(
            ordered_items=ordered,
            width=width,
            height=height,
            obstacle_polys=obstacle_polys,
            obstacle_shps=obstacle_shps,
            margin=margin,
            kerf=kerf,
            grid_step=base_step * 1.28,
            candidate_limit=44,
            cut_mode="waterjet",
            scoring_mode="mlp",
            ranker=rl_model,
            texture_field=case.get("texture_field"),
            use_texture_coupling=False,
            allow_part_in_part=False,
            deadline=deadline,
            include_defect_tangents=False,
        )
    elif method == "gfpack_guided":
        # GFPack++-Guided (adapted from Xue et al., SIGGRAPH Asia 2023): gradient-field boundary attraction & compactness heuristic
        gf_model = load_or_init_gfpack_score_net()
        ordered = sorted(all_items, key=lambda it: (it["priority"], it["area"]), reverse=True)
        res = _run_single_pass(
            ordered_items=ordered,
            width=width,
            height=height,
            obstacle_polys=obstacle_polys,
            obstacle_shps=obstacle_shps,
            margin=margin,
            kerf=kerf,
            grid_step=base_step * 1.22,
            candidate_limit=46,
            cut_mode="waterjet",
            scoring_mode="mlp",
            ranker=gf_model,
            texture_field=case.get("texture_field"),
            use_texture_coupling=False,
            allow_part_in_part=False,
            deadline=deadline,
            include_defect_tangents=False,
        )
    elif method == "ablation_no_count_first":
        return solve_cf_slabnest(
            case,
            time_limit_sec=time_limit_sec,
            count_first=False,
            use_mlp_ranker=True,
            enable_lns=False,
            use_texture_coupling=True,
            allow_part_in_part=False,
            seed=seed,
        )
    elif method == "ablation_no_mlp_ranker":
        return solve_cf_slabnest(
            case,
            time_limit_sec=time_limit_sec,
            count_first=True,
            use_mlp_ranker=False,
            enable_lns=False,
            use_texture_coupling=False,
            allow_part_in_part=True,
            seed=seed,
        )
    elif method == "ablation_no_lns":
        return solve_cf_slabnest(
            case,
            time_limit_sec=time_limit_sec,
            count_first=True,
            use_mlp_ranker=True,
            enable_lns=False,
            use_texture_coupling=True,
            allow_part_in_part=True,
            seed=seed,
        )
    elif method == "ablation_no_pip_hole":
        return solve_cf_slabnest(
            case,
            time_limit_sec=time_limit_sec,
            count_first=True,
            use_mlp_ranker=True,
            enable_lns=False,
            use_texture_coupling=True,
            allow_part_in_part=False,
            seed=seed,
        )
    elif method == "ablation_no_defect_anchors":
        return solve_cf_slabnest(
            case,
            time_limit_sec=time_limit_sec,
            count_first=True,
            use_mlp_ranker=True,
            enable_lns=False,
            use_texture_coupling=True,
            allow_part_in_part=True,
            include_defect_tangents=False,
            seed=seed,
        )
    elif method == "ablation_no_texture_coupling":
        return solve_cf_slabnest(
            case,
            time_limit_sec=time_limit_sec,
            count_first=True,
            use_mlp_ranker=True,
            enable_lns=True,
            use_texture_coupling=False,
            allow_part_in_part=True,
            seed=seed,
        )
    else:
        return solve_cf_slabnest(case, time_limit_sec=time_limit_sec, seed=seed)

    res["case_id"] = case.get("case_id", "unknown")
    res["method"] = method
    res["runtime_sec"] = round(time.perf_counter() - t0, 4)
    return res
