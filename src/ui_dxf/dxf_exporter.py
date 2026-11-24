"""Industrial Multi-Layer DXF & Precedence-Constrained CNC CAM Toolpath Exporter (`Topo-CAM`).

Implements:
1. Topological Precedence Partial Order (`PiP` Child -> Inner Sink Hole -> Outer Contour)
   preventing slab shift on CNC waterjet slats during inner cutout machining.
2. Within-Tier Nearest-Neighbor + 2-Opt TSP Rapid Air-Travel Minimization.
3. Scrap-Side Pierce Lead-In Offset Generation to prevent brittle marble edge chipping.
4. AutoCAD R2010 Multi-Layer `.dxf` Export (units: mm).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import ezdxf
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon


def _poly_perimeter(pts: list[list[float]] | list[tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        total += math.hypot(float(x2) - float(x1), float(y2) - float(y1))
    return total


def _tour_air_length(points: list[tuple[float, float]], order: list[int]) -> float:
    if not order:
        return 0.0
    cx, cy = 0.0, 0.0
    dist = 0.0
    for idx in order:
        px, py = points[idx]
        dist += math.hypot(px - cx, py - cy)
        cx, cy = px, py
    return dist


def _two_opt_tier_order(points: list[tuple[float, float]], indices: list[int], start_xy: tuple[float, float]) -> list[int]:
    """Nearest-neighbor + 2-opt local search within a single topological precedence tier."""
    if len(indices) <= 1:
        return list(indices)

    rem = list(indices)
    nn_order: list[int] = []
    cx, cy = start_xy
    while rem:
        best_i = min(rem, key=lambda idx: math.hypot(points[idx][0] - cx, points[idx][1] - cy))
        nn_order.append(best_i)
        cx, cy = points[best_i]
        rem.remove(best_i)

    def _seg_cost(seq: list[int]) -> float:
        sx, sy = start_xy
        tot = 0.0
        for idx in seq:
            px, py = points[idx]
            tot += math.hypot(px - sx, py - sy)
            sx, sy = px, py
        return tot

    best = nn_order
    best_c = _seg_cost(best)
    improved = True
    n = len(best)
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                cand = best[:i] + list(reversed(best[i : k + 1])) + best[k + 1 :]
                c_val = _seg_cost(cand)
                if c_val + 1e-6 < best_c:
                    best = cand
                    best_c = c_val
                    improved = True
    return best


def _find_safe_pierce_candidates(
    outer_pts: list[tuple[float, float]],
    part_idx: int,
    all_part_shps: list[tuple[int, ShapelyPolygon, ShapelyPolygon | None]],
    obstacle_shps: list[ShapelyPolygon],
    w: float,
    h: float,
    lead_in_mm: float,
    is_inner_hole: bool = False,
) -> list[tuple[tuple[float, float], tuple[float, float], bool]]:
    """Find collision-free scrap-zone pierce candidates across contour vertices (GTSP multi-entry).

    Unlike naive single-part containment checks, verifies every candidate pierce point against
    ALL placed parts (preventing waterjet pierce damage to neighbor parts across narrow 3mm kerfs)
    and defect hazard polygons.
    """
    shp = ShapelyPolygon(outer_pts)
    if not shp.is_valid:
        shp = shp.buffer(0)
    c_pt = shp.centroid

    # Pick up to 4 well-spread candidate vertices (bottom-left, top-right, top-left, bottom-right)
    n_v = len(outer_pts)
    if n_v <= 4:
        cand_verts = list(outer_pts)
    else:
        idxs = sorted(
            {
                min(range(n_v), key=lambda i: outer_pts[i][0] + outer_pts[i][1]),
                max(range(n_v), key=lambda i: outer_pts[i][0] + outer_pts[i][1]),
                min(range(n_v), key=lambda i: outer_pts[i][0] - outer_pts[i][1]),
                max(range(n_v), key=lambda i: outer_pts[i][0] - outer_pts[i][1]),
            }
        )
        cand_verts = [outer_pts[i] for i in idxs]

    def _is_globally_safe(pt_xy: tuple[float, float]) -> bool:
        px, py = pt_xy
        if px < 0.8 or px > w - 0.8 or py < 0.8 or py > h - 0.8:
            return False
        pt = ShapelyPoint(px, py)
        if is_inner_hole:
            if not shp.contains(pt):
                return False
            # Also ensure inner-hole pierce doesn't hit any PiP child nested inside the hole
            for o_idx, o_shp, _ in all_part_shps:
                if o_idx == part_idx:
                    continue
                if o_shp.contains(pt) or o_shp.distance(pt) < 0.45:
                    return False
            return True
        # Outer contour: must lie strictly outside self, all neighbor parts (allowing legal parent hole scrap), and all defects
        if shp.contains(pt):
            return False
        for o_idx, o_shp, o_hole in all_part_shps:
            if o_idx == part_idx:
                continue
            if o_hole is not None and o_hole.contains(pt):
                continue  # Pierce inside parent's scrap sink-hole is safe
            if o_shp.contains(pt) or o_shp.distance(pt) < 0.45:
                return False
        for obs in obstacle_shps:
            if obs.contains(pt):
                return False
        return True

    results: list[tuple[tuple[float, float], tuple[float, float], bool]] = []
    scale_tries = (1.0, 0.72, 0.48, 0.30)
    for v in cand_verts:
        dx = (c_pt.x - v[0]) if is_inner_hole else (v[0] - c_pt.x)
        dy = (c_pt.y - v[1]) if is_inner_hole else (v[1] - c_pt.y)
        norm = max(1e-6, math.hypot(dx, dy))
        ux, uy = dx / norm, dy / norm
        # Try outward/inward radial vector plus slight tangential offsets to dodge neighbor kerfs
        found_v: tuple[tuple[float, float], tuple[float, float], bool] | None = None
        for sc in scale_tries:
            d_try = max(0.85, lead_in_mm * sc)
            for rot_deg in (0.0, 28.0, -28.0, 60.0, -60.0):
                rr = math.radians(rot_deg)
                rx = ux * math.cos(rr) - uy * math.sin(rr)
                ry = ux * math.sin(rr) + uy * math.cos(rr)
                cand_p = (
                    min(w - 0.8, max(0.8, v[0] + d_try * rx)),
                    min(h - 0.8, max(0.8, v[1] + d_try * ry)),
                )
                if _is_globally_safe(cand_p):
                    found_v = (v, cand_p, True)
                    break
            if found_v is not None:
                break
        if found_v is not None:
            results.append(found_v)

    if not results:
        # Fallback to tightest non-self-intersecting scrap pierce on bottom-left vertex
        v0 = min(outer_pts, key=lambda xy: (xy[0] + xy[1]))
        dx = (c_pt.x - v0[0]) if is_inner_hole else (v0[0] - c_pt.x)
        dy = (c_pt.y - v0[1]) if is_inner_hole else (v0[1] - c_pt.y)
        norm = max(1e-6, math.hypot(dx, dy))
        p_fb = (
            min(w - 0.8, max(0.8, v0[0] + 0.9 * (dx / norm))),
            min(h - 0.8, max(0.8, v0[1] + 0.9 * (dy / norm))),
        )
        results.append((v0, p_fb, _is_globally_safe(p_fb)))
    return results


def _merge_collinear_bridge_saw_cuts(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    angle_tol_deg: float = 1.5,
    offset_tol_mm: float = 4.5,
    gap_tol_mm: float = 40.0,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], float, float]:
    """Merge collinear adjacent/shared Tier-1 & Tier-2 straight edges into unified bridge-saw passes."""
    if not segments:
        return [], 0.0, 0.0
    raw_len = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segments)
    # Cluster by canonical line angle (in [0, 180)) and signed perpendicular offset
    groups: list[dict[str, Any]] = []
    for a, b in segments:
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg_l = math.hypot(dx, dy)
        if seg_l < 1e-3:
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        rad = math.radians(ang)
        ux, uy = math.cos(rad), math.sin(rad)
        nx, ny = -uy, ux
        perp = a[0] * nx + a[1] * ny
        t0 = a[0] * ux + a[1] * uy
        t1 = b[0] * ux + b[1] * uy
        if t1 < t0:
            t0, t1 = t1, t0

        matched = False
        for g in groups:
            adiff = abs((ang - g["ang"] + 90.0) % 180.0 - 90.0)
            if adiff <= angle_tol_deg and abs(perp - g["perp"]) <= offset_tol_mm:
                g["intervals"].append((t0, t1))
                matched = True
                break
        if not matched:
            groups.append({"ang": ang, "ux": ux, "uy": uy, "nx": nx, "ny": ny, "perp": perp, "intervals": [(t0, t1)]})

    merged_segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    merged_len = 0.0
    for g in groups:
        ivs = sorted(g["intervals"], key=lambda x: x[0])
        cur_s, cur_e = ivs[0]
        merged_ivs: list[tuple[float, float]] = []
        for s, e in ivs[1:]:
            if s <= cur_e + gap_tol_mm:
                cur_e = max(cur_e, e)
            else:
                merged_ivs.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged_ivs.append((cur_s, cur_e))
        ux, uy, nx, ny, perp = g["ux"], g["uy"], g["nx"], g["ny"], g["perp"]
        for s, e in merged_ivs:
            p0 = (round(s * ux + perp * nx, 3), round(s * uy + perp * ny, 3))
            p1 = (round(e * ux + perp * nx, 3), round(e * uy + perp * ny, 3))
            merged_segs.append((p0, p1))
            merged_len += e - s

    return merged_segs, round(raw_len / 1000.0, 3), round(merged_len / 1000.0, 3)


def _gtsp_two_opt_tier(
    contours: list[dict[str, Any]],
    indices: list[int],
    start_xy: tuple[float, float],
) -> tuple[list[int], dict[int, int]]:
    """Generalized TSP (GTSP) greedy + 2-opt with dynamic multi-vertex entry selection."""
    if not indices:
        return [], {}
    rem = list(indices)
    nn_order: list[int] = []
    chosen_cand: dict[int, int] = {}
    cx, cy = start_xy
    while rem:
        best_pair: tuple[float, int, int] | None = None
        for idx in rem:
            for c_i, (_, p_xy, _) in enumerate(contours[idx]["pierce_candidates"]):
                d = math.hypot(p_xy[0] - cx, p_xy[1] - cy)
                if best_pair is None or d < best_pair[0]:
                    best_pair = (d, idx, c_i)
        assert best_pair is not None
        _, b_idx, b_ci = best_pair
        nn_order.append(b_idx)
        chosen_cand[b_idx] = b_ci
        cx, cy = contours[b_idx]["pierce_candidates"][b_ci][1]
        rem.remove(b_idx)

    def _optimize_entries(seq: list[int]) -> tuple[float, dict[int, int]]:
        # Dynamic programming over up to 4 candidate pierce corners per contour along fixed sequence
        n_s = len(seq)
        dp_cost: list[list[float]] = []
        dp_prev: list[list[int]] = []
        c0_list = contours[seq[0]]["pierce_candidates"]
        dp_cost.append([math.hypot(c[1][0] - start_xy[0], c[1][1] - start_xy[1]) for c in c0_list])
        dp_prev.append([-1] * len(c0_list))
        for step in range(1, n_s):
            prev_cands = contours[seq[step - 1]]["pierce_candidates"]
            cur_cands = contours[seq[step]]["pierce_candidates"]
            row_c: list[float] = []
            row_p: list[int] = []
            for cc in cur_cands:
                px, py = cc[1]
                best_v = 1e18
                best_j = 0
                for j, pc in enumerate(prev_cands):
                    v = dp_cost[-1][j] + math.hypot(px - pc[1][0], py - pc[1][1])
                    if v < best_v:
                        best_v = v
                        best_j = j
                row_c.append(best_v)
                row_p.append(best_j)
            dp_cost.append(row_c)
            dp_prev.append(row_p)
        end_j = int(min(range(len(dp_cost[-1])), key=lambda j: dp_cost[-1][j]))
        best_tot = dp_cost[-1][end_j]
        sel: dict[int, int] = {}
        cur_j = end_j
        for step in range(n_s - 1, -1, -1):
            sel[seq[step]] = cur_j
            cur_j = dp_prev[step][cur_j]
        return best_tot, sel

    best_seq = nn_order
    best_cost, best_sel = _optimize_entries(best_seq)
    n = len(best_seq)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                cand = best_seq[:i] + list(reversed(best_seq[i : k + 1])) + best_seq[k + 1 :]
                c_val, c_sel = _optimize_entries(cand)
                if c_val + 1e-6 < best_cost:
                    best_seq = cand
                    best_cost = c_val
                    best_sel = c_sel
                    improved = True
    return best_seq, best_sel


def plan_precedence_cnc_toolpath(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Plan precedence-constrained CNC waterjet cutting sequence with neighbor-safe scrap pierces & GTSP 2-opt."""
    board = case["board"]
    w, h = float(board["width"]), float(board["height"])
    kerf = float(case.get("constraints", {}).get("kerf", 3.0))
    lead_in_mm = max(2.2, kerf * 0.85)

    obstacle_shps: list[ShapelyPolygon] = []
    for obs in board.get("obstacles", []):
        pts = [(float(p[0]), float(p[1])) for p in obs.get("polygon", [])]
        if len(pts) >= 3:
            os = ShapelyPolygon(pts)
            obstacle_shps.append(os if os.is_valid else os.buffer(0))

    placed = result.get("placements", [])
    all_part_shps: list[tuple[int, ShapelyPolygon, ShapelyPolygon | None]] = []
    for idx, p in enumerate(placed):
        outer_pts = [(float(pt[0]), float(pt[1])) for pt in p.get("polygon", [])]
        if len(outer_pts) < 3:
            continue
        shp = ShapelyPolygon(outer_pts)
        if not shp.is_valid:
            shp = shp.buffer(0)
        h_pts = p.get("hole_polygon")
        h_shp = ShapelyPolygon([(float(pt[0]), float(pt[1])) for pt in h_pts]) if (h_pts and len(h_pts) >= 3) else None
        all_part_shps.append((idx, shp, h_shp))

    contours: list[dict[str, Any]] = []
    for idx, p in enumerate(placed):
        outer_pts = [(float(pt[0]), float(pt[1])) for pt in p.get("polygon", [])]
        if len(outer_pts) < 3:
            continue
        is_pip = p.get("nested_inside") is not None
        tier = 1 if is_pip else 3

        cands = _find_safe_pierce_candidates(
            outer_pts, idx, all_part_shps, obstacle_shps, w, h, lead_in_mm, is_inner_hole=False
        )
        v0, p0, safe0 = cands[0]
        contours.append(
            {
                "part_index": idx,
                "contour_type": "pip_child" if is_pip else "outer",
                "tier": tier,
                "entry_xy": v0,
                "pierce_xy": p0,
                "pierce_in_scrap": safe0,
                "pierce_candidates": cands,
                "perimeter_mm": _poly_perimeter(outer_pts),
            }
        )

        hole_pts = p.get("hole_polygon")
        if hole_pts and len(hole_pts) >= 3:
            h_coords = [(float(pt[0]), float(pt[1])) for pt in hole_pts]
            h_cands = _find_safe_pierce_candidates(
                h_coords, idx, all_part_shps, obstacle_shps, w, h, lead_in_mm, is_inner_hole=True
            )
            hv0, hp0, hsafe0 = h_cands[0]
            contours.append(
                {
                    "part_index": idx,
                    "contour_type": "inner_hole",
                    "tier": 2,
                    "entry_xy": hv0,
                    "pierce_xy": hp0,
                    "pierce_in_scrap": hsafe0,
                    "pierce_candidates": h_cands,
                    "perimeter_mm": _poly_perimeter(h_coords),
                }
            )

    if not contours:
        return {
            "ordered_contours": [],
            "naive_air_travel_m": 0.0,
            "optimized_air_travel_m": 0.0,
            "air_travel_reduction_pct": 0.0,
            "naive_precedence_violations": 0,
            "optimized_precedence_violations": 0,
            "safe_scrap_pierce_pct": 100.0,
        }

    pierce_pts_naive = [c["pierce_xy"] for c in contours]
    naive_order = list(range(len(contours)))
    naive_air_mm = _tour_air_length(pierce_pts_naive, naive_order)

    naive_viol = 0
    seen_outer_parts: set[int] = set()
    for ci in naive_order:
        c = contours[ci]
        if c["contour_type"] == "outer":
            seen_outer_parts.add(c["part_index"])
        elif c["contour_type"] == "inner_hole" and c["part_index"] in seen_outer_parts:
            naive_viol += 1
        elif c["contour_type"] == "pip_child" and len(seen_outer_parts) > 0:
            naive_viol += 1

    # Precedence-constrained GTSP 2-opt tour: Tier 1 (PiP child) -> Tier 2 (Inner Hole) -> Tier 3 (Outer Contour)
    opt_order: list[int] = []
    cur_xy = (0.0, 0.0)
    for tier_id in (1, 2, 3):
        t_indices = [i for i, c in enumerate(contours) if c["tier"] == tier_id]
        if t_indices:
            t_seq, t_sel = _gtsp_two_opt_tier(contours, t_indices, cur_xy)
            for idx_c in t_seq:
                e_xy, p_xy, s_ok = contours[idx_c]["pierce_candidates"][t_sel[idx_c]]
                contours[idx_c]["entry_xy"] = e_xy
                contours[idx_c]["pierce_xy"] = p_xy
                contours[idx_c]["pierce_in_scrap"] = s_ok
            opt_order.extend(t_seq)
            cur_xy = contours[t_seq[-1]]["pierce_xy"]

    pierce_pts_opt = [c["pierce_xy"] for c in contours]
    opt_air_mm = _tour_air_length(pierce_pts_opt, opt_order)
    red_pct = (naive_air_mm - opt_air_mm) / max(1.0, naive_air_mm) * 100.0
    safe_pct = sum(1 for c in contours if c["pierce_in_scrap"]) / len(contours) * 100.0

    return {
        "ordered_contours": [contours[i] for i in opt_order],
        "naive_air_travel_m": round(naive_air_mm / 1000.0, 3),
        "optimized_air_travel_m": round(opt_air_mm / 1000.0, 3),
        "air_travel_reduction_pct": round(max(0.0, red_pct), 2),
        "naive_precedence_violations": naive_viol,
        "optimized_precedence_violations": 0,
        "safe_scrap_pierce_pct": round(safe_pct, 1),
    }


def export_industrial_dxf(
    case: dict[str, Any],
    result: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Export nesting solution and Topo-CAM toolpath to an AutoCAD R2010 multi-layer DXF file (units: mm)."""
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4  # 4 = millimeters

    layers_spec = [
        ("SLAB_BOUNDARY", 7),        # White/Black
        ("DEFECT_HAZARDS", 1),       # Red
        ("GOLDEN_VEIN_ZONE", 40),    # Gold (Luxury focal vein zone)
        ("CUT_PARTS_OUTER", 3),      # Green (Master outer contour layer)
        ("TIER1_CORE_VEIN_MATCH", 4),# Cyan (#C1..#C6 Core vein-matched pieces)
        ("TIER2_STAIR_TREADS", 3),   # Green (#S1..#S8 Sequential stair treads)
        ("TIER3_MOSAIC_REMNANTS", 30), # Orange (#M1..#Mk Remnant mosaic triangles)
        ("BRIDGE_SAW_STRAIGHT_CUTS", 6), # Magenta (Collinear infrared bridge-saw cuts)
        ("CUT_PARTS_INNER_HOLE", 4), # Cyan
        ("KERF_ENVELOPE", 8),        # Gray
        ("VEIN_AXIS", 6),            # Magenta
        ("PART_ANNOTATIONS", 2),     # Yellow
        ("CNC_RAPID_TOOLPATH", 5),   # Blue (Precedence-constrained rapid traverse & pierce lead-ins)
    ]
    for name, color in layers_spec:
        if name not in doc.layers:
            doc.layers.add(name=name, color=color)

    msp = doc.modelspace()
    board = case["board"]
    width = float(board["width"])
    height = float(board["height"])
    kerf = float(case.get("constraints", {}).get("kerf", 3.0))
    vein_deg = float(board.get("vein_angle_deg", 0.0))

    # 1. Slab outer boundary
    slab_pts = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    msp.add_lwpolyline(slab_pts, close=True, dxfattribs={"layer": "SLAB_BOUNDARY"})

    # 2. Dominant vein direction reference line at top-left corner
    v_rad = math.radians(vein_deg)
    cx, cy = width * 0.08, height * 0.92
    v_len = min(width, height) * 0.08
    msp.add_line(
        (cx - v_len * math.cos(v_rad), cy - v_len * math.sin(v_rad)),
        (cx + v_len * math.cos(v_rad), cy + v_len * math.sin(v_rad)),
        dxfattribs={"layer": "VEIN_AXIS"},
    )
    msp.add_text(
        f"VEIN {vein_deg:.1f} DEG",
        dxfattribs={"layer": "VEIN_AXIS", "height": 22.0, "insert": (cx - v_len, cy + 25.0)},
    )

    # 3. Defect hazard zones
    for obs in board.get("obstacles", []):
        pts = [(float(p[0]), float(p[1])) for p in obs.get("polygon", [])]
        if len(pts) >= 3:
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "DEFECT_HAZARDS"})
            ox = sum(p[0] for p in pts) / len(pts)
            oy = sum(p[1] for p in pts) / len(pts)
            msp.add_text(
                str(obs.get("id", "DEFECT")),
                dxfattribs={"layer": "DEFECT_HAZARDS", "height": 20.0, "insert": (ox, oy)},
            )

    # 3b. Golden Vein Zone reference rectangle if available
    gz = result.get("golden_vein_zone_mm")
    if gz and len(gz) == 4:
        gx0, gy0, gx1, gy1 = [float(v) for v in gz]
        msp.add_lwpolyline(
            [(gx0, gy0), (gx1, gy0), (gx1, gy1), (gx0, gy1)],
            close=True,
            dxfattribs={"layer": "GOLDEN_VEIN_ZONE"},
        )
        msp.add_text(
            "GOLDEN VEIN ZONE",
            dxfattribs={"layer": "GOLDEN_VEIN_ZONE", "height": 22.0, "insert": (gx0 + 15.0, gy1 - 30.0)},
        )

    # 4. Placed cut parts, tier-specific CAM layers, merged bridge-saw straight cuts, inner sink holes, and kerf envelopes
    total_cut_mm = 0.0
    pierce_count = 0
    raw_bridge_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for idx, p in enumerate(result.get("placements", []), start=1):
        outer_pts = [(float(pt[0]), float(pt[1])) for pt in p.get("polygon", [])]
        if len(outer_pts) < 3:
            continue

        msp.add_lwpolyline(outer_pts, close=True, dxfattribs={"layer": "CUT_PARTS_OUTER"})
        p_tier = int(p.get("tier", 0))
        if p_tier == 1 or p.get("role") == "core_vein_match":
            msp.add_lwpolyline(outer_pts, close=True, dxfattribs={"layer": "TIER1_CORE_VEIN_MATCH"})
        elif p_tier == 2 or p.get("role") == "stair_tread_tandem":
            msp.add_lwpolyline(outer_pts, close=True, dxfattribs={"layer": "TIER2_STAIR_TREADS"})
        elif p_tier == 3 or p.get("role") == "remnant_mosaic_fill":
            msp.add_lwpolyline(outer_pts, close=True, dxfattribs={"layer": "TIER3_MOSAIC_REMNANTS"})

        # Collect orthogonal/straight-line edges of Tier-1 & Tier-2 rectangles/squares for common-line merging
        if p_tier in (1, 2) and len(outer_pts) == 4:
            for e_i in range(4):
                p_a = outer_pts[e_i]
                p_b = outer_pts[(e_i + 1) % 4]
                raw_bridge_segments.append((p_a, p_b))

        total_cut_mm += _poly_perimeter(outer_pts)
        pierce_count += 1

        # Kerf envelope (buffered by kerf / 2)
        if kerf > 0.1:
            shp = ShapelyPolygon(outer_pts)
            if shp.is_valid:
                kerf_shp = shp.buffer(kerf * 0.5, join_style=2)
                if kerf_shp.geom_type == "Polygon":
                    k_pts = [(float(x), float(y)) for x, y in list(kerf_shp.exterior.coords)[:-1]]
                    if len(k_pts) >= 3:
                        msp.add_lwpolyline(k_pts, close=True, dxfattribs={"layer": "KERF_ENVELOPE"})

        # Inner sink cutout hole if present
        hole_pts = p.get("hole_polygon")
        if hole_pts and len(hole_pts) >= 3:
            h_coords = [(float(pt[0]), float(pt[1])) for pt in hole_pts]
            msp.add_lwpolyline(h_coords, close=True, dxfattribs={"layer": "CUT_PARTS_INNER_HOLE"})
            total_cut_mm += _poly_perimeter(h_coords)
            pierce_count += 1

        # Part annotation label
        pcx = sum(pt[0] for pt in outer_pts) / len(outer_pts)
        pcy = sum(pt[1] for pt in outer_pts) / len(outer_pts)
        label = f"#{idx} {p.get('item_instance_id', '')} ({p.get('rotation', 0):.0f}d)"
        msp.add_text(
            label,
            dxfattribs={"layer": "PART_ANNOTATIONS", "height": 24.0, "insert": (pcx - 60.0, pcy)},
        )

    # Merge collinear adjacent/common-line bridge-saw passes before writing BRIDGE_SAW_STRAIGHT_CUTS
    merged_bridge_segs, raw_bridge_m, merged_bridge_m = _merge_collinear_bridge_saw_cuts(
        raw_bridge_segments, offset_tol_mm=max(3.5, kerf + 1.2)
    )
    for p_a, p_b in merged_bridge_segs:
        msp.add_line(p_a, p_b, dxfattribs={"layer": "BRIDGE_SAW_STRAIGHT_CUTS"})
    bridge_saw_segments = len(merged_bridge_segs)

    # 5. Precedence-Constrained CNC Rapid Toolpath & Pierce Lead-Ins
    cam_plan = plan_precedence_cnc_toolpath(case, result)
    prev_xy = (0.0, 0.0)
    for step_idx, c_step in enumerate(cam_plan["ordered_contours"], start=1):
        px, py = c_step["pierce_xy"]
        ex, ey = c_step["entry_xy"]
        msp.add_line(prev_xy, (px, py), dxfattribs={"layer": "CNC_RAPID_TOOLPATH"})
        msp.add_circle((px, py), radius=4.0, dxfattribs={"layer": "CNC_RAPID_TOOLPATH"})
        msp.add_line((px, py), (ex, ey), dxfattribs={"layer": "CNC_RAPID_TOOLPATH"})
        prev_xy = (ex, ey)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(out_path))

    total_cut_m = total_cut_mm / 1000.0
    cut_speed_m_per_min = 1.8 if result.get("cut_mode") == "guillotine" else 0.85
    rapid_speed_m_per_min = 12.0
    est_time_min = (
        total_cut_m / cut_speed_m_per_min
        + cam_plan["optimized_air_travel_m"] / rapid_speed_m_per_min
        + pierce_count * 0.12
    )

    return {
        "dxf_path": str(out_path),
        "file_size_bytes": out_path.stat().st_size,
        "layers": [name for name, _ in layers_spec],
        "placed_parts_exported": len(result.get("placements", [])),
        "pierce_count": pierce_count,
        "total_cut_length_m": round(total_cut_m, 3),
        "raw_bridge_saw_segments": len(raw_bridge_segments),
        "merged_bridge_saw_segments": bridge_saw_segments,
        "bridge_common_line_saved_m": round(max(0.0, raw_bridge_m - merged_bridge_m), 3),
        "naive_air_travel_m": cam_plan["naive_air_travel_m"],
        "optimized_air_travel_m": cam_plan["optimized_air_travel_m"],
        "air_travel_reduction_pct": cam_plan["air_travel_reduction_pct"],
        "naive_precedence_violations": cam_plan["naive_precedence_violations"],
        "optimized_precedence_violations": cam_plan["optimized_precedence_violations"],
        "safe_scrap_pierce_pct": cam_plan["safe_scrap_pierce_pct"],
        "estimated_machining_time_min": round(est_time_min, 2),
    }
