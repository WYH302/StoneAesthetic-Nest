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


def plan_precedence_cnc_toolpath(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Plan precedence-constrained CNC waterjet cutting sequence and scrap-zone pierce points."""
    board = case["board"]
    w, h = float(board["width"]), float(board["height"])
    kerf = float(case.get("constraints", {}).get("kerf", 3.0))
    lead_in_mm = max(2.5, kerf * 0.9)

    contours: list[dict[str, Any]] = []
    placed = result.get("placements", [])

    for idx, p in enumerate(placed):
        outer_pts = [(float(pt[0]), float(pt[1])) for pt in p.get("polygon", [])]
        if len(outer_pts) < 3:
            continue
        is_pip = p.get("nested_inside") is not None
        tier = 1 if is_pip else 3

        # Choose entry vertex closest to bottom-left corner of part and compute outward scrap pierce lead-in
        v0 = min(outer_pts, key=lambda xy: (xy[0] + xy[1]))
        shp = ShapelyPolygon(outer_pts)
        c_pt = shp.centroid
        dx, dy = v0[0] - c_pt.x, v0[1] - c_pt.y
        norm = max(1e-6, math.hypot(dx, dy))
        px = min(w - 1.0, max(1.0, v0[0] + lead_in_mm * (dx / norm)))
        py = min(h - 1.0, max(1.0, v0[1] + lead_in_mm * (dy / norm)))
        pierce_in_scrap = not shp.contains(ShapelyPoint(px, py))

        contours.append(
            {
                "part_index": idx,
                "contour_type": "pip_child" if is_pip else "outer",
                "tier": tier,
                "entry_xy": v0,
                "pierce_xy": (px, py),
                "pierce_in_scrap": pierce_in_scrap,
                "perimeter_mm": _poly_perimeter(outer_pts),
            }
        )

        hole_pts = p.get("hole_polygon")
        if hole_pts and len(hole_pts) >= 3:
            h_coords = [(float(pt[0]), float(pt[1])) for pt in hole_pts]
            h_shp = ShapelyPolygon(h_coords)
            hv0 = min(h_coords, key=lambda xy: (xy[0] + xy[1]))
            hc = h_shp.centroid
            # For an interior sink hole, the interior of the hole IS scrap! Pierce inward toward hole center.
            hdx, hdy = hc.x - hv0[0], hc.y - hv0[1]
            hnorm = max(1e-6, math.hypot(hdx, hdy))
            hpx = hv0[0] + lead_in_mm * (hdx / hnorm)
            hpy = hv0[1] + lead_in_mm * (hdy / hnorm)
            contours.append(
                {
                    "part_index": idx,
                    "contour_type": "inner_hole",
                    "tier": 2,
                    "entry_xy": hv0,
                    "pierce_xy": (hpx, hpy),
                    "pierce_in_scrap": True,
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

    pierce_pts = [c["pierce_xy"] for c in contours]
    naive_order = list(range(len(contours)))
    naive_air_mm = _tour_air_length(pierce_pts, naive_order)

    # Count precedence violations in naive index order (e.g. outer cut before inner hole or before PiP child)
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

    # Precedence-constrained 2-opt tour: Tier 1 (PiP child) -> Tier 2 (Inner Hole) -> Tier 3 (Outer Contour)
    opt_order: list[int] = []
    cur_xy = (0.0, 0.0)
    for tier_id in (1, 2, 3):
        t_indices = [i for i, c in enumerate(contours) if c["tier"] == tier_id]
        if t_indices:
            t_seq = _two_opt_tier_order(pierce_pts, t_indices, cur_xy)
            opt_order.extend(t_seq)
            cur_xy = pierce_pts[t_seq[-1]]

    opt_air_mm = _tour_air_length(pierce_pts, opt_order)
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

    # 4. Placed cut parts, tier-specific CAM layers, bridge-saw straight cuts, inner sink holes, and kerf envelopes
    total_cut_mm = 0.0
    pierce_count = 0
    bridge_saw_segments = 0

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

        # Export orthogonal/straight-line edges of Tier-1 & Tier-2 rectangles/squares onto BRIDGE_SAW_STRAIGHT_CUTS
        if p_tier in (1, 2) and len(outer_pts) == 4:
            for e_i in range(4):
                p_a = outer_pts[e_i]
                p_b = outer_pts[(e_i + 1) % 4]
                msp.add_line(p_a, p_b, dxfattribs={"layer": "BRIDGE_SAW_STRAIGHT_CUTS"})
                bridge_saw_segments += 1

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
        "naive_air_travel_m": cam_plan["naive_air_travel_m"],
        "optimized_air_travel_m": cam_plan["optimized_air_travel_m"],
        "air_travel_reduction_pct": cam_plan["air_travel_reduction_pct"],
        "naive_precedence_violations": cam_plan["naive_precedence_violations"],
        "optimized_precedence_violations": cam_plan["optimized_precedence_violations"],
        "safe_scrap_pierce_pct": cam_plan["safe_scrap_pierce_pct"],
        "estimated_machining_time_min": round(est_time_min, 2),
    }
