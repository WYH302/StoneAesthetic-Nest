"""Independent Reconstructive Geometric Certificate Validator."""

from __future__ import annotations

from typing import Any

from shapely.geometry import Polygon as ShapelyPolygon

from .geometry_kernel import (
    EPS,
    PolygonCoords,
    bbox,
    expand_case_items,
    polygon_area,
    transform_polygon,
)


def validate_nesting_solution(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Independently verify that a nesting result satisfies all hard geometric constraints.

    Checks:
    1. Every placed item ID exists in the case specification and is placed at most once.
    2. Rotation angle is in the allowed rotation set (respecting grain lock).
    3. Rebuilt polygon from (base_polygon, x, y, rotation) matches serialized coordinates.
    4. All vertices lie strictly inside the slab boundary minus `min_board_distance`.
    5. Zero intersection with any defect hazard polygon.
    6. Pairwise distance between placed parts >= `kerf` (allowing legal Part-in-Part hole containment).
    7. Reported `placed_count` and `utilization` match exact recomputed values.
    """
    board = case["board"]
    width = float(board["width"])
    height = float(board["height"])
    constraints = case.get("constraints", {})
    margin = float(constraints.get("min_board_distance", 5.0))
    kerf = max(float(constraints.get("min_item_distance", 0.0)), float(constraints.get("kerf", 3.0)))

    obstacle_shps: list[ShapelyPolygon] = []
    for obs in board.get("obstacles", []):
        pts = [(float(p[0]), float(p[1])) for p in obs.get("polygon", [])]
        if len(pts) >= 3:
            s = ShapelyPolygon(pts)
            if not s.is_valid:
                s = s.buffer(0)
            obstacle_shps.append(s)

    expected_by_id = {it["item_instance_id"]: it for it in expand_case_items(case)}
    issues: list[str] = []
    verified_records: list[dict[str, Any]] = []
    used_area = 0.0

    placements = result.get("placements", [])
    seen_ids: set[str] = set()

    for p in placements:
        item_id = str(p.get("item_instance_id", ""))
        if item_id in seen_ids:
            issues.append(f"duplicate item_instance_id: {item_id}")
        seen_ids.add(item_id)

        expected = expected_by_id.get(item_id)
        poly: PolygonCoords = [(float(pt[0]), float(pt[1])) for pt in p.get("polygon", [])]
        if len(poly) < 3:
            issues.append(f"degenerate polygon for {item_id}")
            continue

        if expected is None:
            issues.append(f"unknown item_instance_id: {item_id}")
        else:
            rot = float(p.get("rotation", 0.0))
            if not any(abs((rot - float(allowed)) % 360.0) < 1e-2 or abs((float(allowed) - rot) % 360.0) < 1e-2 for allowed in expected["rotations"]):
                issues.append(f"disallowed rotation {rot} for {item_id}; allowed={expected['rotations']}")

            rebuilt = transform_polygon(expected["base_polygon"], float(p["x"]), float(p["y"]), rot)
            if len(rebuilt) != len(poly):
                issues.append(f"vertex count mismatch for {item_id}")
            else:
                max_err = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(rebuilt, poly))
                if max_err > 5e-2:
                    issues.append(f"transform reconstruction error {max_err:.4f}mm for {item_id}")

        min_x, min_y, max_x, max_y = bbox(poly)
        if (
            min_x < margin - 1e-2
            or min_y < margin - 1e-2
            or max_x > width - margin + 1e-2
            or max_y > height - margin + 1e-2
        ):
            issues.append(f"out_of_bounds placement for {item_id}: bbox=({min_x:.1f},{min_y:.1f},{max_x:.1f},{max_y:.1f})")

        shp = ShapelyPolygon(poly)
        if not shp.is_valid:
            shp = shp.buffer(0)

        for obs_idx, obs_shp in enumerate(obstacle_shps):
            if shp.intersects(obs_shp) and shp.intersection(obs_shp).area > 1e-2:
                issues.append(f"defect collision between {item_id} and obstacle_{obs_idx + 1}")

        for other in verified_records:
            other_shp: ShapelyPolygon = other["shapely_outer"]
            other_hole: ShapelyPolygon | None = other.get("shapely_hole")
            this_hole_pts = p.get("hole_polygon")
            this_hole = ShapelyPolygon([(float(pt[0]), float(pt[1])) for pt in this_hole_pts]) if this_hole_pts else None

            # Check if `shp` is legally nested inside `other_hole` or vice versa
            if other_hole is not None and other_hole.buffer(-kerf + 1e-2).contains(shp):
                continue
            if this_hole is not None and this_hole.buffer(-kerf + 1e-2).contains(other_shp):
                continue

            dist = float(shp.distance(other_shp))
            if dist < kerf - 5e-2:
                issues.append(f"kerf violation ({dist:.2f}mm < {kerf:.2f}mm) between {item_id} and {other['item_instance_id']}")

        hole_pts = p.get("hole_polygon")
        hole_shp = ShapelyPolygon([(float(pt[0]), float(pt[1])) for pt in hole_pts]) if hole_pts else None
        net_area = polygon_area(poly) - (polygon_area([(float(pt[0]), float(pt[1])) for pt in hole_pts]) if hole_pts else 0.0)
        used_area += max(0.0, net_area)

        verified_records.append(
            {
                "item_instance_id": item_id,
                "shapely_outer": shp,
                "shapely_hole": hole_shp,
            }
        )

    if int(result.get("placed_count", -1)) != len(placements):
        issues.append(f"placed_count mismatch: reported {result.get('placed_count')} vs actual {len(placements)}")

    computed_util = used_area / max(1.0, width * height)
    if abs(float(result.get("utilization", -1.0)) - computed_util) > 2e-3:
        issues.append(
            f"utilization mismatch: reported {result.get('utilization')} vs computed {computed_util:.6f}"
        )

    return {
        "case_id": case.get("case_id", "unknown"),
        "method": result.get("method", "unknown"),
        "feasible": len(issues) == 0,
        "checked_placements": len(placements),
        "computed_utilization": round(computed_util, 6),
        "issues": issues,
    }
