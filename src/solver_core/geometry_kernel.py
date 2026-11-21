"""Polygonal Geometry Kernel with Minkowski Kerf Compensation & Part-in-Part Support."""

from __future__ import annotations

import math
from typing import Any

from shapely.geometry import Polygon as ShapelyPolygon

Point = tuple[float, float]
PolygonCoords = list[Point]
EPS = 1e-6


def polygon_area(poly: PolygonCoords) -> float:
    if len(poly) < 3:
        return 0.0
    total = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def bbox(poly: PolygonCoords) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def rotate_polygon(poly: PolygonCoords, angle_deg: float) -> PolygonCoords:
    rad = math.radians(angle_deg)
    ca = math.cos(rad)
    sa = math.sin(rad)
    return [(x * ca - y * sa, x * sa + y * ca) for x, y in poly]


def translate_polygon(poly: PolygonCoords, dx: float, dy: float) -> PolygonCoords:
    return [(x + dx, y + dy) for x, y in poly]


def transform_polygon(poly: PolygonCoords, dx: float, dy: float, angle_deg: float) -> PolygonCoords:
    return translate_polygon(rotate_polygon(poly, angle_deg), dx, dy)


def normalize_to_origin(poly: PolygonCoords) -> PolygonCoords:
    min_x, min_y, _, _ = bbox(poly)
    return [(round(x - min_x, 4), round(y - min_y, 4)) for x, y in poly]


def build_part_polygon(spec: dict[str, Any]) -> tuple[PolygonCoords, PolygonCoords | None]:
    """Build outer polygon and optional interior cutout hole polygon from item spec."""
    kind = str(spec.get("shape", "rectangle")).lower()
    hole_poly: PolygonCoords | None = None

    if kind in ("rectangle", "rect", "countertop"):
        w = float(spec.get("width", 600.0))
        h = float(spec.get("height", 400.0))
        outer = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    elif kind == "square":
        s = float(spec.get("size", spec.get("width", 400.0)))
        outer = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)]
    elif kind == "circle":
        r = float(spec.get("radius", float(spec.get("diameter", 500.0)) * 0.5))
        n = int(spec.get("segments", 24))
        outer = [
            (r + r * math.cos(2.0 * math.pi * i / n), r + r * math.sin(2.0 * math.pi * i / n))
            for i in range(n)
        ]
    elif kind == "ellipse":
        rx = float(spec.get("rx", float(spec.get("width", 800.0)) * 0.5))
        ry = float(spec.get("ry", float(spec.get("height", 500.0)) * 0.5))
        n = int(spec.get("segments", 24))
        outer = [
            (rx + rx * math.cos(2.0 * math.pi * i / n), ry + ry * math.sin(2.0 * math.pi * i / n))
            for i in range(n)
        ]
    elif kind in ("l_shape", "l-shape", "l_countertop"):
        w = float(spec.get("width", 900.0))
        h = float(spec.get("height", 700.0))
        leg_w = float(spec.get("leg_width", w * 0.45))
        leg_h = float(spec.get("leg_height", h * 0.45))
        outer = [
            (0.0, 0.0),
            (w, 0.0),
            (w, leg_h),
            (leg_w, leg_h),
            (leg_w, h),
            (0.0, h),
        ]
    elif kind == "hollow_countertop":
        w = float(spec.get("width", 1200.0))
        h = float(spec.get("height", 650.0))
        outer = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        hole_w = float(spec.get("hole_width", w * 0.45))
        hole_h = float(spec.get("hole_height", h * 0.55))
        hx0 = (w - hole_w) * 0.5
        hy0 = (h - hole_h) * 0.5
        hole_poly = [
            (hx0, hy0),
            (hx0 + hole_w, hy0),
            (hx0 + hole_w, hy0 + hole_h),
            (hx0, hy0 + hole_h),
        ]
    elif kind in ("equilateral_triangle", "mosaic_triangle"):
        side = float(spec.get("side", spec.get("width", 180.0)))
        h = side * math.sqrt(3.0) * 0.5
        outer = [(0.0, 0.0), (side, 0.0), (side * 0.5, h)]
    elif kind in ("hexagon", "regular_hexagon", "mosaic_hexagon"):
        r = float(spec.get("radius", spec.get("side", 100.0)))
        pts = [(r * math.cos(math.radians(60.0 * i)), r * math.sin(math.radians(60.0 * i))) for i in range(6)]
        outer = normalize_to_origin(pts)
    elif kind == "triangle":
        if "side" in spec or spec.get("equilateral", False):
            side = float(spec.get("side", spec.get("width", 180.0)))
            h = side * math.sqrt(3.0) * 0.5
            outer = [(0.0, 0.0), (side, 0.0), (side * 0.5, h)]
        else:
            w = float(spec.get("width", 500.0))
            h = float(spec.get("height", 400.0))
            outer = [(0.0, 0.0), (w, 0.0), (w * 0.5, h)]
    elif kind == "rhombus":
        w = float(spec.get("width", 500.0))
        h = float(spec.get("height", 350.0))
        outer = [(w * 0.5, 0.0), (w, h * 0.5), (w * 0.5, h), (0.0, h * 0.5)]
    elif kind == "parallelogram":
        w = float(spec.get("width", 600.0))
        h = float(spec.get("height", 350.0))
        skew = float(spec.get("skew", w * 0.2))
        outer = [(0.0, 0.0), (w, 0.0), (w + skew, h), (skew, h)]
    elif kind == "sector":
        r = float(spec.get("radius", 450.0))
        ang = float(spec.get("angle", 90.0))
        n = max(8, int(spec.get("segments", 16)))
        pts = [(0.0, 0.0)]
        for i in range(n + 1):
            a = math.radians(ang * i / n)
            pts.append((r * math.cos(a), r * math.sin(a)))
        outer = normalize_to_origin(pts)
    elif kind == "star":
        r_out = float(spec.get("outer_radius", 300.0))
        r_in = float(spec.get("inner_radius", r_out * 0.45))
        pts = []
        for i in range(10):
            r = r_out if i % 2 == 0 else r_in
            a = math.radians(-90.0 + i * 36.0)
            pts.append((r_out + r * math.cos(a), r_out + r * math.sin(a)))
        outer = normalize_to_origin(pts)
    elif "polygon" in spec:
        raw = [(float(p[0]), float(p[1])) for p in spec["polygon"]]
        outer = normalize_to_origin(raw)
        if "hole_polygon" in spec and spec["hole_polygon"]:
            min_x, min_y, _, _ = bbox(raw)
            hole_poly = [(round(float(p[0]) - min_x, 4), round(float(p[1]) - min_y, 4)) for p in spec["hole_polygon"]]
    else:
        w = float(spec.get("width", 500.0))
        h = float(spec.get("height", 400.0))
        outer = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]

    return outer, hole_poly


def compute_shape_descriptors(poly: PolygonCoords, hole_poly: PolygonCoords | None = None) -> dict[str, float]:
    shp = ShapelyPolygon(poly)
    if not shp.is_valid:
        shp = shp.buffer(0)
    gross_area = float(shp.area)
    hole_area = polygon_area(hole_poly) if hole_poly else 0.0
    net_area = max(1.0, gross_area - hole_area)
    hull_area = max(1.0, float(shp.convex_hull.area))
    convexity = min(1.0, gross_area / hull_area)
    min_x, min_y, max_x, max_y = bbox(poly)
    w = max(1.0, max_x - min_x)
    h = max(1.0, max_y - min_y)
    aspect = min(w, h) / max(w, h)
    return {
        "gross_area": round(gross_area, 3),
        "net_area": round(net_area, 3),
        "hole_area": round(hole_area, 3),
        "convexity": round(convexity, 4),
        "aspect_ratio": round(aspect, 4),
        "bbox_w": round(w, 3),
        "bbox_h": round(h, 3),
    }


def resolve_allowed_rotations(
    item: dict[str, Any],
    case: dict[str, Any],
    dominant_vein_angle_deg: float | None = None,
) -> list[float]:
    """Determine allowed rotation angles in degrees, respecting grain-lock constraints."""
    grain_locked = bool(item.get("grain_locked", False))
    vein_tol = float(item.get("grain_tolerance_deg", case.get("constraints", {}).get("grain_tolerance_deg", 0.0)))
    vein_base = float(
        dominant_vein_angle_deg
        if dominant_vein_angle_deg is not None
        else case.get("board", {}).get("vein_angle_deg", 0.0)
    )

    if grain_locked:
        base_angles = [round(vein_base % 360.0, 2), round((vein_base + 180.0) % 360.0, 2), 0.0, 180.0]
        if vein_tol > 0.1:
            extra = []
            for b in base_angles[:2]:
                extra.extend([round((b - vein_tol) % 360.0, 2), round((b + vein_tol) % 360.0, 2)])
            base_angles.extend(extra)
        seen: list[float] = []
        for a in base_angles:
            if not any(abs(a - s) < 1e-3 for s in seen):
                seen.append(a)
        return seen

    if "rotations" in item and item["rotations"]:
        return [float(r) for r in item["rotations"]]
    if not item.get("allow_rotation", True):
        return [0.0]
    if str(item.get("shape", "")).lower() in ("equilateral_triangle", "mosaic_triangle") or item.get("equilateral", False):
        return [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    if str(item.get("shape", "")).lower() in ("hexagon", "regular_hexagon", "mosaic_hexagon"):
        return [0.0, 30.0]

    step = float(case.get("constraints", {}).get("rotation_step", 90.0))
    step = max(15.0, step)
    out: list[float] = []
    ang = 0.0
    while ang < 360.0 - 1e-5:
        out.append(round(ang, 2))
        ang += step
    return out


def expand_case_items(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand multi-quantity item specifications into individual sortable item instances."""
    vein_deg = case.get("board", {}).get("vein_angle_deg", None)
    if vein_deg is None and "texture_field" in case:
        vein_deg = case["texture_field"].get("dominant_vein_angle_deg", 0.0)

    expanded: list[dict[str, Any]] = []
    for item in case.get("items", []):
        qty = max(1, int(item.get("quantity", 1)))
        outer_poly, hole_poly = build_part_polygon(item)
        desc = compute_shape_descriptors(outer_poly, hole_poly)
        rotations = resolve_allowed_rotations(item, case, vein_deg)
        grain_locked = bool(item.get("grain_locked", False))
        priority = int(item.get("priority", 2 if desc["gross_area"] >= 350000.0 else 1))

        for idx in range(qty):
            expanded.append(
                {
                    "item_instance_id": f"{item.get('id', item.get('shape', 'part'))}_{idx}",
                    "type_id": str(item.get("id", f"type_{idx}")),
                    "shape": str(item.get("shape", "rectangle")),
                    "base_polygon": outer_poly,
                    "hole_polygon": hole_poly,
                    "area": desc["net_area"],
                    "gross_area": desc["gross_area"],
                    "convexity": desc["convexity"],
                    "aspect_ratio": desc["aspect_ratio"],
                    "rotations": rotations,
                    "grain_locked": grain_locked,
                    "priority": priority,
                    "tier": int(item.get("tier", 1 if priority >= 3 else (2 if priority == 2 else 3))),
                    "role": str(item.get("role", "standard")),
                    "vein_matching": bool(item.get("vein_matching", False)),
                    "golden_zone_required": bool(item.get("golden_zone_required", False)),
                    "tandem_sequence": bool(item.get("tandem_sequence", False)),
                    "grid_layout": item.get("grid_layout"),
                    "fill_remnants": bool(item.get("fill_remnants", False)),
                }
            )
    return expanded


def check_placement_feasible(
    poly: PolygonCoords,
    placed_records: list[dict[str, Any]],
    obstacle_shps: list[ShapelyPolygon],
    width: float,
    height: float,
    margin: float = 5.0,
    kerf: float = 3.0,
    obstacle_clearance: float = 0.0,
    cut_mode: str = "waterjet",
) -> tuple[bool, str | None]:
    """Check if `poly` is feasible against board boundaries, defect hazards, and placed parts.

    Supports Minkowski kerf clearance and Part-in-Part nesting inside interior sink holes
    when `cut_mode == 'waterjet'`.
    Returns `(is_feasible, nested_inside_parent_id)`.
    """
    min_x, min_y, max_x, max_y = bbox(poly)
    if (
        min_x < margin - EPS
        or min_y < margin - EPS
        or max_x > width - margin + EPS
        or max_y > height - margin + EPS
    ):
        return False, None

    shp = ShapelyPolygon(poly)
    if not shp.is_valid:
        shp = shp.buffer(0)

    # Check defect hazards (already buffered by safety_margin_mm, or enforce obstacle_clearance)
    for obs_shp in obstacle_shps:
        if shp.bounds[2] < obs_shp.bounds[0] - obstacle_clearance - EPS:
            continue
        if shp.bounds[0] > obs_shp.bounds[2] + obstacle_clearance + EPS:
            continue
        if shp.bounds[3] < obs_shp.bounds[1] - obstacle_clearance - EPS:
            continue
        if shp.bounds[1] > obs_shp.bounds[3] + obstacle_clearance + EPS:
            continue
        if obstacle_clearance > EPS:
            if shp.distance(obs_shp) < obstacle_clearance - 1e-3:
                return False, None
        else:
            if shp.intersects(obs_shp) and shp.intersection(obs_shp).area > 1e-2:
                return False, None

    nested_parent: str | None = None
    for rec in placed_records:
        other_shp: ShapelyPolygon = rec["shapely_outer"]
        other_bounds = other_shp.bounds
        # Fast AABB rejection
        if (
            max_x < other_bounds[0] - kerf - EPS
            or min_x > other_bounds[2] + kerf + EPS
            or max_y < other_bounds[1] - kerf - EPS
            or min_y > other_bounds[3] + kerf + EPS
        ):
            continue

        # Check Part-in-Part: is `shp` completely inside `rec`'s interior hole with kerf clearance?
        hole_shp: ShapelyPolygon | None = rec.get("shapely_hole")
        if cut_mode == "waterjet" and hole_shp is not None:
            inset_hole = hole_shp.buffer(-kerf)
            if not inset_hole.is_empty and inset_hole.contains(shp):
                nested_parent = rec["item_instance_id"]
                continue

        if cut_mode == "guillotine":
            # In guillotine mode, bounding boxes (expanded by kerf/2) must not overlap
            if not (
                max_x + kerf <= other_bounds[0] + EPS
                or min_x - kerf >= other_bounds[2] - EPS
                or max_y + kerf <= other_bounds[1] + EPS
                or min_y - kerf >= other_bounds[3] - EPS
            ):
                return False, None
        else:
            if shp.distance(other_shp) < kerf - 1e-2:
                return False, None

    return True, nested_parent
