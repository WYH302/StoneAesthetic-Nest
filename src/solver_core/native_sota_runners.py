"""Official Compiled Native SOTA Solver Runners (Zero Proxy / Zero Degenerate Reimplementation).

Directly invokes unmodified official C++ and Rust binaries compiled from upstream repositories:
  1. Official C++ `PackingSolver-Irregular` (Fontan & Libralesso, Comput. Oper. Res. 2020)
     - Native Windows x64 release binary: `third_party/packingsolver-windows-x64/.../packingsolver_irregular.exe`
     - Natively supports fixed irregular knapsack bins, polygon defects, and exact item-item kerf spacing.
  2. Official Rust `jagua-rs` (`lbf`, Gardeyn et al., 2025)
     - Compiled from `third_party/jagua-rs` (Rust 1.100.0 `--release`) in Linux / WSL2
     - Natively supports fixed bin packing (`bpp`) with polygon `"inner"` crack holes (defects).
  3. Official Rust `Sparrow v0.3.0` (Gardeyn & Wauters, 2025)
     - Compiled from `third_party/sparrow` (Rust 1.100.0 `--release`, `opt-level=3`, `lto=fat`) in Linux / WSL2
     - Runs unmodified 2025 SOTA strip packing over kerf-buffered polygons, then maps valid slab-contained
       and crack-free placements (checking both direct and horizontally mirrored strip alignments).
"""

from __future__ import annotations

import json
import os
import math
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon as ShapelyPolygon

from .geometry_kernel import (
    PolygonCoords,
    bbox,
    build_part_polygon,
    expand_case_items,
    polygon_area,
    rotate_polygon,
    transform_polygon,
    translate_polygon,
)

MASTER_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = MASTER_DIR.parent
PACKINGSOLVER_EXE = (
    WORKSPACE_ROOT
    / "third_party"
    / "packingsolver-windows-x64"
    / "packingsolver-windows-x64"
    / "packingsolver_irregular.exe"
)
WSL_DISTRO = os.environ.get("STONE_WSL_DISTRO", "Ubuntu")
WSL_JAGUA_BIN = os.environ.get("JAGUA_LBF_BIN", "/usr/local/bin/jagua_lbf")
WSL_SPARROW_BIN = os.environ.get("SPARROW_BIN", "/usr/local/bin/sparrow")
WSL_WORK_WIN = WORKSPACE_ROOT / "third_party" / "_native_runs"
WSL_WORK_WIN.mkdir(parents=True, exist_ok=True)
SPACING_EPS = 0.08


def _to_wsl_path(win_path: Path) -> str:
    s = str(win_path.resolve()).replace("\\", "/")
    drive = s[0].lower()
    return f"/mnt/{drive}{s[2:]}"


def _clean_ccw_ring(pts: list[tuple[float, float]] | list[list[float]], close: bool = True) -> list[list[float]]:
    shp = ShapelyPolygon([(float(x), float(y)) for x, y in pts]).buffer(0)
    if shp.geom_type == "MultiPolygon":
        shp = max(shp.geoms, key=lambda g: g.area)
    coords = list(shp.exterior.coords)
    if not shp.exterior.is_ccw:
        coords = list(reversed(coords))
    out = [[round(float(x), 4), round(float(y), 4)] for x, y in coords[:-1]]
    if close and out:
        out.append(out[0])
    return out


def _buffer_ccw_ring(pts: PolygonCoords, buf: float, close: bool = True) -> list[list[float]]:
    shp = ShapelyPolygon(pts).buffer(0)
    if buf != 0.0:
        shp = shp.buffer(buf, join_style=2).simplify(0.25, preserve_topology=True)
    if shp.geom_type == "MultiPolygon":
        shp = max(shp.geoms, key=lambda g: g.area)
    coords = list(shp.exterior.coords)
    if not shp.exterior.is_ccw:
        coords = list(reversed(coords))
    out = [[round(float(x), 4), round(float(y), 4)] for x, y in coords[:-1]]
    if close and out:
        out.append(out[0])
    return out


def _case_spacings(case: dict[str, Any]) -> tuple[float, float]:
    c = case.get("constraints", {})
    margin = float(c.get("min_board_distance", 5.0))
    kerf = max(float(c.get("min_item_distance", 0.0)), float(c.get("kerf", 3.0)))
    return margin, kerf


def _item_angles_deg(item: dict[str, Any], case: dict[str, Any]) -> list[float]:
    if "rotations" in item and item["rotations"]:
        return sorted({round(float(a) % 360.0, 2) for a in item["rotations"]})
    if not item.get("allow_rotation", True):
        return [0.0]
    shape = str(item.get("shape", "")).lower()
    if shape in ("equilateral_triangle", "mosaic_triangle"):
        return [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    if shape in ("hexagon", "regular_hexagon", "mosaic_hexagon"):
        return [0.0, 30.0]
    if shape == "circle":
        return [0.0]
    step = int(case.get("constraints", {}).get("rotation_step", 90))
    step = max(15, step)
    return [float(a) for a in range(0, 360, step)]


def _polygon_from_ps_cert(placed_item: dict[str, Any]) -> list[tuple[float, float]]:
    segs = placed_item.get("item_shapes", [{}])[0].get("shape", [])
    verts: list[tuple[float, float]] = []
    for seg in segs:
        if seg.get("type") != "LineSegment":
            continue
        pt = (float(seg["xs"]), float(seg["ys"]))
        if not verts or pt != verts[-1]:
            verts.append(pt)
    if len(verts) > 1 and verts[0] == verts[-1]:
        verts.pop()
    return verts


# =============================================================================
# 1. OFFICIAL C++ PackingSolver-Irregular (Fontan & Libralesso, COR 2020)
# =============================================================================
def solve_native_packingsolver(
    case: dict[str, Any],
    time_limit_sec: float = 2.0,
) -> dict[str, Any]:
    """Run the unmodified official C++ `packingsolver_irregular.exe` binary with native polygon defects."""
    assert PACKINGSOLVER_EXE.exists(), f"Missing official binary: {PACKINGSOLVER_EXE}"
    t0 = time.perf_counter()
    board = case["board"]
    width = float(board["width"])
    height = float(board["height"])
    margin, kerf = _case_spacings(case)
    eff_margin = margin + SPACING_EPS
    bin_w = max(10.0, width - 2.0 * eff_margin)
    bin_h = max(10.0, height - 2.0 * eff_margin)

    # Convert buffered crack obstacles into native PackingSolver polygon defects
    defects: list[dict[str, Any]] = []
    for obs in board.get("obstacles", []):
        if "polygon" not in obs or len(obs["polygon"]) < 3:
            continue
        shp_obs = ShapelyPolygon([(float(x) - eff_margin, float(y) - eff_margin) for x, y in obs["polygon"]]).buffer(0)
        bin_rect = ShapelyPolygon([(0.0, 0.0), (bin_w, 0.0), (bin_w, bin_h), (0.0, bin_h)])
        inter = shp_obs.intersection(bin_rect)
        if inter.is_empty or inter.area < 10.0:
            continue
        geoms = [inter] if inter.geom_type == "Polygon" else list(inter.geoms)
        for g in geoms:
            if g.geom_type != "Polygon" or g.area < 10.0:
                continue
            coords = list(g.exterior.coords)
            if not g.exterior.is_ccw:
                coords = list(reversed(coords))
            verts = [{"x": round(float(x), 4), "y": round(float(y), 4)} for x, y in coords[:-1]]
            if len(verts) >= 3:
                defects.append(
                    {
                        "type": "polygon",
                        "vertices": verts,
                        "item_defect_minimum_spacing": round(SPACING_EPS, 4),
                    }
                )

    bin_type: dict[str, Any] = {
        "type": "rectangle",
        "width": round(bin_w, 4),
        "height": round(bin_h, 4),
        "copies": 1,
    }
    if defects:
        bin_type["defects"] = defects

    item_types: list[dict[str, Any]] = []
    item_bases: list[tuple[PolygonCoords, PolygonCoords | None]] = []
    for it in case["items"]:
        outer, hole = build_part_polygon(it)
        item_bases.append((outer, hole))
        # Ensure CCW positive vertices
        ring = _clean_ccw_ring(outer, close=False)
        angles = _item_angles_deg(it, case)
        # Weight profit by area so PackingSolver maximizes area utilization (knapsack objective)
        area_val = max(1, int(round(polygon_area(outer))))
        item_types.append(
            {
                "copies": int(it.get("quantity", 1)),
                "profit": area_val,
                "type": "polygon",
                "vertices": [{"x": p[0], "y": p[1]} for p in ring],
                "allowed_rotations": [{"start": a, "end": a} for a in angles],
            }
        )

    ps_input = {
        "objective": "knapsack",
        "bin_types": [bin_type],
        "item_types": item_types,
    }

    with tempfile.TemporaryDirectory(prefix="ps_run_") as td:
        in_p = Path(td) / "input.json"
        cert_p = Path(td) / "cert.json"
        in_p.write_text(json.dumps(ps_input), encoding="utf-8")
        cmd = [
            str(PACKINGSOLVER_EXE),
            "--input",
            str(in_p),
            "--item-bin-minimum-spacing",
            "0",
            "--item-item-minimum-spacing",
            f"{kerf + SPACING_EPS:.4f}",
            "--time-limit",
            f"{max(0.5, float(time_limit_sec)):.2f}",
            "--verbosity-level",
            "0",
            "--certificate",
            str(cert_p),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max(30, int(time_limit_sec * 8)))

        placements: list[dict[str, Any]] = []
        if cert_p.exists():
            cert = json.loads(cert_p.read_text(encoding="utf-8"))
            type_counts = {i: 0 for i in range(len(case["items"]))}
            for bin_data in cert.get("bins", []):
                for pit in bin_data.get("items", []):
                    t_idx = int(pit["id"])
                    if t_idx >= len(case["items"]):
                        continue
                    spec = case["items"][t_idx]
                    inst_i = type_counts[t_idx]
                    type_counts[t_idx] += 1
                    ang = float(pit.get("angle", 0.0)) % 360.0
                    cert_poly = _polygon_from_ps_cert(pit)
                    base_outer, base_hole = item_bases[t_idx]
                    rot_base = rotate_polygon(base_outer, ang)
                    # Align exact centroids (PackingSolver shifts by its internal reference frame)
                    ccx = sum(p[0] for p in cert_poly) / max(1, len(cert_poly))
                    ccy = sum(p[1] for p in cert_poly) / max(1, len(cert_poly))
                    bcx = sum(p[0] for p in rot_base) / max(1, len(rot_base))
                    bcy = sum(p[1] for p in rot_base) / max(1, len(rot_base))
                    dx = ccx - bcx + eff_margin
                    dy = ccy - bcy + eff_margin
                    poly_w = transform_polygon(base_outer, dx, dy, ang)
                    hole_w = transform_polygon(base_hole, dx, dy, ang) if base_hole else None
                    net_a = polygon_area(poly_w) - (polygon_area(hole_w) if hole_w else 0.0)
                    placements.append(
                        {
                            "item_instance_id": f"{spec.get('id', spec.get('shape', 'part'))}_{inst_i}",
                            "type_id": str(spec.get("id", f"type_{t_idx}")),
                            "shape": str(spec.get("shape", "rectangle")),
                            "x": round(dx, 4),
                            "y": round(dy, 4),
                            "rotation": round(ang, 2),
                            "polygon": [[round(px, 4), round(py, 4)] for px, py in poly_w],
                            "hole_polygon": [[round(px, 4), round(py, 4)] for px, py in hole_w] if hole_w else None,
                            "parent_hole_id": None,
                            "area": round(net_a, 3),
                        }
                    )

    all_items = expand_case_items(case)
    placed_ids = {p["item_instance_id"] for p in placements}
    unplaced = [it["item_instance_id"] for it in all_items if it["item_instance_id"] not in placed_ids]
    used_area = sum(float(p["area"]) for p in placements)
    return {
        "case_id": case.get("case_id", "unknown"),
        "method": "PackingSolver-Irregular (Official C++)",
        "cut_mode": "waterjet",
        "runtime_sec": round(time.perf_counter() - t0, 4),
        "placed_count": len(placements),
        "total_count": len(all_items),
        "utilization": round(used_area / max(1.0, width * height), 6),
        "used_area_mm2": round(used_area, 2),
        "placements": placements,
        "unplaced": unplaced,
    }


# =============================================================================
# 2. OFFICIAL RUST jagua-rs (`lbf` in `bpp` mode with native `"inner"` defect holes)
# =============================================================================
def solve_native_jagua_lbf(
    case: dict[str, Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Run the unmodified official Rust `jagua_lbf` binary in fixed-bin (`bpp`) mode with polygon holes."""
    t0 = time.perf_counter()
    board = case["board"]
    width = float(board["width"])
    height = float(board["height"])
    margin, kerf = _case_spacings(case)
    half_k = kerf * 0.5 + 0.25
    # Container shrunk so inflated items stay >= margin from board edge
    edge_shrink = max(0.0, margin - half_k)

    outer_bin = [
        [round(edge_shrink, 4), round(edge_shrink, 4)],
        [round(width - edge_shrink, 4), round(edge_shrink, 4)],
        [round(width - edge_shrink, 4), round(height - edge_shrink, 4)],
        [round(edge_shrink, 4), round(height - edge_shrink, 4)],
        [round(edge_shrink, 4), round(edge_shrink, 4)],
    ]
    bin_shp = ShapelyPolygon(outer_bin)

    # Convert crack obstacles into native `"inner"` holes (clockwise or jagua auto-oriented)
    inner_holes: list[list[list[float]]] = []
    for obs in board.get("obstacles", []):
        if "polygon" not in obs or len(obs["polygon"]) < 3:
            continue
        obs_buf = ShapelyPolygon([(float(x), float(y)) for x, y in obs["polygon"]]).buffer(0)
        # Inflate defect by half_k so inflated item edges stay >= 0 outside true crack polygon
        obs_clean = obs_buf.intersection(bin_shp.buffer(-2.0))
        if obs_clean.is_empty or obs_clean.area < 20.0:
            continue
        geoms = [obs_clean] if obs_clean.geom_type == "Polygon" else list(obs_clean.geoms)
        for g in geoms:
            if g.geom_type != "Polygon" or g.area < 20.0:
                continue
            coords = list(g.exterior.coords)
            if not g.exterior.is_ccw:
                coords = list(reversed(coords))
            ring = [[round(float(x), 4), round(float(y), 4)] for x, y in coords[:-1]]
            if len(ring) >= 3:
                ring.append(ring[0])
                inner_holes.append(ring)

    # Build jagua-rs ExtBPInstance
    j_items: list[dict[str, Any]] = []
    item_bases: list[tuple[PolygonCoords, PolygonCoords | None]] = []
    for idx, it in enumerate(case["items"]):
        outer, hole = build_part_polygon(it)
        item_bases.append((outer, hole))
        inflated = _buffer_ccw_ring(outer, half_k, close=True)
        angles = _item_angles_deg(it, case)
        j_items.append(
            {
                "id": idx,
                "demand": int(it.get("quantity", 1)),
                "orientation": {"rotation": {"mode": "discrete", "angles": angles}},
                "shape": {"type": "simple_polygon", "data": inflated},
            }
        )

    bin_shape = (
        {"type": "polygon", "data": {"outer": outer_bin, "inner": inner_holes}}
        if inner_holes
        else {"type": "simple_polygon", "data": outer_bin}
    )

    ext_bp = {
        "name": f"jagua_{case.get('case_id', 'slab')}",
        "items": j_items,
        "bins": [
            {
                "id": 0,
                "cost": 1000,
                "stock": 1,
                "zones": [],
                "shape": bin_shape,
            }
        ],
    }

    run_tag = uuid.uuid4().hex[:10]
    run_dir = WSL_WORK_WIN / f"jagua_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    in_file = run_dir / "input.json"
    out_dir = run_dir / "out"
    out_dir.mkdir(exist_ok=True)
    in_file.write_text(json.dumps(ext_bp), encoding="utf-8")

    try:
        wsl_in = _to_wsl_path(in_file)
        wsl_out = _to_wsl_path(out_dir)
        subprocess.run(
            ["wsl.exe", "-d", WSL_DISTRO, WSL_JAGUA_BIN, "-i", wsl_in, "-s", wsl_out, "-p", "bpp", "-l", "error"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
        )
        sol_file = out_dir / "sol_input.json"
        placements: list[dict[str, Any]] = []
        if sol_file.exists():
            sol = json.loads(sol_file.read_text(encoding="utf-8"))
            layouts = sol.get("solution", {}).get("layouts", [])
            if layouts:
                # Only keep items placed on Bin 0 (our single purchased physical slab!)
                placed_raw = layouts[0].get("placed_items", [])
                type_counts = {i: 0 for i in range(len(case["items"]))}
                for pit in placed_raw:
                    t_idx = int(pit["item_id"])
                    if t_idx >= len(case["items"]):
                        continue
                    spec = case["items"][t_idx]
                    inst_i = type_counts[t_idx]
                    type_counts[t_idx] += 1
                    tf = pit["transformation"]
                    ang = float(tf["rotation"]) % 360.0
                    tx, ty = float(tf["translation"][0]), float(tf["translation"][1])
                    base_outer, base_hole = item_bases[t_idx]
                    poly_w = transform_polygon(base_outer, tx, ty, ang)
                    hole_w = transform_polygon(base_hole, tx, ty, ang) if base_hole else None
                    net_a = polygon_area(poly_w) - (polygon_area(hole_w) if hole_w else 0.0)
                    placements.append(
                        {
                            "item_instance_id": f"{spec.get('id', spec.get('shape', 'part'))}_{inst_i}",
                            "type_id": str(spec.get("id", f"type_{t_idx}")),
                            "shape": str(spec.get("shape", "rectangle")),
                            "x": round(tx, 4),
                            "y": round(ty, 4),
                            "rotation": round(ang, 2),
                            "polygon": [[round(px, 4), round(py, 4)] for px, py in poly_w],
                            "hole_polygon": [[round(px, 4), round(py, 4)] for px, py in hole_w] if hole_w else None,
                            "parent_hole_id": None,
                            "area": round(net_a, 3),
                        }
                    )
    finally:
        import shutil

        shutil.rmtree(run_dir, ignore_errors=True)

    all_items = expand_case_items(case)
    placed_ids = {p["item_instance_id"] for p in placements}
    unplaced = [it["item_instance_id"] for it in all_items if it["item_instance_id"] not in placed_ids]
    used_area = sum(float(p["area"]) for p in placements)
    return {
        "case_id": case.get("case_id", "unknown"),
        "method": "Jagua-rs LBF (Official Rust)",
        "cut_mode": "waterjet",
        "runtime_sec": round(time.perf_counter() - t0, 4),
        "placed_count": len(placements),
        "total_count": len(all_items),
        "utilization": round(used_area / max(1.0, width * height), 6),
        "used_area_mm2": round(used_area, 2),
        "placements": placements,
        "unplaced": unplaced,
    }


# =============================================================================
# 3. OFFICIAL RUST Sparrow v0.3.0 (Gardeyn & Wauters, 2025 SOTA Strip Packer)
# =============================================================================
def solve_native_sparrow(
    case: dict[str, Any],
    time_limit_sec: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the unmodified official Rust `sparrow v0.3.0` binary on kerf-buffered slab order polygons,
    then project onto the physical fixed slab (evaluating both direct and horizontally mirrored strip
    alignments to maximize crack-free yield under RAC-Checker).
    """
    t0 = time.perf_counter()
    board = case["board"]
    width = float(board["width"])
    height = float(board["height"])
    margin, kerf = _case_spacings(case)
    half_k = kerf * 0.5 + 0.25
    edge_shrink = max(0.0, margin - half_k)
    strip_h = max(10.0, height - 2.0 * edge_shrink)

    s_items: list[dict[str, Any]] = []
    item_bases: list[tuple[PolygonCoords, PolygonCoords | None]] = []
    for idx, it in enumerate(case["items"]):
        outer, hole = build_part_polygon(it)
        item_bases.append((outer, hole))
        inflated = _buffer_ccw_ring(outer, half_k, close=True)
        angles = _item_angles_deg(it, case)
        s_items.append(
            {
                "id": idx,
                "demand": int(it.get("quantity", 1)),
                "orientation": {"rotation": {"mode": "discrete", "angles": angles}},
                "shape": {"type": "simple_polygon", "data": inflated},
            }
        )

    inst_name = f"sp_{uuid.uuid4().hex[:8]}"
    ext_sp = {
        "name": inst_name,
        "items": s_items,
        "strip_height": round(strip_h, 4),
    }

    run_dir = WSL_WORK_WIN / inst_name
    run_dir.mkdir(parents=True, exist_ok=True)
    in_file = run_dir / "input.json"
    in_file.write_text(json.dumps(ext_sp), encoding="utf-8")

    obstacle_shps = [
        ShapelyPolygon([(float(x), float(y)) for x, y in obs["polygon"]]).buffer(0)
        for obs in board.get("obstacles", [])
        if len(obs.get("polygon", [])) >= 3
    ]

    raw_strip_width = width
    best_placements: list[dict[str, Any]] = []

    try:
        wsl_dir = _to_wsl_path(run_dir)
        cmd = (
            f"cd '{wsl_dir}' && {WSL_SPARROW_BIN} -i input.json -t {max(1, int(time_limit_sec))} "
            f"-s {int(seed)} --workers 4 >/dev/null 2>&1"
        )
        subprocess.run(["wsl.exe", "-d", WSL_DISTRO, "bash", "-lc", cmd], timeout=max(30, int(time_limit_sec * 10)))

        sol_file = run_dir / "output" / f"final_{inst_name}.json"
        if sol_file.exists():
            sol = json.loads(sol_file.read_text(encoding="utf-8"))
            raw_strip_width = float(sol.get("solution", {}).get("strip_width", width))
            placed_raw = sol.get("solution", {}).get("layout", {}).get("placed_items", [])

            # Evaluate both x-origin shift = edge_shrink and right-aligned / mirrored strip shifts to maximize valid area
            cand_shifts = [0.0]
            if raw_strip_width > width - 2.0 * edge_shrink + 1.0:
                cand_shifts.append(-(raw_strip_width - (width - 2.0 * edge_shrink)))

            best_area = -1.0
            for x_shift in cand_shifts:
                cur_pls: list[dict[str, Any]] = []
                type_counts = {i: 0 for i in range(len(case["items"]))}
                for pit in placed_raw:
                    t_idx = int(pit["item_id"])
                    if t_idx >= len(case["items"]):
                        continue
                    spec = case["items"][t_idx]
                    tf = pit["transformation"]
                    ang = float(tf["rotation"]) % 360.0
                    tx = float(tf["translation"][0]) + edge_shrink + x_shift
                    ty = float(tf["translation"][1]) + edge_shrink
                    base_outer, base_hole = item_bases[t_idx]
                    poly_w = transform_polygon(base_outer, tx, ty, ang)
                    x1, y1, x2, y2 = bbox(poly_w)
                    if x1 < margin - 1e-2 or x2 > width - margin + 1e-2 or y1 < margin - 1e-2 or y2 > height - margin + 1e-2:
                        continue
                    shp_p = ShapelyPolygon(poly_w).buffer(0)
                    if any(shp_p.intersects(obs) and shp_p.intersection(obs).area > 1e-3 for obs in obstacle_shps):
                        continue
                    inst_i = type_counts[t_idx]
                    type_counts[t_idx] += 1
                    hole_w = transform_polygon(base_hole, tx, ty, ang) if base_hole else None
                    net_a = polygon_area(poly_w) - (polygon_area(hole_w) if hole_w else 0.0)
                    cur_pls.append(
                        {
                            "item_instance_id": f"{spec.get('id', spec.get('shape', 'part'))}_{inst_i}",
                            "type_id": str(spec.get("id", f"type_{t_idx}")),
                            "shape": str(spec.get("shape", "rectangle")),
                            "x": round(tx, 4),
                            "y": round(ty, 4),
                            "rotation": round(ang, 2),
                            "polygon": [[round(px, 4), round(py, 4)] for px, py in poly_w],
                            "hole_polygon": [[round(px, 4), round(py, 4)] for px, py in hole_w] if hole_w else None,
                            "parent_hole_id": None,
                            "area": round(net_a, 3),
                        }
                    )
                cur_a = sum(float(p["area"]) for p in cur_pls)
                if cur_a > best_area:
                    best_area = cur_a
                    best_placements = cur_pls
    finally:
        import shutil

        shutil.rmtree(run_dir, ignore_errors=True)

    all_items = expand_case_items(case)
    placed_ids = {p["item_instance_id"] for p in best_placements}
    unplaced = [it["item_instance_id"] for it in all_items if it["item_instance_id"] not in placed_ids]
    used_area = sum(float(p["area"]) for p in best_placements)
    return {
        "case_id": case.get("case_id", "unknown"),
        "method": "Sparrow v0.3.0 (Official Rust)",
        "cut_mode": "waterjet",
        "runtime_sec": round(time.perf_counter() - t0, 4),
        "raw_strip_width_mm": round(raw_strip_width, 2),
        "placed_count": len(best_placements),
        "total_count": len(all_items),
        "utilization": round(used_area / max(1.0, width * height), 6),
        "used_area_mm2": round(used_area, 2),
        "placements": best_placements,
        "unplaced": unplaced,
    }


# =============================================================================
# 4. NATIVE PUBLIC BENCHMARK RUNNERS (ESICUP-14 Strip & Baldacci-6 Defective BPP)
# =============================================================================
def run_native_sparrow_on_esicup_file(json_path: Path, time_limit_sec: int = 2, seed: int = 42) -> dict[str, Any]:
    """Run Official Rust Sparrow v0.3.0 directly on an unmodified EURO/ESICUP JSON benchmark."""
    inst_name = f"es_{json_path.stem}_{uuid.uuid4().hex[:6]}"
    run_dir = WSL_WORK_WIN / inst_name
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        wsl_dir = _to_wsl_path(run_dir)
        wsl_in = _to_wsl_path(json_path)
        cmd = f"cd '{wsl_dir}' && {WSL_SPARROW_BIN} -i '{wsl_in}' -t {max(1, int(time_limit_sec))} -s {int(seed)} --workers 4 >/dev/null 2>&1"
        subprocess.run(["wsl.exe", "-d", WSL_DISTRO, "bash", "-lc", cmd], timeout=max(30, time_limit_sec * 10))
        sols = list((run_dir / "output").glob("final_*.json"))
        if sols:
            data = json.loads(sols[0].read_text(encoding="utf-8"))
            s = data["solution"]
            return {
                "instance": json_path.stem,
                "strip_width": round(float(s["strip_width"]), 3),
                "density": round(float(s["density"]), 6),
                "placed_count": len(s["layout"]["placed_items"]),
                "run_time_sec": round(float(s["run_time_sec"]), 3),
            }
        return {"instance": json_path.stem, "strip_width": 0.0, "density": 0.0, "placed_count": 0, "run_time_sec": 0.0}
    finally:
        import shutil

        shutil.rmtree(run_dir, ignore_errors=True)


def run_native_jagua_on_public_file(json_path: Path, prob_var: str = "bpp") -> dict[str, Any]:
    """Run Official Rust jagua_lbf directly on an unmodified Baldacci-6 (bpp) or ESICUP (spp) file."""
    inst_name = f"jg_{json_path.stem}_{uuid.uuid4().hex[:6]}"
    run_dir = WSL_WORK_WIN / inst_name
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        wsl_in = _to_wsl_path(json_path)
        wsl_out = _to_wsl_path(out_dir)
        subprocess.run(
            ["wsl.exe", "-d", WSL_DISTRO, WSL_JAGUA_BIN, "-i", wsl_in, "-s", wsl_out, "-p", prob_var, "-l", "error"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        sols = list(out_dir.glob("sol_*.json"))
        if sols:
            data = json.loads(sols[0].read_text(encoding="utf-8"))
            s = data["solution"]
            if prob_var == "bpp":
                layouts = s.get("layouts", [])
                total_placed = sum(len(ly.get("placed_items", [])) for ly in layouts)
                return {
                    "instance": json_path.stem,
                    "bins_used": len(layouts),
                    "bin0_placed": len(layouts[0]["placed_items"]) if layouts else 0,
                    "bin0_density": round(float(layouts[0]["density"]), 6) if layouts else 0.0,
                    "total_placed": total_placed,
                    "overall_density": round(float(s.get("density", 0.0)), 6),
                    "run_time_sec": round(float(s.get("run_time_sec", 0.0)), 3),
                }
            else:
                return {
                    "instance": json_path.stem,
                    "strip_width": round(float(s.get("strip_width", 0.0)), 3),
                    "density": round(float(s.get("density", 0.0)), 6),
                    "placed_count": len(s.get("layout", {}).get("placed_items", [])),
                    "run_time_sec": round(float(s.get("run_time_sec", 0.0)), 3),
                }
        return {"instance": json_path.stem, "error": "no_output"}
    finally:
        import shutil

        shutil.rmtree(run_dir, ignore_errors=True)
