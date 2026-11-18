"""Texture-Preserving Original Slab Overlay & Cut-Piece Vein Gallery Renderer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PALETTE_BGR = [
    (235, 99, 37),   # 1: Royal Blue
    (74, 163, 22),   # 2: Emerald Green
    (11, 158, 245),  # 3: Amber Gold
    (237, 58, 124),  # 4: Violet Purple
    (178, 145, 8),   # 5: Cyan Teal
    (119, 39, 219),  # 6: Rose Pink
    (229, 70, 79),   # 7: Indigo
    (136, 148, 13),  # 8: Teal Green
    (12, 88, 234),   # 9: Deep Orange
    (13, 163, 101),  # 10: Lime Green
]

PALETTE_HEX = [
    "#2563eb",
    "#16a34a",
    "#f59e0b",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#4f46e5",
    "#0d9488",
    "#ea580c",
    "#65a30d",
]


def build_canonical_part_map(case: dict[str, Any]) -> dict[str, tuple[int, tuple[int, int, int], str]]:
    """Build a deterministic 1-based canonical (part_number, color_bgr, color_hex) map per item_instance_id."""
    from src.solver_core.solver import expand_case_items

    expanded = expand_case_items(case)
    ordered = sorted(
        expanded,
        key=lambda it: (-int(it.get("priority", 1)), -float(it.get("area", 0.0)), str(it.get("item_instance_id", ""))),
    )
    cmap: dict[str, tuple[int, tuple[int, int, int], str]] = {}
    for idx, it in enumerate(ordered):
        bgr = PALETTE_BGR[idx % len(PALETTE_BGR)]
        hx = PALETTE_HEX[idx % len(PALETTE_HEX)]
        cmap[str(it["item_instance_id"])] = (idx + 1, bgr, hx)
    return cmap


def render_overlay_on_slab_image(
    rectified_bgr: np.ndarray,
    case: dict[str, Any],
    result: dict[str, Any],
    output_png_path: str | Path | None = None,
) -> np.ndarray:
    """Render semi-transparent part layouts, defect hazards, and vein arrows onto the slab photo."""
    canvas = rectified_bgr.copy()
    overlay = rectified_bgr.copy()
    h_px, w_px = canvas.shape[:2]
    canon_map = build_canonical_part_map(case)

    board = case["board"]
    w_mm = float(board["width"])
    h_mm = float(board["height"])
    sx = w_px / max(1.0, w_mm)
    sy = h_px / max(1.0, h_mm)

    def to_px_pts(poly_mm: list[list[float]] | list[tuple[float, float]]) -> np.ndarray:
        # Note: in image coordinates, y=0 is top; we map mm y in [0, h_mm] to image y in [h_px, 0]
        pts = [[int(round(float(x) * sx)), int(round(h_px - float(y) * sy))] for x, y in poly_mm]
        return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

    # 1. Draw Defect Hazards in crimson red with cross-hatching
    for obs in board.get("obstacles", []):
        pts_px = to_px_pts(obs["polygon"])
        cv2.fillPoly(overlay, [pts_px], (38, 38, 220))
        cv2.polylines(canvas, [pts_px], isClosed=True, color=(20, 20, 180), thickness=2, lineType=cv2.LINE_AA)

    # 2. Draw Placed Parts (outer parents first, PiP nested children second) with canonical ID & color
    placements_sorted = sorted(
        result.get("placements", []),
        key=lambda p: (1 if p.get("nested_inside") is not None else 0, canon_map.get(str(p.get("item_instance_id", "")), (999, PALETTE_BGR[0], ""))[0]),
    )
    for idx, p in enumerate(placements_sorted):
        c_num, color_bgr, _ = canon_map.get(str(p.get("item_instance_id", "")), (idx + 1, PALETTE_BGR[idx % len(PALETTE_BGR)], "#2563eb"))
        pts_px = to_px_pts(p["polygon"])

        if p.get("hole_polygon"):
            hole_px = to_px_pts(p["hole_polygon"])
            # Fill outer minus hole on overlay
            mask_part = np.zeros((h_px, w_px), dtype=np.uint8)
            cv2.fillPoly(mask_part, [pts_px], 255)
            cv2.fillPoly(mask_part, [hole_px], 0)
            overlay[mask_part > 0] = color_bgr
            cv2.polylines(canvas, [hole_px], isClosed=True, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        else:
            cv2.fillPoly(overlay, [pts_px], color_bgr)

        cv2.polylines(canvas, [pts_px], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

    # Alpha blend translucent fill over original stone veins
    blended = cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0.0)

    # 2b. Draw Golden Vein Zone (黄金主纹带) boundary if present
    gz = result.get("golden_vein_zone_mm")
    if gz and len(gz) == 4:
        gx0, gy0, gx1, gy1 = [float(v) for v in gz]
        gz_pts = to_px_pts([[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1]])
        cv2.polylines(blended, [gz_pts], isClosed=True, color=(0, 215, 255), thickness=3, lineType=cv2.LINE_AA)
        gx_tl = int(round(gx0 * sx)) + 8
        gy_tl = int(round(h_px - gy1 * sy)) + 22
        cv2.putText(blended, "GOLDEN VEIN ZONE", (gx_tl, gy_tl), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blended, "GOLDEN VEIN ZONE", (gx_tl, gy_tl), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 225, 255), 1, cv2.LINE_AA)

    # 3. Draw crisp borders, grain direction arrows, and part labels on top
    for obs in board.get("obstacles", []):
        pts_px = to_px_pts(obs["polygon"])
        cv2.polylines(blended, [pts_px], isClosed=True, color=(0, 0, 220), thickness=2, lineType=cv2.LINE_AA)
        M = cv2.moments(pts_px)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(
                blended,
                str(obs.get("id", "CRACK")),
                (cx - 28, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                blended,
                str(obs.get("id", "CRACK")),
                (cx - 28, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 200),
                1,
                cv2.LINE_AA,
            )

    for idx, p in enumerate(placements_sorted):
        c_num, color_bgr, _ = canon_map.get(str(p.get("item_instance_id", "")), (idx + 1, PALETTE_BGR[idx % len(PALETTE_BGR)], "#2563eb"))
        pts_px = to_px_pts(p["polygon"])
        cv2.polylines(blended, [pts_px], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)
        if p.get("hole_polygon"):
            hole_px = to_px_pts(p["hole_polygon"])
            cv2.polylines(blended, [hole_px], isClosed=True, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        raw_pts = pts_px.reshape(-1, 2)
        cx = int(np.mean(raw_pts[:, 0]))
        if p.get("hole_polygon"):
            y_min_px, y_max_px = int(np.min(raw_pts[:, 1])), int(np.max(raw_pts[:, 1]))
            cy = int(y_min_px + 0.15 * (y_max_px - y_min_px))
        else:
            cy = int(np.mean(raw_pts[:, 1]))

        # Draw grain direction arrow if grain_locked or Tier 1/2
        if p.get("grain_locked", True) or int(p.get("tier", 0)) in (1, 2):
            rot_rad = math.radians(float(p.get("rotation", 0.0)))
            arrow_len = 26.0 if int(p.get("tier", 0)) != 3 else 14.0
            dx = int(round(arrow_len * math.cos(rot_rad)))
            dy = int(round(-arrow_len * math.sin(rot_rad)))
            cv2.arrowedLine(
                blended,
                (cx - dx, cy - dy),
                (cx + dx, cy + dy),
                (255, 255, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )

        tag = f"#{p['aesthetic_tag']}" if p.get("aesthetic_tag") else f"#{c_num}"
        f_scale = 0.44 if str(p.get("aesthetic_tag", "")).startswith("M") else 0.52
        cv2.putText(blended, tag, (cx - 16, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blended, tag, (cx - 16, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, f_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Top status banner
    banner_txt = (
        f"StoneVL-Nest | Placed: {result['placed_count']}/{result['total_count']} | "
        f"Util: {result['utilization'] * 100:.1f}% | Mode: {result.get('cut_mode', 'waterjet').upper()}"
    )
    cv2.rectangle(blended, (8, 8), (min(w_px - 8, 620), 40), (20, 24, 35), -1)
    cv2.putText(blended, banner_txt, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if output_png_path is not None:
        out_p = Path(output_png_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", blended)
        if ok:
            out_p.write_bytes(buf.tobytes())

    return blended


def render_cut_piece_gallery(
    rectified_bgr: np.ndarray,
    case: dict[str, Any],
    result: dict[str, Any],
    output_png_path: str | Path | None = None,
    max_pieces: int = 8,
) -> np.ndarray:
    """Crop each placed part from the stone slab image to preview its real marble vein pattern."""
    h_px, w_px = rectified_bgr.shape[:2]
    w_mm = float(case["board"]["width"])
    h_mm = float(case["board"]["height"])
    sx = w_px / max(1.0, w_mm)
    sy = h_px / max(1.0, h_mm)

    placements = result.get("placements", [])[:max_pieces]
    cols = min(4, max(1, len(placements)))
    rows = max(1, math.ceil(max(1, len(placements)) / cols))
    cell_w, cell_h = 240, 180
    gallery = np.full((rows * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)

    for idx, p in enumerate(placements):
        r = idx // cols
        c = idx % cols
        x0_cell, y0_cell = c * cell_w, r * cell_h

        pts_px = np.array(
            [[int(round(float(x) * sx)), int(round(h_px - float(y) * sy))] for x, y in p["polygon"]],
            dtype=np.int32,
        )
        mask = np.zeros((h_px, w_px), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_px.reshape(-1, 1, 2)], 255)
        if p.get("hole_polygon"):
            h_pts = np.array(
                [[int(round(float(x) * sx)), int(round(h_px - float(y) * sy))] for x, y in p["hole_polygon"]],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [h_pts.reshape(-1, 1, 2)], 0)

        bx, by, bw, bh = cv2.boundingRect(pts_px)
        bx = max(0, bx)
        by = max(0, by)
        bw = max(1, min(w_px - bx, bw))
        bh = max(1, min(h_px - by, bh))

        crop_img = rectified_bgr[by : by + bh, bx : bx + bw].copy()
        crop_mask = mask[by : by + bh, bx : bx + bw]
        crop_img[crop_mask == 0] = (235, 238, 242)

        scale = min((cell_w - 30) / float(bw), (cell_h - 48) / float(bh))
        nw = max(10, int(round(bw * scale)))
        nh = max(10, int(round(bh * scale)))
        resized = cv2.resize(crop_img, (nw, nh), interpolation=cv2.INTER_AREA)

        ox = x0_cell + (cell_w - nw) // 2
        oy = y0_cell + 12 + (cell_h - 44 - nh) // 2
        gallery[oy : oy + nh, ox : ox + nw] = resized
        cv2.rectangle(gallery, (ox, oy), (ox + nw, oy + nh), PALETTE_BGR[idx % len(PALETTE_BGR)], 2)

        caption = f"#{idx + 1} {p['shape'][:6]} ({p['rotation']:.0f}deg)"
        cv2.putText(
            gallery,
            caption,
            (x0_cell + 14, y0_cell + cell_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (30, 41, 59),
            1,
            cv2.LINE_AA,
        )

    if output_png_path is not None:
        out_p = Path(output_png_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", gallery)
        if ok:
            out_p.write_bytes(buf.tobytes())

    return gallery


def render_overlay_svg(
    case: dict[str, Any],
    result: dict[str, Any],
    output_svg_path: str | Path,
) -> str:
    """Render vector SVG layout diagram for CAD inspection and publication figures."""
    board = case["board"]
    w = float(board["width"])
    h = float(board["height"])

    def svg_pts(poly: list[list[float]] | list[tuple[float, float]]) -> str:
        return " ".join(f"{float(x):.2f},{h - float(y):.2f}" for x, y in poly)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="100%" height="100%">',
        f'<rect x="0" y="0" width="{w:.0f}" height="{h:.0f}" fill="#f1f5f9" stroke="#0f172a" stroke-width="6"/>',
    ]

    # Obstacles
    for obs in board.get("obstacles", []):
        lines.append(
            f'<polygon points="{svg_pts(obs["polygon"])}" fill="#ef4444" fill-opacity="0.42" stroke="#991b1b" stroke-width="4" stroke-dasharray="10,5"/>'
        )

    # Placements
    for idx, p in enumerate(result.get("placements", [])):
        col = PALETTE_HEX[idx % len(PALETTE_HEX)]
        lines.append(
            f'<polygon points="{svg_pts(p["polygon"])}" fill="{col}" fill-opacity="0.30" stroke="{col}" stroke-width="4"/>'
        )
        if p.get("hole_polygon"):
            lines.append(
                f'<polygon points="{svg_pts(p["hole_polygon"])}" fill="#f1f5f9" stroke="#0284c7" stroke-width="3" stroke-dasharray="8,4"/>'
            )
        cx = sum(float(pt[0]) for pt in p["polygon"]) / len(p["polygon"])
        cy = h - sum(float(pt[1]) for pt in p["polygon"]) / len(p["polygon"])
        lines.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" dominant-baseline="middle" font-family="Arial" font-size="36" font-weight="bold" fill="#0f172a">#{idx + 1}</text>'
        )

    lines.append("</svg>")
    svg_text = "\n".join(lines)
    out_p = Path(output_svg_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(svg_text, encoding="utf-8")
    return svg_text


def extract_oriented_rect_patch(
    rectified_bgr: np.ndarray,
    poly_mm: list[list[float]] | list[tuple[float, float]],
    width_mm: float,
    height_mm: float,
    out_w_px: int,
    out_h_px: int,
) -> np.ndarray:
    """Warp the exact 4-vertex cut rectangle/square from the slab photo into its installation frame
    [0..out_w_px, 0..out_h_px], preserving true stone grain orientation as cut on the CNC bed.
    """
    h_px, w_px = rectified_bgr.shape[:2]
    sx = w_px / max(1.0, width_mm)
    sy = h_px / max(1.0, height_mm)
    pts_img = np.array([[float(x) * sx, float(h_px) - float(y) * sy] for x, y in poly_mm], dtype=np.float32)

    if len(pts_img) == 4:
        # In base_polygon: v0=(0,0) bottom-left, v1=(w,0) bottom-right, v2=(w,h) top-right, v3=(0,h) top-left
        # In image target patch (top-left origin): top-left is v3, top-right is v2, bottom-right is v1, bottom-left is v0
        src = np.array([pts_img[3], pts_img[2], pts_img[1], pts_img[0]], dtype=np.float32)
        dst = np.array(
            [[0.0, 0.0], [float(out_w_px - 1), 0.0], [float(out_w_px - 1), float(out_h_px - 1)], [0.0, float(out_h_px - 1)]],
            dtype=np.float32,
        )
        H_mat = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(rectified_bgr, H_mat, (out_w_px, out_h_px), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    bx, by, bw, bh = cv2.boundingRect(pts_img.astype(np.int32))
    bx, by = max(0, bx), max(0, by)
    bw, bh = max(1, min(w_px - bx, bw)), max(1, min(h_px - by, bh))
    return cv2.resize(rectified_bgr[by : by + bh, bx : bx + bw], (out_w_px, out_h_px), interpolation=cv2.INTER_LINEAR)


def render_dry_lay_preview(
    rectified_bgr: np.ndarray,
    case: dict[str, Any],
    result: dict[str, Any],
    output_png_path: str | Path | None = None,
    title_label: str = "StoneAesthetic-Nest Dry-Lay Preview (Customer Sign-Off)",
) -> np.ndarray:
    """Render an architectural Dry-Lay Preview (客户干铺连纹预演效果图) assembling:
    1) Left Panel: 3x2 (or RxC) Vein-Matched Core Feature Panel / Tea-Table Top (#C1..#C6)
    2) Right Top Panel: Sequential #S1..#S8 Continuous-Grain Staircase Treads (20x80cm)
    3) Right Bottom Panel: Recovered Remnant Equilateral Triangle Honeycomb Mosaic Strip (#M1..#Mk)
    """
    from src.solver_core.solver import expand_case_items

    w_mm = float(case["board"]["width"])
    h_mm = float(case["board"]["height"])
    all_items = expand_case_items(case)
    meta = {it["item_instance_id"]: it for it in all_items}

    pls = result.get("placements", [])
    t1_pls = [p for p in pls if meta.get(p["item_instance_id"], {}).get("tier") == 1 or meta.get(p["item_instance_id"], {}).get("role") == "core_vein_match"]
    t2_pls = [p for p in pls if p not in t1_pls and (meta.get(p["item_instance_id"], {}).get("tier") == 2 or meta.get(p["item_instance_id"], {}).get("role") == "stair_tread_tandem")]
    t3_pls = [p for p in pls if p not in t1_pls and p not in t2_pls]

    # Fallback if generic single-tier case: treat top 6 largest as Tier 1 and next 8 as Tier 2
    if not t1_pls and pls:
        pls_by_area = sorted(pls, key=lambda p: float(p.get("area", 0.0)), reverse=True)
        t1_pls = pls_by_area[: min(6, len(pls_by_area))]
        t2_pls = pls_by_area[min(6, len(pls_by_area)) : min(14, len(pls_by_area))]
        t3_pls = pls_by_area[min(14, len(pls_by_area)) :]

    canvas_w, canvas_h = 1360, 760
    board_img = np.full((canvas_h, canvas_w, 3), (244, 245, 247), dtype=np.uint8)

    # Top Dark Header Bar
    cv2.rectangle(board_img, (0, 0), (canvas_w, 56), (24, 30, 42), -1)
    short_title = title_label if len(title_label) <= 52 else title_label[:49] + "..."
    cv2.putText(board_img, short_title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    aes = result.get("aesthetic_metrics", {})
    if aes:
        stat_str = (
            f"Seam: {aes.get('seam_vein_continuity', 0)*100:.1f}% | "
            f"Gold: {aes.get('golden_zone_hit_rate', 0)*100:.1f}% | "
            f"StairErr: {aes.get('stair_vein_error_deg', 0):.1f}d | "
            f"CNY {aes.get('gross_output_cny', 0):,.0f}"
        )
        cv2.putText(board_img, stat_str, (735, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (110, 231, 183), 1, cv2.LINE_AA)

    # --- SECTION 1 (Left): 3x2 Core Vein-Matched Feature Wall / Tea-Table Dry-Lay ---
    sec1_x, sec1_y = 28, 96
    cv2.putText(
        board_img,
        "1. Core 50x50cm Vein-Matched Dry-Lay (3x2 Feature Panel)",
        (sec1_x, sec1_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (15, 23, 42),
        2,
        cv2.LINE_AA,
    )
    tile_sz = 196
    grout = 3
    grid_cols, grid_rows = 3, 2
    panel_w = grid_cols * tile_sz + (grid_cols + 1) * grout
    panel_h = grid_rows * tile_sz + (grid_rows + 1) * grout
    # Dark architectural stone shadow frame
    cv2.rectangle(board_img, (sec1_x - 4, sec1_y - 4), (sec1_x + panel_w + 4, sec1_y + panel_h + 4), (45, 55, 72), -1)

    def _t1_order_key(p: dict[str, Any]) -> tuple[int, int]:
        if p.get("grid_rc"):
            return (int(p["grid_rc"][0]), int(p["grid_rc"][1]))
        xs = [pt[0] for pt in p["polygon"]]
        ys = [pt[1] for pt in p["polygon"]]
        return (int(round(-np.mean(ys) / 450.0)), int(round(np.mean(xs))))

    t1_sorted = sorted(t1_pls, key=_t1_order_key)
    for idx_cell in range(grid_rows * grid_cols):
        r_c = idx_cell // grid_cols
        c_c = idx_cell % grid_cols
        tx = sec1_x + grout + c_c * (tile_sz + grout)
        ty = sec1_y + grout + r_c * (tile_sz + grout)
        if idx_cell < len(t1_sorted):
            p = t1_sorted[idx_cell]
            patch = extract_oriented_rect_patch(rectified_bgr, p["polygon"], w_mm, h_mm, tile_sz, tile_sz)
            board_img[ty : ty + tile_sz, tx : tx + tile_sz] = patch
            lbl = p.get("aesthetic_tag", f"C{idx_cell + 1}")
            cv2.rectangle(board_img, (tx + 6, ty + 6), (tx + 48, ty + 28), (15, 23, 42), -1)
            cv2.putText(board_img, f"#{lbl}", (tx + 10, ty + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(board_img, (tx, ty), (tx + tile_sz, ty + tile_sz), (203, 213, 225), -1)
            cv2.putText(board_img, "MISSING", (tx + 55, ty + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (185, 28, 28), 2, cv2.LINE_AA)

    # Remnant Mosaic Preview below Section 1
    mos_y = sec1_y + panel_h + 44
    cv2.putText(
        board_img,
        f"3. Remnant Equilateral Mosaic Recovery ({len(t3_pls)} Pcs Cut from Offcuts)",
        (sec1_x, mos_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (15, 23, 42),
        2,
        cv2.LINE_AA,
    )
    mos_box_h = canvas_h - mos_y - 22
    cv2.rectangle(board_img, (sec1_x, mos_y), (sec1_x + panel_w, mos_y + mos_box_h), (226, 232, 240), -1)
    cv2.rectangle(board_img, (sec1_x, mos_y), (sec1_x + panel_w, mos_y + mos_box_h), (100, 116, 139), 1)

    h_img, w_img = rectified_bgr.shape[:2]
    sx_i, sy_i = w_img / max(1.0, w_mm), h_img / max(1.0, h_mm)
    tri_side_px = 92
    tri_h_px = int(round(tri_side_px * math.sqrt(3.0) * 0.5))
    max_mos_show = min(9, len(t3_pls))
    start_mx = sec1_x + 28
    base_my = mos_y + (mos_box_h + tri_h_px) // 2 - 4

    for m_i in range(max_mos_show):
        p = t3_pls[m_i]
        pts_src = np.array([[float(x) * sx_i, float(h_img) - float(y) * sy_i] for x, y in p["polygon"]], dtype=np.float32)
        if len(pts_src) >= 3:
            upward = (m_i % 2 == 0)
            cx_m = start_mx + m_i * (tri_side_px // 2 + 4)
            if upward:
                dst_tri = np.array(
                    [[cx_m, base_my], [cx_m + tri_side_px, base_my], [cx_m + tri_side_px // 2, base_my - tri_h_px]],
                    dtype=np.float32,
                )
            else:
                dst_tri = np.array(
                    [[cx_m + tri_side_px, base_my - tri_h_px], [cx_m, base_my - tri_h_px], [cx_m + tri_side_px // 2, base_my]],
                    dtype=np.float32,
                )
            M_aff = cv2.getAffineTransform(pts_src[:3], dst_tri)
            warped_tri = cv2.warpAffine(rectified_bgr, M_aff, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            tri_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            cv2.fillConvexPoly(tri_mask, dst_tri.astype(np.int32), 255)
            board_img[tri_mask > 0] = warped_tri[tri_mask > 0]
            cv2.polylines(board_img, [dst_tri.astype(np.int32)], True, (51, 65, 85), 2, cv2.LINE_AA)

    # --- SECTION 2 (Right): Sequential #S1..#S8 Staircase Treads (20x80cm) ---
    sec2_x, sec2_y = 668, 96
    cv2.putText(
        board_img,
        "2. Grain-Aligned 20x80cm Stair Treads (#S1-#S8 Dry-Lay)",
        (sec2_x, sec2_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (15, 23, 42),
        2,
        cv2.LINE_AA,
    )
    tread_w_px, tread_h_px = 590, 68
    step_gap = 8
    t2_sorted = sorted(t2_pls, key=lambda p: int(p.get("sequence_index", 99)))
    for s_i in range(8):
        sy_pos = sec2_y + s_i * (tread_h_px + step_gap)
        # Subtle staircase cascade indent (6px per step)
        sx_pos = sec2_x + s_i * 8
        cv2.rectangle(
            board_img,
            (sx_pos - 2, sy_pos - 2),
            (sx_pos + tread_w_px + 2, sy_pos + tread_h_px + 2),
            (51, 65, 85),
            -1,
        )
        if s_i < len(t2_sorted):
            p = t2_sorted[s_i]
            t_patch = extract_oriented_rect_patch(rectified_bgr, p["polygon"], w_mm, h_mm, tread_w_px, tread_h_px)
            board_img[sy_pos : sy_pos + tread_h_px, sx_pos : sx_pos + tread_w_px] = t_patch
            lbl = p.get("aesthetic_tag", f"S{s_i + 1}")
            rot_err = float(p.get("vein_angle_error_deg", 0.0))
            badge_col = (21, 128, 61) if rot_err <= 12.0 else (185, 28, 28)
            cv2.rectangle(board_img, (sx_pos + 6, sy_pos + 6), (sx_pos + 128, sy_pos + 28), badge_col, -1)
            cv2.putText(
                board_img,
                f"Step #{lbl} ({rot_err:.0f}d err)",
                (sx_pos + 10, sy_pos + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.rectangle(board_img, (sx_pos, sy_pos), (sx_pos + tread_w_px, sy_pos + tread_h_px), (203, 213, 225), -1)
            cv2.putText(board_img, f"Step #S{s_i + 1} UNPLACED", (sx_pos + 210, sy_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (185, 28, 28), 2, cv2.LINE_AA)

    if output_png_path is not None:
        out_p = Path(output_png_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", board_img)
        if ok:
            out_p.write_bytes(buf.tobytes())

    return board_img
