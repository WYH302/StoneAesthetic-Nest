"""Real-World Yunfu Stone Factory Slab Case-Study Pipeline (升档 ③).

Processes 3-5 genuine smartphone/camera photos of natural stone slabs taken in
Yunfu workshops (with or without printable A4 ArUco DICT_4X4_50 corner markers):
  1. Automatic ArUco 4-corner homography (IDs 0,1,2,3), manual corners_px, or
     dual-polarity contour fallback -> true millimeter rectified slab.
  2. Fiducial marker suppression + crack segmentation via either:
       - RTX 3090 `CrackSeg-MV3-clDice` (3.17M, trained on DeepCrack-300), OR
       - Zero-GPU `Black-Hat Adaptive` fallback.
  3. 96x60 anisotropic texture field + Golden Vein Zone + 3-Tier polymorphic
     `StoneAesthetic-Nest` solving + independent `RAC-Checker` 100% audit.
  4. 13-layer industrial CNC `.DXF` (4-corner GTSP 2-Opt + bridge-saw common-line
     merging + collision-free pierce points) + Client Dry-Lay Preview + combined
     4-column IJAMT publication figure (`fig_real_yunfu_workshop_cases.png/.pdf`).

Usage:
  # End-to-end self-check on a synthetic tilted ArUco workshop photo:
  python experiments/run_real_yunfu_slab_case_study.py --smoke-test

  # Run on real Yunfu factory photos placed in real_yunfu_slabs/photos/:
  python experiments/run_real_yunfu_slab_case_study.py --crack-backend auto
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MASTER_DIR = Path(__file__).resolve().parents[1]
if str(MASTER_DIR) not in sys.path:
    sys.path.insert(0, str(MASTER_DIR))

from src.ui_dxf.app import run_commercial_pipeline
from src.vision.vision_parser import _order_quad_corners, detect_slab_corners_auto, rectify_slab_perspective

REAL_SLAB_DIR = MASTER_DIR / "real_yunfu_slabs"
DEFAULT_PHOTOS_DIR = REAL_SLAB_DIR / "photos"
DEFAULT_MANIFEST = REAL_SLAB_DIR / "slabs_manifest.json"
DEFAULT_OUT_DIR = REAL_SLAB_DIR / "outputs"
_CKPT_CANDIDATES = [
    MASTER_DIR / "experiments" / "results" / "crackseg_cnx_sdm_fp16.pt",
    MASTER_DIR / "checkpoints" / "crackseg_cnx_sdm_fp16.pt",
    MASTER_DIR / "experiments" / "results" / "crackseg_mv3_cldice.pt",
    MASTER_DIR / "checkpoints" / "crackseg_mv3_cldice.pt",
]
DEFAULT_CRACK_CKPT = next((p for p in _CKPT_CANDIDATES if p.exists()), _CKPT_CANDIDATES[-1])


def _imread_unicode(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to decode image: {path}")
    return img


def _imwrite_unicode(path: Path, img_bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(buf.tobytes())


def detect_corners_with_metadata(
    image_bgr: np.ndarray,
    manual_corners_px: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Detect slab 4 corners with priority: Manual -> ArUco DICT_4X4_50 -> Dual-Polarity Contour.

    Also returns detected ArUco marker quads in raw pixel coordinates so they can be
    masked out from crack detection after homography warping.
    """
    if manual_corners_px is not None and len(manual_corners_px) == 4:
        ordered = _order_quad_corners(np.array(manual_corners_px, dtype=np.float32))
        return {
            "corners_px": ordered.tolist(),
            "calibration_mode": "manual_4corners",
            "aruco_quads_px": [],
            "aruco_ids": [],
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "aruco"):
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners_list, ids, _ = detector.detectMarkers(gray)
            if ids is not None and len(ids) >= 4:
                id_map: dict[int, np.ndarray] = {}
                for c, mid in zip(corners_list, ids.flatten().tolist()):
                    id_map[int(mid)] = c[0].astype(np.float32)

                if all(k in id_map for k in (0, 1, 2, 3)):
                    chosen_quads = [id_map[0], id_map[1], id_map[2], id_map[3]]
                    chosen_ids = [0, 1, 2, 3]
                else:
                    # Pick 4 markers and sort spatially as TL, TR, BR, BL
                    raw_quads = [c[0].astype(np.float32) for c in corners_list[:4]]
                    centers = np.array([q.mean(axis=0) for q in raw_quads], dtype=np.float32)
                    ordered_centers = _order_quad_corners(centers)
                    chosen_quads = []
                    chosen_ids = []
                    for oc in ordered_centers:
                        best_i = int(np.argmin(np.linalg.norm(centers - oc, axis=1)))
                        chosen_quads.append(raw_quads[best_i])
                        chosen_ids.append(int(ids.flatten()[best_i]))

                # Select the outermost corner vertex of each marker (flush with slab outer edges)
                tl_q, tr_q, br_q, bl_q = chosen_quads
                tl_pt = tl_q[int(np.argmin(tl_q[:, 0] + tl_q[:, 1]))]
                tr_pt = tr_q[int(np.argmax(tr_q[:, 0] - tr_q[:, 1]))]
                br_pt = br_q[int(np.argmax(br_q[:, 0] + br_q[:, 1]))]
                bl_pt = bl_q[int(np.argmin(bl_q[:, 0] - bl_q[:, 1]))]
                ordered = _order_quad_corners(np.array([tl_pt, tr_pt, br_pt, bl_pt], dtype=np.float32))
                return {
                    "corners_px": ordered.tolist(),
                    "calibration_mode": "aruco_4corners",
                    "aruco_quads_px": [q.tolist() for q in chosen_quads],
                    "aruco_ids": chosen_ids,
                }
        except Exception:
            pass

    auto_pts = detect_slab_corners_auto(image_bgr)
    h, w = image_bgr.shape[:2]
    is_full_frame = (
        abs(auto_pts[0][0]) < 1.0
        and abs(auto_pts[0][1]) < 1.0
        and abs(auto_pts[2][0] - w) < 1.0
        and abs(auto_pts[2][1] - h) < 1.0
    )
    return {
        "corners_px": auto_pts,
        "calibration_mode": "full_frame_pre_cropped" if is_full_frame else "dual_polarity_contour",
        "aruco_quads_px": [],
        "aruco_ids": [],
    }


def build_rectified_marker_exclusion_mask(
    aruco_quads_px: list[list[list[float]]],
    h_px_to_rect: np.ndarray,
    rect_w: int,
    rect_h: int,
    dilate_px: int = 24,
) -> np.ndarray:
    """Project raw ArUco fiducial quads into rectified space so markers are never flagged as cracks."""
    mask = np.zeros((rect_h, rect_w), dtype=np.uint8)
    if not aruco_quads_px:
        return mask
    for quad in aruco_quads_px:
        pts = np.array(quad, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, h_px_to_rect).reshape(-1, 2)
        cv2.fillConvexPoly(mask, np.round(warped).astype(np.int32), 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask


class _CrackSegMV3Inference:
    """Offline-safe wrapper around trained CrackSeg-MV3-clDice (3.17M) for RTX 3090 inference."""

    def __init__(self, ckpt_path: Path, device_str: str = "cuda") -> None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.models as tv_models
        from experiments.run_exp_e_sota_crack_gpu import ConvBNSiLU, DualStripAttention

        self.torch = torch
        self.device = torch.device(device_str)

        class _OfflineMV3(nn.Module):
            def __init__(self, fpn_c: int = 64) -> None:
                super().__init__()
                mv3 = tv_models.mobilenet_v3_large(weights=None).features
                self.s2 = nn.Sequential(*mv3[:2])
                self.s4 = nn.Sequential(*mv3[2:4])
                self.s8 = nn.Sequential(*mv3[4:7])
                self.s16 = nn.Sequential(*mv3[7:13])
                self.s32 = nn.Sequential(*mv3[13:])
                self.detail_1x = nn.Sequential(ConvBNSiLU(3, 24, 3), ConvBNSiLU(24, 32, 3))
                self.lat32 = nn.Conv2d(960, fpn_c, 1, bias=False)
                self.lat16 = nn.Conv2d(112, fpn_c, 1, bias=False)
                self.lat8 = nn.Conv2d(40, fpn_c, 1, bias=False)
                self.lat4 = nn.Conv2d(24, fpn_c, 1, bias=False)
                self.lat2 = nn.Conv2d(16, fpn_c, 1, bias=False)
                self.att16 = DualStripAttention(fpn_c)
                self.att8 = DualStripAttention(fpn_c)
                self.att4 = DualStripAttention(fpn_c)
                self.smooth2 = ConvBNSiLU(fpn_c, fpn_c, 3)
                self.refine_1x = nn.Sequential(
                    ConvBNSiLU(fpn_c + 32, 48, 3),
                    DualStripAttention(48),
                    ConvBNSiLU(48, 32, 3),
                )
                self.head = nn.Conv2d(32, 1, 1)
                self.aux4 = nn.Conv2d(fpn_c, 1, 1)
                self.aux8 = nn.Conv2d(fpn_c, 1, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h, w = x.shape[2:]
                d1 = self.detail_1x(x)
                f2 = self.s2(x)
                f4 = self.s4(f2)
                f8 = self.s8(f4)
                f16 = self.s16(f8)
                f32 = self.s32(f16)
                p32 = self.lat32(f32)
                p16 = self.att16(self.lat16(f16) + F.interpolate(p32, size=f16.shape[2:], mode="bilinear", align_corners=False))
                p8 = self.att8(self.lat8(f8) + F.interpolate(p16, size=f8.shape[2:], mode="bilinear", align_corners=False))
                p4 = self.att4(self.lat4(f4) + F.interpolate(p8, size=f4.shape[2:], mode="bilinear", align_corners=False))
                p2 = self.smooth2(self.lat2(f2) + F.interpolate(p4, size=f2.shape[2:], mode="bilinear", align_corners=False))
                up1 = F.interpolate(p2, size=(h, w), mode="bilinear", align_corners=False)
                return self.head(self.refine_1x(torch.cat([up1, d1], dim=1)))

        if "cnx_sdm" in ckpt_path.name.lower():
            from experiments.run_exp_e_crackseg_sdm_sota import CrackSegCNXSDM

            base_m = CrackSegCNXSDM(fpn_c=64, pretrained=False)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            base_m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()}, strict=True)
            base_m.to(self.device).eval()

            class _CNXWrap(nn.Module):
                def __init__(self, net: nn.Module) -> None:
                    super().__init__()
                    self.net = net

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.net(x)[0]

            self.model = _CNXWrap(base_m).to(self.device).eval()
            self.in_hw = (512, 512)
            self.model_label = "CrackSeg-CNX-SDM"
        else:
            model = _OfflineMV3(fpn_c=64)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.to(self.device).eval()
            self.model = model
            self.in_hw = (384, 512)
            self.model_label = "CrackSeg-MV3-clDice"

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # Warm up CUDA kernels once so reported per-slab latency reflects steady-state inference
        with self.torch.no_grad():
            dummy = self.torch.zeros((1, 3, self.in_hw[0], self.in_hw[1]), device=self.device)
            _ = self.model(dummy)
            if self.device.type == "cuda":
                self.torch.cuda.synchronize()

    def predict_mask(
        self,
        rectified_bgr: np.ndarray,
        threshold: float = 0.55,
        marker_exclusion_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run 4-view flip-TTA crack segmentation and return (binary_mask_u8, latency_ms)."""
        h0, w0 = rectified_bgr.shape[:2]
        rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
        rgb_rs = cv2.resize(rgb, (self.in_hw[1], self.in_hw[0]), interpolation=cv2.INTER_LINEAR)
        x_np = ((rgb_rs.astype(np.float32) / 255.0) - self.mean) / self.std
        x_t = self.torch.from_numpy(x_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        with self.torch.no_grad():
            p0 = self.torch.sigmoid(self.model(x_t))
            p1 = self.torch.flip(self.torch.sigmoid(self.model(self.torch.flip(x_t, dims=[3]))), dims=[3])
            p2 = self.torch.flip(self.torch.sigmoid(self.model(self.torch.flip(x_t, dims=[2]))), dims=[2])
            p3 = self.torch.flip(self.torch.sigmoid(self.model(self.torch.flip(x_t, dims=[2, 3]))), dims=[2, 3])
            prob = (0.25 * (p0 + p1 + p2 + p3)).squeeze().cpu().numpy()
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0

        prob_full = cv2.resize(prob.astype(np.float32), (w0, h0), interpolation=cv2.INTER_LINEAR)
        mask = (prob_full >= threshold).astype(np.uint8) * 255

        # Suppress 3% slab outer edge & any ArUco fiducial zones
        bx, by = max(4, int(0.03 * w0)), max(4, int(0.03 * h0))
        mask[:by, :] = 0
        mask[h0 - by :, :] = 0
        mask[:, :bx] = 0
        mask[:, w0 - bx :] = 0
        if marker_exclusion_mask is not None and marker_exclusion_mask.shape == mask.shape:
            mask[marker_exclusion_mask > 0] = 0

        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        return mask, round(ms, 2)


def build_default_yunfu_order(width_mm: float, height_mm: float) -> str:
    """Build a proportional Yunfu 3-tier luxury stone order matching the slab dimensions."""
    scale = min(width_mm / 3000.0, height_mm / 1800.0)
    tile_cm = max(35, min(55, int(round(50 * scale / 5.0) * 5)))
    stair_w_cm = max(16, min(22, int(round(20 * scale / 2.0) * 2)))
    stair_l_cm = max(60, min(85, int(round(80 * scale / 5.0) * 5)))
    mosaic_cm = max(14, min(20, int(round(18 * scale / 2.0) * 2)))
    return (
        f"主件：切8个{stair_w_cm}x{stair_l_cm}厘米长方形（楼梯踏步，必须顺纹连排、纹路不断）；"
        f"核心件：切6个{tile_cm}x{tile_cm}厘米正方形（茶几/电视背景墙主拼花，必须落在黄金主纹带，支持连纹对花）；"
        f"边角料余料套裁：在前面大件定好、花纹锁死之后，剩下的空隙和边角料，尽量帮我切一些等边三角形马赛克拼花砖（边长{mosaic_cm}厘米），"
        "锯缝留3mm，边距5mm，水刀加桥切。"
    )


def render_combined_ijamt_figure(
    case_records: list[dict[str, Any]],
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Render a 4-column IJAMT-grade visual gallery across all processed real Yunfu slabs."""
    n_rows = len(case_records)
    if n_rows == 0:
        return

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(n_rows, 4, figsize=(16.0, 3.65 * n_rows), dpi=300)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "(a) Raw Factory Photo & 4-Corner Homography",
        "(b) Rectified Slab, Vein Flow & Defect Mask",
        "(c) StoneAesthetic-Nest Layout & GTSP Toolpath",
        "(d) Virtual Dry-Lay Preview & Commercial Yield",
    ]

    for r_idx, rec in enumerate(case_records):
        raw_img = _imread_unicode(Path(rec["annotated_raw_png"]))
        rect_img = _imread_unicode(Path(rec["annotated_rect_png"]))
        overlay_img = _imread_unicode(Path(rec["artifacts"]["overlay_png"]))
        drylay_img = _imread_unicode(Path(rec["artifacts"]["dry_lay_preview_png"]))

        imgs = [raw_img, rect_img, overlay_img, drylay_img]
        sub_notes = [
            f"{rec['stone_name_en']} | {int(rec['width_mm'])}x{int(rec['height_mm'])} mm\nMode: {rec['calibration_mode']}",
            f"Vein: {rec['dominant_vein_angle_deg']:.1f} deg | Defects: {rec['detected_defects_count']} ({rec['crack_backend_used']})",
            f"RAC: {'PASS' if rec['rac_feasible'] else 'FAIL'} | Util: {rec['utilization_pct']:.1f}% | GTSP: -{rec['gtsp_reduction_pct']:.1f}%",
            f"Seam: {rec['seam_continuity_pct']:.1f}% | Gold: {rec['golden_hit_pct']:.1f}% | Mosaics: {rec['mosaic_count']} pcs",
        ]

        for c_idx in range(4):
            ax = axes[r_idx, c_idx]
            ax.imshow(cv2.cvtColor(imgs[c_idx], cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#cbd5e1")
                sp.set_linewidth(0.8)
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=10.5, fontweight="bold", pad=8, color="#0f172a")
            ax.set_xlabel(sub_notes[c_idx], fontsize=8.4, labelpad=5, color="#1e293b")

    fig.tight_layout(pad=1.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _create_smoke_test_photo(smoke_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Synthesize a realistic perspective-skewed Yunfu workshop photo with 4 ArUco markers + fissure."""
    sample_candidates = [
        MASTER_DIR / "data" / "ambientcg_pbr" / "Onyx011_Jade_Onyx.jpg",
        MASTER_DIR / "examples" / "sample_slabs" / "Onyx011_Jade_Onyx.jpg",
    ]
    slab_src = None
    for cand in sample_candidates:
        if cand.exists():
            slab_src = _imread_unicode(cand)
            break
    if slab_src is None:
        # Procedural fallback marble
        yy, xx = np.mgrid[0:900, 0:1500]
        vein = np.sin(0.012 * xx + 0.018 * yy + 2.0 * np.sin(0.008 * yy))
        base = np.clip(210 - 55 * np.abs(vein), 40, 245).astype(np.uint8)
        slab_src = np.stack([base, np.clip(base + 5, 0, 255).astype(np.uint8), base], axis=-1)

    slab_w, slab_h = 1500, 900
    slab = cv2.resize(slab_src, (slab_w, slab_h), interpolation=cv2.INTER_AREA)

    # Inject one realistic dark diagonal fissure near upper-right quadrant (so crack detector has a real target!)
    pts_crack = np.array(
        [[1050, 180], [1110, 235], [1165, 290], [1235, 360], [1280, 415]],
        dtype=np.int32,
    )
    cv2.polylines(slab, [pts_crack], False, (18, 16, 15), thickness=6, lineType=cv2.LINE_AA)
    cv2.polylines(slab, [pts_crack + np.array([2, -1])], False, (36, 32, 30), thickness=3, lineType=cv2.LINE_AA)

    # Paste 4 genuine DICT_4X4_50 ArUco markers (IDs 0, 1, 2, 3) flush at the 4 slab corners
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m_size = 96
    pad = 12
    box = m_size + 2 * pad
    placements = [
        (0, 6, 6),
        (1, slab_w - box - 6, 6),
        (2, slab_w - box - 6, slab_h - box - 6),
        (3, 6, slab_h - box - 6),
    ]
    for mid, x0, y0 in placements:
        sheet = np.full((box, box, 3), 250, dtype=np.uint8)
        m_gray = cv2.aruco.generateImageMarker(aruco_dict, mid, m_size)
        sheet[pad : pad + m_size, pad : pad + m_size] = cv2.cvtColor(m_gray, cv2.COLOR_GRAY2BGR)
        slab[y0 : y0 + box, x0 : x0 + box] = sheet

    # Warp slab onto a concrete workshop background at a realistic ~25-degree handheld perspective tilt
    bg_w, bg_h = 1920, 1280
    rng = np.random.default_rng(42)
    bg = np.clip(68 + rng.normal(0, 10, (bg_h, bg_w, 3)), 30, 120).astype(np.uint8)
    src_quad = np.array([[0, 0], [slab_w, 0], [slab_w, slab_h], [0, slab_h]], dtype=np.float32)
    dst_quad = np.array([[260, 185], [1690, 130], [1780, 1090], [150, 1140]], dtype=np.float32)
    h_mat = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(slab, h_mat, (bg_w, bg_h))
    mask = cv2.warpPerspective(np.full((slab_h, slab_w), 255, dtype=np.uint8), h_mat, (bg_w, bg_h))
    scene = np.where(mask[:, :, None] > 127, warped, bg)

    smoke_dir.mkdir(parents=True, exist_ok=True)
    photo_path = smoke_dir / "smoke_yunfu_jade_onyx_aruco.jpg"
    _imwrite_unicode(photo_path, scene)
    meta = {
        "case_id": "smoke_yunfu_01_aruco_onyx",
        "filename": photo_path.name,
        "stone_name": "SmokeTest Jade Onyx (ArUco Tilted)",
        "width_mm": 3000.0,
        "height_mm": 1800.0,
        "corners_px": None,
        "slab_price_rmb_m2": 850.0,
        "order_text": None,
    }
    return photo_path, meta


def process_single_slab(
    photo_path: Path,
    meta: dict[str, Any],
    out_dir: Path,
    crack_backend: str,
    gpu_segmenter: _CrackSegMV3Inference | None,
    crack_threshold: float = 0.55,
) -> dict[str, Any]:
    """Run full real-slab visual calibration -> defect segmentation -> aesthetic nesting -> DXF."""
    case_id = str(meta["case_id"])
    width_mm = float(meta.get("width_mm", 2800.0))
    height_mm = float(meta.get("height_mm", 1600.0))
    stone_name = str(meta.get("stone_name", case_id))
    stone_name_en = stone_name.split("(")[0].strip() or case_id
    order_text = meta.get("order_text") or build_default_yunfu_order(width_mm, height_mm)

    raw_bgr = _imread_unicode(photo_path)
    calib_det = detect_corners_with_metadata(raw_bgr, manual_corners_px=meta.get("corners_px"))
    corners_px = calib_det["corners_px"]

    # Rectify early so we can suppress ArUco marker patches cleanly before crack segmentation & nesting
    rect_info = rectify_slab_perspective(
        image_bgr=raw_bgr,
        width_mm=width_mm,
        height_mm=height_mm,
        corners_px=corners_px,
    )
    rect_bgr = rect_info["rectified_bgr"]
    h_px_to_rect = np.array(rect_info["homography_px_to_rect"], dtype=np.float32)
    marker_excl = build_rectified_marker_exclusion_mask(
        calib_det["aruco_quads_px"],
        h_px_to_rect,
        rect_info["rect_width_px"],
        rect_info["rect_height_px"],
        dilate_px=24,
    )

    # Clean inpaint inside the tiny ArUco corner stickers so they neither trigger fake cracks nor bias veins
    rect_clean_bgr = rect_bgr.copy()
    if np.any(marker_excl > 0):
        rect_clean_bgr = cv2.inpaint(rect_clean_bgr, marker_excl, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Run crack segmentation (GPU CrackSeg-MV3-clDice OR CPU Black-Hat Adaptive)
    external_mask = None
    crack_latency_ms = 0.0
    crack_backend_used = "Black-Hat Adaptive (CPU)"
    if crack_backend == "gpu" and gpu_segmenter is not None:
        external_mask, crack_latency_ms = gpu_segmenter.predict_mask(
            rect_clean_bgr,
            threshold=crack_threshold,
            marker_exclusion_mask=marker_excl,
        )
        crack_backend_used = f"{gpu_segmenter.model_label} (GPU {crack_latency_ms:.1f}ms)"

    # Save cleaned rectified image so `run_commercial_pipeline` works directly on exact mm coordinates
    case_out_dir = out_dir / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)
    clean_rect_path = case_out_dir / f"{case_id}_rectified_clean.png"
    _imwrite_unicode(clean_rect_path, rect_clean_bgr)

    rh, rw = rect_clean_bgr.shape[:2]
    full_rect_corners = [[0.0, 0.0], [float(rw), 0.0], [float(rw), float(rh)], [0.0, float(rh)]]

    t0 = time.perf_counter()
    report = run_commercial_pipeline(
        image_input=clean_rect_path,
        order_text=order_text,
        output_dir=case_out_dir,
        width_mm=width_mm,
        height_mm=height_mm,
        corners_px=full_rect_corners,
        safety_margin_mm=5.0,
        case_id=case_id,
        external_crack_mask=external_mask,
    )
    total_sec = round(time.perf_counter() - t0, 2)

    # Save Column-1 visual: Raw photo + detected homography quad + corner badges
    ann_raw = raw_bgr.copy()
    quad_i = np.round(np.array(corners_px, dtype=np.float32)).astype(np.int32)
    color_quad = (34, 197, 94) if calib_det["calibration_mode"] == "aruco_4corners" else (245, 158, 11)
    cv2.polylines(ann_raw, [quad_i], True, color_quad, max(4, min(ann_raw.shape[:2]) // 180), cv2.LINE_AA)
    for idx, (pt, lbl) in enumerate(zip(quad_i, ("TL", "TR", "BR", "BL"))):
        cv2.circle(ann_raw, tuple(pt), max(10, min(ann_raw.shape[:2]) // 70), (220, 38, 38), -1)
        cv2.putText(
            ann_raw,
            lbl,
            (int(pt[0]) + 12, int(pt[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.7, min(ann_raw.shape[:2]) / 900.0),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    ann_raw_path = case_out_dir / f"{case_id}_raw_calibrated.png"
    _imwrite_unicode(ann_raw_path, ann_raw)

    # Save Column-2 visual: Rectified slab + vein angle arrows + defect mask tint
    ann_rect = rect_clean_bgr.copy()
    vein_deg = float(report.get("dominant_vein_angle_deg", 0.0))
    vein_rad = math.radians(vein_deg)
    # Draw subtle vein direction vector grid across rectified slab
    step_x = max(80, rw // 10)
    step_y = max(80, rh // 6)
    arrow_len = max(24, min(step_x, step_y) // 3)
    for cy in range(step_y // 2, rh - step_y // 2, step_y):
        for cx in range(step_x // 2, rw - step_x // 2, step_x):
            dx = int(round(arrow_len * math.cos(vein_rad)))
            dy = int(round(-arrow_len * math.sin(vein_rad)))
            cv2.arrowedLine(
                ann_rect,
                (cx - dx // 2, cy - dy // 2),
                (cx + dx // 2, cy + dy // 2),
                (56, 189, 248),
                2,
                tipLength=0.28,
            )
    if external_mask is not None and np.any(external_mask > 0):
        red_tint = ann_rect.copy()
        red_tint[external_mask > 0] = (38, 38, 239)
        ann_rect = cv2.addWeighted(ann_rect, 0.55, red_tint, 0.45, 0)
    ann_rect_path = case_out_dir / f"{case_id}_rect_vein_defects.png"
    _imwrite_unicode(ann_rect_path, ann_rect)

    am = report["nesting_result"].get("aesthetic_metrics", {})
    dxf_info = report.get("dxf_export", {})
    quote = report.get("commercial_quote", {})

    return {
        "case_id": case_id,
        "stone_name": stone_name,
        "stone_name_en": stone_name_en,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "calibration_mode": calib_det["calibration_mode"],
        "aruco_ids": calib_det["aruco_ids"],
        "crack_backend_used": crack_backend_used,
        "dominant_vein_angle_deg": vein_deg,
        "detected_defects_count": int(report.get("detected_defects_count", 0)),
        "rac_feasible": bool(report["certificate"].get("feasible", False)),
        "placed_parts": int(report["nesting_result"].get("placed_count", 0)),
        "utilization_pct": round(float(report["nesting_result"].get("utilization", 0.0)) * 100.0, 2),
        "seam_continuity_pct": round(float(am.get("seam_vein_continuity", 0.0)) * 100.0, 2),
        "golden_hit_pct": round(float(am.get("golden_zone_hit_rate", 0.0)) * 100.0, 2),
        "stair_vein_error_deg": round(float(am.get("stair_vein_error_deg", 0.0)), 2),
        "mosaic_count": int(am.get("tier3_mosaic_count", 0)),
        "gtsp_reduction_pct": round(float(dxf_info.get("air_travel_reduction_pct", 0.0)), 2),
        "common_line_saved_m": round(float(dxf_info.get("common_line_saved_m", 0.0)), 3),
        "pierce_safe_pct": round(float(dxf_info.get("pierce_safety_rate_pct", 100.0)), 1),
        "net_gross_margin_rmb": float(quote.get("net_gross_margin_rmb", 0.0)),
        "pipeline_runtime_sec": total_sec,
        "annotated_raw_png": str(ann_raw_path),
        "annotated_rect_png": str(ann_rect_path),
        "artifacts": report["artifacts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real Yunfu Stone Factory Slab Case-Study Runner")
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_PHOTOS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--crack-backend", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--crack-threshold", type=float, default=0.55)
    parser.add_argument("--smoke-test", action="store_true", help="Run end-to-end self-check on a synthetic tilted ArUco photo")
    args = parser.parse_args()

    # Resolve crack segmentation backend
    gpu_segmenter = None
    active_backend = args.crack_backend
    if args.crack_backend in ("auto", "gpu"):
        try:
            import torch

            if torch.cuda.is_available() and DEFAULT_CRACK_CKPT.exists():
                gpu_segmenter = _CrackSegMV3Inference(DEFAULT_CRACK_CKPT, device_str="cuda")
                active_backend = "gpu"
                print(f"[INFO] Loaded RTX 3090 CrackSeg-MV3-clDice from {DEFAULT_CRACK_CKPT.name}")
            else:
                active_backend = "cpu"
                print("[INFO] CUDA/checkpoint unavailable -> falling back to zero-GPU Black-Hat Adaptive")
        except Exception as exc:
            active_backend = "cpu"
            print(f"[WARN] GPU CrackSeg init skipped ({exc}) -> using zero-GPU Black-Hat Adaptive")

    jobs: list[tuple[Path, dict[str, Any]]] = []
    out_dir = args.out_dir

    if args.smoke_test:
        out_dir = REAL_SLAB_DIR / "smoke_test_outputs"
        photo_p, meta = _create_smoke_test_photo(out_dir / "input")
        jobs.append((photo_p, meta))
    else:
        manifest_by_file: dict[str, dict[str, Any]] = {}
        if args.manifest.exists():
            data = json.loads(args.manifest.read_text(encoding="utf-8"))
            for entry in data.get("slabs", []):
                manifest_by_file[str(entry["filename"]).lower()] = entry

        if args.photos_dir.exists():
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
            photo_files = sorted([p for p in args.photos_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
            for idx, p in enumerate(photo_files, start=1):
                entry = manifest_by_file.get(p.name.lower())
                if entry is None:
                    entry = {
                        "case_id": f"yunfu_real_{idx:02d}_{p.stem}",
                        "filename": p.name,
                        "stone_name": p.stem,
                        "width_mm": 2800.0,
                        "height_mm": 1600.0,
                        "corners_px": None,
                        "slab_price_rmb_m2": 720.0,
                        "order_text": None,
                    }
                jobs.append((p, entry))

    if not jobs:
        print(f"[INFO] No real slab photos found in {args.photos_dir}.")
        print("       Drop 3-5 smartphone photos (.jpg/.png) there and re-run, or run with --smoke-test.")
        return

    records: list[dict[str, Any]] = []
    for photo_path, meta in jobs:
        print(f"\n>>> Processing {photo_path.name} ({meta['width_mm']:.0f}x{meta['height_mm']:.0f} mm)...")
        rec = process_single_slab(
            photo_path=photo_path,
            meta=meta,
            out_dir=out_dir,
            crack_backend=active_backend,
            gpu_segmenter=gpu_segmenter,
            crack_threshold=args.crack_threshold,
        )
        records.append(rec)
        print(
            f"    [OK] Mode={rec['calibration_mode']} | RAC={rec['rac_feasible']} | "
            f"Seam={rec['seam_continuity_pct']:.1f}% | Gold={rec['golden_hit_pct']:.1f}% | "
            f"Util={rec['utilization_pct']:.1f}% | GTSP=-{rec['gtsp_reduction_pct']:.1f}% | "
            f"Time={rec['pipeline_runtime_sec']:.2f}s"
        )

    fig_png = out_dir / "fig_real_yunfu_workshop_cases.png"
    fig_pdf = out_dir / "fig_real_yunfu_workshop_cases.pdf"
    render_combined_ijamt_figure(records, fig_png, fig_pdf)

    summary_path = out_dir / "real_yunfu_case_study_summary.json"
    summary_payload = {
        "n_slabs": len(records),
        "crack_backend": active_backend,
        "all_rac_feasible": all(r["rac_feasible"] for r in records),
        "mean_seam_continuity_pct": round(float(np.mean([r["seam_continuity_pct"] for r in records])), 2),
        "mean_golden_hit_pct": round(float(np.mean([r["golden_hit_pct"] for r in records])), 2),
        "mean_utilization_pct": round(float(np.mean([r["utilization_pct"] for r in records])), 2),
        "mean_gtsp_reduction_pct": round(float(np.mean([r["gtsp_reduction_pct"] for r in records])), 2),
        "figure_png": str(fig_png),
        "figure_pdf": str(fig_pdf),
        "cases": records,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] Real Yunfu case study saved -> {summary_path}")
    print(f"       Publication 4-column figure  -> {fig_png}")


if __name__ == "__main__":
    main()
