"""Experiment E-v3 (RTX 3090): CrackSeg-SDM-clDice — Pure-PyTorch Structural-Directional SOTA Upgrade.

Synthesizes 2026 structural-directional crack modeling insights ("以我为主，为我所用", zero external
datasets, zero custom CUDA Mamba operators) directly into our industrial stone perception stack:
  1. Structural-Directional Modeling (SDM, k=13): Local 3x3 sharpness branch S + long multi-axis
     continuity D = DW1x13 + DW13x1 + DilatedDW3x3(d=3) + structure-aware directional gate
     Y = S + sigmoid(f_g(S + D)) * D, preventing marble vein background coupling.
  2. Batch-Invariant Normalization: GroupNorm(<=8) + SiLU throughout decoder & 1x detail stem;
     benchmarks both MobileNetV3-Large (~3.9M, ultra-light) and truncated ConvNeXt-Tiny Stages 1-3
     (~13.2M, 100% LayerNorm/GroupNorm zero-BatchNorm).
  3. Gated Multi-Scale Structure Fusion (MSF) + calibrated BCE/Dice/soft-clDice + 512x512 native
     training + 90-deg & +-20-deg oblique rotation augmentation + EMA (decay=0.992).
  4. Transparent Dual-Protocol Audit:
       - Protocol 1: Hard Official DeepCrack 300 train / 237 unseen test (zero scene mixing)
       - Protocol 2: Exact CVPR 2025/2026 (SCSegamba / MixerCSeg / RIFT) 7:1:2 @ 512x512 split
         with bit-for-bit identical `SCSegamba/eval/evaluate.py` metric formulas (ODS/OIS/mIoU/F1).
  5. Native PyTorch FP16 + `torch.cuda.CUDAGraph()` pure GPU latency & VRAM audit at 512x512.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tv_models

MASTER_DIR = Path(__file__).resolve().parents[1]
if str(MASTER_DIR) not in sys.path:
    sys.path.insert(0, str(MASTER_DIR))

from experiments.run_exp_e_sota_crack_gpu import compute_2class_miou, soft_cldice_loss
from experiments.run_rtx3090_deepcrack_benchmark import (
    DEEPCRACK_ROOT,
    SUBSET20_ROOT,
    compute_metrics,
    load_deepcrack_split,
)

RESULTS_DIR = MASTER_DIR / "experiments" / "results"
OUT_JSON = RESULTS_DIR / "exp_e_crackseg_sdm_sota.json"


# ==============================================================================
# 1. Pure-PyTorch SDM + MSF + GroupNorm Building Blocks
# ==============================================================================


def _pick_gn_groups(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    """Batch-size-invariant Conv2d + Adaptive GroupNorm + SiLU block."""

    def __init__(self, in_c: int, out_c: int, k: int = 3, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, stride=stride, padding=k // 2, groups=groups, bias=False),
            nn.GroupNorm(_pick_gn_groups(out_c), out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SDMBlock(nn.Module):
    """Structural-Directional Modeling Block (RIFT-inspired, k=13, pure PyTorch).

    Separates crisp local fracture cues S = DW3x3(x) from long anisotropic continuity
    D = DW1x13(x) + DW13x1(x) + DilatedDW3x3(d=3)(x), modulating D through a learned
    structure-aware gate G = sigmoid(f_g(S + D)) so Y = S + G * D prevents background
    coupling along decorative natural stone veins.
    """

    def __init__(self, channels: int, k: int = 13, expand: float = 2.0) -> None:
        super().__init__()
        c_h = max(channels, int(round(channels * expand)))
        dil = max(1, k // 4)
        self.norm = nn.GroupNorm(_pick_gn_groups(channels), channels)
        self.pw_in = nn.Sequential(
            nn.Conv2d(channels, c_h, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        # Local structural preservation (sharp crack vs smooth marble vein)
        self.dw_local = nn.Conv2d(c_h, c_h, 3, padding=1, groups=c_h, bias=False)
        # Cooperative long multi-axis directional continuity (H + V + oblique)
        self.dw_h = nn.Conv2d(c_h, c_h, kernel_size=(1, k), padding=(0, k // 2), groups=c_h, bias=False)
        self.dw_v = nn.Conv2d(c_h, c_h, kernel_size=(k, 1), padding=(k // 2, 0), groups=c_h, bias=False)
        self.dw_diag = nn.Conv2d(c_h, c_h, kernel_size=3, padding=dil, dilation=dil, groups=c_h, bias=False)
        # Structure-aware directional modulation gate
        g_mid = max(16, c_h // 4)
        self.gate = nn.Sequential(
            nn.Conv2d(c_h, g_mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(g_mid, c_h, 1, bias=True),
            nn.Sigmoid(),
        )
        self.pw_out = nn.Sequential(
            nn.Conv2d(c_h, channels, 1, bias=False),
            nn.GroupNorm(_pick_gn_groups(channels), channels),
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xh = self.pw_in(self.norm(x))
        s = self.dw_local(xh)
        d = self.dw_h(xh) + self.dw_v(xh) + self.dw_diag(xh)
        g = self.gate(s + d)
        y = self.pw_out(s + g * d)
        return x + self.gamma * y


class MSFFusion(nn.Module):
    """Multi-Scale Structure Fusion: learned spatial gate blending lateral detail & coarse semantics."""

    def __init__(self, c_lat: int, c_up: int, c_out: int, k: int = 13) -> None:
        super().__init__()
        self.proj_lat = ConvGNAct(c_lat, c_out, 1)
        self.proj_up = ConvGNAct(c_up, c_out, 1) if c_up != c_out else nn.Identity()
        self.alpha_gate = nn.Sequential(
            nn.Conv2d(c_out * 2, max(16, c_out // 2), 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(16, c_out // 2), c_out, 1, bias=True),
            nn.Sigmoid(),
        )
        self.sdm = SDMBlock(c_out, k=k, expand=2.0)

    def forward(self, lat: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        l_feat = self.proj_lat(lat)
        u_feat = self.proj_up(F.interpolate(up, size=lat.shape[2:], mode="bilinear", align_corners=False))
        alpha = self.alpha_gate(torch.cat([l_feat, u_feat], dim=1))
        fused = alpha * l_feat + (1.0 - alpha) * u_feat
        return self.sdm(fused)


class CrackSegMV3SDM(nn.Module):
    """Ultra-Lightweight MobileNetV3-Large + 1x GN Detail Stem + SDM(k=13) + MSF (~3.9M params)."""

    def __init__(self, fpn_c: int = 64, pretrained: bool = True) -> None:
        super().__init__()
        w = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        mv3 = tv_models.mobilenet_v3_large(weights=w).features
        self.s2 = nn.Sequential(*mv3[:2])     # 1/2,  16 ch
        self.s4 = nn.Sequential(*mv3[2:4])    # 1/4,  24 ch
        self.s8 = nn.Sequential(*mv3[4:7])    # 1/8,  40 ch
        self.s16 = nn.Sequential(*mv3[7:13])  # 1/16, 112 ch
        self.s32 = nn.Sequential(*mv3[13:])   # 1/32, 960 ch

        # Lean full-res 1x GroupNorm structural detail stem (no heavy 512x512 expand=2 SDM!)
        self.detail_1x = nn.Sequential(
            ConvGNAct(3, 20, 3),
            ConvGNAct(20, 24, 3),
        )
        self.top32 = nn.Sequential(ConvGNAct(960, fpn_c, 1), SDMBlock(fpn_c, k=13, expand=2.0))
        self.msf16 = MSFFusion(112, fpn_c, fpn_c, k=13)
        self.msf8 = MSFFusion(40, fpn_c, fpn_c, k=13)
        self.msf4 = MSFFusion(24, fpn_c, fpn_c, k=13)
        # Lean 1/2 (256x256) and 1x (512x512) refinement preserving 1px cracks at ~10x lower VRAM & latency
        self.fuse2 = nn.Sequential(ConvGNAct(16 + fpn_c, 40, 3), ConvGNAct(40, 32, 3))
        self.fuse1 = nn.Sequential(
            ConvGNAct(24 + 32, 32, 3),
            nn.Conv2d(32, 32, kernel_size=(1, 7), padding=(0, 3), groups=32, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ConvGNAct(32, 24, 3),
        )

        self.head = nn.Conv2d(24, 1, 1)
        self.aux4 = nn.Conv2d(fpn_c, 1, 1)
        self.aux8 = nn.Conv2d(fpn_c, 1, 1)

    def encoder_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.s2.parameters())
            + list(self.s4.parameters())
            + list(self.s8.parameters())
            + list(self.s16.parameters())
            + list(self.s32.parameters())
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h, w = x.shape[2], x.shape[3]
        d1 = self.detail_1x(x)
        f2 = self.s2(x)
        f4 = self.s4(f2)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        f32 = self.s32(f16)

        p32 = self.top32(f32)
        p16 = self.msf16(f16, p32)
        p8 = self.msf8(f8, p16)
        p4 = self.msf4(f4, p8)
        p2 = self.fuse2(torch.cat([f2, F.interpolate(p4, size=f2.shape[2:], mode="bilinear", align_corners=False)], dim=1))
        p1 = self.fuse1(torch.cat([d1, F.interpolate(p2, size=(h, w), mode="bilinear", align_corners=False)], dim=1))

        out = self.head(p1)
        a4 = F.interpolate(self.aux4(p4), size=(h, w), mode="bilinear", align_corners=False)
        a8 = F.interpolate(self.aux8(p8), size=(h, w), mode="bilinear", align_corners=False)
        return out, [a4, a8]


class CrackSegCNXSDM(nn.Module):
    """Zero-BatchNorm Truncated ConvNeXt-Tiny (Stages 1-3) + Lean 1x/2x GN Detail Stem + SDM(k=13) + MSF (~13.0M)."""

    def __init__(self, fpn_c: int = 64, pretrained: bool = True) -> None:
        super().__init__()
        w = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        feats = tv_models.convnext_tiny(weights=w).features
        # Strictly Stages 1..3 (100% LayerNorm, drops bloated 15.5M Stage 4!)
        self.c4 = nn.Sequential(feats[0], feats[1])  # 1/4,  96 ch
        self.c8 = nn.Sequential(feats[2], feats[3])  # 1/8,  192 ch
        self.c16 = nn.Sequential(feats[4], feats[5]) # 1/16, 384 ch

        # Lean high-res 1x and 1/2 GroupNorm detail pyramid compensating ConvNeXt's 4x4 patch stem
        self.detail_1x = nn.Sequential(
            ConvGNAct(3, 20, 3),
            ConvGNAct(20, 24, 3),
        )
        self.detail_2x = nn.Sequential(
            ConvGNAct(24, 32, 3, stride=2),
            ConvGNAct(32, 32, 3),
        )

        self.top16 = nn.Sequential(ConvGNAct(384, fpn_c, 1), SDMBlock(fpn_c, k=13, expand=2.0))
        self.msf8 = MSFFusion(192, fpn_c, fpn_c, k=13)
        self.msf4 = MSFFusion(96, fpn_c, fpn_c, k=13)
        self.fuse2 = nn.Sequential(ConvGNAct(32 + fpn_c, 40, 3), ConvGNAct(40, 32, 3))
        self.fuse1 = nn.Sequential(
            ConvGNAct(24 + 32, 32, 3),
            nn.Conv2d(32, 32, kernel_size=(7, 1), padding=(3, 0), groups=32, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            ConvGNAct(32, 24, 3),
        )

        self.head = nn.Conv2d(24, 1, 1)
        self.aux4 = nn.Conv2d(fpn_c, 1, 1)
        self.aux8 = nn.Conv2d(fpn_c, 1, 1)

    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.c4.parameters()) + list(self.c8.parameters()) + list(self.c16.parameters())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h, w = x.shape[2], x.shape[3]
        d1 = self.detail_1x(x)
        d2 = self.detail_2x(d1)
        f4 = self.c4(x)
        f8 = self.c8(f4)
        f16 = self.c16(f8)

        p16 = self.top16(f16)
        p8 = self.msf8(f8, p16)
        p4 = self.msf4(f4, p8)
        p2 = self.fuse2(torch.cat([d2, F.interpolate(p4, size=d2.shape[2:], mode="bilinear", align_corners=False)], dim=1))
        p1 = self.fuse1(torch.cat([d1, F.interpolate(p2, size=(h, w), mode="bilinear", align_corners=False)], dim=1))

        out = self.head(p1)
        a4 = F.interpolate(self.aux4(p4), size=(h, w), mode="bilinear", align_corners=False)
        a8 = F.interpolate(self.aux8(p8), size=(h, w), mode="bilinear", align_corners=False)
        return out, [a4, a8]


# ==============================================================================
# 2. Rich Rotation/Scale Augmentation Dataset & Calibrated Topology Loss
# ==============================================================================


class CrackDataset512(Dataset):
    """512x512 crack dataset with 90-deg + oblique +-20-deg rotations, scale crop, and photometric jitter."""

    def __init__(self, pairs: list[tuple[str, np.ndarray, np.ndarray]], size: int = 512, augment: bool = True) -> None:
        self.pairs = pairs
        self.size = size
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, bgr, mask = self.pairs[idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            # 1. Flips + 90-degree rotations (4x directional diversity for arbitrary stone veins!)
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
                m = np.ascontiguousarray(m[:, ::-1])
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[::-1, :])
                m = np.ascontiguousarray(m[::-1, :])
            k_rot = int(np.random.randint(0, 4))
            if k_rot > 0:
                rgb = np.ascontiguousarray(np.rot90(rgb, k_rot))
                m = np.ascontiguousarray(np.rot90(m, k_rot))
            # 2. Oblique +-20 deg rotation (critical for diagonal cracks!)
            if np.random.rand() < 0.5:
                ang = float(np.random.uniform(-20.0, 20.0))
                mat = cv2.getRotationMatrix2D((self.size * 0.5, self.size * 0.5), ang, 1.0)
                rgb = cv2.warpAffine(rgb, mat, (self.size, self.size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                m = cv2.warpAffine(m, mat, (self.size, self.size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # 3. Random scale crop (0.72..1.0)
            if np.random.rand() < 0.45:
                sc = float(np.random.uniform(0.72, 1.0))
                cs = int(self.size * sc)
                y0 = int(np.random.randint(0, max(1, self.size - cs + 1)))
                x0 = int(np.random.randint(0, max(1, self.size - cs + 1)))
                rgb = cv2.resize(rgb[y0 : y0 + cs, x0 : x0 + cs], (self.size, self.size), interpolation=cv2.INTER_LINEAR)
                m = cv2.resize(m[y0 : y0 + cs, x0 : x0 + cs], (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            # 4. Photometric & contrast jitter
            if np.random.rand() < 0.55:
                alpha = float(np.random.uniform(0.78, 1.24))
                beta = float(np.random.uniform(-22.0, 22.0))
                rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.30:
                gamma = float(np.random.uniform(0.82, 1.22))
                rgb = np.clip(((rgb.astype(np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.15:
                rgb = cv2.GaussianBlur(rgb, (3, 3), 0.6)

        x = ((rgb.astype(np.float32) / 255.0) - self.mean) / self.std
        x_t = torch.from_numpy(x.transpose(2, 0, 1).copy())
        y_t = torch.from_numpy((m > 0).astype(np.float32).copy()).unsqueeze(0)
        return x_t, y_t


def calibrated_sdm_loss(logits: torch.Tensor, target: torch.Tensor, use_cldice: bool = True) -> torch.Tensor:
    """Calibrated BCE + Per-Image/Batch Dice + optional soft-clDice loss for sparse thin cracks."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter_img = (prob * target).sum(dim=(1, 2, 3))
    denom_img = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_img = 1.0 - ((2.0 * inter_img + 1.0) / (denom_img + 1.0)).mean()

    inter_b = (prob * target).sum()
    denom_b = prob.sum() + target.sum()
    dice_batch = 1.0 - (2.0 * inter_b + 1.0) / (denom_b + 1.0)

    if use_cldice:
        cl = soft_cldice_loss(logits, target)
        return 0.25 * bce + 0.35 * dice_img + 0.28 * dice_batch + 0.12 * cl
    return 0.28 * bce + 0.40 * dice_img + 0.32 * dice_batch


class ModelEMA:
    """Exponential Moving Average for stable small-dataset generalization."""

    def __init__(self, model: nn.Module, decay: float = 0.992) -> None:
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                v.copy_(src)


# ==============================================================================
# 3. Exact `Karl1109/SCSegamba/eval/evaluate.py` + Strict Foreground Metrics
# ==============================================================================


def eval_exact_scsegamba_and_strict(
    prob_maps: list[np.ndarray],
    gts: list[np.ndarray],
) -> dict[str, float]:
    """Compute BOTH exact `Karl1109/SCSegamba/eval/evaluate.py` metrics (100 thresholds 0.00..0.99)
    AND our strict fixed-0.50 foreground IoU / Strict F1 / Safety Recall.
    """
    n_imgs = len(gts)
    thresholds = np.arange(0.0, 1.0, 0.01)  # 100 exact thresholds matching SCSegamba eval.py
    n_t = len(thresholds)

    ods_f1_sum = np.zeros(n_t, dtype=np.float64)
    miou_sum = np.zeros(n_t, dtype=np.float64)
    ois_best = np.zeros(n_imgs, dtype=np.float64)

    # Global micro stats for prf_metrics (pred/max > 0.5) & segment_metrics (pred >= 0.5)
    tp_peak = fp_peak = fn_peak = 0
    tp_g1 = fp_g1 = fn_g1 = tn_g0 = 0

    preds_05: list[np.ndarray] = []

    for idx, (prob, gt) in enumerate(zip(prob_maps, gts)):
        # Match SCSegamba uint8 quantization (pred_u8 / 255.0 > thresh)
        pred_u8 = np.clip(np.round(prob * 255.0), 0, 255).astype(np.uint8)
        p_norm = pred_u8.astype(np.float32) / 255.0
        g = gt > 0
        g_sum = int(g.sum())
        n_pix = g.size

        # Exact 100-threshold sweep via 256-bin histogram (O(N_pix) instead of O(100 * N_pix)!)
        hist_pos = np.bincount(pred_u8[g], minlength=256)
        hist_neg = np.bincount(pred_u8[~g], minlength=256)
        # Cumulative sum from high to low: counts pixels with pred_u8 >= k
        cum_tp = np.cumsum(hist_pos[::-1])[::-1]
        cum_fp = np.cumsum(hist_neg[::-1])[::-1]

        # Threshold predicate `pred_u8 / 255.0 > thr` <=> `pred_u8 >= floor(thr * 255) + 1`
        cut_idx = np.floor(thresholds * 255.0 + 1e-9).astype(np.int32) + 1
        tp_arr = np.where(cut_idx < 256, cum_tp[np.minimum(cut_idx, 255)], 0).astype(np.float64)
        fp_arr = np.where(cut_idx < 256, cum_fp[np.minimum(cut_idx, 255)], 0).astype(np.float64)
        fn_arr = float(g_sum) - tp_arr
        tn_arr = float(n_pix - g_sum) - fp_arr

        # Exact SCSegamba precision/recall/F1 conventions
        p_acc = np.where((tp_arr == 0) & (fp_arr == 0), 1.0, tp_arr / np.maximum(1.0, tp_arr + fp_arr))
        r_acc = np.where((tp_arr + fn_arr) == 0, 0.0, tp_arr / np.maximum(1.0, tp_arr + fn_arr))
        f1_arr = np.where((p_acc + r_acc) == 0, 0.0, 2.0 * p_acc * r_acc / np.maximum(1e-12, p_acc + r_acc))

        ods_f1_sum += f1_arr
        ois_best[idx] = float(np.max(f1_arr))

        # Exact SCSegamba cal_mIoU_metrics
        denom1 = fn_arr + fp_arr + tp_arr
        iou1 = np.where(denom1 <= 0, 0.0, tp_arr / np.maximum(1.0, denom1))
        iou0 = np.where(denom1 <= 0, 0.0, tn_arr / np.maximum(1.0, tn_arr + fp_arr + fn_arr))
        miou_sum += np.where(denom1 <= 0, 0.0, 0.5 * (iou1 + iou0))

        # Exact SCSegamba prf_metrics (`pred / max(pred) > 0.5`)
        pmax = float(pred_u8.max())
        p_pk = (pred_u8.astype(np.float32) / max(1.0, pmax)) > 0.5
        tp_peak += int(np.logical_and(p_pk, g).sum())
        fp_peak += int(np.logical_and(p_pk, ~g).sum())
        fn_peak += int(np.logical_and(~p_pk, g).sum())

        # Fixed 0.50 metrics
        p05 = p_norm >= 0.5
        preds_05.append(p05.astype(np.uint8))
        tp05 = int(np.logical_and(p05, g).sum())
        fp05 = int(np.logical_and(p05, ~g).sum())
        fn05 = g_sum - tp05
        tn05 = (n_pix - g_sum) - fp05
        tp_g1 += tp05
        fp_g1 += fp05
        fn_g1 += fn05
        tn_g0 += tn05

    ods_pct = float(np.max(ods_f1_sum / n_imgs) * 100.0)
    ois_pct = float(np.mean(ois_best) * 100.0)
    miou_scs_pct = float(np.max(miou_sum / n_imgs) * 100.0)

    p_pk_acc = tp_peak / max(1.0, tp_peak + fp_peak)
    r_pk_acc = tp_peak / max(1.0, tp_peak + fn_peak)
    f1_peak_pct = (2.0 * p_pk_acc * r_pk_acc / max(1e-12, p_pk_acc + r_pk_acc)) * 100.0

    # Global corpus 2-class mIoU (SCSegamba segment_metrics)
    giou1 = tp_g1 / max(1.0, tp_g1 + fp_g1 + fn_g1)
    giou0 = tn_g0 / max(1.0, tn_g0 + fp_g1 + fn_g1)
    miou_global_pct = 0.5 * (giou1 + giou0) * 100.0

    strict_m = compute_metrics(preds_05, gts)
    per_img_miou_05 = compute_2class_miou(preds_05, gts)

    return {
        "scsegamba_ods_pct": round(ods_pct, 2),
        "scsegamba_ois_pct": round(ois_pct, 2),
        "scsegamba_miou_pct": round(miou_scs_pct, 2),
        "scsegamba_f1_peaknorm_pct": round(f1_peak_pct, 2),
        "global_2class_miou_05_pct": round(miou_global_pct, 2),
        "per_img_2class_miou_05_pct": per_img_miou_05,
        "strict_fg_iou_05_pct": strict_m["strict_mean_iou_pct"],
        "strict_fg_f1_05_pct": strict_m["strict_mean_f1_pct"],
        "strict_global_f1_05_pct": strict_m["strict_global_f1_pct"],
        "safety_recall_05_pct": strict_m["safety_crack_recall_pct"],
    }


def predict_split(
    model: nn.Module,
    pairs: list[tuple[str, np.ndarray, np.ndarray]],
    device: torch.device,
    size: int = 512,
    use_tta: bool = False,
    eval_at_512: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run inference across split (single-pass or 4-view flip+rot TTA)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    model.eval()
    probs: list[np.ndarray] = []
    gts_out: list[np.ndarray] = []

    with torch.no_grad():
        for _, bgr, gt in pairs:
            oh, ow = gt.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(((im.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if not use_tta:
                    p = torch.sigmoid(model(x)[0])
                else:
                    # 4-view flip TTA (batched B=4)
                    xb = torch.cat([x, torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])], dim=0)
                    pb = torch.sigmoid(model(xb)[0])
                    p = 0.25 * (
                        pb[0:1]
                        + torch.flip(pb[1:2], [3])
                        + torch.flip(pb[2:3], [2])
                        + torch.flip(pb[3:4], [2, 3])
                    )
            prob_np = p[0, 0].float().cpu().numpy()
            if eval_at_512:
                gt_rs = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
                probs.append(prob_np)
                gts_out.append((gt_rs > 0).astype(np.uint8))
            else:
                probs.append(cv2.resize(prob_np, (ow, oh), interpolation=cv2.INTER_LINEAR))
                gts_out.append(gt)
    return probs, gts_out


# ==============================================================================
# 4. Training Loop + FP16 CUDAGraph Hardware Speed Audit
# ==============================================================================


def train_model(
    model_name: str,
    model_cls: type[nn.Module],
    train_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    val_pairs: list[tuple[str, np.ndarray, np.ndarray]] | None,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 8,
    enc_lr: float = 2.5e-4,
    dec_lr: float = 1.2e-3,
    seed: int = 42,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(fpn_c=64, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ema = ModelEMA(model, decay=0.992)

    ds_tr = CrackDataset512(train_pairs, size=512, augment=True)
    loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    enc_params = model.encoder_parameters()
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": enc_lr}, {"params": dec_params, "lr": dec_lr}],
        weight_decay=1e-2,
    )
    warmup_ep = 5

    def _lr_lambda(ep: int) -> float:
        if ep < warmup_ep:
            return float(ep + 1) / float(warmup_ep)
        prog = float(ep - warmup_ep) / float(max(1, epochs - warmup_ep))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    best_val_score = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                main_l, sides = model(xb)
                loss = calibrated_sdm_loss(main_l, yb, use_cldice=True) + 0.22 * sum(
                    calibrated_sdm_loss(s, yb, use_cldice=False) for s in sides
                ) / len(sides)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ema.update(model)
            ep_loss += float(loss.item())
        sched.step()

        # Optional strict validation checkpoint selection (Protocol 2: val NEVER touches test!)
        if val_pairs is not None and (ep + 1 >= 45) and ((ep + 1) % 5 == 0 or ep == epochs - 1):
            vp, vg = predict_split(ema.ema, val_pairs, device, size=512, use_tta=False, eval_at_512=True)
            vm = eval_exact_scsegamba_and_strict(vp, vg)
            score = vm["scsegamba_miou_pct"]
            if score > best_val_score:
                best_val_score = score
                best_state = copy.deepcopy(ema.ema.state_dict())

        if (ep + 1) % 20 == 0 or ep == 0 or ep == epochs - 1:
            v_str = f" | best_val_mIoU={best_val_score:.2f}%" if val_pairs is not None else ""
            print(
                f"    [{model_name}] Ep {ep+1:3d}/{epochs} | loss={ep_loss/len(loader):.4f}{v_str} | {time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    torch.cuda.synchronize()
    train_sec = round(time.perf_counter() - t0, 1)
    train_vram = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1)

    if best_state is not None:
        ema.ema.load_state_dict(best_state)
    return ema.ema, {
        "params": int(n_params),
        "params_m": round(n_params / 1e6, 2),
        "epochs": epochs,
        "train_sec": train_sec,
        "train_peak_vram_mib": train_vram,
        "best_val_miou_pct": round(best_val_score, 2) if val_pairs is not None else None,
    }


def benchmark_fp16_cudagraph_512(model: nn.Module, device: torch.device) -> dict[str, Any]:
    """Measure pure GPU FP16 + CUDAGraph hardware latency & VRAM at 512x512 on RTX 3090."""
    m = copy.deepcopy(model).to(device=device, dtype=torch.float16, memory_format=torch.channels_last).eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    static_in = torch.randn((1, 3, 512, 512), device=device, dtype=torch.float16).to(memory_format=torch.channels_last)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(15):
            _ = torch.sigmoid(m(static_in)[0])
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        static_out = torch.sigmoid(m(static_in)[0])
    torch.cuda.synchronize()

    for _ in range(25):
        graph.replay()
    torch.cuda.synchronize()

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    n_runs = 200
    ev_s.record()
    for _ in range(n_runs):
        graph.replay()
    ev_e.record()
    torch.cuda.synchronize()

    ms = float(ev_s.elapsed_time(ev_e) / n_runs)
    vram = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1)
    return {
        "resolution": "512x512",
        "fp16_cudagraph_ms": round(ms, 2),
        "fp16_cudagraph_fps": round(1000.0 / max(1e-6, ms), 1),
        "fp16_cudagraph_vram_mib": vram,
    }


def main() -> None:
    assert torch.cuda.is_available(), "CUDA RTX 3090 required!"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"=== Exp E-v3: CrackSeg-SDM-clDice Dual-Protocol SOTA Run on {torch.cuda.get_device_name(0)} ===", flush=True)

    # Load Official DeepCrack 300 / 237
    p1_train = load_deepcrack_split(DEEPCRACK_ROOT / "train_img", DEEPCRACK_ROOT / "train_lab")
    p1_test = load_deepcrack_split(DEEPCRACK_ROOT / "test_img", DEEPCRACK_ROOT / "test_lab")
    subset20 = load_deepcrack_split(SUBSET20_ROOT / "images", SUBSET20_ROOT / "masks")

    # Build Protocol 2: Exact CVPR 2025/2026 (SCSegamba/MixerCSeg/RIFT) 7:1:2 split across all 537 images (seed 42)
    all_537 = sorted(p1_train + p1_test, key=lambda t: t[0])
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(all_537))
    n_tr = int(round(0.7 * len(all_537)))  # 376
    n_va = int(round(0.1 * len(all_537)))  # 54
    p2_train = [all_537[i] for i in perm[:n_tr]]
    p2_val = [all_537[i] for i in perm[n_tr : n_tr + n_va]]
    p2_test = [all_537[i] for i in perm[n_tr + n_va :]]  # 107
    print(f"Protocol 1 (Official Liu'19): train={len(p1_train)}, test={len(p1_test)}", flush=True)
    print(f"Protocol 2 (CVPR25/26 7:1:2): train={len(p2_train)}, val={len(p2_val)}, test={len(p2_test)}", flush=True)

    results: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "protocol_1_official_300_237": {},
        "protocol_2_cvpr_7_1_2_512": {},
    }

    candidates = [
        ("CrackSeg-MV3-SDM", CrackSegMV3SDM, 2.5e-4, 1.2e-3),
        ("CrackSeg-CNX-SDM", CrackSegCNXSDM, 1.5e-4, 1.0e-3),
    ]

    # ---- Protocol 1: Hard Official 300 / 237 ----
    print("\n--- [Protocol 1] Official DeepCrack (300 Train / 237 Unseen Test) ---", flush=True)
    p1_models: dict[str, nn.Module] = {}
    for name, cls, elr, dlr in candidates:
        ema_m, tr_info = train_model(
            f"{name}-P1", cls, p1_train, None, device, epochs=85, batch_size=8, enc_lr=elr, dec_lr=dlr, seed=42
        )
        p1_models[name] = ema_m
        sp_probs, sp_gts = predict_split(ema_m, p1_test, device, size=512, use_tta=False, eval_at_512=False)
        tta_probs, tta_gts = predict_split(ema_m, p1_test, device, size=512, use_tta=True, eval_at_512=False)
        sub_probs, sub_gts = predict_split(ema_m, subset20, device, size=512, use_tta=False, eval_at_512=False)

        m_single = eval_exact_scsegamba_and_strict(sp_probs, sp_gts)
        m_tta = eval_exact_scsegamba_and_strict(tta_probs, tta_gts)
        m_sub = eval_exact_scsegamba_and_strict(sub_probs, sub_gts)
        speed = benchmark_fp16_cudagraph_512(ema_m, device)

        ckpt_file = RESULTS_DIR / f"{name.lower().replace('-', '_')}_p1_300_237.pt"
        torch.save(ema_m.state_dict(), ckpt_file)

        results["protocol_1_official_300_237"][name] = {
            **tr_info,
            **speed,
            "single_pass": m_single,
            "tta_4view": m_tta,
            "subset20": m_sub,
            "checkpoint": ckpt_file.name,
        }
        print(
            f"  >> P1 [{name}] ({tr_info['params_m']}M, {speed['fp16_cudagraph_ms']}ms / {speed['fp16_cudagraph_fps']} FPS) | "
            f"Single: FG-IoU={m_single['strict_fg_iou_05_pct']}% ODS={m_single['scsegamba_ods_pct']}% mIoU={m_single['scsegamba_miou_pct']}% | "
            f"TTA: FG-IoU={m_tta['strict_fg_iou_05_pct']}% ODS={m_tta['scsegamba_ods_pct']}% OIS={m_tta['scsegamba_ois_pct']}% mIoU={m_tta['scsegamba_miou_pct']}% F1pk={m_tta['scsegamba_f1_peaknorm_pct']}%",
            flush=True,
        )

    # ---- Protocol 2: Exact CVPR 2025/2026 7:1:2 @ 512x512 ----
    print("\n--- [Protocol 2] Exact CVPR 2025/2026 Protocol (DeepCrack 7:1:2 @ 512x512, seed=42) ---", flush=True)
    for name, cls, elr, dlr in candidates:
        ema_m, tr_info = train_model(
            f"{name}-P2", cls, p2_train, p2_val, device, epochs=85, batch_size=8, enc_lr=elr, dec_lr=dlr, seed=42
        )
        sp_probs, sp_gts = predict_split(ema_m, p2_test, device, size=512, use_tta=False, eval_at_512=True)
        tta_probs, tta_gts = predict_split(ema_m, p2_test, device, size=512, use_tta=True, eval_at_512=True)
        m_single = eval_exact_scsegamba_and_strict(sp_probs, sp_gts)
        m_tta = eval_exact_scsegamba_and_strict(tta_probs, tta_gts)
        speed = results["protocol_1_official_300_237"][name]["fp16_cudagraph_ms"]
        fps = results["protocol_1_official_300_237"][name]["fp16_cudagraph_fps"]

        results["protocol_2_cvpr_7_1_2_512"][name] = {
            **tr_info,
            "fp16_cudagraph_ms": speed,
            "fp16_cudagraph_fps": fps,
            "single_pass": m_single,
            "tta_4view": m_tta,
        }
        print(
            f"  >> P2 [{name}] ({tr_info['params_m']}M, {speed}ms / {fps} FPS) | "
            f"Single: mIoU={m_single['scsegamba_miou_pct']}% ODS={m_single['scsegamba_ods_pct']}% OIS={m_single['scsegamba_ois_pct']}% F1={m_single['scsegamba_f1_peaknorm_pct']}% | "
            f"TTA: mIoU={m_tta['scsegamba_miou_pct']}% ODS={m_tta['scsegamba_ods_pct']}% OIS={m_tta['scsegamba_ois_pct']}% F1={m_tta['scsegamba_f1_peaknorm_pct']}%",
            flush=True,
        )

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved full dual-protocol CrackSeg-SDM SOTA audit -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
