"""Experiment E-v2 (RTX 3090): CrackSeg-MV3-clDice (Pretrained MobileNetV3-Large + 1x Detail Stem + Dual-Attention FPN + soft-clDice).

Trains strictly on Official DeepCrack-300 train split and evaluates strictly on held-out
Official DeepCrack-237 test split (fixed threshold 0.50, zero test leakage) + 20-image subset.
Reports BOTH clean pure-GPU forward latency/FPS AND end-to-end pipeline latency, plus
both strict foreground crack IoU/F1 and 2-class mIoU for apples-to-apples 2025-2026 SOTA comparison.
"""

from __future__ import annotations

import json
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

from experiments.run_rtx3090_deepcrack_benchmark import (
    DEEPCRACK_ROOT,
    SUBSET20_ROOT,
    compute_metrics,
    load_deepcrack_split,
)
from experiments.run_yunfu_aesthetic_benchmark_and_figures import RESULTS_DIR

OUT_JSON = RESULTS_DIR / "exp_e_sota_crack_mv3_cldice.json"
CKPT_OUT = RESULTS_DIR / "crackseg_mv3_cldice.pt"


class RichCrackDataset(Dataset):
    """DeepCrack-300 dataset with realistic geometric + photometric augmentation."""

    def __init__(self, pairs: list[tuple[str, np.ndarray, np.ndarray]], size: tuple[int, int] = (384, 512), augment: bool = True) -> None:
        self.pairs = pairs
        self.h, self.w = size
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, bgr, mask = self.pairs[idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
                m = np.ascontiguousarray(m[:, ::-1])
            if np.random.rand() < 0.5:
                rgb = np.ascontiguousarray(rgb[::-1, :])
                m = np.ascontiguousarray(m[::-1, :])
            if np.random.rand() < 0.35:
                # Random scale crop (0.78..1.0) preserving hairline crack detail
                sc = float(np.random.uniform(0.78, 1.0))
                ch, cw = int(self.h * sc), int(self.w * sc)
                y0 = np.random.randint(0, max(1, self.h - ch + 1))
                x0 = np.random.randint(0, max(1, self.w - cw + 1))
                rgb = cv2.resize(rgb[y0 : y0 + ch, x0 : x0 + cw], (self.w, self.h), interpolation=cv2.INTER_LINEAR)
                m = cv2.resize(m[y0 : y0 + ch, x0 : x0 + cw], (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            if np.random.rand() < 0.5:
                alpha = float(np.random.uniform(0.82, 1.20))
                beta = float(np.random.uniform(-18.0, 18.0))
                rgb = np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.25:
                gamma = float(np.random.uniform(0.85, 1.18))
                rgb = np.clip(((rgb.astype(np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)

        x = ((rgb.astype(np.float32) / 255.0) - self.mean) / self.std
        x_t = torch.from_numpy(x.transpose(2, 0, 1).copy())
        y_t = torch.from_numpy(m.astype(np.float32).copy()).unsqueeze(0)
        return x_t, y_t


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iters: int = 4) -> torch.Tensor:
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Topological centerline Dice loss (Shit et al., CVPR 2021) preserving thin crack connectivity."""
    prob = torch.sigmoid(logits)
    skel_pred = soft_skel(prob, iters=3)
    skel_true = soft_skel(target, iters=3)
    tprec = (torch.sum(skel_pred * target, dim=(1, 2, 3)) + smooth) / (torch.sum(skel_pred, dim=(1, 2, 3)) + smooth)
    tsens = (torch.sum(skel_true * prob, dim=(1, 2, 3)) + smooth) / (torch.sum(skel_true, dim=(1, 2, 3)) + smooth)
    cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-6)
    return (1.0 - cl_dice).mean()


def crack_topology_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
    cl = soft_cldice_loss(logits, target)
    return 0.40 * bce + 0.42 * dice + 0.18 * cl


class DualStripAttention(nn.Module):
    """Lightweight Channel + Horizontal/Vertical Strip Attention tailored for elongated hairline cracks."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        mid = max(16, ch // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
            nn.Sigmoid(),
        )
        self.strip_h = nn.Conv2d(ch, ch, kernel_size=(1, 5), padding=(0, 2), groups=ch, bias=False)
        self.strip_v = nn.Conv2d(ch, ch, kernel_size=(5, 1), padding=(2, 0), groups=ch, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.se(x)
        s = self.strip_h(y) + self.strip_v(y)
        return x + self.fuse(s)


class ConvBNSiLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrackSegMV3CLDice(nn.Module):
    """Lightweight Pretrained MobileNetV3-Large + 1x Full-Res Detail Stem + Dual-Strip-Attention FPN (~3.4M params)."""

    def __init__(self, fpn_c: int = 64) -> None:
        super().__init__()
        mv3 = tv_models.mobilenet_v3_large(weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT).features
        # Stage indices for MobileNetV3-Large:
        # 0..1 -> stride 2 (16 ch), 2..3 -> stride 4 (24 ch), 4..6 -> stride 8 (40 ch), 7..12 -> stride 16 (112 ch), 13..16 -> stride 32 (960 ch)
        self.s2 = nn.Sequential(*mv3[:2])     # 1/2,  16 ch
        self.s4 = nn.Sequential(*mv3[2:4])    # 1/4,  24 ch
        self.s8 = nn.Sequential(*mv3[4:7])    # 1/8,  40 ch
        self.s16 = nn.Sequential(*mv3[7:13])  # 1/16, 112 ch
        self.s32 = nn.Sequential(*mv3[13:])   # 1/32, 960 ch

        # Native 1x detail stem preserving sub-millimeter 1-2px hairline cracks lost by stride-2 stem
        self.detail_1x = nn.Sequential(
            ConvBNSiLU(3, 24, 3),
            ConvBNSiLU(24, 32, 3),
        )

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
        out = self.head(self.refine_1x(torch.cat([up1, d1], dim=1)))
        a4 = F.interpolate(self.aux4(p4), size=(h, w), mode="bilinear", align_corners=False)
        a8 = F.interpolate(self.aux8(p8), size=(h, w), mode="bilinear", align_corners=False)
        return out, [a4, a8]


def compute_2class_miou(preds: list[np.ndarray], gts: list[np.ndarray]) -> float:
    mious = []
    for p_u8, g_u8 in zip(preds, gts):
        p = p_u8 > 0
        g = g_u8 > 0
        iou_fg = float(np.logical_and(p, g).sum()) / max(1.0, float(np.logical_or(p, g).sum()))
        iou_bg = float(np.logical_and(~p, ~g).sum()) / max(1.0, float(np.logical_or(~p, ~g).sum()))
        mious.append(0.5 * (iou_fg + iou_bg))
    return round(float(np.mean(mious) * 100.0), 2)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA RTX 3090 required!"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"=== Experiment E-v2: CrackSeg-MV3-clDice on {torch.cuda.get_device_name(0)} ===", flush=True)
    train_pairs = load_deepcrack_split(DEEPCRACK_ROOT / "train_img", DEEPCRACK_ROOT / "train_lab")
    test_pairs = load_deepcrack_split(DEEPCRACK_ROOT / "test_img", DEEPCRACK_ROOT / "test_lab")
    subset_pairs = load_deepcrack_split(SUBSET20_ROOT / "images", SUBSET20_ROOT / "masks")
    print(f"Loaded DeepCrack: train={len(train_pairs)}, test={len(test_pairs)}, subset20={len(subset_pairs)}", flush=True)

    model = CrackSegMV3CLDice(fpn_c=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params/1e6:.2f}M)", flush=True)

    ds_tr = RichCrackDataset(train_pairs, size=(384, 512), augment=True)
    loader = DataLoader(ds_tr, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

    epochs = 65
    # Differential learning rate: gentler on ImageNet pretrained encoder, faster on FPN/attention decoder
    enc_params = list(model.s2.parameters()) + list(model.s4.parameters()) + list(model.s8.parameters()) + list(model.s16.parameters()) + list(model.s32.parameters())
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": 3.0e-4}, {"params": dec_params, "lr": 1.5e-3}],
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                main_l, sides = model(xb)
                loss = crack_topology_loss(main_l, yb) + 0.25 * sum(crack_topology_loss(s, yb) for s in sides) / len(sides)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += float(loss.item())
        sched.step()
        if (ep + 1) % 15 == 0 or ep == epochs - 1:
            print(f"  Epoch {ep+1:2d}/{epochs} | loss={ep_loss/len(loader):.4f} | elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    torch.cuda.synchronize()
    train_sec = round(time.perf_counter() - t0, 2)
    train_vram = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1)
    torch.save(model.state_dict(), CKPT_OUT)

    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 1. Clean pure-GPU forward latency benchmark (batch=1, 384x512, bfloat16, CUDA events)
    dummy = torch.randn(1, 3, 384, 512, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(30):
            _ = model(dummy)[0]
    torch.cuda.synchronize()
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    n_runs = 200
    ev_s.record()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(n_runs):
            _ = model(dummy)[0]
    ev_e.record()
    torch.cuda.synchronize()
    pure_gpu_ms = round(ev_s.elapsed_time(ev_e) / n_runs, 2)
    pure_gpu_fps = round(1000.0 / max(1e-6, pure_gpu_ms), 1)
    infer_vram = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1)

    # 2. Full evaluation on strictly held-out DeepCrack-237 & 20-image subset (single pass + flip TTA)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _eval_split(pairs: list[tuple[str, np.ndarray, np.ndarray]], use_tta: bool = False) -> tuple[dict[str, float], float, float]:
        preds = []
        ts = time.perf_counter()
        with torch.no_grad():
            for _, bgr, gt in pairs:
                oh, ow = gt.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                im = cv2.resize(rgb, (512, 384), interpolation=cv2.INTER_LINEAR)
                x = torch.from_numpy(((im.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p0 = torch.sigmoid(model(x)[0])
                    if use_tta:
                        p1 = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[3]))[0]), dims=[3])
                        p2 = torch.flip(torch.sigmoid(model(torch.flip(x, dims=[2]))[0]), dims=[2])
                        prob = ((p0 + p1 + p2) / 3.0)[0, 0].float().cpu().numpy()
                    else:
                        prob = p0[0, 0].float().cpu().numpy()
                prob_o = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
                preds.append((prob_o >= 0.5).astype(np.uint8))
        torch.cuda.synchronize()
        dt = time.perf_counter() - ts
        e2e_ms = round(dt / len(pairs) * 1000.0, 2)
        gts = [g for _, _, g in pairs]
        m = compute_metrics(preds, gts)
        m["two_class_miou_pct"] = compute_2class_miou(preds, gts)
        return m, e2e_ms, round(len(pairs) / max(1e-6, dt), 1)

    m_test_single, e2e_ms_single, e2e_fps_single = _eval_split(test_pairs, use_tta=False)
    m_test_tta, e2e_ms_tta, e2e_fps_tta = _eval_split(test_pairs, use_tta=True)
    m_sub_single, _, _ = _eval_split(subset_pairs, use_tta=False)

    out = {
        "gpu_name": torch.cuda.get_device_name(0),
        "method": "CrackSeg-MV3-clDice (Ours, Opt. GPU SOTA)",
        "params": int(n_params),
        "params_m": round(n_params / 1e6, 2),
        "train_epochs": epochs,
        "train_gpu_sec": train_sec,
        "train_peak_vram_mib": train_vram,
        "infer_peak_vram_mib": infer_vram,
        "pure_gpu_infer_ms": pure_gpu_ms,
        "pure_gpu_infer_fps": pure_gpu_fps,
        "e2e_single_ms": e2e_ms_single,
        "e2e_single_fps": e2e_fps_single,
        "e2e_tta_ms": e2e_ms_tta,
        "e2e_tta_fps": e2e_fps_tta,
        "deepcrack_237_single": m_test_single,
        "deepcrack_237_tta": m_test_tta,
        "deepcrack_20_subset": m_sub_single,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== CrackSeg-MV3-clDice RESULTS ===")
    print(f"Params: {out['params_m']:.2f}M | Train: {train_sec:.1f}s (VRAM {train_vram:.0f} MiB)")
    print(f"Pure GPU Forward: {pure_gpu_ms:.2f} ms ({pure_gpu_fps:.1f} FPS, VRAM {infer_vram:.0f} MiB) | E2E Single: {e2e_ms_single:.2f} ms ({e2e_fps_single:.1f} FPS)")
    print(f"DeepCrack-237 (Single): FG-IoU={m_test_single['strict_mean_iou_pct']:.2f}% | 2-Class-mIoU={m_test_single['two_class_miou_pct']:.2f}% | FG-F1={m_test_single['strict_mean_f1_pct']:.2f}% | GlobF1={m_test_single['strict_global_f1_pct']:.2f}% | SafeRec={m_test_single['safety_crack_recall_pct']:.2f}%")
    print(f"DeepCrack-237 (Flip-TTA): FG-IoU={m_test_tta['strict_mean_iou_pct']:.2f}% | 2-Class-mIoU={m_test_tta['two_class_miou_pct']:.2f}% | FG-F1={m_test_tta['strict_mean_f1_pct']:.2f}% | GlobF1={m_test_tta['strict_global_f1_pct']:.2f}% | SafeRec={m_test_tta['safety_crack_recall_pct']:.2f}%")
    print(f"DeepCrack-20 Subset: FG-IoU={m_sub_single['strict_mean_iou_pct']:.2f}% | SafeRec={m_sub_single['safety_crack_recall_pct']:.2f}%")
    print(f"SAVED -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
