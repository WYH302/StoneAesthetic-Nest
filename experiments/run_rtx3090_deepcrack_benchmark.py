"""Genuine RTX 3090 Crack Segmentation Benchmark on Official Full DeepCrack (300 train / 237 test).

Compares 5 real methods head-to-head under identical strict & CNC-buffered metrics,
measuring exact Params, Peak RTX 3090 VRAM (MiB), GPU Train Time (s), and Inference FPS:
  1. Otsu + Canny (CPU, 0 params, 0 VRAM)
  2. Luminance-Adaptive Black-Hat (Ours, Zero-GPU CPU, 0 params, 0 VRAM)
  3. CrackUNet-Lite (~0.49M params, trained on DeepCrack-300 on RTX 3090)
  4. ResNet34-UNet (~24.4M params, ImageNet-pretrained, fine-tuned on DeepCrack-300 on RTX 3090)
  5. Meta SAM 2.1 Hiera-Tiny (38.9M params, official checkpoint, zero-shot prompted on RTX 3090)
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
WORKSPACE_ROOT = MASTER_DIR.parent
DEEPCRACK_ROOT = WORKSPACE_ROOT / "dataset" / "DeepCrack_Official" / "extracted"
SUBSET20_ROOT = WORKSPACE_ROOT / "dataset" / "DeepCrack"
SAM2_REPO = WORKSPACE_ROOT / "third_party" / "sam2_repo"
SAM2_CKPT = WORKSPACE_ROOT / "third_party" / "sam2" / "checkpoints" / "sam2.1_hiera_tiny.pt"
OUT_JSON = MASTER_DIR / "experiments" / "results" / "rtx3090_deepcrack_benchmark.json"


def _imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(raw, flags)
    if img is None:
        raise RuntimeError(f"Failed to decode {path}")
    return img


def load_deepcrack_split(img_dir: Path, lab_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    for ip in img_files:
        stem = ip.stem
        lp = lab_dir / f"{stem}.png"
        if not lp.exists():
            lp = lab_dir / f"{stem}_mask.png"
        if not lp.exists():
            continue
        bgr = _imread_unicode(ip, cv2.IMREAD_COLOR)
        gt = _imread_unicode(lp, cv2.IMREAD_GRAYSCALE)
        pairs.append((stem, bgr, (gt > 127).astype(np.uint8)))
    return pairs


def compute_metrics(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict[str, float]:
    """Compute strict pixel-level and 3-px CNC safety-buffer IoU/F1/Recall."""
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    strict_ious, strict_f1s, strict_recs = [], [], []
    buf_ious, buf_f1s, buf_recs = [], [], []
    tot_inter = tot_union = tot_p = tot_g = 0.0

    for p_u8, g_u8 in zip(preds, gts):
        p = p_u8 > 0
        g = g_u8 > 0
        inter = float(np.logical_and(p, g).sum())
        union = float(np.logical_or(p, g).sum())
        ps = float(p.sum())
        gs = float(g.sum())
        strict_ious.append(inter / max(1.0, union))
        strict_f1s.append(2.0 * inter / max(1.0, ps + gs))
        strict_recs.append(inter / max(1.0, gs))
        tot_inter += inter
        tot_union += union
        tot_p += ps
        tot_g += gs

        # 3-px dilated manufacturing safety envelope (mirrors 3-5 mm Minkowski buffer)
        pb = cv2.dilate(p_u8, k3) > 0
        gb = cv2.dilate(g_u8, k5) > 0
        ib = float(np.logical_and(pb, gb).sum())
        ub = float(np.logical_or(pb, gb).sum())
        # Safety recall: fraction of ground-truth crack pixels covered by dilated prediction
        cov = float(np.logical_and(pb, g).sum()) / max(1.0, gs)
        buf_ious.append(ib / max(1.0, ub))
        buf_f1s.append(2.0 * ib / max(1.0, float(pb.sum() + gb.sum())))
        buf_recs.append(cov)

    return {
        "strict_mean_iou_pct": round(float(np.mean(strict_ious) * 100.0), 2),
        "strict_mean_f1_pct": round(float(np.mean(strict_f1s) * 100.0), 2),
        "strict_global_iou_pct": round(float(tot_inter / max(1.0, tot_union) * 100.0), 2),
        "strict_global_f1_pct": round(float(2.0 * tot_inter / max(1.0, tot_p + tot_g) * 100.0), 2),
        "strict_recall_pct": round(float(np.mean(strict_recs) * 100.0), 2),
        "buf3px_iou_pct": round(float(np.mean(buf_ious) * 100.0), 2),
        "buf3px_f1_pct": round(float(np.mean(buf_f1s) * 100.0), 2),
        "safety_crack_recall_pct": round(float(np.mean(buf_recs) * 100.0), 2),
    }


def predict_otsu_canny(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    med = float(np.median(gray))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 60, 150)
    mask = ((otsu > 0) & (gray < min(110.0, med - 28.0))) | ((edges > 0) & (gray < med - 35.0))
    return mask.astype(np.uint8)


def predict_blackhat_adaptive(bgr: np.ndarray) -> np.ndarray:
    """Our exact zero-GPU luminance-adaptive multi-scale Black-Hat operator (vision_parser.py)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    med = float(np.median(gray))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bh9 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k9)
    if med <= 135.0:
        tau_d = max(18.0, min(55.0, med - 35.0))
        raw = (bh9 > 24) & (gray < tau_d)
    else:
        k23 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        k45 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        bh23 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k23)
        bh45 = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k45)
        bh_max = np.maximum(np.maximum(bh9, bh23), bh45)
        tau_d = min(115.0, med - 42.0)
        raw = (bh_max > 28) & (gray < tau_d)
    # Remove tiny isolated speckle noise (<18 px)
    u8 = raw.astype(np.uint8)
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    clean = np.zeros_like(u8)
    for c in range(1, n_cc):
        if stats[c, cv2.CC_STAT_AREA] >= 18:
            clean[labels == c] = 1
    return clean


# =============================================================================
# Lightweight Depthwise-Separable CrackUNet-Lite (~0.49M params)
# =============================================================================
class DSConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrackUNetLite(nn.Module):
    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(inplace=True),
        )
        self.e1 = DSConvBlock(base, base)
        self.e2 = DSConvBlock(base, base * 2)
        self.e3 = DSConvBlock(base * 2, base * 4)
        self.e4 = DSConvBlock(base * 4, base * 8)
        self.bot = DSConvBlock(base * 8, base * 8)
        self.d4 = DSConvBlock(base * 16, base * 4)
        self.d3 = DSConvBlock(base * 8, base * 2)
        self.d2 = DSConvBlock(base * 4, base)
        self.d1 = DSConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        x1 = self.e1(s0)
        x2 = self.e2(F.max_pool2d(x1, 2))
        x3 = self.e3(F.max_pool2d(x2, 2))
        x4 = self.e4(F.max_pool2d(x3, 2))
        b = self.bot(F.max_pool2d(x4, 2))
        u4 = self.d4(torch.cat([F.interpolate(b, size=x4.shape[2:], mode="bilinear", align_corners=False), x4], dim=1))
        u3 = self.d3(torch.cat([F.interpolate(u4, size=x3.shape[2:], mode="bilinear", align_corners=False), x3], dim=1))
        u2 = self.d2(torch.cat([F.interpolate(u3, size=x2.shape[2:], mode="bilinear", align_corners=False), x2], dim=1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=x1.shape[2:], mode="bilinear", align_corners=False), x1], dim=1))
        return self.head(u1)


# =============================================================================
# Heavy Pretrained ResNet34-UNet (~24.4M params)
# =============================================================================
class ResNet34UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        enc = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.l1 = enc.layer1  # 64
        self.l2 = enc.layer2  # 128
        self.l3 = enc.layer3  # 256
        self.l4 = enc.layer4  # 512

        def _up(ic: int, oc: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )

        self.d3 = _up(512 + 256, 256)
        self.d2 = _up(256 + 128, 128)
        self.d1 = _up(128 + 64, 64)
        self.d0 = _up(64 + 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        s0 = self.stem(x)
        x1 = self.l1(self.pool(s0))
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        u3 = self.d3(torch.cat([F.interpolate(x4, size=x3.shape[2:], mode="bilinear", align_corners=False), x3], dim=1))
        u2 = self.d2(torch.cat([F.interpolate(u3, size=x2.shape[2:], mode="bilinear", align_corners=False), x2], dim=1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=x1.shape[2:], mode="bilinear", align_corners=False), x1], dim=1))
        u0 = self.d0(torch.cat([F.interpolate(u1, size=s0.shape[2:], mode="bilinear", align_corners=False), s0], dim=1))
        out = F.interpolate(self.head(u0), size=(h, w), mode="bilinear", align_corners=False)
        return out


class CrackDataset(Dataset):
    def __init__(self, items: list[tuple[str, np.ndarray, np.ndarray]], size: tuple[int, int] = (384, 512), augment: bool = False) -> None:
        self.items = items
        self.h, self.w = size
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, bgr, gt = self.items[idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(gt, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                msk = np.ascontiguousarray(msk[:, ::-1])
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[::-1, :, :])
                msk = np.ascontiguousarray(msk[::-1, :])
            if np.random.rand() < 0.4:
                alpha = float(np.random.uniform(0.85, 1.15))
                beta = float(np.random.uniform(-12.0, 12.0))
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        x_t = torch.from_numpy(x.transpose(2, 0, 1))
        y_t = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)
        return x_t, y_t


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor([4.0], device=logits.device))
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6
    dice = 1.0 - (2.0 * inter / denom).mean()
    return 0.5 * bce + 0.5 * dice


def train_and_eval_gpu_model(
    name: str,
    model: nn.Module,
    train_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    test_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    subset_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    epochs: int,
    lr: float,
    batch_size: int = 8,
) -> dict[str, Any]:
    device = torch.device("cuda")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ds_tr = CrackDataset(train_pairs, size=(384, 512), augment=True)
    loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    t_tr0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss = dice_bce_loss(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
    torch.cuda.synchronize()
    train_sec = round(time.perf_counter() - t_tr0, 2)
    train_peak_vram_mib = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1)

    # Measure clean inference peak VRAM & latency
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _infer_split(pairs: list[tuple[str, np.ndarray, np.ndarray]], warmup: int = 5) -> tuple[list[np.ndarray], float]:
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for wi in range(min(warmup, len(pairs))):
                rgb = cv2.cvtColor(pairs[wi][1], cv2.COLOR_BGR2RGB)
                im = cv2.resize(rgb, (512, 384))
                x = torch.from_numpy(((im.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _, bgr, gt in pairs:
                orig_h, orig_w = gt.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                im = cv2.resize(rgb, (512, 384))
                x = torch.from_numpy(((im.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    prob = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
                prob_orig = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                preds.append((prob_orig >= 0.5).astype(np.uint8))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        return preds, elapsed

    test_preds, test_sec = _infer_split(test_pairs)
    infer_peak_vram_mib = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1)
    sub_preds, _ = _infer_split(subset_pairs, warmup=2)

    m_test = compute_metrics(test_preds, [g for _, _, g in test_pairs])
    m_sub = compute_metrics(sub_preds, [g for _, _, g in subset_pairs])
    ms_per_img = round(test_sec / len(test_pairs) * 1000.0, 2)
    fps = round(len(test_pairs) / max(1e-6, test_sec), 1)

    del model, opt, sched
    torch.cuda.empty_cache()

    res = {
        "method": name,
        "device": "RTX 3090 (CUDA)",
        "params": int(n_params),
        "params_m": round(n_params / 1e6, 2),
        "train_epochs": epochs,
        "train_gpu_sec": train_sec,
        "train_peak_vram_mib": train_peak_vram_mib,
        "infer_peak_vram_mib": infer_peak_vram_mib,
        "infer_ms_per_img": ms_per_img,
        "infer_fps": fps,
        "deepcrack_237_test": m_test,
        "deepcrack_20_subset": m_sub,
    }
    print(
        f"[{name}] params={res['params_m']:.2f}M | train={train_sec:.1f}s (VRAM {train_peak_vram_mib:.0f} MiB) | "
        f"infer={ms_per_img:.2f} ms ({fps:.1f} FPS, VRAM {infer_peak_vram_mib:.0f} MiB) | "
        f"237-test IoU={m_test['strict_mean_iou_pct']:.2f}% F1={m_test['strict_mean_f1_pct']:.2f}% "
        f"BufF1={m_test['buf3px_f1_pct']:.2f}% SafeRec={m_test['safety_crack_recall_pct']:.2f}%",
        flush=True,
    )
    return res


def eval_sam2_zeroshot_gpu(
    test_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    subset_pairs: list[tuple[str, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Run official Meta SAM 2.1 Hiera-Tiny natively on RTX 3090, prompted automatically by Black-Hat peaks."""
    if str(SAM2_REPO) not in sys.path:
        sys.path.insert(0, str(SAM2_REPO))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(SAM2_CKPT), device=device)
    n_params = sum(p.numel() for p in sam2_model.parameters())
    predictor = SAM2ImagePredictor(sam2_model)

    def _run_split(pairs: list[tuple[str, np.ndarray, np.ndarray]], warmup: int = 3) -> tuple[list[np.ndarray], float]:
        preds: list[np.ndarray] = []

        def _predict_one(bgr: np.ndarray) -> np.ndarray:
            bh_mask = predict_blackhat_adaptive(bgr)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            ys, xs = np.where(bh_mask > 0)
            if len(xs) < 4:
                return bh_mask
            # Pick up to 6 spatially spread foreground crack points + 3 background non-crack points (zero GT peeking!)
            idx_fg = np.linspace(0, len(xs) - 1, min(6, len(xs)), dtype=int)
            pts = [[float(xs[i]), float(ys[i])] for i in idx_fg]
            labs = [1] * len(pts)
            bg_ys, bg_xs = np.where(bh_mask == 0)
            if len(bg_xs) >= 3:
                idx_bg = np.linspace(0, len(bg_xs) - 1, 3, dtype=int)
                for i in idx_bg:
                    pts.append([float(bg_xs[i]), float(bg_ys[i])])
                    labs.append(0)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(rgb)
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(pts, dtype=np.float32),
                    point_labels=np.array(labs, dtype=np.int32),
                    multimask_output=True,
                )
            # Pick the SAM 2 mask with highest overlap agreement against thin crack geometry (prevent full-slab flood)
            best_m = None
            best_score = -1.0
            for m, sc in zip(masks, scores):
                area_frac = float(m.mean())
                if area_frac > 0.35:
                    continue
                if float(sc) > best_score:
                    best_score = float(sc)
                    best_m = m
            if best_m is None:
                return bh_mask
            return (best_m > 0.5).astype(np.uint8)

        for wi in range(min(warmup, len(pairs))):
            _ = _predict_one(pairs[wi][1])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _, bgr, _ in pairs:
            preds.append(_predict_one(bgr))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return preds, elapsed

    test_preds, test_sec = _run_split(test_pairs, warmup=3)
    infer_peak_vram_mib = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1)
    sub_preds, _ = _run_split(subset_pairs, warmup=1)

    m_test = compute_metrics(test_preds, [g for _, _, g in test_pairs])
    m_sub = compute_metrics(sub_preds, [g for _, _, g in subset_pairs])
    ms_per_img = round(test_sec / len(test_pairs) * 1000.0, 2)
    fps = round(len(test_pairs) / max(1e-6, test_sec), 1)

    del predictor, sam2_model
    torch.cuda.empty_cache()

    res = {
        "method": "Meta SAM 2.1 Hiera-Tiny (Zero-Shot)",
        "device": "RTX 3090 (CUDA)",
        "params": int(n_params),
        "params_m": round(n_params / 1e6, 2),
        "train_epochs": 0,
        "train_gpu_sec": 0.0,
        "train_peak_vram_mib": 0.0,
        "infer_peak_vram_mib": infer_peak_vram_mib,
        "infer_ms_per_img": ms_per_img,
        "infer_fps": fps,
        "deepcrack_237_test": m_test,
        "deepcrack_20_subset": m_sub,
    }
    print(
        f"[SAM 2.1 Hiera-Tiny] params={res['params_m']:.2f}M | infer={ms_per_img:.2f} ms ({fps:.1f} FPS, VRAM {infer_peak_vram_mib:.0f} MiB) | "
        f"237-test IoU={m_test['strict_mean_iou_pct']:.2f}% F1={m_test['strict_mean_f1_pct']:.2f}% "
        f"BufF1={m_test['buf3px_f1_pct']:.2f}% SafeRec={m_test['safety_crack_recall_pct']:.2f}%",
        flush=True,
    )
    return res


def eval_cpu_method(
    name: str,
    fn,
    test_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    subset_pairs: list[tuple[str, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    for wi in range(min(5, len(test_pairs))):
        _ = fn(test_pairs[wi][1])
    t0 = time.perf_counter()
    test_preds = [fn(bgr) for _, bgr, _ in test_pairs]
    test_sec = time.perf_counter() - t0
    sub_preds = [fn(bgr) for _, bgr, _ in subset_pairs]

    m_test = compute_metrics(test_preds, [g for _, _, g in test_pairs])
    m_sub = compute_metrics(sub_preds, [g for _, _, g in subset_pairs])
    ms_per_img = round(test_sec / len(test_pairs) * 1000.0, 2)
    fps = round(len(test_pairs) / max(1e-6, test_sec), 1)

    res = {
        "method": name,
        "device": "CPU (Zero-GPU)",
        "params": 0,
        "params_m": 0.0,
        "train_epochs": 0,
        "train_gpu_sec": 0.0,
        "train_peak_vram_mib": 0.0,
        "infer_peak_vram_mib": 0.0,
        "infer_ms_per_img": ms_per_img,
        "infer_fps": fps,
        "deepcrack_237_test": m_test,
        "deepcrack_20_subset": m_sub,
    }
    print(
        f"[{name}] params=0.00M | infer={ms_per_img:.2f} ms ({fps:.1f} FPS, VRAM 0 MiB) | "
        f"237-test IoU={m_test['strict_mean_iou_pct']:.2f}% F1={m_test['strict_mean_f1_pct']:.2f}% "
        f"BufF1={m_test['buf3px_f1_pct']:.2f}% SafeRec={m_test['safety_crack_recall_pct']:.2f}%",
        flush=True,
    )
    return res


def main() -> None:
    assert torch.cuda.is_available(), "CUDA GPU is required for Experiment C!"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"=== Experiment C: Official DeepCrack RTX 3090 Benchmark on {gpu_name} ===", flush=True)

    train_pairs = load_deepcrack_split(DEEPCRACK_ROOT / "train_img", DEEPCRACK_ROOT / "train_lab")
    test_pairs = load_deepcrack_split(DEEPCRACK_ROOT / "test_img", DEEPCRACK_ROOT / "test_lab")
    subset_pairs = load_deepcrack_split(SUBSET20_ROOT / "images", SUBSET20_ROOT / "masks")
    print(f"Loaded DeepCrack Official: train={len(train_pairs)}, test={len(test_pairs)}, subset20={len(subset_pairs)}", flush=True)

    t_all = time.perf_counter()
    rows = []
    rows.append(eval_cpu_method("Otsu + Canny (CPU)", predict_otsu_canny, test_pairs, subset_pairs))
    rows.append(eval_cpu_method("Black-Hat Adaptive (Ours, Zero-GPU)", predict_blackhat_adaptive, test_pairs, subset_pairs))
    rows.append(train_and_eval_gpu_model("CrackUNet-Lite (RTX 3090)", CrackUNetLite(base=32), train_pairs, test_pairs, subset_pairs, epochs=35, lr=1.8e-3, batch_size=8))
    rows.append(train_and_eval_gpu_model("ResNet34-UNet (RTX 3090)", ResNet34UNet(), train_pairs, test_pairs, subset_pairs, epochs=25, lr=4.0e-4, batch_size=8))
    rows.append(eval_sam2_zeroshot_gpu(test_pairs, subset_pairs))

    total_gpu_sec = round(sum(r["train_gpu_sec"] for r in rows) + sum(r["infer_ms_per_img"] * len(test_pairs) / 1000.0 for r in rows if "RTX 3090" in r["device"]), 2)
    summary = {
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "deepcrack_official_train_n": len(train_pairs),
        "deepcrack_official_test_n": len(test_pairs),
        "deepcrack_subset_n": len(subset_pairs),
        "total_wall_sec": round(time.perf_counter() - t_all, 2),
        "total_gpu_sec": total_gpu_sec,
        "total_gpu_hours": round(total_gpu_sec / 3600.0, 4),
        "methods": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSAVED Experiment C results -> {OUT_JSON} (total_wall={summary['total_wall_sec']}s, total_gpu={total_gpu_sec}s)", flush=True)


if __name__ == "__main__":
    main()
