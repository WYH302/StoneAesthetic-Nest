"""Experiment E (RTX 3090): Canonical DeepCrack-Net (Liu et al., 2019) & Deep-Supervised CrackUNet-Lite-DS.

Runs strictly inside dedicated RTX 3090 VRAM (batch=4, <1.8 GB peak VRAM) on the Official Full
DeepCrack benchmark (300 train / 237 strictly held-out test) to report clean unthrottled GPU speed,
VRAM footprint, and accuracy alongside Otsu, Black-Hat Adaptive (Ours, 0-GPU), ResNet34-UNet, and Meta SAM 2.1.
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
from torch.utils.data import DataLoader
import torchvision.models as tv_models

MASTER_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = MASTER_DIR.parent
if str(MASTER_DIR) not in sys.path:
    sys.path.insert(0, str(MASTER_DIR))

from experiments.run_rtx3090_deepcrack_benchmark import (
    DEEPCRACK_ROOT,
    SUBSET20_ROOT,
    CrackDataset,
    DSConvBlock,
    compute_metrics,
    dice_bce_loss,
    load_deepcrack_split,
)
from experiments.run_yunfu_aesthetic_benchmark_and_figures import RESULTS_DIR

OUT_JSON = RESULTS_DIR / "exp_e_deepcrack_sota_gpu.json"


class DeepCrackNetVGG16(nn.Module):
    """Canonical multi-scale side-output fusion network following DeepCrack (Liu et al., Neurocomputing 2019)."""

    def __init__(self) -> None:
        super().__init__()
        vgg = tv_models.vgg16_bn(weights=tv_models.VGG16_BN_Weights.DEFAULT).features
        self.s1 = nn.Sequential(*vgg[:6])    # 64, 1x
        self.s2 = nn.Sequential(*vgg[6:13])  # 128, 1/2
        self.s3 = nn.Sequential(*vgg[13:23]) # 256, 1/4
        self.s4 = nn.Sequential(*vgg[23:33]) # 512, 1/8
        self.s5 = nn.Sequential(*vgg[33:43]) # 512, 1/16

        self.side1 = nn.Conv2d(64, 1, 1)
        self.side2 = nn.Conv2d(128, 1, 1)
        self.side3 = nn.Conv2d(256, 1, 1)
        self.side4 = nn.Conv2d(512, 1, 1)
        self.side5 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h, w = x.shape[2:]
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        f5 = self.s5(f4)
        o1 = self.side1(f1)
        o2 = F.interpolate(self.side2(f2), size=(h, w), mode="bilinear", align_corners=False)
        o3 = F.interpolate(self.side3(f3), size=(h, w), mode="bilinear", align_corners=False)
        o4 = F.interpolate(self.side4(f4), size=(h, w), mode="bilinear", align_corners=False)
        o5 = F.interpolate(self.side5(f5), size=(h, w), mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([o1, o2, o3, o4, o5], dim=1))
        return fused, [o1, o2, o3, o4, o5]


class CrackUNetLiteDS(nn.Module):
    """0.41M-param Depthwise-Separable U-Net with Multi-Scale Deep Supervision."""

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
        self.aux3 = nn.Conv2d(base * 2, 1, 1)
        self.aux2 = nn.Conv2d(base, 1, 1)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h, w = x.shape[2:]
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
        out = self.head(u1)
        a3 = F.interpolate(self.aux3(u3), size=(h, w), mode="bilinear", align_corners=False)
        a2 = F.interpolate(self.aux2(u2), size=(h, w), mode="bilinear", align_corners=False)
        return out, [a2, a3]


def run_ds_model(
    name: str,
    model: nn.Module,
    train_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    test_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    subset_pairs: list[tuple[str, np.ndarray, np.ndarray]],
    epochs: int,
    lr: float,
    batch_size: int = 4,
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

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                main_l, sides = model(xb)
                loss = dice_bce_loss(main_l, yb) + 0.25 * sum(dice_bce_loss(s, yb) for s in sides) / len(sides)
            loss.backward()
            opt.step()
        sched.step()
    torch.cuda.synchronize()
    train_sec = round(time.perf_counter() - t0, 2)
    train_vram = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1)

    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _infer(pairs: list[tuple[str, np.ndarray, np.ndarray]], warmup: int = 5) -> tuple[list[np.ndarray], float]:
        preds = []
        with torch.no_grad():
            for wi in range(min(warmup, len(pairs))):
                rgb = cv2.cvtColor(pairs[wi][1], cv2.COLOR_BGR2RGB)
                im = cv2.resize(rgb, (512, 384))
                x = torch.from_numpy(((im.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(x)[0]
            torch.cuda.synchronize()
            ts = time.perf_counter()
            for _, bgr, gt in pairs:
                oh, ow = gt.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                im = cv2.resize(rgb, (512, 384))
                x = torch.from_numpy(((im.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    prob = torch.sigmoid(model(x)[0])[0, 0].float().cpu().numpy()
                prob_o = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
                preds.append((prob_o >= 0.5).astype(np.uint8))
            torch.cuda.synchronize()
            dt = time.perf_counter() - ts
        return preds, dt

    test_preds, test_dt = _infer(test_pairs)
    infer_vram = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1)
    sub_preds, _ = _infer(subset_pairs, warmup=2)

    m_test = compute_metrics(test_preds, [g for _, _, g in test_pairs])
    m_sub = compute_metrics(sub_preds, [g for _, _, g in subset_pairs])
    ms_img = round(test_dt / len(test_pairs) * 1000.0, 2)
    fps = round(len(test_pairs) / max(1e-6, test_dt), 1)

    del model, opt, sched
    torch.cuda.empty_cache()

    res = {
        "method": name,
        "device": "RTX 3090 (CUDA)",
        "params": int(n_params),
        "params_m": round(n_params / 1e6, 2),
        "train_epochs": epochs,
        "train_gpu_sec": train_sec,
        "train_peak_vram_mib": train_vram,
        "infer_peak_vram_mib": infer_vram,
        "infer_ms_per_img": ms_img,
        "infer_fps": fps,
        "deepcrack_237_test": m_test,
        "deepcrack_20_subset": m_sub,
    }
    print(
        f"[{name}] params={res['params_m']:.2f}M | train={train_sec:.1f}s (VRAM {train_vram:.0f} MiB) | "
        f"infer={ms_img:.2f} ms ({fps:.1f} FPS, VRAM {infer_vram:.0f} MiB) | "
        f"237-test IoU={m_test['strict_mean_iou_pct']:.2f}% F1={m_test['strict_mean_f1_pct']:.2f}% "
        f"GlobF1={m_test['strict_global_f1_pct']:.2f}% SafeRec={m_test['safety_crack_recall_pct']:.2f}%",
        flush=True,
    )
    return res


def main() -> None:
    assert torch.cuda.is_available()
    print(f"=== Experiment E: Canonical DeepCrack-Net & CrackUNet-Lite-DS on {torch.cuda.get_device_name(0)} ===", flush=True)
    train_pairs = load_deepcrack_split(DEEPCRACK_ROOT / "train_img", DEEPCRACK_ROOT / "train_lab")
    test_pairs = load_deepcrack_split(DEEPCRACK_ROOT / "test_img", DEEPCRACK_ROOT / "test_lab")
    subset_pairs = load_deepcrack_split(SUBSET20_ROOT / "images", SUBSET20_ROOT / "masks")

    rows = [
        run_ds_model("CrackUNet-Lite-DS (Ours, Opt. GPU)", CrackUNetLiteDS(base=32), train_pairs, test_pairs, subset_pairs, epochs=45, lr=2.0e-3, batch_size=4),
        run_ds_model("DeepCrack-Net (VGG16-MS, Liu et al. 2019)", DeepCrackNetVGG16(), train_pairs, test_pairs, subset_pairs, epochs=22, lr=2.5e-4, batch_size=4),
    ]
    out = {
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "methods": rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSAVED Experiment E -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
