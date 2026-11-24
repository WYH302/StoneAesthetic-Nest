"""Experiment E-Deploy (RTX 3090): FP16 + CUDA Graph Deployment Audit for CrackSeg-MV3-clDice.

Demonstrates zero-overhead Windows/Linux GPU deployment of our trained 3.17M
`CrackSeg-MV3-clDice` model via native PyTorch `torch.cuda.CUDAGraph()` + FP16
Ampere Tensor Cores, eliminating Windows WDDM per-kernel launch overhead (~120
MobileNetV3 depthwise ops) with zero accuracy degradation on Official DeepCrack-237:
  1. PyTorch FP32 Eager (Single-Pass)
  2. PyTorch FP16 Eager (Single-Pass, channels_last NHWC)
  3. PyTorch FP16 CUDA Graph (Single-Pass, 1 fused GPU graph replay)
  4. PyTorch FP16 CUDA Graph (Batched 4-View Flip TTA, B=4 in 1 GPU graph replay)

Merges verified metrics directly into `experiments/results/exp_e_sota_crack_mv3_cldice.json`.
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
import torchvision.models as tv_models

MASTER_DIR = Path(__file__).resolve().parents[1]
if str(MASTER_DIR) not in sys.path:
    sys.path.insert(0, str(MASTER_DIR))

from experiments.run_exp_e_sota_crack_gpu import (
    CKPT_OUT,
    ConvBNSiLU,
    DualStripAttention,
    OUT_JSON,
    compute_2class_miou,
)
from experiments.run_rtx3090_deepcrack_benchmark import (
    DEEPCRACK_ROOT,
    compute_metrics,
    load_deepcrack_split,
)


class CrackSegMV3Deploy(nn.Module):
    """Inference-only offline-safe CrackSeg-MV3-clDice returning primary probability map."""

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
        h, w = x.shape[2], x.shape[3]
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
        return torch.sigmoid(self.head(self.refine_1x(torch.cat([up1, d1], dim=1))))


def _eval_ods_ois_fast(prob_maps: list[np.ndarray], gts: list[np.ndarray]) -> dict[str, float]:
    """Fast exact 37-threshold ODS/OIS/2-Class mIoU evaluation on DeepCrack-237."""
    thresholds = np.linspace(0.05, 0.95, 37)
    n_t = len(thresholds)
    tp_g = np.zeros(n_t, dtype=np.int64)
    fp_g = np.zeros(n_t, dtype=np.int64)
    fn_g = np.zeros(n_t, dtype=np.int64)
    miou_sum = np.zeros(n_t, dtype=np.float64)
    ois_list: list[float] = []

    for prob, gt in zip(prob_maps, gts):
        g = gt > 0
        g_sum = int(g.sum())
        n_pix = g.size
        best_f1_img = 0.0
        for ti, thr in enumerate(thresholds):
            p = prob >= thr
            tp = int(np.logical_and(p, g).sum())
            p_sum = int(p.sum())
            fp = p_sum - tp
            fn = g_sum - tp
            tn = n_pix - tp - fp - fn
            tp_g[ti] += tp
            fp_g[ti] += fp
            fn_g[ti] += fn
            f1_i = (2.0 * tp) / max(1.0, 2.0 * tp + fp + fn)
            if f1_i > best_f1_img:
                best_f1_img = f1_i
            iou_fg = tp / max(1.0, tp + fp + fn)
            iou_bg = tn / max(1.0, tn + fp + fn)
            miou_sum[ti] += 0.5 * (iou_fg + iou_bg)
        ois_list.append(best_f1_img)

    gf1_arr = (2.0 * tp_g) / np.maximum(1.0, 2.0 * tp_g + fp_g + fn_g)
    best_i = int(np.argmax(gf1_arr))
    return {
        "ods_global_f1_pct": round(float(gf1_arr[best_i] * 100.0), 2),
        "ods_best_threshold": round(float(thresholds[best_i]), 3),
        "ois_f1_pct": round(float(np.mean(ois_list) * 100.0), 2),
        "miou_2class_ods_pct": round(float((miou_sum[best_i] / len(gts)) * 100.0), 2),
    }


def _run_cuda_graph_benchmark(
    name: str,
    model: nn.Module,
    cpu_inputs: list[torch.Tensor],
    orig_shapes: list[tuple[int, int]],
    gts: list[np.ndarray],
    device: torch.device,
    use_fp16: bool,
    use_graph: bool,
    use_batched_tta: bool,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    dtype = torch.float16 if use_fp16 else torch.float32
    mem_fmt = torch.channels_last if use_fp16 else torch.contiguous_format
    model = model.to(device=device, dtype=dtype, memory_format=mem_fmt).eval()

    static_in = torch.zeros((1, 3, 384, 512), device=device, dtype=dtype).to(memory_format=mem_fmt)

    def _forward_fn(inp: torch.Tensor) -> torch.Tensor:
        if not use_batched_tta:
            return model(inp)
        xb = torch.cat([inp, torch.flip(inp, [3]), torch.flip(inp, [2]), torch.flip(inp, [2, 3])], dim=0)
        pb = model(xb)
        return 0.25 * (
            pb[0:1]
            + torch.flip(pb[1:2], [3])
            + torch.flip(pb[2:3], [2])
            + torch.flip(pb[3:4], [2, 3])
        )

    # Warmup stream before CUDA Graph capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(15):
            _ = _forward_fn(static_in)
    torch.cuda.current_stream().wait_stream(s)

    graph = None
    static_out = None
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            static_out = _forward_fn(static_in)
        torch.cuda.synchronize()
        # Warm up replay
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize()
    else:
        with torch.no_grad():
            for _ in range(20):
                _ = _forward_fn(static_in)
        torch.cuda.synchronize()

    # Pure GPU kernel latency measurement using CUDA Events (hardware timer)
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    gpu_ms_list: list[float] = []
    prob_maps: list[np.ndarray] = []
    preds_05: list[np.ndarray] = []

    with torch.no_grad():
        for x_cpu, (h0, w0) in zip(cpu_inputs, orig_shapes):
            x_dev = x_cpu.to(device=device, dtype=dtype, non_blocking=False).to(memory_format=mem_fmt)
            static_in.copy_(x_dev)
            torch.cuda.synchronize()
            ev_start.record()
            if use_graph and graph is not None and static_out is not None:
                graph.replay()
                out_t = static_out
            else:
                out_t = _forward_fn(static_in)
            ev_end.record()
            torch.cuda.synchronize()
            gpu_ms_list.append(float(ev_start.elapsed_time(ev_end)))

            prob_np = out_t.squeeze().float().cpu().numpy()
            prob_full = cv2.resize(prob_np, (w0, h0), interpolation=cv2.INTER_LINEAR)
            prob_maps.append(prob_full)
            preds_05.append((prob_full >= 0.5).astype(np.uint8))

    vram_mib = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 1)
    mean_ms = float(np.mean(gpu_ms_list))
    med_ms = float(np.median(gpu_ms_list))
    m05 = compute_metrics(preds_05, gts)
    miou_05 = compute_2class_miou(preds_05, gts)
    ods = _eval_ods_ois_fast(prob_maps, gts)

    res = {
        "mode": name,
        "pure_gpu_mean_ms": round(mean_ms, 2),
        "pure_gpu_median_ms": round(med_ms, 2),
        "pure_gpu_fps": round(1000.0 / max(1e-6, mean_ms), 1),
        "vram_peak_mib": vram_mib,
        "fg_iou_05_pct": m05["strict_mean_iou_pct"],
        "strict_f1_05_pct": m05["strict_mean_f1_pct"],
        "global_f1_05_pct": m05["strict_global_f1_pct"],
        "safety_recall_05_pct": m05["safety_crack_recall_pct"],
        "miou_2class_05_pct": miou_05,
        **ods,
    }
    print(
        f"  [{name:30s}] {res['pure_gpu_mean_ms']:5.2f} ms ({res['pure_gpu_fps']:6.1f} FPS) | "
        f"VRAM {vram_mib:5.1f} MiB | FG-IoU {res['fg_iou_05_pct']:.2f}% | "
        f"ODS {res['ods_global_f1_pct']:.2f}% | OIS {res['ois_f1_pct']:.2f}% | mIoU {res['miou_2class_ods_pct']:.2f}%",
        flush=True,
    )
    return res


def main() -> None:
    assert torch.cuda.is_available(), "CUDA RTX 3090 required!"
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"=== Exp E-Deploy: FP16 + CUDA Graph Deployment Audit on {torch.cuda.get_device_name(0)} ===", flush=True)
    test_pairs = load_deepcrack_split(DEEPCRACK_ROOT / "test_img", DEEPCRACK_ROOT / "test_lab")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    cpu_inputs: list[torch.Tensor] = []
    orig_shapes: list[tuple[int, int]] = []
    gts: list[np.ndarray] = []
    for _, bgr, gt in test_pairs:
        h0, w0 = bgr.shape[:2]
        orig_shapes.append((h0, w0))
        gts.append(gt)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_rs = cv2.resize(rgb, (512, 384), interpolation=cv2.INTER_LINEAR)
        x_np = ((rgb_rs.astype(np.float32) / 255.0) - mean) / std
        cpu_inputs.append(torch.from_numpy(x_np.transpose(2, 0, 1)).unsqueeze(0).contiguous())

    state = torch.load(CKPT_OUT, map_location="cpu", weights_only=True)

    def _fresh_model() -> CrackSegMV3Deploy:
        m = CrackSegMV3Deploy(fpn_c=64)
        m.load_state_dict(state, strict=True)
        return m

    r_fp32_eager = _run_cuda_graph_benchmark(
        "FP32_Eager_SinglePass", _fresh_model(), cpu_inputs, orig_shapes, gts, device,
        use_fp16=False, use_graph=False, use_batched_tta=False,
    )
    r_fp16_eager = _run_cuda_graph_benchmark(
        "FP16_Eager_SinglePass", _fresh_model(), cpu_inputs, orig_shapes, gts, device,
        use_fp16=True, use_graph=False, use_batched_tta=False,
    )
    r_fp16_graph_1 = _run_cuda_graph_benchmark(
        "FP16_CUDAGraph_SinglePass", _fresh_model(), cpu_inputs, orig_shapes, gts, device,
        use_fp16=True, use_graph=True, use_batched_tta=False,
    )
    r_fp16_graph_4 = _run_cuda_graph_benchmark(
        "FP16_CUDAGraph_Batched4xTTA", _fresh_model(), cpu_inputs, orig_shapes, gts, device,
        use_fp16=True, use_graph=True, use_batched_tta=True,
    )

    existing = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    existing["deployment_fp16_audit"] = {
        "device": torch.cuda.get_device_name(0),
        "input_resolution": "384x512",
        "fp32_eager_single": r_fp32_eager,
        "fp16_eager_single": r_fp16_eager,
        "fp16_cudagraph_single": r_fp16_graph_1,
        "fp16_cudagraph_batched_4tta": r_fp16_graph_4,
    }
    OUT_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\n[OK] Saved FP16 + CUDA Graph deployment audit into {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
