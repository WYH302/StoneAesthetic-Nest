# Real Workshop Stone Slab Calibration & Case-Study Kit

Drop 3–5 smartphone or gantry-camera photos of real natural stone slabs into `photos/`, record each slab's physical `width_mm` and `height_mm` in `slabs_manifest.json` (optional; defaults to `2800 × 1600 mm`), and run:

```bash
# 1. Verify end-to-end pipeline on a synthetic perspective-tilted ArUco slab photo:
python experiments/run_real_yunfu_slab_case_study.py --smoke-test

# 2. Process real workshop photos in real_yunfu_slabs/photos/:
python experiments/run_real_yunfu_slab_case_study.py --crack-backend auto
```

- `ArUco_4Corners_A4_Print.pdf`: Ready-to-print A4 calibration sheets (Page 1 visual guide; Pages 2–5 `DICT_4X4_50` markers `ID 0=TL, 1=TR, 2=BR, 3=BL`).
- Works with **or without** printed ArUco markers (falls back to dual-polarity slab boundary contour detection or explicit `corners_px` in `slabs_manifest.json`).
