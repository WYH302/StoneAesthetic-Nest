"""Minimal End-to-End Quickstart Demo for StoneAesthetic-Nest.

Runs the 4-stage natural stone workflow on a sample CC0 raw onyx slab:
  1. Slab defect buffering + 2D structure-tensor flow + Golden Vein Zone detection
  2. Colloquial 3-tier workshop order compilation (cm/mm unit conversion)
  3. Neural-combinatorial 3-tier nesting (3x2 vein-matching + tandem stair strips + honeycomb mosaics)
  4. Independent RAC-Checker audit + 13-layer AutoCAD R2010 .DXF + Client Dry-Lay Preview
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ui_dxf.app import run_commercial_pipeline


def main() -> None:
    sample_img = REPO_ROOT / "examples" / "sample_slabs" / "Onyx011_Jade_Onyx.jpg"
    out_dir = REPO_ROOT / "outputs_demo"

    order_text = (
        "主件：切8个20x80厘米长方形（楼梯踏步，必须顺纹连排、纹路不断）；"
        "核心件：切6个50x50厘米正方形（茶几/地面主拼花，必须落在黄金主纹带，支持连纹对花）；"
        "边角料余料套裁：在前面大件定好、花纹锁死之后，剩下的空隙和边角料，尽量帮我切一些等边三角形马赛克拼花砖（边长18厘米），"
        "锯缝留3mm，边距5mm，水刀加桥切。"
    )

    report = run_commercial_pipeline(
        image_input=sample_img,
        order_text=order_text,
        output_dir=out_dir,
        width_mm=3000.0,
        height_mm=1800.0,
        safety_margin_mm=5.0,
        case_id="demo_jade_onyx",
    )

    am = report["nesting_result"].get("aesthetic_metrics", {})
    print("=== StoneAesthetic-Nest Demo Completed ===")
    print(f"  RAC-Checker Feasible : {report['certificate']['feasible']}")
    print(f"  3x2 Seam Continuity  : {am.get('seam_vein_continuity', 0.0) * 100:.1f}%")
    print(f"  Golden-Zone Hit Rate : {am.get('golden_zone_hit_rate', 0.0) * 100:.1f}%")
    print(f"  Stair Vein Error     : {am.get('stair_vein_error_deg', 0.0):.1f} deg")
    print(f"  Recovered Mosaics    : {am.get('tier3_mosaic_count', 0)} pcs")
    print(f"  DXF Toolpath Saved   : {report['artifacts']['cutting_dxf']}")
    print(f"  Dry-Lay Preview Saved: {report['artifacts']['dry_lay_preview_png']}")


if __name__ == "__main__":
    main()
