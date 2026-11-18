"""Commercial Chinese Stone Slab Nesting Software MVP (云浮石材智能避障排板收费软件系统).

Provides:
1. Full end-to-end pipeline `run_commercial_pipeline` (Photo -> 4-Corner Calibration ->
   Crack & Vein Perception -> Chinese Order Compilation -> Count-First Nesting ->
   Solver-in-the-Loop Feedback -> Overlay & Vein Gallery -> Multi-Layer DXF Export ->
   Commercial Profit Quote).
2. Interactive Web UI (`create_fastapi_app` / Gradio launcher) for stone merchants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from ..llm_compiler.feedback_loop import apply_relaxation_plan, diagnose_and_suggest_relaxations
from ..llm_compiler.order_compiler import (
    apply_interactive_command,
    compile_chinese_order,
    compile_to_solver_case,
)
from ..solver_core.solver import solve_cf_slabnest
from ..solver_core.validator import validate_nesting_solution
from ..vision.vision_parser import parse_slab_image
from .dxf_exporter import export_industrial_dxf
from .overlay_renderer import (
    render_cut_piece_gallery,
    render_dry_lay_preview,
    render_overlay_on_slab_image,
    render_overlay_svg,
)


def calculate_commercial_quote(
    case: dict[str, Any],
    result: dict[str, Any],
    dxf_info: dict[str, Any],
    slab_unit_price_rmb_m2: float = 680.0,
    finished_part_multiplier: float = 1.65,
) -> dict[str, Any]:
    """Calculate commercial material cost, machining cost, and net profit for stone merchants."""
    w_m = float(case["board"]["width"]) / 1000.0
    h_m = float(case["board"]["height"]) / 1000.0
    slab_area_m2 = w_m * h_m
    used_area_m2 = float(result.get("used_area_mm2", 0.0)) / 1e6

    slab_cost_rmb = slab_area_m2 * slab_unit_price_rmb_m2
    cut_len_m = float(dxf_info.get("total_cut_length_m", 0.0))
    rate_per_m = 18.0 if result.get("cut_mode") == "waterjet" else 8.5
    machining_fee_rmb = cut_len_m * rate_per_m + float(dxf_info.get("pierce_count", 0)) * 3.0

    finished_revenue_rmb = used_area_m2 * slab_unit_price_rmb_m2 * finished_part_multiplier
    # Estimated extra material saved vs traditional 70% manual layout
    saved_area_m2 = max(0.0, used_area_m2 - slab_area_m2 * 0.65)
    saved_material_value_rmb = saved_area_m2 * slab_unit_price_rmb_m2

    return {
        "slab_area_m2": round(slab_area_m2, 3),
        "used_area_m2": round(used_area_m2, 3),
        "utilization_pct": round(float(result.get("utilization", 0.0)) * 100.0, 2),
        "slab_raw_cost_rmb": round(slab_cost_rmb, 2),
        "machining_fee_rmb": round(machining_fee_rmb, 2),
        "estimated_order_revenue_rmb": round(finished_revenue_rmb, 2),
        "saved_material_value_rmb": round(saved_material_value_rmb, 2),
        "net_gross_margin_rmb": round(finished_revenue_rmb - slab_cost_rmb - machining_fee_rmb, 2),
    }


def run_commercial_pipeline(
    image_input: str | Path,
    order_text: str,
    output_dir: str | Path,
    width_mm: float = 2800.0,
    height_mm: float = 1600.0,
    corners_px: list[list[float]] | None = None,
    safety_margin_mm: float = 5.0,
    auto_apply_relaxation: bool = False,
    case_id: str = "commercial_job_001",
    external_crack_mask: Any = None,
) -> dict[str, Any]:
    """Execute complete commercial workflow from slab photo & Chinese order to DXF & Overlay."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_arr = None
    if external_crack_mask is not None:
        if isinstance(external_crack_mask, (str, Path)):
            import cv2
            import numpy as np

            raw_m = Path(external_crack_mask).read_bytes()
            mask_arr = cv2.imdecode(np.frombuffer(raw_m, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            mask_arr = external_crack_mask

    # 1. Visual perception & texture field extraction
    vis = parse_slab_image(
        image_input=image_input,
        width_mm=width_mm,
        height_mm=height_mm,
        corners_px=corners_px,
        safety_margin_mm=safety_margin_mm,
        external_crack_mask=mask_arr,
    )

    # 2. Natural language order compilation
    order_state = compile_chinese_order(order_text)
    solver_case = compile_to_solver_case(case_id=case_id, visual_state=vis, order_state=order_state)

    # 3. Aesthetic Multi-Tier or Count-First Defect-Aware Nesting
    if solver_case.get("is_aesthetic_multi_tier", False):
        from ..solver_core.aesthetic_engine import solve_aesthetic_stone_nesting

        nesting_res = solve_aesthetic_stone_nesting(solver_case, rectified_bgr=vis["rectified_bgr"])
    else:
        nesting_res = solve_cf_slabnest(solver_case, time_limit_sec=2.5)
    cert = validate_nesting_solution(solver_case, nesting_res)

    # 4. Solver-in-the-Loop Feedback (skip relaxation when open-ended remnant mosaic pool is used)
    has_open_remnant = any(it.get("fill_remnants", False) for it in solver_case.get("items", []))
    feedback = diagnose_and_suggest_relaxations(solver_case, nesting_res)
    if auto_apply_relaxation and not has_open_remnant and not feedback["is_fully_placed"] and feedback.get("verified_relaxations"):
        rel_case, rel_res = apply_relaxation_plan(feedback, 0)
        if rel_case is not None and rel_res is not None:
            solver_case = rel_case
            nesting_res = rel_res
            cert = validate_nesting_solution(solver_case, nesting_res)

    # 5. Export Overlay PNG, Customer Dry-Lay Preview PNG, Cut-Piece Vein Gallery PNG, Vector SVG, and Multi-Layer DXF
    overlay_png = out_dir / f"{case_id}_overlay.png"
    dry_lay_png = out_dir / f"{case_id}_dry_lay_preview.png"
    gallery_png = out_dir / f"{case_id}_vein_gallery.png"
    overlay_svg = out_dir / f"{case_id}_layout.svg"
    dxf_file = out_dir / f"{case_id}_cutting.dxf"
    report_json = out_dir / f"{case_id}_report.json"

    render_overlay_on_slab_image(vis["rectified_bgr"], solver_case, nesting_res, overlay_png)
    render_dry_lay_preview(vis["rectified_bgr"], solver_case, nesting_res, dry_lay_png)
    render_cut_piece_gallery(vis["rectified_bgr"], solver_case, nesting_res, gallery_png)
    render_overlay_svg(solver_case, nesting_res, overlay_svg)
    dxf_info = export_industrial_dxf(solver_case, nesting_res, dxf_file)
    quote = calculate_commercial_quote(solver_case, nesting_res, dxf_info)

    summary = {
        "case_id": case_id,
        "order_text": order_text,
        "compiled_order": order_state,
        "dominant_vein_angle_deg": vis["texture_field"]["dominant_vein_angle_deg"],
        "detected_defects_count": len(vis["defects"]),
        "nesting_result": nesting_res,
        "certificate": cert,
        "solver_feedback": {
            "is_fully_placed": feedback["is_fully_placed"],
            "unplaced_count": feedback["unplaced_count"],
            "diagnosis_summary_cn": feedback["diagnosis_summary_cn"],
            "verified_relaxation_count": len(feedback.get("verified_relaxations", [])),
        },
        "dxf_export": dxf_info,
        "commercial_quote": quote,
        "artifacts": {
            "overlay_png": str(overlay_png),
            "dry_lay_preview_png": str(dry_lay_png),
            "vein_gallery_png": str(gallery_png),
            "layout_svg": str(overlay_svg),
            "cutting_dxf": str(dxf_file),
            "report_json": str(report_json),
        },
    }
    report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def create_fastapi_app() -> FastAPI:
    """Create the commercial web application server for Yunfu stone merchants."""
    app = FastAPI(title="StoneVL-Nest 云浮石材智能避障排板系统", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>StoneVL-Nest 云浮石材智能避障排板与报价工业系统 v2.0</title>
<style>
  body { font-family: "Microsoft YaHei", sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }
  .header { background: linear-gradient(90deg, #1e3a8a, #0f766e); padding: 20px 28px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #38bdf8; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 20px; }
  .badge { display: inline-block; padding: 4px 10px; border-radius: 6px; background: #0284c7; font-size: 13px; margin-right: 8px; }
  textarea { width: 100%; height: 72px; background: #090d16; color: #f8fafc; border: 1px solid #475569; border-radius: 8px; padding: 10px; font-size: 14px; }
  button { background: #16a34a; color: #fff; border: none; padding: 10px 18px; border-radius: 8px; font-weight: bold; cursor: pointer; margin-top: 10px; }
  pre { background: #090d16; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 12.5px; color: #4ade80; border: 1px solid #334155; }
</style>
</head>
<body>
  <div class="header">
    <h1>StoneVL-Nest 云浮石材智能视觉-语言避障排板系统 (商业生产版 v2.0)</h1>
    <p><span class="badge">四点透视毫米标定</span><span class="badge">SAM-2.1+形态学裂纹避障</span><span class="badge">结构张量顺纹耦合</span><span class="badge">件数优先+水刀盆孔套切(PiP)</span><span class="badge">7层AutoCAD R2010 DXF直出</span></p>
  </div>
  <div class="grid">
    <div class="card">
      <h3>1. 口语订单 AST 编译与石板视觉输入</h3>
      <textarea id="orderInput">切1块1100x600的带水槽台面要顺纹，再来3个直径480的圆几，3块220x220的正方形方垫，避开中间裂纹，锯缝留3mm，水刀模式</textarea>
      <button onclick="compileOrder()">实时编译口语订单 (AST Schema)</button>
      <pre id="astOut">点击按钮查看实时编译的结构化几何约束 JSON...</pre>
    </div>
    <div class="card">
      <h3>2. 独立几何验收契约 (RAC-Checker) 与商业报价输出</h3>
      <p>内置独立重构式几何核验器（100% 零重叠、零越界、零踩裂纹），一键生成原图花纹透视 Overlay 预览、切件花纹画廊、人民币成本/毛利核算表与 AutoCAD R2010 标准 7 层 DXF 加工刀路文件。</p>
    </div>
  </div>
  <script>
    async function compileOrder() {
      const text = document.getElementById('orderInput').value;
      const res = await fetch('/api/compile_order', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({order_text: text})
      });
      const data = await res.json();
      document.getElementById('astOut').textContent = JSON.stringify(data, null, 2);
    }
  </script>
</body>
</html>"""

    @app.post("/api/compile_order")
    def api_compile(payload: dict[str, Any]) -> JSONResponse:
        text = str(payload.get("order_text", ""))
        compiled = compile_chinese_order(text)
        return JSONResponse(compiled)

    return app
