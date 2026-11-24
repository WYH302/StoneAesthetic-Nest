"""Schema-Constrained Chinese Stone Order Compiler (自然语言几何约束编译器).

Translates colloquial Yunfu stone fabrication orders into strict geometric JSON
schemas, supporting compound clauses, grain-lock scoping, kerf compensation,
sink cutouts, and interactive conversational adjustments.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from ..solver_core.geometry_kernel import bbox, transform_polygon
from ..solver_core.solver import solve_cf_slabnest

CN_NUMS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def parse_cn_int(token: str | None, default: int = 1) -> int:
    if not token:
        return default
    token = token.strip()
    if token.isdigit():
        return int(token)
    if token in CN_NUMS:
        return CN_NUMS[token]
    if token.startswith("十"):
        return 10 + CN_NUMS.get(token[1:], 0)
    if token.endswith("十"):
        return CN_NUMS.get(token[:-1], 1) * 10
    if "十" in token:
        left, right = token.split("十", 1)
        return CN_NUMS.get(left, 1) * 10 + CN_NUMS.get(right, 0)
    return default


def _has_grain_lock(clause: str, global_default: bool = False) -> bool:
    if any(k in clause for k in ("不限纹", "不限方向", "自由旋转", "任意角度", "可旋转", "不用顺纹", "不要顺纹", "不用对纹", "不对纹", "无需顺纹")):
        return False
    if any(k in clause for k in ("顺纹", "对纹", "追纹", "锁纹理", "按纹理", "同纹")):
        return True
    return global_default


def _preprocess_colloquial_chinese(text: str) -> str:
    """Normalize colloquial Yunfu workshop shorthand into canonical metric expressions before AST parsing."""
    s = text.strip()
    # Colloquial dimension idioms: 半米见方 -> 50x50厘米正方形; 一米二 -> 1200mm; 一米五 -> 1500mm
    s = re.sub(r"半米见方", "50x50厘米正方形", s)
    s = re.sub(r"一米([一二三四五六七八九])", lambda m: f"{1000 + CN_NUMS[m.group(1)] * 100}mm", s)
    s = re.sub(r"两米([一二三四五六七八九])", lambda m: f"{2000 + CN_NUMS[m.group(1)] * 100}mm", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*米见方", lambda m: f"{round(float(m.group(1)) * 1000.0):.0f}x{round(float(m.group(1)) * 1000.0):.0f}mm正方形", s)
    s = re.sub(r"(?<![一二三四五六七八九\d])一米(?![一二三四五六七八九\d见])", "1000mm", s)
    s = re.sub(r"(?<![一二三四五六七八九\d])两米(?![一二三四五六七八九\d见])", "2000mm", s)
    s = re.sub(r"(?<![一二三四五六七八九\d])半米(?!见方)", "500mm", s)
    # Spoken Chinese two-digit dimensions like 六十公分 / 八十厘米 -> 60公分 / 80厘米
    s = re.sub(
        r"([一二两三四五六七八九]?十[一二三四五六七八九]?)\s*(公分|厘米|毫米)",
        lambda m: f"{parse_cn_int(m.group(1))}{m.group(2)}",
        s,
    )
    # Normalize "长120宽60厘米" or "1200mm长600mm宽" -> "120x60厘米" / "1200mmx600mm"
    s = re.sub(
        r"长\s*(\d+(?:\.\d+)?)\s*(mm|毫米|cm|厘米|公分|m|米)?\s*宽\s*(\d+(?:\.\d+)?)\s*(mm|毫米|cm|厘米|公分|m|米)?",
        lambda m: f"{m.group(1)}{m.group(2) or ''}x{m.group(3)}{m.group(4) or ''}",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"(\d+(?:\.\d+)?)\s*(mm|毫米|cm|厘米|公分|m|米)?\s*长\s*(\d+(?:\.\d+)?)\s*(mm|毫米|cm|厘米|公分|m|米)?\s*宽",
        lambda m: f"{m.group(1)}{m.group(2) or ''}x{m.group(3)}{m.group(4) or ''}",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"常规水刀(?:留缝)?", "水刀，锯缝留3mm", s)
    return s


def validate_order_schema(order_state: dict[str, Any]) -> tuple[bool, list[str]]:
    """Formally validate a compiled order schema before dispatching to the geometric solver."""
    errors: list[str] = []
    items = order_state.get("items", [])
    if not items:
        errors.append("empty_items")
    for idx, it in enumerate(items):
        qty = int(it.get("quantity", 0))
        if qty < 1 or qty > 400:
            errors.append(f"invalid_quantity_at_{idx}:{qty}")
        for dk in ("width", "height", "size", "radius", "side"):
            if dk in it and (float(it[dk]) < 20.0 or float(it[dk]) > 3500.0):
                errors.append(f"out_of_range_{dk}_at_{idx}:{it[dk]}")
        if it.get("shape") == "hollow_countertop":
            if float(it.get("hole_width", 0.0)) >= float(it.get("width", 1.0)) - 20.0:
                errors.append(f"invalid_hole_width_at_{idx}")
            if float(it.get("hole_height", 0.0)) >= float(it.get("height", 1.0)) - 20.0:
                errors.append(f"invalid_hole_height_at_{idx}")
    c = order_state.get("constraints", {})
    if float(c.get("kerf", 3.0)) < 0.5 or float(c.get("kerf", 3.0)) > 20.0:
        errors.append("invalid_kerf")
    return len(errors) == 0, errors


def _unit_scale_mm(unit_str: str | None) -> float:
    if not unit_str:
        return 1.0
    u = unit_str.strip().lower()
    if u in ("cm", "厘米", "公分"):
        return 10.0
    if u in ("m", "米"):
        return 1000.0
    return 1.0


def _infer_vein_grid(qty: int, clause: str = "") -> list[int]:
    if "四联" in clause or "1x4" in clause:
        return [1, 4]
    if "双联" in clause or "双盆" in clause or "1x2" in clause:
        return [1, 2]
    if "田字" in clause or "2x2" in clause:
        return [2, 2]
    if qty == 6:
        return [2, 3]
    if qty == 4:
        return [2, 2]
    if qty == 9:
        return [3, 3]
    if qty == 8:
        return [2, 4]
    return [1, max(1, qty)]


def compile_chinese_order(text: str, *, dynamic_remnant_cap: bool = False) -> dict[str, Any]:
    """Compile a colloquial Chinese stone order into a validated 3-Tier aesthetic & geometric schema."""
    raw = text.strip()
    pre = _preprocess_colloquial_chinese(raw)
    # Keep modifiers inside parentheses together with their parent item clause
    def _clean_parens(m: re.Match[str]) -> str:
        inner = m.group(1).replace("，", " ").replace(",", " ").replace("、", " ").replace("；", " ")
        return f"({inner})"

    grouped = re.sub(r"[（(]([^）)]*)[）)]", _clean_parens, pre)
    norm = (
        grouped.replace("，", ",")
        .replace("；", ",")
        .replace("。", ",")
        .replace("、", ",")
        .replace("×", "x")
        .replace("＊", "x")
        .replace("*", "x")
        .replace("X", "x")
        .replace("乘以", "x")
        .replace("乘", "x")
        .replace("$", "")
        .replace("\\times", "x")
    )

    # Extract global constraints first
    kerf_mm = 3.0
    m_kerf = re.search(r"(?:锯缝|刀缝|切缝|水刀缝|留缝)\s*(?:留|为|设|改)?\s*(\d+(?:\.\d+)?)\s*(cm|厘米|mm|毫米)?", norm, re.IGNORECASE)
    if m_kerf:
        kerf_mm = float(m_kerf.group(1)) * _unit_scale_mm(m_kerf.group(2))

    margin_mm = 5.0
    m_margin = re.search(r"(?:边距|安全边距|留边)\s*(?:留|为|设|改)?\s*(\d+(?:\.\d+)?)\s*(cm|厘米|公分|mm|毫米)?", norm, re.IGNORECASE)
    if m_margin:
        margin_mm = float(m_margin.group(1)) * _unit_scale_mm(m_margin.group(2))

    negate_waterjet = any(k in norm for k in ("不用水刀", "不要水刀", "不走水刀", "只用桥切", "纯桥切"))
    has_waterjet = (not negate_waterjet) and (
        "水刀" in norm
        or any(
            (k in c and not any(neg in c for neg in ("不要", "别切", "不用", "不需要", "不切", "取消")))
            for c in norm.split(",")
            for k in ("三角形", "马赛克", "六边形", "六角砖")
        )
    )
    cut_mode = "guillotine" if (negate_waterjet or (not has_waterjet and any(k in norm for k in ("桥切", "一刀切", "直切")))) else "waterjet"
    avoid_cracks = not any(k in norm for k in ("忽略裂纹", "不避裂纹"))
    global_grain = any(k in norm for k in ("全部顺纹", "统一顺纹", "整体顺纹", "都要顺纹"))

    clauses = [c.strip() for c in norm.split(",") if c.strip()]
    items: list[dict[str, Any]] = []
    qty_pat = r"(?P<q>\d+|[一二两俩三四五六七八九十]{1,3})\s*(?:个|块|件|张|套|片|条)?"
    pending_tier3_remnant = False

    for idx, clause in enumerate(clauses):
        # Skip explicit item-exclusion / negation clauses (e.g., "不要马赛克", "别切三角形", "不需要余料")
        if any(neg in clause for neg in ("不要", "别切", "不用", "不需要", "不切", "取消")):
            if any(k in clause for k in ("马赛克", "三角形", "六边形", "六角砖", "余料", "边角料", "水刀")):
                pending_tier3_remnant = False
                continue
        # Modifier clauses attaching to the previous item
        if items and not re.search(r"\d", clause) and not any(s in clause for s in ("三角形", "马赛克", "正方形", "长方形", "矩形", "圆", "台面")):
            if any(k in clause for k in ("黄金主纹", "黄金带", "主纹带", "连纹对花", "对花", "主拼花")):
                items[-1]["tier"] = 1
                items[-1]["priority"] = 3
                items[-1]["role"] = "core_vein_match"
                items[-1]["vein_matching"] = True
                items[-1]["golden_zone_required"] = True
                items[-1]["grain_locked"] = True
                items[-1]["grid_layout"] = _infer_vein_grid(int(items[-1]["quantity"]))
                continue
            if any(k in clause for k in ("顺纹连排", "纹路不断", "楼梯踏步", "连排")):
                items[-1]["tier"] = 2
                items[-1]["priority"] = max(2, int(items[-1].get("priority", 2)))
                items[-1]["role"] = "stair_tread_tandem"
                items[-1]["grain_locked"] = True
                items[-1]["allow_rotation"] = False
                items[-1]["tandem_sequence"] = True
                continue
            if any(clause in (k, f"要{k}", f"需{k}") for k in ("顺纹", "对纹", "追纹", "锁纹理")):
                items[-1]["grain_locked"] = True
                continue
            if any(clause in (k, f"可{k}") for k in ("自由旋转", "任意旋转", "不限方向")):
                items[-1]["grain_locked"] = False
                items[-1]["allow_rotation"] = True
                continue

        if any(k in clause for k in ("边角料", "余料套裁", "剩下的空隙", "剩余空隙", "余料")):
            pending_tier3_remnant = True

        grain_flag = _has_grain_lock(clause, global_default=global_grain)

        # 1. Hollow countertop (带水槽/挖孔/中岛台面) - supports mm, cm/厘米/公分, and m/米
        m_hollow = re.search(
            qty_pat
            + r"\s*(?P<w>\d{1,4}(?:\.\d+)?)\s*(?P<uw>mm|毫米|cm|厘米|公分|m|米)?\s*x\s*(?P<h>\d{1,4}(?:\.\d+)?)\s*(?P<uh>mm|毫米|cm|厘米|公分|m|米)?\s*(?:的)?(?:带水槽|挖孔|内挖孔|水槽|中岛挖孔)(?:中岛)?(?:台面|板)?(?:(?:\(|（|\s)*内孔\s*(?P<hw>\d{1,4}(?:\.\d+)?)\s*(?P<uhw>mm|毫米|cm|厘米|公分|m|米)?\s*x\s*(?P<hh>\d{1,4}(?:\.\d+)?)\s*(?P<uhole>mm|毫米|cm|厘米|公分|m|米)?(?:\)|）)?)?",
            clause,
            re.IGNORECASE,
        )
        if m_hollow:
            uw_h = m_hollow.group("uw")
            uh_h = m_hollow.group("uh")
            sc_w = _unit_scale_mm(uw_h or uh_h)
            sc_h = _unit_scale_mm(uh_h or uw_h)
            raw_w, raw_h = float(m_hollow.group("w")), float(m_hollow.group("h"))
            if uw_h is None and uh_h is None and max(raw_w, raw_h) <= 260.0 and any(u in clause for u in ("厘米", "公分", "cm")):
                sc_w = sc_h = 10.0
            w = round(raw_w * sc_w, 1)
            h = round(raw_h * sc_h, 1)
            if m_hollow.group("hw") and m_hollow.group("hh"):
                sc_hw = _unit_scale_mm(m_hollow.group("uhw") or m_hollow.group("uhole"))
                sc_hh = _unit_scale_mm(m_hollow.group("uhole") or m_hollow.group("uhw"))
                if m_hollow.group("uhw") is None and m_hollow.group("uhole") is None and float(m_hollow.group("hw")) <= 150.0 and sc_w == 10.0:
                    sc_hw = sc_hh = 10.0
                hw = round(float(m_hollow.group("hw")) * sc_hw, 1)
                hh = round(float(m_hollow.group("hh")) * sc_hh, 1)
            else:
                hw = round(w * 0.42, 1)
                hh = round(h * 0.54, 1)
            is_core_h = any(k in clause for k in ("核心件", "黄金主纹", "黄金带", "主纹带", "中岛"))
            items.append(
                {
                    "id": f"hollow_countertop_{len(items) + 1}",
                    "shape": "hollow_countertop",
                    "width": w,
                    "height": h,
                    "hole_width": hw,
                    "hole_height": hh,
                    "quantity": parse_cn_int(m_hollow.group("q")),
                    "grain_locked": True if is_core_h else grain_flag,
                    "allow_rotation": not (is_core_h or grain_flag),
                    "priority": 3,
                    "tier": 1 if is_core_h else 1,
                    "role": "core_vein_match" if is_core_h else "standard",
                    "vein_matching": is_core_h,
                    "golden_zone_required": is_core_h,
                    "grid_layout": [1, 1] if is_core_h else None,
                }
            )
            continue

        # 2. L-shape countertop (L型台面/转角吧台/拐角) - supports both mm and cm/厘米
        m_lshape = re.search(
            qty_pat
            + r"\s*(?P<w>\d{2,4}(?:\.\d+)?)\s*(?P<uw>cm|厘米|公分|mm|毫米)?\s*x\s*(?P<h>\d{2,4}(?:\.\d+)?)\s*(?P<uh>cm|厘米|公分|mm|毫米)?\s*(?:的)?(?:L型|L形|转角|拐角)(?:转角)?(?:吧台|台面|板)?",
            clause,
            re.IGNORECASE,
        )
        if m_lshape:
            sc_l = _unit_scale_mm(m_lshape.group("uh") or m_lshape.group("uw"))
            raw_w, raw_h = float(m_lshape.group("w")), float(m_lshape.group("h"))
            if sc_l == 1.0 and max(raw_w, raw_h) <= 250.0 and any(u in clause for u in ("厘米", "公分", "cm")):
                sc_l = 10.0
            w = round(raw_w * sc_l, 1)
            h = round(raw_h * sc_l, 1)
            is_core_l = any(k in clause for k in ("核心件", "黄金主纹", "黄金带", "主纹带", "吧台"))
            items.append(
                {
                    "id": f"l_shape_{len(items) + 1}",
                    "shape": "l_shape",
                    "width": w,
                    "height": h,
                    "leg_width": round(w * 0.45, 1),
                    "leg_height": round(h * 0.45, 1),
                    "quantity": parse_cn_int(m_lshape.group("q")),
                    "grain_locked": True if is_core_l else grain_flag,
                    "allow_rotation": not (is_core_l or grain_flag),
                    "priority": 3 if is_core_l else 2,
                    "tier": 1 if is_core_l else 2,
                    "role": "core_vein_match" if is_core_l else "standard",
                    "vein_matching": is_core_l,
                    "golden_zone_required": is_core_l,
                    "grid_layout": [1, 1] if is_core_l else None,
                }
            )
            continue

        # 3. Ellipse (椭圆)
        m_ellipse = re.search(
            qty_pat + r"\s*(?P<w>\d{3,4})\s*x\s*(?P<h>\d{3,4})\s*(?:的)?椭圆(?:台面|桌|几|板)?",
            clause,
        )
        if m_ellipse:
            w = float(m_ellipse.group("w"))
            h = float(m_ellipse.group("h"))
            items.append(
                {
                    "id": f"ellipse_{len(items) + 1}",
                    "shape": "ellipse",
                    "rx": round(w * 0.5, 1),
                    "ry": round(h * 0.5, 1),
                    "width": w,
                    "height": h,
                    "quantity": parse_cn_int(m_ellipse.group("q")),
                    "grain_locked": grain_flag,
                    "allow_rotation": not grain_flag,
                    "priority": 2,
                }
            )
            continue

        # 4. Circle (圆桌/圆几/圆盘/圆) - supports both 直径 and 半径 in mm, cm/厘米, or m/米
        m_circle = re.search(
            qty_pat + r"\s*(?P<mode>直径|半径|Ø|Φ)?\s*(?P<val>\d{1,4}(?:\.\d+)?)\s*(?P<uc>mm|毫米|cm|厘米|公分|m|米)?(?:的)?(?:圆桌|圆几|圆形茶几|圆茶几|圆台|圆盘|小圆垫|圆形|圆)",
            clause,
            re.IGNORECASE,
        )
        if m_circle:
            sc_c = _unit_scale_mm(m_circle.group("uc"))
            raw_val = float(m_circle.group("val"))
            if sc_c == 1.0 and raw_val <= 150.0 and any(u in clause for u in ("厘米", "公分", "cm")):
                sc_c = 10.0
            val = raw_val * sc_c
            mode_str = m_circle.group("mode") or "半径"
            radius = val * 0.5 if mode_str in ("直径", "Ø", "Φ") else val
            items.append(
                {
                    "id": f"circle_{len(items) + 1}",
                    "shape": "circle",
                    "radius": round(radius, 1),
                    "diameter": round(radius * 2.0, 1),
                    "quantity": parse_cn_int(m_circle.group("q")),
                    "grain_locked": False,
                    "allow_rotation": False,
                    "priority": 2 if radius >= 240 else 1,
                    "tier": 2 if radius >= 240 else 3,
                }
            )
            continue

        # 5. Sector (扇形)
        m_sector = re.search(
            qty_pat + r"\s*(?:半径)?(?P<r>\d{2,4})\s*(?:的)?(?:(?P<a>\d{2,3})度)?扇形(?:角架|板)?",
            clause,
        )
        if m_sector:
            r = float(m_sector.group("r"))
            ang = float(m_sector.group("a") or 90.0)
            items.append(
                {
                    "id": f"sector_{len(items) + 1}",
                    "shape": "sector",
                    "radius": r,
                    "angle": ang,
                    "quantity": parse_cn_int(m_sector.group("q")),
                    "grain_locked": grain_flag,
                    "allow_rotation": True,
                    "priority": 1,
                }
            )
            continue

        # 6. Square (正方形/方几/主拼花/余料小方砖/茶盘垫) - supports cm/厘米/mm, Tier-1 Vein-Match & Tier-3 Remnant Fill
        sq_qty_pat = r"(?P<q>\d+|[一二两俩三四五六七八九十]{1,3})\s*(?:个|块|件|张|套|片|条)"
        sq_nouns = r"(?:正方形|方几|方板|小方砖|防滑小地砖|方砖|小方垫|茶盘垫|杯垫|主拼花|田字格地拼|地拼)"
        m_sq = re.search(
            r"(?:" + sq_qty_pat + r"\s*)?(?P<s>\d{1,4}(?:\.\d+)?)(?:\s*x\s*(?P=s))?\s*(?P<u1>mm|毫米|cm|厘米|公分|m|米)?(?:的)?(?:背景墙|田字格|大堂)?" + sq_nouns,
            clause,
            re.IGNORECASE,
        ) or re.search(
            r"(?:" + sq_qty_pat + r"\s*)?" + sq_nouns + r"\s*(?P<s>\d{1,4}(?:\.\d+)?)\s*(?P<u1>mm|毫米|cm|厘米|公分|m|米)",
            clause,
            re.IGNORECASE,
        )
        if m_sq:
            scale = _unit_scale_mm(m_sq.group("u1"))
            raw_s = float(m_sq.group("s"))
            if scale == 1.0 and 10.0 <= raw_s <= 95.0 and any(u in clause for u in ("厘米", "公分", "cm")):
                scale = 10.0
            s = round(raw_s * scale, 1)
            is_remnant_sq = (
                pending_tier3_remnant
                or any(k in clause for k in ("边角料", "余料", "空隙", "尽量", "一些", "小方砖", "茶盘垫", "杯垫", "防滑小地砖"))
            ) and s <= 260.0
            if is_remnant_sq and (not m_sq.group("q") or any(k in clause for k in ("一些", "尽量"))):
                qty = 34
            else:
                qty = parse_cn_int(m_sq.group("q"), default=1)
            is_core_vein = (not is_remnant_sq) and any(
                k in clause for k in ("核心件", "黄金主纹", "黄金带", "主纹带", "连纹对花", "对花", "主拼花", "背景墙", "田字格", "地拼")
            )
            items.append(
                {
                    "id": f"core_square_{len(items) + 1}" if is_core_vein else f"square_{len(items) + 1}",
                    "shape": "square",
                    "size": s,
                    "width": s,
                    "height": s,
                    "quantity": qty,
                    "grain_locked": True if is_core_vein else grain_flag,
                    "allow_rotation": not is_core_vein,
                    "priority": 3 if is_core_vein else (2 if s >= 450.0 else 1),
                    "tier": 1 if is_core_vein else (2 if s >= 450.0 else 3),
                    "role": "core_vein_match" if is_core_vein else ("remnant_mosaic_fill" if is_remnant_sq else "standard"),
                    "vein_matching": is_core_vein,
                    "golden_zone_required": is_core_vein,
                    "grid_layout": _infer_vein_grid(qty, clause) if is_core_vein else None,
                    "fill_remnants": is_remnant_sq,
                }
            )
            if is_remnant_sq:
                pending_tier3_remnant = False
            continue

        # 7. Rectangle / Four-Panel Screen / Twin Vanity / Stair Tread / Window Sill / Threshold
        m_rect = re.search(
            qty_pat
            + r"\s*(?P<w>\d{1,4}(?:\.\d+)?)\s*(?P<uw>mm|毫米|cm|厘米|公分|m|米)?\s*x\s*(?P<h>\d{1,4}(?:\.\d+)?)\s*(?P<uh>mm|毫米|cm|厘米|公分|m|米)?(?:的)?(?:四联屏|双联洗手台|双盆洗手台|洗手台面|主台面|台面|长方形|矩形|楼梯踏步|踏步板|踏步|茶几板|门槛石|过门石|飘窗板|飘窗台|窗台板|护墙板条|护墙板|踢脚线|侧板|挡水条|背景墙|屏风|板)?",
            clause,
            re.IGNORECASE,
        )
        if m_rect:
            uw_s = m_rect.group("uw")
            uh_s = m_rect.group("uh")
            scale_w = _unit_scale_mm(uw_s or uh_s)
            scale_h = _unit_scale_mm(uh_s or uw_s)
            raw_w = float(m_rect.group("w"))
            raw_h = float(m_rect.group("h"))
            if uw_s is None and uh_s is None and max(raw_w, raw_h) <= 180.0 and any(u in clause for u in ("厘米", "公分", "cm")):
                scale_w = scale_h = 10.0
            w = round(raw_w * scale_w, 1)
            h = round(raw_h * scale_h, 1)
            qty = parse_cn_int(m_rect.group("q"))
            is_core_rect = any(
                k in clause for k in ("核心件", "黄金主纹", "黄金带", "主纹带", "连纹对花", "四联屏", "双联", "双盆")
            )
            is_tandem = (not is_core_rect) and any(
                k in clause for k in ("楼梯踏步", "踏步", "顺纹连排", "纹路不断", "连排", "飘窗", "窗台", "门槛石", "过门石", "护墙", "挡水条")
            )
            if is_tandem or (grain_flag and not is_core_rect):
                w, h = max(w, h), min(w, h)
            is_equal_square = abs(w - h) <= 1.0 and any(
                k in clause for k in ("正方形", "主拼花", "田字格", "地拼", "方板", "方砖", "方几")
            )
            area = w * h
            item_dict: dict[str, Any] = {
                "id": (
                    f"core_square_{len(items) + 1}"
                    if (is_equal_square and is_core_rect)
                    else (
                        f"core_panel_{len(items) + 1}"
                        if is_core_rect
                        else (f"stair_tread_{len(items) + 1}" if is_tandem else f"rectangle_{len(items) + 1}")
                    )
                ),
                "shape": "square" if is_equal_square else "rectangle",
                "width": w,
                "height": h,
                "quantity": qty,
                "grain_locked": True if (is_core_rect or is_tandem) else grain_flag,
                "allow_rotation": False if (is_core_rect or is_tandem or grain_flag) else True,
                "priority": 3 if is_core_rect else (2 if (is_tandem or area >= 150000.0) else 1),
                "tier": 1 if is_core_rect else (2 if (is_tandem or area >= 150000.0) else 3),
                "role": "core_vein_match" if is_core_rect else ("stair_tread_tandem" if is_tandem else "standard"),
                "vein_matching": is_core_rect,
                "golden_zone_required": is_core_rect,
                "grid_layout": _infer_vein_grid(qty, clause) if is_core_rect else None,
                "tandem_sequence": is_tandem,
            }
            if is_equal_square:
                item_dict["size"] = w
            items.append(item_dict)
            continue

        # 8. Regular Hexagon Mosaic (正六边形/六边形/六角砖)
        if any(k in clause for k in ("正六边形", "六边形", "六角砖", "六角拼花")):
            m_q = re.search(r"(?P<q>\d+|[一二两俩三四五六七八九十]{1,3})\s*(?:个|块|件|片)", clause)
            m_rad = re.search(
                r"(?:边长|半径|外接半径|尺寸|规格)\s*(?P<r>\d{1,4}(?:\.\d+)?)\s*(?P<u>cm|厘米|公分|mm|毫米)?",
                clause,
                re.IGNORECASE,
            ) or re.search(
                r"(?P<r>\d{1,4}(?:\.\d+)?)\s*(?P<u>cm|厘米|公分|mm|毫米)(?!\s*(?:个|块|件|片))",
                clause,
                re.IGNORECASE,
            )
            if m_rad and float(m_rad.group("r")) >= 5.0:
                sc_h = _unit_scale_mm(m_rad.group("u"))
                if sc_h == 1.0 and float(m_rad.group("r")) <= 35.0 and any(u in clause for u in ("厘米", "公分", "cm")):
                    sc_h = 10.0
                rad_mm = round(float(m_rad.group("r")) * sc_h, 1)
            else:
                rad_mm = 100.0
            is_rem_hex = pending_tier3_remnant or any(k in clause for k in ("边角料", "余料", "空隙", "尽量", "一些", "拼花", "马赛克"))
            qty_hex = parse_cn_int(m_q.group("q"), default=30) if (m_q and not any(k in clause for k in ("一些", "尽量"))) else 30
            items.append(
                {
                    "id": f"mosaic_hexagon_{len(items) + 1}",
                    "shape": "regular_hexagon",
                    "radius": rad_mm,
                    "side": rad_mm,
                    "width": round(2.0 * rad_mm, 1),
                    "height": round(1.73205 * rad_mm, 1),
                    "quantity": qty_hex,
                    "grain_locked": False,
                    "allow_rotation": True,
                    "rotations": [0.0, 30.0],
                    "priority": 1,
                    "tier": 3,
                    "role": "remnant_mosaic_fill" if is_rem_hex else "standard",
                    "fill_remnants": is_rem_hex,
                }
            )
            pending_tier3_remnant = False
            continue

        # 9. Equilateral Triangle / Remnant Mosaic Tile (等边三角形/马赛克拼花砖 - supports explicit qty or open remnant fill)
        if any(k in clause for k in ("等边三角形", "三角形", "马赛克")):
            m_q = re.search(r"(?P<q>\d+|[一二两俩三四五六七八九十]{1,3})\s*(?:个|块|件|片)", clause)
            m_side = re.search(
                r"(?:边长|尺寸|规格)\s*(?P<s>\d{1,4}(?:\.\d+)?)\s*(?P<u>cm|厘米|公分|mm|毫米)?",
                clause,
                re.IGNORECASE,
            ) or re.search(
                r"(?P<s>\d{1,4}(?:\.\d+)?)\s*(?P<u>cm|厘米|公分|mm|毫米)(?!\s*(?:个|块|件|片))",
                clause,
                re.IGNORECASE,
            )
            if m_side and float(m_side.group("s")) >= 8.0:
                sc = _unit_scale_mm(m_side.group("u"))
                if sc == 1.0 and float(m_side.group("s")) <= 40.0 and any(u in clause for u in ("厘米", "公分", "cm")):
                    sc = 10.0
                side_mm = round(float(m_side.group("s")) * sc, 1)
            else:
                side_mm = 180.0

            is_remnant = pending_tier3_remnant or any(
                k in clause for k in ("边角料", "余料", "空隙", "尽量", "一些", "马赛克")
            )
            qty_tri = parse_cn_int(m_q.group("q"), default=42) if (m_q and not any(k in clause for k in ("一些", "尽量"))) else 42
            items.append(
                {
                    "id": f"mosaic_triangle_{len(items) + 1}",
                    "shape": "equilateral_triangle",
                    "side": side_mm,
                    "width": side_mm,
                    "height": round(side_mm * 0.8660254, 2),
                    "equilateral": True,
                    "quantity": qty_tri,
                    "grain_locked": False,
                    "allow_rotation": True,
                    "rotations": [0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
                    "priority": 1,
                    "tier": 3,
                    "role": "remnant_mosaic_fill" if is_remnant else "standard",
                    "fill_remnants": is_remnant,
                }
            )
            pending_tier3_remnant = False
            continue

    has_multi_tier_aesthetic = any(
        it.get("role") in ("core_vein_match", "stair_tread_tandem", "remnant_mosaic_fill") for it in items
    )
    bookmatch_requested = any(k in norm for k in ("对开", "蝴蝶对花", "双板镜像", "镜像对花", "bookmatch"))
    if dynamic_remnant_cap:
        # Dynamically size open-ended Tier-3 remnant quantity to slab fill capacity instead of fixed magic numbers
        for it in items:
            if it.get("fill_remnants", False):
                est_area = max(4000.0, float(it.get("width", 180.0)) * float(it.get("height", 160.0)))
                it["quantity"] = max(int(it.get("quantity", 30)), min(180, int(2_400_000.0 // est_area)))
                it["fill_to_capacity"] = True

    warnings: list[str] = []
    if not items:
        warnings.append("未能从输入语句中解析出有效石材零件，请检查尺寸与形状描述。")

    out_schema = {
        "raw_text": raw,
        "items": items,
        "objective": "aesthetic_multi_tier_lexicographic" if has_multi_tier_aesthetic else "count_first_lexicographic",
        "is_aesthetic_multi_tier": has_multi_tier_aesthetic,
        "bookmatch_requested": bookmatch_requested,
        "constraints": {
            "avoid_cracks": avoid_cracks,
            "kerf": kerf_mm,
            "min_board_distance": margin_mm,
            "rotation_step": 60.0 if any(it.get("shape") == "equilateral_triangle" for it in items) else 90.0,
            "cut_mode": cut_mode,
        },
        "compiler_version": "StoneVL-AestheticCompiler-v3",
        "warnings": warnings,
    }
    ok_schema, schema_errs = validate_order_schema(out_schema)
    out_schema["schema_valid"] = ok_schema
    out_schema["schema_errors"] = schema_errs
    return out_schema


def compile_naive_baseline(text: str) -> dict[str, Any]:
    """Unconstrained naive regex/prompt baseline for experimental comparison.

    Lacks diameter-to-radius conversion, hollow/L-shape recognition, and local
    clause grain-lock scoping.
    """
    norm = text.replace("×", "x").replace("*", "x").replace("X", "x")
    items: list[dict[str, Any]] = []
    for m in re.finditer(r"(\d+)\s*(?:个|块|件)?\s*(\d{2,4})\s*x\s*(\d{2,4})", norm):
        items.append(
            {
                "id": f"rect_{len(items) + 1}",
                "shape": "rectangle",
                "width": float(m.group(2)),
                "height": float(m.group(3)),
                "quantity": int(m.group(1)),
                "grain_locked": False,
                "allow_rotation": True,
            }
        )
    for m in re.finditer(r"(\d+)\s*(?:个|块|件)?\s*(?:半径|直径)?(\d{2,4})(?:的)?圆", norm):
        items.append(
            {
                "id": f"circle_{len(items) + 1}",
                "shape": "circle",
                "radius": float(m.group(2)),
                "quantity": int(m.group(1)),
                "grain_locked": False,
                "allow_rotation": False,
            }
        )
    return {
        "raw_text": text,
        "items": items,
        "objective": "maximize_count",
        "constraints": {
            "avoid_cracks": "裂纹" in norm,
            "kerf": 5.0,
            "min_board_distance": 10.0,
            "rotation_step": 90.0,
            "cut_mode": "waterjet",
        },
    }


def compile_to_solver_case(
    case_id: str,
    visual_state: dict[str, Any],
    order_state: dict[str, Any],
) -> dict[str, Any]:
    """Fuse visual slab perception state and compiled language order into a solver case."""
    board_info = visual_state["board"]
    width = float(board_info["width_mm"])
    height = float(board_info["height_mm"])
    vein_deg = float(visual_state.get("texture_field", {}).get("dominant_vein_angle_deg", 0.0))

    obstacles = []
    if order_state.get("constraints", {}).get("avoid_cracks", True):
        for idx, d in enumerate(visual_state.get("defects", []), start=1):
            obstacles.append(
                {
                    "id": d.get("id", f"defect_{idx}"),
                    "type": d.get("type", "crack"),
                    "polygon": d["polygon_mm"],
                }
            )

    c_spec = order_state.get("constraints", {})
    return {
        "case_id": case_id,
        "unit": "mm",
        "board": {
            "width": width,
            "height": height,
            "vein_angle_deg": vein_deg,
            "obstacles": obstacles,
        },
        "texture_field": visual_state.get("texture_field"),
        "constraints": {
            "kerf": float(c_spec.get("kerf", 3.0)),
            "min_item_distance": 0.0,
            "min_board_distance": float(c_spec.get("min_board_distance", 5.0)),
            "rotation_step": float(c_spec.get("rotation_step", 90.0)),
            "cut_mode": str(c_spec.get("cut_mode", "waterjet")),
            "grid_step": max(35.0, min(width, height) / 22.0),
        },
        "items": deepcopy(order_state.get("items", [])),
        "is_aesthetic_multi_tier": bool(order_state.get("is_aesthetic_multi_tier", False)),
        "objective": (
            "aesthetic_multi_tier_lexicographic"
            if order_state.get("is_aesthetic_multi_tier", False)
            else "lexicographic:max_count,max_area,min_texture_cost"
        ),
    }


def apply_interactive_command(
    case: dict[str, Any],
    current_result: dict[str, Any],
    command_text: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Execute an interactive natural language refinement command from the operator."""
    updated_case = deepcopy(case)
    updated_res = deepcopy(current_result)
    cmd = command_text.strip()

    # Pattern 1: Move specific part (#k) by (dx, dy)
    m_move = re.search(
        r"把?\s*第?\s*(\d+)\s*号?(?:大件|小件|台面|圆桌|零件|件)?\s*往?(左|右|上|下)\s*(?:移|挪|偏)?\s*(\d+(?:\.\d+)?)",
        cmd,
    )
    if m_move and updated_res.get("placements"):
        idx = int(m_move.group(1)) - 1
        direction = m_move.group(2)
        dist = float(m_move.group(3))
        if 0 <= idx < len(updated_res["placements"]):
            p = updated_res["placements"][idx]
            dx = -dist if direction == "左" else (dist if direction == "右" else 0.0)
            dy = -dist if direction == "下" else (dist if direction == "上" else 0.0)
            w, h = float(updated_case["board"]["width"]), float(updated_case["board"]["height"])
            margin = float(updated_case["constraints"].get("min_board_distance", 5.0))
            pts = [(float(pt[0]) + dx, float(pt[1]) + dy) for pt in p["polygon"]]
            bx1, by1, bx2, by2 = bbox(pts)
            if bx1 >= margin and by1 >= margin and bx2 <= w - margin and by2 <= h - margin:
                p["x"] = round(float(p["x"]) + dx, 3)
                p["y"] = round(float(p["y"]) + dy, 3)
                p["polygon"] = [[round(pt[0], 3), round(pt[1], 3)] for pt in pts]
                if p.get("hole_polygon"):
                    p["hole_polygon"] = [
                        [round(float(hp[0]) + dx, 3), round(float(hp[1]) + dy, 3)]
                        for hp in p["hole_polygon"]
                    ]
                return updated_case, updated_res, f"已将第 {idx + 1} 号零件向{direction}平移 {dist:.1f} mm 并通过边界复核。"

    # Pattern 2: Modify kerf width
    m_kerf = re.search(r"(?:锯缝|刀缝|切缝)\s*(?:改为|改成|设为|调为|调成)\s*(\d+(?:\.\d+)?)", cmd)
    if m_kerf:
        new_kerf = float(m_kerf.group(1))
        updated_case["constraints"]["kerf"] = new_kerf
        new_res = solve_cf_slabnest(updated_case, time_limit_sec=2.5)
        return updated_case, new_res, f"已将刀缝宽度调整为 {new_kerf:.1f} mm 并完成全局重排（排入 {new_res['placed_count']}/{new_res['total_count']} 件）。"

    # Pattern 3: Unlock grain rotation or add angular tolerance
    if any(k in cmd for k in ("允许旋转", "自由旋转", "解除顺纹", "微调角度", "放宽角度")):
        updated_case["constraints"]["grain_tolerance_deg"] = 5.0
        for it in updated_case["items"]:
            if "自由旋转" in cmd or "解除顺纹" in cmd:
                it["grain_locked"] = False
                it["allow_rotation"] = True
        new_res = solve_cf_slabnest(updated_case, time_limit_sec=2.5)
        return updated_case, new_res, f"已放宽纹理旋转角度约束并重新求解（排入 {new_res['placed_count']}/{new_res['total_count']} 件）。"

    # Default: re-run solver with higher LNS budget
    new_res = solve_cf_slabnest(updated_case, time_limit_sec=3.0, seed=99)
    return updated_case, new_res, f"已根据指令『{cmd}』执行大邻域重排优化（当前排入 {new_res['placed_count']}/{new_res['total_count']} 件）。"
