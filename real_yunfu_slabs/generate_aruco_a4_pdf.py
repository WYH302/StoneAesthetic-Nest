"""Generate printable A4 ArUco 4-Corner Calibration PDF for Yunfu Stone Workshops.

Creates `ArUco_4Corners_A4_Print.pdf` with:
  - Page 1: Concise visual placement guide (ArUco mode + No-Printer fallback mode)
  - Pages 2-5: High-contrast 150x150 mm DICT_4X4_50 markers (ID 0=TL, ID 1=TR, ID 2=BR, ID 3=BL)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
PDF_PATH = OUT_DIR / "ArUco_4Corners_A4_Print.pdf"

# Configure Chinese-compatible font fallback on Windows
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

A4_W_IN = 210.0 / 25.4
A4_H_IN = 297.0 / 25.4


def _make_aruco_marker(marker_id: int, side_px: int = 800) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_px)


def build_pdf(pdf_path: Path = PDF_PATH) -> Path:
    corners_meta = [
        (0, "左上角 (TOP-LEFT)", "TL", "把本页【左上白边】贴齐大板左上外角", (0.06, 0.94)),
        (1, "右上角 (TOP-RIGHT)", "TR", "把本页【右上白边】贴齐大板右上外角", (0.94, 0.94)),
        (2, "右下角 (BOTTOM-RIGHT)", "BR", "把本页【右下白边】贴齐大板右下外角", (0.94, 0.06)),
        (3, "左下角 (BOTTOM-LEFT)", "BL", "把本页【左下白边】贴齐大板左下外角", (0.06, 0.06)),
    ]

    with PdfPages(pdf_path) as pdf:
        # ---- Page 1: Quick Workshop Guide ----
        fig = plt.figure(figsize=(A4_W_IN, A4_H_IN), dpi=300)
        ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 145)
        ax.axis("off")

        ax.text(
            50,
            139,
            "云浮石材大板实拍标定指南 (StoneAesthetic-Nest)",
            ha="center",
            va="top",
            fontsize=17,
            fontweight="bold",
            color="#0f172a",
        )
        ax.text(
            50,
            132.5,
            "ArUco DICT_4X4_50 Four-Corner Perspective Calibration Kit (IDs 0, 1, 2, 3)",
            ha="center",
            va="top",
            fontsize=10.5,
            color="#334155",
        )

        # Schematic diagram of slab with 4 ArUco markers at corners
        slab_rect = patches.FancyBboxPatch(
            (12, 74),
            76,
            48,
            boxstyle="round,pad=0.3",
            facecolor="#e2e8f0",
            edgecolor="#1e293b",
            linewidth=2.0,
        )
        ax.add_patch(slab_rect)
        # Subtle diagonal marble vein lines
        for offset in (-18, -6, 6, 18, 30):
            xs = np.linspace(14, 86, 60)
            ys = np.clip(98 + 0.35 * (xs - 50) + offset + 2.2 * np.sin(xs * 0.18), 76, 120)
            ax.plot(xs, ys, color="#94a3b8", linewidth=1.2, alpha=0.65)
        ax.text(
            50,
            98,
            "天然大理石 / 奢石大板\n(例如 280 x 160 cm 或 320 x 190 cm)\n拍全四个外角即可，允许手机轻微斜拍",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#64748b", alpha=0.92),
        )

        marker_boxes = [
            (13.5, 111.5, "ID 0\n左上 TL"),
            (76.5, 111.5, "ID 1\n右上 TR"),
            (76.5, 75.5, "ID 2\n右下 BR"),
            (13.5, 75.5, "ID 3\n左下 BL"),
        ]
        for bx, by, label in marker_boxes:
            m_rect = patches.Rectangle((bx, by), 10, 9, facecolor="#0f172a", edgecolor="#ef4444", linewidth=2.0)
            ax.add_patch(m_rect)
            ax.text(bx + 5, by + 4.5, label, ha="center", va="center", fontsize=8.2, fontweight="bold", color="white")

        # Instructions box
        steps_text = (
            "【方式 A：打印第 2~5 页标定纸（亚毫米精度，推荐）】\n"
            "  1. 用普通 A4 黑白打印机打印本 PDF 的第 2~5 页（无需彩印、无需过塑）。\n"
            "  2. 将 4 张纸分别贴/压在大板四个外角（ID 0 左上、ID 1 右上、ID 2 右下、ID 3 左下）。\n"
            "  3. 用手机正对或微斜拍摄整块大板，确保 4 张二维码纸都在画面内，关闭美颜滤镜。\n"
            "  4. 微信发送【原图】并备注大板实际尺寸（例如：鱼肚白 310x185 厘米）。\n\n"
            "【方式 B：没打印机时的零门槛快拍（同样支持！）】\n"
            "  1. 直接拿手机站在大板正前方/龙门架下，拍全整块大板的 4 个边角。\n"
            "  2. 尽量避免强反光遮挡主纹路，确保大板边缘与地面/铁架有清晰边界。\n"
            "  3. 微信发【原图】+ 报一下卷尺量出的长×宽（厘米）即可。\n\n"
            "【选板建议（只需 3~5 块不同品种，论文直接升档！）】\n"
            "  • 建议挑 3~5 块不同纹路：如 鱼肚白/卡拉拉白、黑白根/山水画、青玉/翡翠绿、\n"
            "    黄洞石/罗马洞石、带 1~2 条天然裂纹或色斑的大板（最能体现本算法避障+对纹实力！）。"
        )
        ax.text(
            6,
            66,
            steps_text,
            ha="left",
            va="top",
            fontsize=10.2,
            linespacing=1.55,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.2),
        )
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        # ---- Pages 2-5: 4 Full-Size A4 Markers (150x150 mm) ----
        for mid, cn_title, short_code, align_tip, _ in corners_meta:
            fig = plt.figure(figsize=(A4_W_IN, A4_H_IN), dpi=300)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.set_xlim(0, 210.0)
            ax.set_ylim(0, 297.0)
            ax.axis("off")

            # Outer dashed cutting/alignment border (10 mm margin)
            border = patches.Rectangle(
                (10, 10),
                190,
                277,
                fill=False,
                edgecolor="#94a3b8",
                linewidth=1.0,
                linestyle="--",
            )
            ax.add_patch(border)

            # Red L-corner indicator matching the physical corner of the slab
            if short_code == "TL":
                ax.plot([10, 42], [287, 287], color="#dc2626", linewidth=4.5)
                ax.plot([10, 10], [287, 255], color="#dc2626", linewidth=4.5)
            elif short_code == "TR":
                ax.plot([168, 200], [287, 287], color="#dc2626", linewidth=4.5)
                ax.plot([200, 200], [287, 255], color="#dc2626", linewidth=4.5)
            elif short_code == "BR":
                ax.plot([168, 200], [10, 10], color="#dc2626", linewidth=4.5)
                ax.plot([200, 200], [10, 42], color="#dc2626", linewidth=4.5)
            else:  # BL
                ax.plot([10, 42], [10, 10], color="#dc2626", linewidth=4.5)
                ax.plot([10, 10], [10, 42], color="#dc2626", linewidth=4.5)

            # Header
            ax.text(
                105,
                268,
                f"ArUco ID = {mid}   |   {cn_title}",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
                color="#0f172a",
            )
            ax.text(
                105,
                253,
                "Dictionary: DICT_4X4_50   |   Marker Size: 150 x 150 mm",
                ha="center",
                va="center",
                fontsize=11,
                color="#475569",
            )

            # High-contrast 150x150 mm ArUco marker centered at (105, 152) -> [30..180, 77..227]
            marker_img = _make_aruco_marker(mid, side_px=800)
            ax.imshow(marker_img, cmap="gray", vmin=0, vmax=255, extent=[30, 180, 77, 227], interpolation="nearest")

            # Subtle crosshair at marker center
            ax.plot([101, 109], [152, 152], color="#ef4444", linewidth=0.8, alpha=0.6)
            ax.plot([105, 105], [148, 156], color="#ef4444", linewidth=0.8, alpha=0.6)

            # Footer tip
            ax.text(
                105,
                48,
                align_tip,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="#dc2626",
            )
            ax.text(
                105,
                32,
                "StoneAesthetic-Nest Workshop Calibration Sheet  |  Keep white border clean & unoccluded",
                ha="center",
                va="center",
                fontsize=9.5,
                color="#64748b",
            )

            pdf.savefig(fig, dpi=300)
            plt.close(fig)

    return pdf_path


if __name__ == "__main__":
    out = build_pdf()
    print(f"[OK] Generated printable A4 ArUco kit -> {out} ({out.stat().st_size / 1024:.1f} KB)")
