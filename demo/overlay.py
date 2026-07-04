"""Rendering for the demo UI — thermal panels with detection overlays.

Dark neutral theme (slate / teal / amber / red — deliberately no purple).
All rendering produces PIL Images; tkinter just displays them.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---- palette ---------------------------------------------------------------
BG = "#0e1116"
PANEL = "#151b22"
PANEL_EDGE = "#232b35"
TEXT = "#e8eef4"
MUTED = "#77848f"
TEAL = "#2dd4bf"        # person boxes
AMBER = "#fbbf24"       # fire
RED = "#ef4444"         # contact alarm
GREEN = "#34d399"       # all-clear

_TEAL_BGR = (191, 212, 45)
_AMBER_BGR = (36, 191, 251)


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---- thermal panel ----------------------------------------------------------

def render_panel(raw: np.ndarray | None,
                 boxes: list,
                 fire: bool,
                 fire_bbox,
                 scale: int = 6) -> Image.Image:
    """One camera feed: inferno colormap + person/fire boxes."""
    if raw is None:
        img = np.full((62 * scale, 80 * scale, 3), 22, np.uint8)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        d.text((img.shape[1] // 2 - 40, img.shape[0] // 2 - 8),
               "waiting for frames", font=_font(14), fill=MUTED)
        return pil

    lo, hi = np.percentile(raw, (2.0, 99.0))
    u8 = np.clip((raw - lo) / max(hi - lo, 1e-3) * 255, 0, 255).astype(np.uint8)
    # light edge-preserving smoothing for display only (detectors see raw)
    u8 = cv2.bilateralFilter(u8, d=5, sigmaColor=40, sigmaSpace=3)
    img = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    img = cv2.resize(img, (80 * scale, 62 * scale),
                     interpolation=cv2.INTER_CUBIC)

    for (x, y, w, h), score in boxes:
        p1 = (int(x * scale), int(y * scale))
        p2 = (int((x + w) * scale), int((y + h) * scale))
        cv2.rectangle(img, p1, p2, _TEAL_BGR, 2)
        label = f"person {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(p1[1] - 6, th + 4)
        cv2.rectangle(img, (p1[0], ty - th - 4), (p1[0] + tw + 8, ty + 4),
                      _TEAL_BGR, -1)
        cv2.putText(img, label, (p1[0] + 4, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (16, 20, 24), 1, cv2.LINE_AA)

    if fire and fire_bbox is not None:
        x, y, w, h = fire_bbox
        p1 = (int(x * scale), int(y * scale))
        p2 = (int((x + w) * scale), int((y + h) * scale))
        cv2.rectangle(img, p1, p2, _AMBER_BGR, 3)
        label = "FIRE"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (p1[0], p2[1] + 2), (p1[0] + tw + 10, p2[1] + th + 12),
                      _AMBER_BGR, -1)
        cv2.putText(img, label, (p1[0] + 5, p2[1] + th + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (16, 20, 24), 2, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ---- contact strip -----------------------------------------------------------

def render_contact_strip(history: list, conf, alarmed: bool, threshold: float,
                         width: int, height: int = 96) -> Image.Image:
    """Bottom strip: confidence sparkline + threshold line + status pill."""
    pil = Image.new("RGB", (width, height), _hex_rgb(PANEL))
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, width - 1, height - 1], outline=_hex_rgb(PANEL_EDGE))

    pill_w = 240
    chart_l, chart_r = 16, width - pill_w - 24
    chart_t, chart_b = 14, height - 16
    ch = chart_b - chart_t

    # threshold line
    ty = chart_b - int(threshold * ch)
    for x in range(chart_l, chart_r, 10):
        d.line([(x, ty), (x + 5, ty)], fill=_hex_rgb(MUTED), width=1)
    d.text((chart_l, ty - 14), f"alarm threshold {threshold:.2f}",
           font=_font(10), fill=MUTED)

    # sparkline (most recent right)
    if history:
        n = len(history)
        span = chart_r - chart_l
        pts = []
        for i, c in enumerate(history):
            x = chart_r - (n - 1 - i) * max(1, span // 239)
            if x < chart_l:
                continue
            pts.append((x, chart_b - int(min(max(c, 0.0), 1.0) * ch)))
        if len(pts) >= 2:
            color = RED if alarmed else TEAL
            # area fill
            poly = pts + [(pts[-1][0], chart_b), (pts[0][0], chart_b)]
            base = _hex_rgb(color)
            d.polygon(poly, fill=tuple(int(v * 0.25 + 14) for v in base))
            d.line(pts, fill=color, width=2)

    # status pill
    px1, px2 = width - pill_w - 8, width - 16
    py1, py2 = 18, height - 18
    if conf is None:
        pill, label, sub = PANEL_EDGE, "WARMING UP", "collecting frames"
    elif alarmed:
        pill, label, sub = RED, "CONTACT DETECTED", f"confidence {conf:.2f}"
    else:
        pill, label, sub = "#1d2733", "NO CONTACT", f"confidence {conf:.2f}"
    d.rounded_rectangle([px1, py1, px2, py2], radius=10, fill=_hex_rgb(pill))
    lab_col = TEXT if alarmed or conf is None else GREEN
    d.text((px1 + 18, py1 + 8), label, font=_font(16, bold=True), fill=lab_col)
    d.text((px1 + 18, py1 + 32), sub, font=_font(11), fill=(
        TEXT if alarmed else MUTED))
    return pil
