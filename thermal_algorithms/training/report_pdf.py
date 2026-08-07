"""Minimal markdown + LaTeX-math -> PDF renderer, built because this machine
has no LaTeX toolchain at all (no pdflatex/xelatex/tectonic found anywhere
on PATH or in common install locations) and the existing reports/*.tex files
were evidently compiled elsewhere. Rather than depend on installing a
multi-hundred-MB TeX distribution (or a network fetch that may not be
available), this renders math via matplotlib's mathtext (a real, if smaller,
subset of LaTeX math syntax -- covers everything actually used in
reports/algorithm_derivations.md: subscripts/superscripts, \\frac, \\sum,
\\nabla, Greek letters, etc.) to small PNGs, embedded inline into ReportLab
Platypus flowables. Tables and bar charts use ReportLab Table and matplotlib
respectively.

This is a deliberately narrow markdown subset (headers, paragraphs, bullet
lists, inline $...$ and display $$...$$ math, **bold**) -- exactly what
reports/algorithm_derivations.md actually uses -- not a general CommonMark
implementation.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import LETTER  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

_MATH_DPI = 300
_INLINE_FONTSIZE = 11
_DISPLAY_FONTSIZE = 15

# matplotlib's mathtext is a LIMITED subset of LaTeX -- these substitutions
# cover the constructs that show up repeatedly in hand-written derivations
# but aren't part of that subset (equation-numbering \tag, \Bigl/\Bigr-style
# auto-sizing delimiters, the \le/\ge short aliases, \texttt, and a stray
# backslash before a bare * that isn't even valid full LaTeX either).
_LATEX_SANITIZE = [
    (re.compile(r"\\tag\{[^}]*\}"), ""),
    (re.compile(r"\\[Bb]igg?[lr]?\("), "("),
    (re.compile(r"\\[Bb]igg?[lr]?\)"), ")"),
    (re.compile(r"\\[Bb]igg?[lr]?\["), "["),
    (re.compile(r"\\[Bb]igg?[lr]?\]"), "]"),
    (re.compile(r"\\[Bb]igg?[lr]?\\\{"), r"\{"),
    (re.compile(r"\\[Bb]igg?[lr]?\\\}"), r"\}"),
    (re.compile(r"\\le\b"), r"\leq"),
    (re.compile(r"\\ge\b"), r"\geq"),
    (re.compile(r"\\texttt\{"), r"\text{"),
    (re.compile(r"\^\\\*"), "^*"),
    (re.compile(r"\\\*"), "*"),
]


def _sanitize_latex(latex_body: str) -> str:
    for pattern, repl in _LATEX_SANITIZE:
        latex_body = pattern.sub(repl, latex_body)
    return latex_body


def render_math_png(latex_body: str, *, fontsize: float, color: str = "black") -> tuple[bytes, float, float]:
    """Render a mathtext expression (no surrounding $) to a tightly cropped
    PNG. Returns (png_bytes, width_pt, height_pt) so the caller can size the
    <img> tag to match the surrounding text's line height.
    """
    latex_body = _sanitize_latex(latex_body)
    fig = plt.figure()
    try:
        fig.text(0, 0, f"${latex_body}$", fontsize=fontsize, color=color)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_MATH_DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    finally:
        plt.close(fig)
    buf.seek(0)
    from PIL import Image as PILImage

    with PILImage.open(buf) as im:
        w_px, h_px = im.size
    buf.seek(0)
    scale = 72.0 / _MATH_DPI
    return buf.getvalue(), w_px * scale, h_px * scale


class _ImageCache:
    """Renders each distinct math span once, writes it to a temp dir (
    ReportLab's <img> tag needs a file path, not in-memory bytes), and
    reuses the file for repeated spans."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple[str, float], tuple[Path, float, float]] = {}
        self._n = 0

    def get(self, latex_body: str, fontsize: float) -> tuple[Path, float, float]:
        key = (latex_body, fontsize)
        if key in self._cache:
            return self._cache[key]
        png_bytes, w_pt, h_pt = render_math_png(latex_body, fontsize=fontsize)
        self._n += 1
        path = self.out_dir / f"eq_{self._n:04d}.png"
        path.write_bytes(png_bytes)
        self._cache[key] = (path, w_pt, h_pt)
        return path, w_pt, h_pt


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+?)`")
_INLINE_MATH_RE = re.compile(r"\$([^$]+?)\$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*\|?\s*$")


def _is_table_row(line: str) -> bool:
    return "|" in line.strip()


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped.replace("|", "").replace(":", "").replace(" ", "")) <= {"-"} and "-" in stripped


def _split_table_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [cell.strip() for cell in s.split("|")]


def _escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_markup(text: str, images: _ImageCache) -> str:
    """Convert **bold**, `code`, and $inline math$ into ReportLab's
    paragraph-XML mini-markup (escaping everything else first)."""
    out_parts: list[str] = []
    pos = 0
    for m in _INLINE_MATH_RE.finditer(text):
        out_parts.append(text[pos:m.start()])
        out_parts.append(("MATH", m.group(1)))
        pos = m.end()
    out_parts.append(text[pos:])

    rendered: list[str] = []
    for part in out_parts:
        if isinstance(part, tuple):
            _, latex_body = part
            try:
                path, w_pt, h_pt = images.get(latex_body, _INLINE_FONTSIZE)
                rendered.append(f'<img src="{path.as_posix()}" width="{w_pt:.1f}" height="{h_pt:.1f}" valign="-20%"/>')
            except Exception:
                # matplotlib's mathtext is a LIMITED subset of LaTeX -- a few
                # spans in hand-written derivations use syntax it can't parse
                # (e.g. an escaped \* that full LaTeX also doesn't need).
                # Fall back to plain monospace rather than losing the whole
                # report to one bad equation.
                rendered.append(f'<font face="Courier">{_escape_xml(latex_body)}</font>')
        else:
            escaped = _escape_xml(part)
            escaped = _BOLD_RE.sub(r"<b>\1</b>", escaped)
            escaped = _CODE_RE.sub(r'<font face="Courier">\1</font>', escaped)
            rendered.append(escaped)
    return "".join(rendered)


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], spaceBefore=18, spaceAfter=10))
    styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle("H3c", parent=styles["Heading3"], spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle("Bodyc", parent=styles["BodyText"], spaceBefore=4, spaceAfter=8, leading=15))
    styles.add(ParagraphStyle("Bulletc", parent=styles["BodyText"], leading=14))
    return styles


def markdown_to_flowables(md_text: str, images: _ImageCache, styles) -> list:
    flowables: list = []
    lines = md_text.splitlines()
    i = 0
    bullet_buf: list[str] = []
    para_buf: list[str] = []

    def flush_para():
        # Markdown paragraphs are logically one block of text even when
        # hand-wrapped across several source lines -- **bold**/$math$ spans
        # routinely straddle those line breaks, so inline markup MUST run
        # on the whole joined paragraph, not line-by-line (a regex applied
        # per physical line will never find a **/** pair that spans two of
        # them, leaving the asterisks showing up literally in the PDF).
        if para_buf:
            joined = " ".join(para_buf)
            flowables.append(Paragraph(_inline_markup(joined, images), styles["Bodyc"]))
            para_buf.clear()

    def flush_bullets():
        if bullet_buf:
            items = [
                ListItem(Paragraph(_inline_markup(b, images), styles["Bulletc"]))
                for b in bullet_buf
            ]
            flowables.append(ListFlowable(items, bulletType="bullet", leftIndent=18))
            bullet_buf.clear()

    def flush_all():
        flush_para()
        flush_bullets()

    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            flush_all()
            i += 1
            continue

        if (
            _is_table_row(line)
            and i + 1 < len(lines)
            and _is_table_separator(lines[i + 1])
        ):
            flush_all()
            header = _split_table_row(line)
            i += 2  # skip header + separator
            body_rows = []
            while i < len(lines) and _is_table_row(lines[i]) and lines[i].strip():
                body_rows.append(_split_table_row(lines[i]))
                i += 1
            data = [[Paragraph(_inline_markup(c, images), styles["Bulletc"]) for c in header]]
            for row in body_rows:
                row = row + [""] * (len(header) - len(row))
                data.append([Paragraph(_inline_markup(c, images), styles["Bulletc"]) for c in row[:len(header)]])
            table = Table(data, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2f38")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f5")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            flowables.append(Spacer(1, 4))
            flowables.append(table)
            flowables.append(Spacer(1, 8))
            continue

        if line.strip().startswith("```"):
            flush_all()
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code_style = ParagraphStyle(
                "Code", parent=styles["Bodyc"], fontName="Courier", fontSize=8.5,
                leftIndent=10, backColor=colors.HexColor("#f2f3f5"), leading=11,
            )
            escaped = "<br/>".join(_escape_xml(cl) for cl in code_lines)
            flowables.append(Paragraph(escaped, code_style))
            flowables.append(Spacer(1, 6))
            continue

        if line.strip() == "$$" or (line.strip().startswith("$$") and line.strip().endswith("$$") and len(line.strip()) > 4):
            flush_all()
            if line.strip() == "$$":
                body_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() != "$$":
                    body_lines.append(lines[i])
                    i += 1
                i += 1
                body = " ".join(body_lines)
            else:
                body = line.strip()[2:-2]
                i += 1
            try:
                path, w_pt, h_pt = images.get(body, _DISPLAY_FONTSIZE)
                max_w = 6.0 * inch
                if w_pt > max_w:
                    scale = max_w / w_pt
                    w_pt, h_pt = w_pt * scale, h_pt * scale
                img = Image(str(path), width=w_pt, height=h_pt)
                img.hAlign = "CENTER"
                flowables.append(Spacer(1, 6))
                flowables.append(img)
                flowables.append(Spacer(1, 6))
            except Exception:
                # Unrendered (mathtext couldn't parse it even after
                # sanitizing) -- show the source itself, not a stack trace.
                centered = ParagraphStyle("MathFallback", parent=styles["Bodyc"], alignment=1, fontName="Courier")
                flowables.append(Paragraph(_escape_xml(body), centered))
            continue

        if line.startswith("# "):
            flush_all()
            flowables.append(Paragraph(_inline_markup(line[2:], images), styles["H1c"]))
        elif line.startswith("## "):
            flush_all()
            flowables.append(Paragraph(_inline_markup(line[3:], images), styles["H2c"]))
        elif line.startswith("### "):
            flush_all()
            flowables.append(Paragraph(_inline_markup(line[4:], images), styles["H3c"]))
        elif line.strip().startswith(("- ", "* ")):
            flush_para()
            bullet_buf.append(line.strip()[2:])
        else:
            flush_bullets()
            para_buf.append(line.strip())
        i += 1

    flush_all()
    return flowables


# ---------------------------------------------------------------------------
# Results tables and bar charts
# ---------------------------------------------------------------------------

METRIC_COLUMNS = ["variant", "accuracy", "precision", "recall", "f1", "mean_iou", "mean_inference_ms", "p95_inference_ms"]


def results_table(rows: list[dict], styles, title: Optional[str] = None) -> list:
    flowables = []
    if title:
        flowables.append(Paragraph(title, styles["H3c"]))

    header = ["Variant", "Acc", "Prec", "Rec", "F1", "IoU", "Mean ms", "p95 ms"]
    data = [header]
    for r in rows:
        def fmt_pct(v):
            return f"{v:.1%}" if isinstance(v, (int, float)) else "N/A"

        def fmt_ms(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"

        data.append([
            r.get("variant", ""),
            fmt_pct(r.get("accuracy")),
            fmt_pct(r.get("precision")),
            fmt_pct(r.get("recall")),
            fmt_pct(r.get("f1")),
            fmt_pct(r.get("mean_iou")) if r.get("mean_iou") is not None else "N/A",
            fmt_ms(r.get("mean_inference_ms")),
            fmt_ms(r.get("p95_inference_ms")),
        ])

    table = Table(data, hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2f38")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f5")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    flowables.append(table)
    flowables.append(Spacer(1, 10))
    return flowables


def bar_chart_image(
    labels: list[str], values: list[float], *, ylabel: str, title: str, out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3))
    bars = ax.bar(labels, values, color="#3d6fd1")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.2g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
