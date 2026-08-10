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
# 'cm' (Computer Modern) mathtext font, not matplotlib's default DejaVu Sans
# -- this is what actually makes rendered equations look like a real LaTeX
# paper (CVPR's own template renders in Computer Modern) instead of a web
# stylesheet's math widget.
matplotlib.rcParams["mathtext.fontset"] = "cm"
# Match chart text (axis labels, ticks, titles) to the same serif family as
# the surrounding paper body text -- a sans-serif chart embedded in an
# otherwise Times-set document is exactly the kind of inconsistency that
# breaks the "looks like a real paper" illusion.
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.mathtext as mathtext  # noqa: E402
import matplotlib.font_manager  # noqa: E402
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
_INLINE_FONTSIZE = 9.5
_DISPLAY_FONTSIZE = 13
_INLINE_MAX_HEIGHT_RATIO = 1.75  # covers ~all real inline spans (measured); only the rare
# inline \sum-with-sub/superscript (mathtext stacks it like displaystyle even inline,
# unlike real LaTeX's compact textstyle -- there's no \nolimits support to fix that
# properly) gets mildly scaled down rather than left to overflow the line.

# matplotlib's mathtext is a LIMITED subset of LaTeX -- these substitutions
# cover the constructs that show up repeatedly in hand-written derivations
# but aren't part of that subset (equation-numbering \tag, \Bigl/\Bigr-style
# auto-sizing delimiters, the \le/\ge short aliases, \texttt, and a stray
# backslash before a bare * that isn't even valid full LaTeX either).
_LATEX_SANITIZE = [
    (re.compile(r"\\tag\{[^}]*\}"), lambda m: ""),
    (re.compile(r"\\[Bb]igg?[lr]?\("), lambda m: "("),
    (re.compile(r"\\[Bb]igg?[lr]?\)"), lambda m: ")"),
    (re.compile(r"\\[Bb]igg?[lr]?\["), lambda m: "["),
    (re.compile(r"\\[Bb]igg?[lr]?\]"), lambda m: "]"),
    (re.compile(r"\\[Bb]igg?[lr]?\\\{"), lambda m: "\\{"),
    (re.compile(r"\\[Bb]igg?[lr]?\\\}"), lambda m: "\\}"),
    # No trailing \b: same issue as \tfrac below -- these are routinely
    # followed by a digit (\ge0, \le1), which is a word char too, so \b
    # never matches there. A negative lookahead for another LETTER (not
    # \b's word-char test) avoids the boundary bug while still refusing to
    # match inside \left/\leq-already/etc.
    (re.compile(r"\\le(?![a-zA-Z])"), lambda m: "\\leq"),
    (re.compile(r"\\ge(?![a-zA-Z])"), lambda m: "\\geq"),
    (re.compile(r"\\iff(?![a-zA-Z])"), lambda m: "\\Leftrightarrow"),
    # No \b after these two: they're routinely followed by a digit
    # (\tfrac12), which is a word character too, so \b never matches there
    # and the substitution would silently no-op.
    (re.compile(r"\\tfrac"), lambda m: "\\frac"),
    (re.compile(r"\\dfrac"), lambda m: "\\frac"),
    # Unlike full LaTeX, mathtext requires \frac{num}{den} braces even for
    # single-token arguments -- \frac12 / \frac1n (real LaTeX's bare-token
    # shorthand) raise "Expected \frac{num}{den}" without them.
    (re.compile(r"\\frac(\w)(\w)\b"), lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"),
    (re.compile(r"\\texttt\{"), lambda m: "\\text{"),
    (re.compile(r"\^\\\*"), lambda m: "^*"),
    (re.compile(r"\\\*"), lambda m: "*"),
    # mathtext has no \underbrace/\overbrace -- drop the brace, keep the
    # base expression, and fold the label into a trailing \text{(...)} so
    # the annotation isn't silently lost, just repositioned.
    (re.compile(r"\\underbrace\{(.+?)\}_\{(.+?)\}"), lambda m: f"{m.group(1)}\\ \\text{{[{m.group(2)}]}}"),
    (re.compile(r"\\overbrace\{(.+?)\}\^\{(.+?)\}"), lambda m: f"{m.group(1)}\\ \\text{{[{m.group(2)}]}}"),
]


def _sanitize_latex(latex_body: str) -> str:
    for pattern, repl in _LATEX_SANITIZE:
        latex_body = pattern.sub(repl, latex_body)
    return latex_body


def render_math_png(
    latex_body: str, *, fontsize: float, color: str = "black",
) -> tuple[bytes, float, float, float]:
    """Render a mathtext expression (no surrounding $) to a tightly cropped
    PNG via matplotlib's own math_to_image (not a manually-cropped
    Figure.savefig -- that hack under-reported real glyph extents and gave
    no baseline info at all). Returns (png_bytes, width_pt, height_pt,
    depth_pt): depth_pt is the descent below the text baseline (e.g. how
    far a fraction's denominator or a subscript hangs below the line the
    surrounding text sits on), which the caller needs to align the image
    against the paragraph's baseline instead of guessing a fixed offset --
    guessing is exactly what caused inline equations to visually collide
    with the line above/below them.
    """
    latex_body = _sanitize_latex(latex_body)
    buf = io.BytesIO()
    depth_px = mathtext.math_to_image(
        f"${latex_body}$", buf, dpi=_MATH_DPI, format="png", color=color,
        prop=matplotlib.font_manager.FontProperties(size=fontsize),
    )
    buf.seek(0)
    from PIL import Image as PILImage

    with PILImage.open(buf) as im:
        w_px, h_px = im.size
    buf.seek(0)
    scale = 72.0 / _MATH_DPI
    return buf.getvalue(), w_px * scale, h_px * scale, depth_px * scale


class _ImageCache:
    """Renders each distinct math span once, writes it to a temp dir (
    ReportLab's <img> tag needs a file path, not in-memory bytes), and
    reuses the file for repeated spans."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple[str, float], tuple[Path, float, float, float]] = {}
        self._n = 0

    def get(self, latex_body: str, fontsize: float) -> tuple[Path, float, float, float]:
        key = (latex_body, fontsize)
        if key in self._cache:
            return self._cache[key]
        png_bytes, w_pt, h_pt, depth_pt = render_math_png(latex_body, fontsize=fontsize)
        self._n += 1
        path = self.out_dir / f"eq_{self._n:04d}.png"
        path.write_bytes(png_bytes)
        self._cache[key] = (path, w_pt, h_pt, depth_pt)
        return path, w_pt, h_pt, depth_pt


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
                path, w_pt, h_pt, depth_pt = images.get(latex_body, _INLINE_FONTSIZE)
                # Cap inline height so a tall nested fraction/subscript can
                # never grow past the paragraph's line spacing and visually
                # collide with the line above/below -- this, not a wrong
                # valign, was the main cause of "math slides into other
                # text": every inline image used to be placed at a fixed
                # -20% offset regardless of its actual size, so anything
                # taller than one text line intruded into its neighbors.
                max_h = _INLINE_MAX_HEIGHT_RATIO * _INLINE_FONTSIZE
                if h_pt > max_h:
                    scale = max_h / h_pt
                    w_pt, h_pt, depth_pt = w_pt * scale, h_pt * scale, depth_pt * scale
                # valign is the offset of the image's BOTTOM edge from the
                # text baseline; a mathtext glyph's bottom edge sits `depth`
                # below its own baseline, so shift down by exactly that much
                # (negative) to line the two baselines up precisely, instead
                # of the old fixed "-20%" guess.
                rendered.append(
                    f'<img src="{path.as_posix()}" width="{w_pt:.1f}" height="{h_pt:.1f}" valign="{-depth_pt:.1f}"/>'
                )
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
    """Times-family serif throughout (CVPR/IEEE-style papers are set in
    Times, not a sans-serif UI font), justified body text, and leading
    generous enough (~1.4x font size) to hold the capped inline-math height
    (_INLINE_MAX_HEIGHT_RATIO * _INLINE_FONTSIZE) without the image
    intruding into the line above/below -- ReportLab's Paragraph leading is
    fixed for the whole paragraph (unlike real LaTeX, which nudges
    line-to-line spacing per line when tall inline content appears), so
    this has to be generous enough for the worst case up front.
    """
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "H1c", fontName="Times-Bold", fontSize=15, leading=18,
        spaceBefore=16, spaceAfter=8, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        "H2c", fontName="Times-Bold", fontSize=12, leading=15,
        spaceBefore=12, spaceAfter=6, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        "H3c", fontName="Times-BoldItalic", fontSize=10.5, leading=13,
        spaceBefore=9, spaceAfter=4, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        "Bodyc", fontName="Times-Roman", fontSize=9.5, leading=17,
        alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "Bulletc", fontName="Times-Roman", fontSize=9.5, leading=17, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        "TitleC", fontName="Times-Bold", fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        "SubtitleC", fontName="Times-Italic", fontSize=13, leading=17, alignment=TA_CENTER, spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        "AbstractC", fontName="Times-Roman", fontSize=9.5, leading=17, alignment=TA_JUSTIFY,
        leftIndent=24, rightIndent=24, spaceBefore=6, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "CaptionC", fontName="Times-Italic", fontSize=8.3, leading=10.5, alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=10,
    ))
    return styles


def markdown_to_flowables(md_text: str, images: _ImageCache, styles, max_width_pt: float = 6.0 * inch) -> list:
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
                path, w_pt, h_pt, _depth_pt = images.get(body, _DISPLAY_FONTSIZE)
                max_w = max_width_pt
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

METRIC_COLUMNS = ["variant", "accuracy", "precision", "recall", "f1", "mean_iou",
                  "mean_inference_ms", "p95_inference_ms", "device"]


_TABLE_FONT = "Times-Roman"
_TABLE_HEADER_FONT = "Times-Bold"


def results_table(
    rows: list[dict], styles, title: Optional[str] = None, *, table_num: Optional[int] = None,
) -> list:
    """Renders a "booktabs"-style table (horizontal rules only, no grid
    lines, no shaded header/striped rows) -- the standard academic-paper
    table convention CVPR/IEEE use, and a deliberate departure from the
    filled-header/striped-row look used elsewhere in this codebase's own
    report scripts, which reads as a web dashboard, not a paper table.
    """
    flowables = []
    caption_text = f"Table{f' {table_num}' if table_num else ''}. {title}" if title else None

    # "Measured on" is not decoration: a latency is uninterpretable without it
    # (8 ms on a laptop GPU and 8 ms on CPU imply completely different things
    # about whether this runs on a Pi), and hand-written prose about which
    # execution provider ran had previously drifted out of sync with the data.
    header = ["Variant", "Acc", "Prec", "Rec", "F1", "IoU", "Mean ms", "p95 ms", "Measured on"]
    data = [header]
    for r in rows:
        def fmt_pct(v):
            return f"{v:.1%}" if isinstance(v, (int, float)) else "N/A"

        def fmt_ms(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"

        # A row may be a real measurement or a deliberate "this variant cannot
        # exist" placeholder (see generate_full_report.py's VARIANT_UNAVAILABLE):
        # the latter carries `unavailable` and renders as N/A across the board
        # with its reason in the caption, rather than being silently omitted.
        dev = r.get("device") or ("--" if r.get("unavailable") else "N/A")
        if r.get("device_source") == "inferred":
            dev += "*"
        data.append([
            r.get("variant", ""),
            fmt_pct(r.get("accuracy")),
            fmt_pct(r.get("precision")),
            fmt_pct(r.get("recall")),
            fmt_pct(r.get("f1")),
            fmt_pct(r.get("mean_iou")) if r.get("mean_iou") is not None else "N/A",
            fmt_ms(r.get("mean_inference_ms")),
            fmt_ms(r.get("p95_inference_ms")),
            dev,
        ])

    n_rows = len(data)
    table = Table(data, hAlign="CENTER", repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), _TABLE_HEADER_FONT),
        ("FONTNAME", (0, 1), (-1, -1), _TABLE_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        # Booktabs rule weights: a heavier line above/below the whole table
        # and under the header row, no other lines at all.
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, n_rows - 1), (-1, n_rows - 1), 1.0, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    flowables.append(table)
    if caption_text:
        flowables.append(Paragraph(caption_text, styles["CaptionC"]))
    else:
        flowables.append(Spacer(1, 10))
    return flowables


def bar_chart_image(
    labels: list[str], values: list[float], *, ylabel: str, title: str, out_path: Path,
    figsize: tuple[float, float] = (3.2, 2.2),
) -> Path:
    """Sized to fit one paper column (default ~3.2in) by default; pass a
    wider figsize for a full-width figure."""
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, values, color="#5b5b5b", edgecolor="black", linewidth=0.6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7, rotation=20)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.2g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=6.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path
