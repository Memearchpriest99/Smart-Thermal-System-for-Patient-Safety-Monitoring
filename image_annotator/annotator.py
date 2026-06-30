"""
YOLO Image Annotator — Thermal Edition (Waveshare)
====================================================
Single-file desktop tool for drawing YOLO bounding boxes across up to 3
simultaneous camera / folder views. Loads a "father" dataset directory
containing ch0_frames/, ch1_frames/, and/or ch2_frames/ subdirectories
(the layout produced by scripts/reorganize_waveshare.py). Class names are
managed via classes.txt in the father directory, with a persistent history
in ~/.annotator_settings.json.

This is a rebuild of the original multi-camera annotator, kept visually
identical, with four additions for the thermal Waveshare dataset:

  1. A single per-frame TOUCH button that labels all three views at once
     (binary contact 0/1), hotkey ``T``.
  2. A recommender (Suggest toggle, hotkey ``S``) that reads the per-channel
     thermal .npz and proposes conservative ghost boxes (fire / person);
     Accept (hotkey ``A``) commits the active channel's ghosts. Tuned for
     precision — misses are fine, false positives are not.
  3. Export Labels: auto-generates labels.xlsx (per-session sheet) AND a
     per-session contact_labels.csv from the drawn boxes + touch values.
  4. Loads the thermal .npz alongside the PNGs to drive the recommender.

Dependencies: Python 3, Tkinter (stdlib), Pillow, NumPy, openpyxl
  pip install Pillow numpy openpyxl

Usage: python annotator.py
"""

import base64
import csv
import json
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

try:
    import openpyxl
except Exception:  # pragma: no cover - only needed for Export Labels
    openpyxl = None

# ── Reuse the project's validated detection pipeline for person suggestions ─
# The annotator lives inside the Final Project repo; add the repo root to the
# path so we can import the §4.4.1 / §4.4.2.1 algorithms. If unavailable
# (e.g. opencv not installed), person suggestions degrade gracefully to off.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
    from thermal_algorithms.core.types import Frame as _ThermalFrame
    from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
    from thermal_algorithms.human_detection.adaptive_threshold import (
        AdaptiveThresholdDetector,
    )
    _HAVE_PIPELINE = True
except Exception as _exc:  # pragma: no cover
    print(f"[annotator] person recommender disabled (import failed): {_exc}")
    _HAVE_PIPELINE = False

# Homography solver is pure NumPy (no cv2), so import it independently of the
# person pipeline — manual calibration should work even if opencv is missing.
try:
    from thermal_algorithms.contact_detection.multi_view.homography import (
        solve_homography_from_markers,
        project_foot_point,
    )
    _HAVE_HOMOGRAPHY = True
except Exception as _exc:  # pragma: no cover
    print(f"[annotator] homography calibration disabled (import failed): {_exc}")
    _HAVE_HOMOGRAPHY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_COLORS = [
    "#ff4444", "#44dd44", "#4488ff", "#ffdd00", "#ff44ff",
    "#00ddff", "#ff8800", "#88ff00", "#0088ff", "#ff0088",
]

MAX_VIEWS     = 3
CHANNEL_DIRS  = ["ch0_frames", "ch1_frames", "ch2_frames"]
NPZ_NAMES     = ["ch0_raw_data.npz", "ch1_raw_data.npz", "ch2_raw_data.npz"]
CHAN_NAMES    = ["Ch 0", "Ch 1", "Ch 2"]
CAM_ACCENT    = ["#e07050", "#50a0e0", "#60d080"]
IMG_EXTS      = (".jpg", ".jpeg", ".png")

# Global settings stored in the user's home directory
SETTINGS_PATH = Path.home() / ".annotator_settings.json"
MAX_CLASS_HISTORY = 10

# ── Display rendering (from the thermal .npz, not the preview PNGs) ─────────
# Default output resolution the raw (62×80) thermal frame is resized to before
# display. The original PNG previews were 320×240, so this preserves the look.
DEFAULT_RESIZE_W   = 320
DEFAULT_RESIZE_H   = 240
DEFAULT_GAUSSIAN_SIGMA = 1.0   # px, on the native thermal grid, when enabled

# Exact matplotlib 'inferno' 256-entry RGB LUT (the colormap the original PNGs
# used), baked in so display needs neither matplotlib nor cv2.
_INFERNO_LUT_B64 = (
    "AAAEAQAFAQEGAQEIAgEKAgIMAgIOAwIQBAMSBAMUBQQXBgQZBwUbCAUdCQYfCgciCwckDAgmDQgp"
    "DgkrEAktEQowEgoyFAs0FQs3Fgs5GAw8GQw+GwxBHAxDHgxFHwxIIQxKIwxMJAxPJgxRKAtTKQtV"
    "KwtXLQtZLwpbMQpcMgpeNApfNglhOAliOQljOwlkPQllPglmQApnQgpoRApoRQppRwtqSQtqSgxr"
    "TAxrTQ1sTw1sUQ5sUg5tVA9tVQ9tVxBuWRBuWhFuXBJuXRJuXxNuYRNuYhRuZBVuZRVuZxZuaRZu"
    "ahdubBhubRhubxlucRluchpudBpudRtudxxteBxteh1tfB1tfR5tfx5sgB9sgiBshCBrhSFrhyFr"
    "iCJqiiJqjCNpjSNpjyRpkCVokiVokyZnlSZnlydmmCdmmihlmylknSlknypjoCpjoitioyxhpSxg"
    "pi1gqC5fqS5eqy9erTBdrjBcsDFbsTJaszJatDNZtjRYtzVXuTVWujZVvDdUvThTvzlSwDpRwTpQ"
    "wztPxDxOxj1Nxz5MyD9LykBKy0FJzEJIzkNHz0RG0EVF0kZE00dD1EhC1UpB10s/2Ew+2U092k48"
    "21A73VE63lI431M34FU24VY14lc041kz5Fox5Vww5l0v514u6GAt6WEr6mMq62Qp62Yo7Gcm7Wkl"
    "7mok72wj724h8G8g8XEf8XMd8nQc83Yb83gZ9HkY9XsX9X0V9n4U9oAT94IS94QQ+IUP+IcO+IkM"
    "+YsL+YwK+Y4J+pAI+pIH+pQH+5YG+5cG+5kG+5sG+50H/J8H/KEI/KMJ/KUK/KYM/KgN/KoP/KwR"
    "/K4S/LAU/LIW/LQY+7Ya+7gd+7of+7wh+74j+sAm+sIo+sQq+sYt+ccv+cky+cs1+M03+M8699E9"
    "99NA9tVD9tdG9dlJ9dtM9N1P9N9T9OFW8+Na8+Vd8uZh8uhl8upp8ext8e1x8e918fF58vJ98vSC"
    "8/WG8/aK9PiO9fmS9vqW+Pua+fyd+v2h/P+k"
)
INFERNO_LUT = np.frombuffer(
    base64.b64decode(_INFERNO_LUT_B64), dtype=np.uint8
).reshape(256, 3)


def _gaussian_blur(frame: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur on a 2-D float array (reflect-padded)."""
    if sigma <= 0:
        return frame
    radius = max(1, int(round(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    pad = np.pad(frame, ((0, 0), (radius, radius)), mode="reflect")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 1, pad)
    pad2 = np.pad(tmp, ((radius, radius), (0, 0)), mode="reflect")
    return np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 0, pad2)


def render_thermal_frame(frame: np.ndarray, out_w: int, out_h: int) -> Image.Image:
    """
    Turn a (H, W) degC thermal frame into a display RGB image.

    Pipeline:  bicubic resize to (out_w, out_h)  ->  per-frame min-max
               normalise  ->  inferno colormap.

    Any pre-processing (Gaussian filter, Tateno residual) is applied to `frame`
    on the native grid by the caller BEFORE this resize. Normalisation and
    colouring match the original PNG previews (per-frame min-max + 'inferno').
    """
    f = np.asarray(frame, dtype=np.float32)
    # Bicubic resize via PIL's 32-bit float mode (no cv2 dependency).
    f = np.asarray(
        Image.fromarray(f, mode="F").resize((out_w, out_h), Image.BICUBIC),
        dtype=np.float32,
    )
    lo, hi = float(f.min()), float(f.max())
    if hi - lo < 1e-6:
        idx = np.zeros(f.shape, dtype=np.uint8)
    else:
        idx = ((f - lo) / (hi - lo) * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(INFERNO_LUT[idx], mode="RGB")

# YOLO class convention (matches thermal_algorithms/training/label_io.py)
FIRE_CLASS_ID   = 0
PERSON_CLASS_ID = 1

# ── Recommender configuration ──────────────────────────────────────────────
# FIRE: cleanly separable by absolute temperature — nothing in the empty/person
# scenes exceeds ~37 degC, while ignition sources run far hotter. A simple
# connected-component threshold gives ~zero false positives. (No background
# model needed.)
T_FIRE_C      = 50.0   # a pixel hotter than this is a fire candidate
FIRE_AREA_MIN = 1      # min connected hot pixels to emit a fire box
# Empirically validated on the Waveshare set: no human/empty frame has even a
# single pixel > 50 degC, so area-1 fire boxes catch cigarettes/lighters with
# zero false positives. Heaters/larger fires produce multi-pixel clusters.
#
# PERSON: reuses the project's §4.4.1 + §4.4.2.1 pipeline — Tateno background
# subtraction (fit on an empty-room recording) followed by the adaptive-
# threshold detector on the residual. Local thresholding + background removal
# is what keeps warm static equipment (a ~33 degC PC screen) from being labelled
# a person. The parameters below were tuned on the Waveshare sessions so that
# empty/calibrate rooms yield 0 detections while every occupied scene detects.
# Splits/merges still happen (one person may yield 2 boxes); these are review-
# able ghost suggestions, not committed labels.
BACKGROUND_SESSIONS = ("empty_room", "calibrate_room")  # sibling dirs, in order
PERSON_DET_KWARGS = dict(
    c_offset=0.5,
    morph_kernel_size=9,
    min_area_pixels=60,
    min_solidity=0.30,
    min_aspect_ratio=0.10,
    max_aspect_ratio=8.0,
    pixel_value_bounds=(1.5, 1000.0),   # mean residual (degC) gate; >= noise floor
)


# ---------------------------------------------------------------------------
# Recommender — fire heuristic (pure NumPy; operates on a (H, W) degC frame)
# ---------------------------------------------------------------------------
def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected component labelling on a boolean mask (small frames)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            pix: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                pix.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(pix)
    return comps


def _bbox_of(pix: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    ys = [p[0] for p in pix]
    xs = [p[1] for p in pix]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1  # x1, y1, x2, y2 (thermal px)


def suggest_fire(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return fire bounding boxes (thermal px) for very hot, sizeable clusters."""
    mask = frame > T_FIRE_C
    out = []
    for pix in _connected_components(mask):
        if len(pix) >= FIRE_AREA_MIN:
            out.append(_bbox_of(pix))
    return out


# ---------------------------------------------------------------------------
# Per-view state container
# ---------------------------------------------------------------------------
class ViewState:
    """
    Encapsulates every piece of mutable state that belongs to ONE camera view.

    Coordinate spaces
    -----------------
      Canvas space        – pixel position on this view's tk.Canvas widget
      Image space         – absolute pixel in the ORIGINAL image file (PNG)
      Thermal space       – pixel in the (H, W) degC .npz frame
      YOLO-normalised     – [0, 1] relative to image dimensions (on disk)
    """

    def __init__(self, slot: int) -> None:
        self.slot: int = slot   # 0 = Ch 0, 1 = Ch 1, 2 = Ch 2

        # ── Tk widget references (assigned in _build_view_frames) ──────────
        self.frame:      tk.Frame     | None = None
        self.header_var: tk.StringVar | None = None
        self.canvas:     tk.Canvas    | None = None

        # ── file list ──────────────────────────────────────────────────────
        self.image_paths:   list[str] = []
        self.current_index: int       = 0

        # ── PIL / Tk image references ──────────────────────────────────────
        self.current_pil:  Image.Image        | None = None
        self.photo_image:  ImageTk.PhotoImage | None = None

        # ── thermal cube for the recommender: shape (N, H, W) degC ─────────
        self.thermal: np.ndarray | None = None

        # ── coordinate mapping (recalculated on every _redraw_view call) ──
        self.scale_factor: float = 1.0
        self.offset_x:     int   = 0
        self.offset_y:     int   = 0
        self.orig_w:       int   = 0
        self.orig_h:       int   = 0

        # ── annotation store ───────────────────────────────────────────────
        # { absolute_image_path : [(class_id, x1, y1, x2, y2), ...] }
        # Coordinates are always in ORIGINAL IMAGE PIXELS.
        self.annotations: dict[str, list[tuple]] = {}

        # ── recommender ghost boxes for the current frame (image px) ───────
        self.suggestions: list[tuple] = []

        # ── rubber-band / drawing state ────────────────────────────────────
        self.drag_start_x: int       = 0
        self.drag_start_y: int       = 0
        self.rubber_band:  int | None = None

    @property
    def loaded(self) -> bool:
        return bool(self.image_paths)

    @property
    def current_path(self) -> str | None:
        return self.image_paths[self.current_index] if self.image_paths else None

    @property
    def current_filename(self) -> str | None:
        p = self.current_path
        return os.path.basename(p) if p else None

    def thermal_frame(self) -> np.ndarray | None:
        """The degC frame aligned to current_index, or None."""
        if self.thermal is None or self.current_index >= len(self.thermal):
            return None
        return self.thermal[self.current_index]


# ---------------------------------------------------------------------------
# Class Selection Dialog
# ---------------------------------------------------------------------------
class ClassSelectionDialog(tk.Toplevel):
    """Modal dialog shown when a father directory has no classes.txt."""

    def __init__(self, parent: tk.Tk, history: list[list[str]]) -> None:
        super().__init__(parent)
        self.result: list[str] | None = None
        self.history = history

        self.title("Configure Classes")
        self.resizable(False, False)
        self.grab_set()   # modal – block the main window
        self.configure(bg="#252526")

        self._build()

        # Centre on parent
        self.update_idletasks()
        px, py = parent.winfo_x(), parent.winfo_y()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        w, h   = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - w)//2}+{py + (ph - h)//2}")

    def _build(self) -> None:
        BG, FG = "#252526", "white"

        tk.Label(
            self,
            text="No classes.txt found in this dataset.\n"
                 "Choose a previous configuration or enter a new one.",
            bg=BG, fg="#aaaaaa", font=("Segoe UI", 9), justify=tk.LEFT,
        ).pack(anchor="w", padx=16, pady=(14, 4))

        # ── Section A: history ─────────────────────────────────────────────
        self._hist_listbox: tk.Listbox | None = None
        if self.history:
            tk.Label(self, text="A  Previous configurations:",
                     bg=BG, fg=FG, font=("Segoe UI", 9, "bold")).pack(
                         anchor="w", padx=16, pady=(8, 2))

            hist_frame = tk.Frame(self, bg=BG)
            hist_frame.pack(fill=tk.X, padx=16)

            sb = tk.Scrollbar(hist_frame)
            sb.pack(side=tk.RIGHT, fill=tk.Y)

            lb = tk.Listbox(
                hist_frame, yscrollcommand=sb.set,
                height=min(len(self.history), 6), width=56,
                bg="#1e1e1e", fg="#cccccc", font=("Consolas", 9),
                selectbackground="#094771", activestyle="none",
                relief=tk.FLAT, borderwidth=0,
            )
            for cfg in self.history:
                lb.insert(tk.END, ", ".join(cfg))
            lb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            sb.config(command=lb.yview)
            lb.bind("<<ListboxSelect>>", self._on_hist_select)
            self._hist_listbox = lb

        # ── Section B: new entry ───────────────────────────────────────────
        tk.Label(self, text="B  New classes (comma-separated):",
                 bg=BG, fg=FG, font=("Segoe UI", 9, "bold")).pack(
                     anchor="w", padx=16, pady=(12, 2))

        self._new_var = tk.StringVar()
        if self.history:
            self._new_var.set(", ".join(self.history[0]))

        entry = tk.Entry(
            self, textvariable=self._new_var, font=("Segoe UI", 10),
            bg="#3c3c3c", fg="white", insertbackground="white",
            relief=tk.FLAT, width=56,
        )
        entry.pack(padx=16, pady=(0, 4), fill=tk.X)
        entry.bind("<Return>", lambda _: self._apply())

        tk.Label(self, text='e.g.  "fire, person"',
                 bg=BG, fg="#666", font=("Segoe UI", 8)).pack(
                     anchor="w", padx=16, pady=(0, 12))

        # ── Buttons ────────────────────────────────────────────────────────
        tk.Frame(self, height=1, bg="#444").pack(fill=tk.X)
        btn_row = tk.Frame(self, bg="#2a2a2a")
        btn_row.pack(fill=tk.X)

        tk.Button(
            btn_row, text="Cancel", command=self.destroy,
            bg="#3c3c3c", fg="white", activebackground="#555",
            relief=tk.FLAT, padx=14, pady=8,
        ).pack(side=tk.RIGHT, padx=(4, 12), pady=8)

        tk.Button(
            btn_row, text="Apply", command=self._apply,
            bg="#0e639c", fg="white", activebackground="#1177bb",
            relief=tk.FLAT, padx=14, pady=8,
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.RIGHT, pady=8)

    def _on_hist_select(self, _: tk.Event) -> None:
        if self._hist_listbox is None:
            return
        sel = self._hist_listbox.curselection()
        if sel:
            self._new_var.set(", ".join(self.history[sel[0]]))

    def _apply(self) -> None:
        raw = self._new_var.get().strip()
        if not raw:
            messagebox.showerror("No classes",
                                 "Please enter at least one class name.",
                                 parent=self)
            return
        names = [n.strip() for n in raw.split(",") if n.strip()]
        if not names:
            messagebox.showerror("Invalid input",
                                 "Could not parse any class names from the entry.",
                                 parent=self)
            return
        self.result = names
        self.destroy()


# ---------------------------------------------------------------------------
# Homography Calibration Dialog
# ---------------------------------------------------------------------------
class HomographyCalibrationDialog(tk.Toplevel):
    """Manual top-down (bird's-eye) homography calibration (§ 4.4.3.1, GUI).

    The user never types a floor coordinate. Instead they click *matching
    points* — the same physical floor location seen in each camera — and the
    software computes the transform to a true overhead floor plane.

    Workflow
    --------
    1. Enter the floor **rectangle**'s width × height (any unit; real cm/m makes
       the plane metric, otherwise a bare ratio like ``2 × 1`` still yields
       correct overhead geometry).
    2. Click the rectangle's 4 corners — **TL → TR → BR → BL** — in each camera,
       plus any number of extra matching points. These 4 corners define the
       floor coordinate frame; the extras improve the fit.
    3. *Solve & Save*:
         • The 4 corners get canonical floor coords ``TL=(0,0)``, ``TR=(w,0)``,
           ``BR=(w,h)``, ``BL=(0,h)`` → the **reference** camera's rectifying
           homography ``H_ref`` (image → top-down floor).
         • Every extra point the reference camera also saw is projected through
           ``H_ref`` to obtain its floor coordinate — computed, not typed.
         • Each camera is then fit (DLT+SVD, ``solve_homography_from_markers``)
           to that same floor plane using whichever of those floor points it saw
           (≥ 4 needed). All cameras share one overhead plane.
       The result is written to ``homography_calibration.npz`` in the session
       directory, in the same format as ``examples/calibrate_homography.py`` so
       the contact-detection stack loads it unchanged.

    Coordinate spaces
    -----------------
    Clicks are stored in **thermal pixel** coordinates (the native (H, W) °C
    grid) — the space the contact detector's foot-points live in
    (``fusion.project_foot_point(det.foot_point, H)``). The on-screen canvas is
    a fixed integer upscale of that grid, so canvas→thermal is a plain divide.
    """

    DISP_SCALE = 5          # canvas px per thermal px
    MARK_COLOR = "#00e5ff"
    MARK_ACTIVE = "#ffea00"
    CORNER_LABELS = ["TL", "TR", "BR", "BL"]   # click order; defines the frame
    N_CORNERS = 4

    def __init__(self, parent: "tk.Tk", app: "ImageAnnotator") -> None:
        super().__init__(parent)
        self.app = app
        self.title("Homography Calibration — top-down floor")
        self.configure(bg="#1e1e1e")
        self.grab_set()

        # Cameras that have both images and a thermal cube loaded.
        self.slots: list[int] = [
            v.slot for v in app.views if v.loaded and v.thermal is not None
        ]
        self.frame_idx: int = app._master_index() or 0

        # Matching points: indices 0..3 are the rectangle corners (fixed),
        # 4.. are extra points. ``clicks[slot][point_index] = (u, v)`` thermal px.
        self.point_labels: list[str] = list(self.CORNER_LABELS)
        self.clicks: dict[int, dict[int, tuple[float, float]]] = {
            s: {} for s in self.slots
        }
        self.active_point: int | None = None

        # Display options (mirrors the main window): optional Gaussian filter and
        # a read-only overlay of the boxes already drawn for this frame.
        self._use_gaussian = tk.BooleanVar(
            value=app.display_mode_var.get() == "gaussian")
        self._sigma_var = tk.StringVar(value=app.gaussian_sigma_var.get())
        self._show_boxes = tk.BooleanVar(value=True)

        self.canvases: dict[int, tk.Canvas] = {}
        self.count_vars: dict[int, tk.StringVar] = {}
        self._photo: dict[int, ImageTk.PhotoImage] = {}

        self._build()
        self._refresh_points_box(select=0)
        self._render_all()

        self.update_idletasks()
        px, py = parent.winfo_x(), parent.winfo_y()
        self.geometry(f"+{px + 40}+{py + 40}")

    # ── construction ────────────────────────────────────────────────────────
    def _build(self) -> None:
        BG, FG = "#1e1e1e", "white"

        tk.Label(
            self,
            text="Click the 4 floor-rectangle corners (TL→TR→BR→BL) in each "
                 "camera, plus any extra matching points. The software computes "
                 "the top-down floor transform — you only type the rectangle size.",
            bg=BG, fg="#aaaaaa", font=("Segoe UI", 9), justify=tk.LEFT,
            wraplength=900,
        ).pack(anchor="w", padx=12, pady=(10, 6))

        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=12)

        # ── Left: rectangle, reference, matching-point list ──────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        tk.Label(left, text="FLOOR RECTANGLE", font=("Segoe UI", 7),
                 fg="#666", bg=BG).pack(anchor="w")
        rect_row = tk.Frame(left, bg=BG)
        rect_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(rect_row, text="W", bg=BG, fg="#aaa",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._w_var = tk.StringVar(value="1")
        tk.Entry(rect_row, textvariable=self._w_var, width=6, bg="#3c3c3c",
                 fg="white", insertbackground="white", relief=tk.FLAT,
                 justify=tk.CENTER).pack(side=tk.LEFT, padx=(2, 6))
        tk.Label(rect_row, text="× H", bg=BG, fg="#aaa",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._h_var = tk.StringVar(value="1")
        tk.Entry(rect_row, textvariable=self._h_var, width=6, bg="#3c3c3c",
                 fg="white", insertbackground="white", relief=tk.FLAT,
                 justify=tk.CENTER).pack(side=tk.LEFT, padx=(2, 0))
        tk.Label(left, text="real cm/m → metric; else any ratio",
                 bg=BG, fg="#666", font=("Segoe UI", 7)).pack(anchor="w")

        tk.Label(left, text="REFERENCE CAMERA", font=("Segoe UI", 7),
                 fg="#666", bg=BG).pack(anchor="w", pady=(8, 0))
        self._ref_var = tk.StringVar(value=CHAN_NAMES[self.slots[0]])
        ref_om = tk.OptionMenu(left, self._ref_var,
                               *[CHAN_NAMES[s] for s in self.slots])
        ref_om.config(bg="#3c3c3c", fg="white", activebackground="#555",
                      relief=tk.FLAT, font=("Segoe UI", 9), anchor="w",
                      highlightthickness=0)
        ref_om["menu"].config(bg="#3c3c3c", fg="white",
                              activebackground="#094771")
        ref_om.pack(fill=tk.X, pady=(2, 0))
        tk.Label(left, text="must see all 4 corners; defines the plane",
                 bg=BG, fg="#666", font=("Segoe UI", 7)).pack(anchor="w")

        tk.Label(left, text="MATCHING POINTS", font=("Segoe UI", 7),
                 fg="#666", bg=BG).pack(anchor="w", pady=(8, 0))
        self._points_box = tk.Listbox(
            left, height=12, width=24, bg="#1e1e1e", fg="#cccccc",
            font=("Consolas", 9), selectbackground="#094771",
            activestyle="none", relief=tk.FLAT, exportselection=False,
            highlightthickness=1, highlightbackground="#444",
        )
        self._points_box.pack(fill=tk.Y, expand=True, pady=(2, 4))
        self._points_box.bind("<<ListboxSelect>>", self._on_point_select)

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill=tk.X)
        tk.Button(btn_row, text="+ Extra point", command=self._add_point,
                  bg="#0e639c", fg="white", activebackground="#1177bb",
                  relief=tk.FLAT, padx=6, pady=4).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(btn_row, text="🗑 Remove", command=self._remove_point,
                  bg="#8b1a1a", fg="white", activebackground="#aa2020",
                  relief=tk.FLAT, padx=6, pady=4).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        # ── Right: camera canvases + frame nav ───────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        nav = tk.Frame(right, bg=BG)
        nav.pack(fill=tk.X, pady=(0, 4))
        tk.Button(nav, text="◀ Prev", command=lambda: self._step(-1),
                  bg="#3c3c3c", fg="white", activebackground="#555",
                  relief=tk.FLAT, padx=8, pady=2).pack(side=tk.LEFT)
        tk.Button(nav, text="Next ▶", command=lambda: self._step(+1),
                  bg="#3c3c3c", fg="white", activebackground="#555",
                  relief=tk.FLAT, padx=8, pady=2).pack(side=tk.LEFT, padx=(4, 8))
        self._frame_var = tk.StringVar()
        tk.Label(nav, textvariable=self._frame_var, bg=BG, fg="#aaa",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)

        # Display options — Gaussian filter + read-only annotation overlay.
        tk.Checkbutton(
            nav, text="Gaussian σ", variable=self._use_gaussian,
            command=self._render_all, bg=BG, fg="white",
            activebackground=BG, activeforeground="white", selectcolor="#3c3c3c",
            font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(12, 0))
        sigma_entry = tk.Entry(nav, textvariable=self._sigma_var, width=4,
                               bg="#3c3c3c", fg="white", insertbackground="white",
                               relief=tk.FLAT, justify=tk.CENTER,
                               font=("Segoe UI", 8))
        sigma_entry.pack(side=tk.LEFT, padx=(4, 0))
        sigma_entry.bind("<Return>", lambda _: self._render_all())
        tk.Checkbutton(
            nav, text="Show boxes", variable=self._show_boxes,
            command=self._render_all, bg=BG, fg="white",
            activebackground=BG, activeforeground="white", selectcolor="#3c3c3c",
            font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(nav, text="left-click: place selected point • right-click: clear",
                 bg=BG, fg="#666", font=("Segoe UI", 8)).pack(side=tk.RIGHT)

        cams = tk.Frame(right, bg=BG)
        cams.pack(fill=tk.BOTH, expand=True)
        for slot in self.slots:
            col = tk.Frame(cams, bg="#111")
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
            tk.Label(col, text=CHAN_NAMES[slot], bg="#2a2a2a",
                     fg=CAM_ACCENT[slot], font=("Segoe UI", 8, "bold"),
                     anchor="w", padx=6).pack(fill=tk.X)
            frame = self.app.views[slot].thermal[self.frame_idx]
            th_h, th_w = frame.shape
            canvas = tk.Canvas(
                col, bg="#2d2d2d", cursor="crosshair", highlightthickness=1,
                highlightbackground="#444",
                width=th_w * self.DISP_SCALE, height=th_h * self.DISP_SCALE,
            )
            canvas.pack()
            canvas.bind("<ButtonPress-1>",
                        lambda e, s=slot: self._on_canvas_click(e, s))
            canvas.bind("<ButtonPress-3>",
                        lambda e, s=slot: self._on_canvas_clear(e, s))
            self.canvases[slot] = canvas
            cv = tk.StringVar(value="0 points")
            self.count_vars[slot] = cv
            tk.Label(col, textvariable=cv, bg="#111", fg="#888",
                     font=("Segoe UI", 8)).pack(fill=tk.X)

        # ── Bottom: solve / save + status ────────────────────────────────────
        tk.Frame(self, height=1, bg="#444").pack(fill=tk.X, pady=(8, 0))
        bottom = tk.Frame(self, bg="#2a2a2a")
        bottom.pack(fill=tk.X)
        self._status_var = tk.StringVar(
            value="Click the 4 corners (≥ in the reference camera) + extras, "
                  "then Solve & Save.")
        tk.Label(bottom, textvariable=self._status_var, bg="#2a2a2a",
                 fg="#aaaaaa", font=("Segoe UI", 8), justify=tk.LEFT,
                 wraplength=560).pack(side=tk.LEFT, padx=12, pady=8)
        tk.Button(bottom, text="Close", command=self.destroy,
                  bg="#3c3c3c", fg="white", activebackground="#555",
                  relief=tk.FLAT, padx=14, pady=8).pack(
                      side=tk.RIGHT, padx=(4, 12), pady=8)
        tk.Button(bottom, text="💾  Solve & Save", command=self._solve_and_save,
                  bg="#2d7d2d", fg="white", activebackground="#3a9a3a",
                  font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
                  padx=14, pady=8).pack(side=tk.RIGHT, pady=8)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _ref_slot(self) -> int:
        """The slot index of the camera currently chosen as reference."""
        name = self._ref_var.get()
        for s in self.slots:
            if CHAN_NAMES[s] == name:
                return s
        return self.slots[0]

    def _cam_count(self, pi: int) -> int:
        """How many cameras have a click for matching point ``pi``."""
        return sum(1 for s in self.slots if pi in self.clicks[s])

    # ── matching-point list ───────────────────────────────────────────────────
    def _add_point(self) -> None:
        n_extra = len(self.point_labels) - self.N_CORNERS + 1
        self.point_labels.append(f"E{n_extra}")
        self._refresh_points_box(select=len(self.point_labels) - 1)

    def _remove_point(self) -> None:
        idx = self.active_point
        if idx is None or idx < self.N_CORNERS:
            self._status_var.set("Only extra points can be removed "
                                 "(the 4 corners are fixed).")
            return
        self.point_labels.pop(idx)
        # Drop this point's clicks and renumber higher indices down by one.
        for slot in self.slots:
            new: dict[int, tuple[float, float]] = {}
            for pi, uv in self.clicks[slot].items():
                if pi == idx:
                    continue
                new[pi - 1 if pi > idx else pi] = uv
            self.clicks[slot] = new
        # Re-label extras so they stay E1, E2, ... in order.
        for j in range(self.N_CORNERS, len(self.point_labels)):
            self.point_labels[j] = f"E{j - self.N_CORNERS + 1}"
        self.active_point = None
        self._refresh_points_box()
        self._render_all()

    def _refresh_points_box(self, *, select: int | None = None) -> None:
        self._points_box.delete(0, tk.END)
        for i, lbl in enumerate(self.point_labels):
            kind = "corner" if i < self.N_CORNERS else "extra "
            self._points_box.insert(
                tk.END, f"{lbl:<3} {kind} · {self._cam_count(i)} cam")
        if select is not None and 0 <= select < len(self.point_labels):
            self._points_box.selection_clear(0, tk.END)
            self._points_box.selection_set(select)
            self.active_point = select
        self._update_counts()

    def _on_point_select(self, _: "tk.Event") -> None:
        sel = self._points_box.curselection()
        self.active_point = sel[0] if sel else None
        self._render_all()

    # ── frame navigation ─────────────────────────────────────────────────────
    def _step(self, delta: int) -> None:
        n = min(len(self.app.views[s].thermal) for s in self.slots)
        self.frame_idx = (self.frame_idx + delta) % n
        self._render_all()

    # ── canvas clicks ────────────────────────────────────────────────────────
    def _on_canvas_click(self, event: "tk.Event", slot: int) -> None:
        if self.active_point is None:
            self._status_var.set("Select a matching point first (left list).")
            return
        frame = self.app.views[slot].thermal[self.frame_idx]
        th_h, th_w = frame.shape
        u = max(0.0, min(event.x / self.DISP_SCALE, float(th_w)))
        v = max(0.0, min(event.y / self.DISP_SCALE, float(th_h)))
        self.clicks[slot][self.active_point] = (u, v)
        self._render_view(slot)
        self._refresh_points_box(select=self.active_point)

    def _on_canvas_clear(self, event: "tk.Event", slot: int) -> None:
        if self.active_point is None:
            return
        self.clicks[slot].pop(self.active_point, None)
        self._render_view(slot)
        self._refresh_points_box(select=self.active_point)

    def _update_counts(self) -> None:
        for slot in self.slots:
            n = len(self.clicks[slot])
            mark = "✓" if n >= 4 else " "
            self.count_vars[slot].set(f"{mark} {n} point(s)")
        self._frame_var.set(
            f"frame {self.frame_idx + 1} / "
            f"{min(len(self.app.views[s].thermal) for s in self.slots)}")

    # ── rendering ────────────────────────────────────────────────────────────
    def _render_all(self) -> None:
        for slot in self.slots:
            self._render_view(slot)
        self._update_counts()

    def _dlg_sigma(self) -> float:
        try:
            s = float(self._sigma_var.get())
            return s if s > 0 else DEFAULT_GAUSSIAN_SIGMA
        except (ValueError, tk.TclError):
            return DEFAULT_GAUSSIAN_SIGMA

    def _frame_boxes(self, slot: int) -> list[tuple]:
        """Read-only boxes for this slot's current frame, as normalised
        ``(cid, nx1, ny1, nx2, ny2)``.

        Prefers the main window's in-memory annotations (reflects unsaved
        edits); otherwise reads the YOLO ``.txt`` from disk.
        """
        view = self.app.views[slot]
        if self.frame_idx >= len(view.image_paths):
            return []
        path = view.image_paths[self.frame_idx]
        if path in view.annotations:
            ow, oh = self.app._cur_res
            return [
                (cid, x1 / ow, y1 / oh, x2 / ow, y2 / oh)
                for cid, x1, y1, x2, y2 in view.annotations[path]
            ]
        txt = os.path.splitext(path)[0] + ".txt"
        boxes: list[tuple] = []
        if os.path.isfile(txt):
            try:
                with open(txt, encoding="utf-8") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        cid = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])
                        boxes.append((cid, xc - bw / 2, yc - bh / 2,
                                      xc + bw / 2, yc + bh / 2))
            except Exception:
                pass
        return boxes

    def _render_view(self, slot: int) -> None:
        canvas = self.canvases[slot]
        frame = self.app.views[slot].thermal[self.frame_idx]
        th_h, th_w = frame.shape
        cw, ch = th_w * self.DISP_SCALE, th_h * self.DISP_SCALE
        shown = (_gaussian_blur(frame, self._dlg_sigma())
                 if self._use_gaussian.get() else frame)
        disp = render_thermal_frame(shown, cw, ch)
        self._photo[slot] = ImageTk.PhotoImage(disp)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=self._photo[slot])

        # Read-only annotation overlay (boxes the user already drew).
        if self._show_boxes.get():
            for cid, nx1, ny1, nx2, ny2 in self._frame_boxes(slot):
                color = CLASS_COLORS[cid % len(CLASS_COLORS)]
                bx1, by1 = nx1 * cw, ny1 * ch
                bx2, by2 = nx2 * cw, ny2 * ch
                canvas.create_rectangle(bx1, by1, bx2, by2,
                                        outline=color, width=2)
                lbl = (self.app.class_names[cid]
                       if 0 <= cid < len(self.app.class_names) else str(cid))
                canvas.create_text(bx1 + 2, by1 + 6, anchor=tk.W, text=lbl,
                                   fill=color, font=("Segoe UI", 7, "bold"))

        for pi, (u, v) in self.clicks[slot].items():
            cx, cy = u * self.DISP_SCALE, v * self.DISP_SCALE
            active = pi == self.active_point
            col = self.MARK_ACTIVE if active else self.MARK_COLOR
            r = 6
            canvas.create_line(cx - r, cy, cx + r, cy, fill=col, width=2)
            canvas.create_line(cx, cy - r, cx, cy + r, fill=col, width=2)
            lbl = (self.point_labels[pi] if pi < len(self.point_labels)
                   else str(pi))
            canvas.create_text(cx + r + 2, cy - r, anchor=tk.W,
                               text=lbl, fill=col,
                               font=("Segoe UI", 8, "bold"))

    # ── solve + save ─────────────────────────────────────────────────────────
    def _solve_and_save(self) -> None:
        if not self.app.father_dir:
            self._status_var.set("No dataset directory — cannot save.")
            return

        try:
            w = float(self._w_var.get())
            h = float(self._h_var.get())
            if w <= 0 or h <= 0:
                raise ValueError
        except ValueError:
            self._status_var.set("Enter a positive rectangle width and height.")
            return

        ref = self._ref_slot()
        ref_clicks = self.clicks[ref]
        if not all(i in ref_clicks for i in range(self.N_CORNERS)):
            self._status_var.set(
                f"Reference camera {CHAN_NAMES[ref]} must have all 4 corners "
                "(TL, TR, BR, BL) clicked.")
            return

        # Canonical floor coordinates for the rectangle corners.
        floor: dict[int, tuple[float, float]] = {
            0: (0.0, 0.0), 1: (w, 0.0), 2: (w, h), 3: (0.0, h),
        }

        # Reference camera's rectifying homography (image -> top-down floor).
        try:
            ref_pairs = [(ref_clicks[i], floor[i]) for i in range(self.N_CORNERS)]
            H_ref = solve_homography_from_markers([(ref, ref_pairs)])[ref]
        except Exception as exc:
            self._status_var.set(f"Reference solve failed: {exc} "
                                 "(are the 4 corners non-collinear?)")
            return

        # Extra points seen by the reference camera get a computed floor coord.
        for pi in range(self.N_CORNERS, len(self.point_labels)):
            if pi in ref_clicks:
                floor[pi] = project_foot_point(ref_clicks[pi], H_ref)

        # Fit every camera to that shared floor plane.
        correspondences: list[tuple[int, list]] = []
        solved_slots: list[int] = []
        for slot in self.slots:
            pairs = [
                (self.clicks[slot][pi], floor[pi])
                for pi in sorted(self.clicks[slot])
                if pi in floor
            ]
            if len(pairs) >= 4:
                correspondences.append((slot, pairs))
                solved_slots.append(slot)

        try:
            H = solve_homography_from_markers(correspondences)
        except Exception as exc:
            self._status_var.set(f"Solver failed: {exc}")
            return

        # Per-camera reprojection residual (in floor units).
        residuals: dict[int, float] = {}
        for slot, pairs in correspondences:
            errs = [
                ((px - xw) ** 2 + (py - yw) ** 2) ** 0.5
                for (uv, (xw, yw)) in pairs
                for (px, py) in [project_foot_point(uv, H[slot])]
            ]
            residuals[slot] = float(np.mean(errs)) if errs else 0.0

        # Save — same NPZ schema as examples/calibrate_homography.py.
        n_pts = len(self.point_labels)
        world = np.full((n_pts, 2), np.nan, dtype=np.float64)
        for pi, xy in floor.items():
            world[pi] = xy
        save_kwargs = {
            "h1": H.h1, "h2": H.h2, "h3": H.h3,
            "world_positions": world,
            "rect_wh": np.array([w, h], dtype=np.float64),
            "ref_camera": np.int64(ref),
        }
        for slot in (0, 1, 2):
            cl = self.clicks.get(slot, {})
            arr = np.array(
                [[u, v, *floor[pi]] for pi, (u, v) in sorted(cl.items())
                 if pi in floor],
                dtype=np.float64,
            ) if cl else np.empty((0, 4), dtype=np.float64)
            save_kwargs[f"pairs_cam{slot}"] = arr if arr.size else \
                np.empty((0, 4), dtype=np.float64)

        out_path = os.path.join(self.app.father_dir, "homography_calibration.npz")
        try:
            np.savez(out_path, **save_kwargs)
        except Exception as exc:
            self._status_var.set(f"Save failed: {exc}")
            return

        skipped = [s for s in self.slots if s not in solved_slots]
        res_txt = "  ".join(
            f"{CHAN_NAMES[s]}:{residuals[s]:.2f}" for s in solved_slots)
        msg = (f"Saved {os.path.basename(out_path)} (ref {CHAN_NAMES[ref]}) — "
               f"solved {', '.join(CHAN_NAMES[s] for s in solved_slots)}.  "
               f"Residual (floor units) {res_txt}.")
        if skipped:
            msg += (f"  Skipped (< 4 floor pts, identity): "
                    f"{', '.join(CHAN_NAMES[s] for s in skipped)}.")
        self._status_var.set(msg)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class ImageAnnotator:
    """Multi-camera YOLO image annotator with father-directory dataset loading."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("YOLO Image Annotator — Thermal")
        self.root.geometry("1440x860")
        self.root.minsize(900, 600)

        self.views: list[ViewState] = [ViewState(i) for i in range(MAX_VIEWS)]
        self.active_slot: int = 0
        self.father_dir: str | None = None
        self.class_names: list[str] = []

        # Per-frame touch labels for the loaded session: {frame_index: 0/1}
        self.touch_labels: dict[int, int] = {}

        # Person recommender: a shared (stateless) adaptive-threshold detector
        # plus a per-channel Tateno background subtractor fitted on load.
        self._human_detector = (
            AdaptiveThresholdDetector(WAVESHARE_26984, **PERSON_DET_KWARGS)
            if _HAVE_PIPELINE else None
        )
        self._bg_pre: dict[int, object] = {}   # slot -> fitted TatenoPipeline
        self._bg_source: str | None = None     # which sibling session gave the bg

        self.scale_to_fit_var = tk.BooleanVar(value=True)
        self.suggest_var = tk.BooleanVar(value=False)

        # Display rendering controls (render from the .npz thermal frames).
        self.resize_w_var = tk.StringVar(value=str(DEFAULT_RESIZE_W))
        self.resize_h_var = tk.StringVar(value=str(DEFAULT_RESIZE_H))
        # Pre-processing applied to the frame BEFORE resize: one of
        # 'none' (raw), 'gaussian' (Gaussian filter), 'tateno' (Tateno residual).
        self.display_mode_var = tk.StringVar(value="none")
        self.gaussian_sigma_var = tk.StringVar(value=str(DEFAULT_GAUSSIAN_SIGMA))
        # Currently-applied output resolution (drives annotation pixel space).
        self._cur_res: tuple[int, int] = (DEFAULT_RESIZE_W, DEFAULT_RESIZE_H)

        self._resize_job: str | None = None

        self._build_ui()

    # ═══════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # The sidebar is scrollable: it holds more controls than fit at small
        # window heights, so wrap it in a Canvas + Scrollbar. `sidebar` below is
        # the inner frame — every control still packs into it unchanged.
        sidebar_container = tk.Frame(self.root, width=236, bg="#252526")
        sidebar_container.grid(row=0, column=0, sticky="ns")
        sidebar_container.grid_propagate(False)

        sb_canvas = tk.Canvas(sidebar_container, bg="#252526",
                              highlightthickness=0, width=220)
        sb_scroll = tk.Scrollbar(sidebar_container, orient="vertical",
                                 command=sb_canvas.yview, troughcolor="#333")
        sb_canvas.configure(yscrollcommand=sb_scroll.set)
        sb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        sb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(sb_canvas, bg="#252526", padx=10, pady=10)
        _sb_win = sb_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind(
            "<Configure>",
            lambda e: sb_canvas.configure(scrollregion=sb_canvas.bbox("all")))
        sb_canvas.bind(
            "<Configure>",
            lambda e: sb_canvas.itemconfigure(_sb_win, width=e.width))
        # Mouse-wheel scrolls the sidebar only while the pointer is over it.
        def _sb_wheel(e):
            sb_canvas.yview_scroll(int(-e.delta / 120), "units")
        sb_canvas.bind("<Enter>", lambda e: sb_canvas.bind_all("<MouseWheel>", _sb_wheel))
        sb_canvas.bind("<Leave>", lambda e: sb_canvas.unbind_all("<MouseWheel>"))

        self.canvas_area = tk.Frame(self.root, bg="#1a1a1a")
        self.canvas_area.grid(row=0, column=1, sticky="nsew")

        self._build_view_frames()

        S = {"bg": "#252526", "fg": "white"}

        tk.Label(sidebar, text="YOLO Annotator", font=("Segoe UI", 13, "bold"),
                 **S).pack(pady=(4, 12), anchor="w")

        # ── Dataset loader ─────────────────────────────────────────────────
        tk.Label(sidebar, text="DATASET", font=("Segoe UI", 7),
                 fg="#666", bg="#252526").pack(anchor="w")
        self.load_btn = tk.Button(
            sidebar,
            text="📂  Open Dataset Directory",
            command=self._open_dataset_directory,
            bg="#0e639c", fg="white", activebackground="#1177bb",
            relief=tk.FLAT, padx=6, pady=7,
            font=("Segoe UI", 9, "bold"),
        )
        self.load_btn.pack(fill=tk.X, pady=(2, 0))

        self.dataset_var = tk.StringVar(value="No dataset loaded")
        tk.Label(sidebar, textvariable=self.dataset_var, font=("Segoe UI", 7),
                 fg="#666", bg="#252526", wraplength=195,
                 justify=tk.LEFT).pack(anchor="w", pady=(3, 0))

        tk.Frame(sidebar, height=1, bg="#444").pack(fill=tk.X, pady=8)

        # ── Image counter (active channel) ─────────────────────────────────
        self.counter_var = tk.StringVar(value="No channels loaded")
        tk.Label(sidebar, textvariable=self.counter_var, font=("Segoe UI", 8),
                 fg="#aaaaaa", bg="#252526", wraplength=195,
                 justify=tk.LEFT).pack(anchor="w")

        # ── Prev / Next ────────────────────────────────────────────────────
        nav = tk.Frame(sidebar, bg="#252526")
        nav.pack(fill=tk.X, pady=(6, 0))
        tk.Button(nav, text="◀ Prev", command=self.prev_image,
                  bg="#3c3c3c", fg="white", activebackground="#555",
                  relief=tk.FLAT, padx=4, pady=5).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        tk.Button(nav, text="Next ▶", command=self.next_image,
                  bg="#3c3c3c", fg="white", activebackground="#555",
                  relief=tk.FLAT, padx=4, pady=5).pack(
                      side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        tk.Frame(sidebar, height=1, bg="#444").pack(fill=tk.X, pady=10)

        # ── TOUCH (single per-frame, all three views) ──────────────────────
        tk.Label(sidebar, text="CONTACT", font=("Segoe UI", 7),
                 fg="#666", bg="#252526").pack(anchor="w")
        self.touch_btn = tk.Button(
            sidebar, text="✋  Touch:  0", command=self.toggle_touch,
            bg="#3c3c3c", fg="white", activebackground="#555",
            relief=tk.FLAT, padx=6, pady=7, font=("Segoe UI", 9, "bold"),
        )
        self.touch_btn.pack(fill=tk.X, pady=(2, 0))
        self.touch_hint_var = tk.StringVar(value="")
        tk.Label(sidebar, textvariable=self.touch_hint_var, font=("Segoe UI", 7),
                 fg="#888", bg="#252526").pack(anchor="w")

        tk.Frame(sidebar, height=1, bg="#444").pack(fill=tk.X, pady=10)

        # ── Rendering + recommender ─────────────────────────────────────────
        tk.Label(sidebar, text="RENDERING", font=("Segoe UI", 7),
                 fg="#666", bg="#252526").pack(anchor="w")
        tk.Checkbutton(
            sidebar, text="Scale to Fit Canvas",
            variable=self.scale_to_fit_var,
            command=self._on_scale_toggle,
            bg="#252526", fg="white",
            activebackground="#252526", activeforeground="white",
            selectcolor="#3c3c3c", font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))

        # ── Resize resolution (the thermal frame is bicubic-resized to this) ─
        res_row = tk.Frame(sidebar, bg="#252526")
        res_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(res_row, text="Resize", font=("Segoe UI", 8),
                 fg="#aaaaaa", bg="#252526").pack(side=tk.LEFT)
        tk.Entry(res_row, textvariable=self.resize_w_var, width=5,
                 bg="#3c3c3c", fg="white", insertbackground="white",
                 relief=tk.FLAT, justify=tk.CENTER,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(6, 1))
        tk.Label(res_row, text="×", font=("Segoe UI", 9),
                 fg="#aaaaaa", bg="#252526").pack(side=tk.LEFT)
        tk.Entry(res_row, textvariable=self.resize_h_var, width=5,
                 bg="#3c3c3c", fg="white", insertbackground="white",
                 relief=tk.FLAT, justify=tk.CENTER,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(1, 4))
        tk.Button(res_row, text="Apply", command=self._on_resolution_change,
                  bg="#3c3c3c", fg="white", activebackground="#555",
                  relief=tk.FLAT, padx=6, pady=1,
                  font=("Segoe UI", 8)).pack(side=tk.LEFT)

        # ── Pre-processing mode (applied on the native grid BEFORE resizing) ─
        tk.Label(sidebar, text="PROCESSING", font=("Segoe UI", 7),
                 fg="#666", bg="#252526").pack(anchor="w", pady=(6, 0))
        radio_kw = dict(
            variable=self.display_mode_var, command=self._on_display_mode_change,
            bg="#252526", fg="white", activebackground="#252526",
            activeforeground="white", selectcolor="#3c3c3c",
            font=("Segoe UI", 9), anchor="w",
        )
        tk.Radiobutton(sidebar, text="None (raw)", value="none",
                       **radio_kw).pack(anchor="w")
        g_row = tk.Frame(sidebar, bg="#252526")
        g_row.pack(fill=tk.X)
        tk.Radiobutton(g_row, text="Gaussian (σ)", value="gaussian",
                       **radio_kw).pack(side=tk.LEFT)
        sigma_entry = tk.Entry(g_row, textvariable=self.gaussian_sigma_var, width=4,
                               bg="#3c3c3c", fg="white", insertbackground="white",
                               relief=tk.FLAT, justify=tk.CENTER,
                               font=("Segoe UI", 9))
        sigma_entry.pack(side=tk.LEFT, padx=(6, 0))
        sigma_entry.bind("<Return>", lambda _: self._on_display_mode_change())
        tk.Radiobutton(sidebar, text="Tateno (residual)", value="tateno",
                       **radio_kw).pack(anchor="w")

        tk.Checkbutton(
            sidebar, text="Suggest (thermal)",
            variable=self.suggest_var,
            command=self._on_suggest_toggle,
            bg="#252526", fg="white",
            activebackground="#252526", activeforeground="white",
            selectcolor="#3c3c3c", font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))
        tk.Button(sidebar, text="✓  Accept Suggestions", command=self.accept_suggestions,
                  bg="#3c3c3c", fg="white", activebackground="#555",
                  relief=tk.FLAT, padx=6, pady=5).pack(fill=tk.X, pady=(4, 0))

        tk.Frame(sidebar, height=1, bg="#444").pack(fill=tk.X, pady=10)

        # ── Calibration ─────────────────────────────────────────────────────
        tk.Label(sidebar, text="CALIBRATION", font=("Segoe UI", 7),
                 fg="#666", bg="#252526").pack(anchor="w")
        tk.Button(sidebar, text="🎯  Homography Calibration",
                  command=self.open_homography_calibration,
                  bg="#3c3c3c", fg="white", activebackground="#555",
                  relief=tk.FLAT, padx=6, pady=7).pack(fill=tk.X, pady=(2, 0))

        tk.Frame(sidebar, height=1, bg="#444").pack(fill=tk.X, pady=10)

        # ── Class selector ──────────────────────────────────────────────────
        tk.Label(sidebar, text="CLASS", font=("Segoe UI", 7),
                 fg="#666", bg="#252526").pack(anchor="w")

        self.class_name_var = tk.StringVar(value="")
        self.class_selector = tk.OptionMenu(sidebar, self.class_name_var, "")
        self.class_selector.config(
            bg="#3c3c3c", fg="white", activebackground="#555",
            relief=tk.FLAT, font=("Segoe UI", 9), anchor="w",
            highlightthickness=0, padx=6,
        )
        self.class_selector["menu"].config(bg="#3c3c3c", fg="white",
                                            activebackground="#094771")
        self.class_selector.pack(fill=tk.X, pady=(2, 0))

        self.swatch = tk.Label(sidebar, text="  —  ", bg="#444",
                               fg="white", font=("Segoe UI", 8, "bold"))
        self.swatch.pack(fill=tk.X, pady=(4, 0))
        self.class_name_var.trace_add("write", self._update_swatch)

        tk.Frame(sidebar, height=1, bg="#444").pack(fill=tk.X, pady=10)

        # ── Annotations listbox ────────────────────────────────────────────
        self.active_label_var = tk.StringVar(value="ANNOTATIONS")
        tk.Label(sidebar, textvariable=self.active_label_var,
                 font=("Segoe UI", 7), fg="#666", bg="#252526").pack(anchor="w")

        list_frame = tk.Frame(sidebar, bg="#252526")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        scrollbar = tk.Scrollbar(list_frame, troughcolor="#333")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.ann_listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            bg="#1e1e1e", fg="#cccccc", font=("Consolas", 8),
            selectbackground="#094771", activestyle="none",
            relief=tk.FLAT, borderwidth=0,
        )
        self.ann_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.ann_listbox.yview)

        tk.Button(sidebar, text="🗑  Delete Selected", command=self.delete_selected,
                  bg="#8b1a1a", fg="white", activebackground="#aa2020",
                  relief=tk.FLAT, padx=6, pady=5).pack(fill=tk.X, pady=(4, 0))

        tk.Frame(sidebar, height=1, bg="#444").pack(fill=tk.X, pady=8)

        tk.Button(sidebar, text="💾  Save All Channels", command=self._save_all,
                  bg="#2d7d2d", fg="white", activebackground="#3a9a3a",
                  font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, padx=6, pady=8).pack(fill=tk.X)
        tk.Button(sidebar, text="📊  Export Labels", command=self.export_labels,
                  bg="#2d7d2d", fg="white", activebackground="#3a9a3a",
                  font=("Segoe UI", 9, "bold"),
                  relief=tk.FLAT, padx=6, pady=8).pack(fill=tk.X, pady=(4, 0))

        self.status_var = tk.StringVar(value="Open a dataset directory to begin.")
        tk.Label(sidebar, textvariable=self.status_var, font=("Segoe UI", 8),
                 fg="#888888", bg="#252526", wraplength=195,
                 justify=tk.LEFT).pack(pady=(8, 0), anchor="w")

        # ── Keyboard shortcuts ─────────────────────────────────────────────
        self.root.bind("<Right>",    lambda _: self.next_image())
        self.root.bind("<Left>",     lambda _: self.prev_image())
        self.root.bind("<Delete>",   lambda _: self.delete_selected())
        self.root.bind("<Control-s>", lambda _: self._save_all())
        self.root.bind("t",          lambda _: self.toggle_touch())
        self.root.bind("T",          lambda _: self.toggle_touch())
        self.root.bind("a",          lambda _: self.accept_suggestions())
        self.root.bind("A",          lambda _: self.accept_suggestions())
        self.root.bind("s",          lambda _: self._toggle_suggest_key())
        self.root.bind("S",          lambda _: self._toggle_suggest_key())
        self.root.bind("<Configure>", self._on_window_resize)

    def _build_view_frames(self) -> None:
        for view in self.views:
            frame = tk.Frame(self.canvas_area, bg="#111")
            view.frame = frame

            hdr_var = tk.StringVar(value=CHAN_NAMES[view.slot])
            view.header_var = hdr_var
            tk.Label(
                frame, textvariable=hdr_var,
                bg="#2a2a2a", fg=CAM_ACCENT[view.slot],
                font=("Segoe UI", 8, "bold"), anchor="w", padx=6,
            ).pack(fill=tk.X)

            canvas = tk.Canvas(
                frame, bg="#2d2d2d", cursor="crosshair",
                highlightthickness=2,
                highlightbackground="#444",
                highlightcolor=CAM_ACCENT[view.slot],
            )
            canvas.pack(fill=tk.BOTH, expand=True)
            view.canvas = canvas

            canvas.bind("<ButtonPress-1>",
                        lambda e, v=view: self._on_press(e, v))
            canvas.bind("<B1-Motion>",
                        lambda e, v=view: self._on_drag(e, v))
            canvas.bind("<ButtonRelease-1>",
                        lambda e, v=view: self._on_release(e, v))

    def _refresh_canvas_layout(self) -> None:
        for view in self.views:
            view.frame.pack_forget()
        for view in self.views:
            if view.loaded:
                view.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                                padx=1, pady=1)

    # ═══════════════════════════════════════════════════════════════════════
    # DATASET LOADING — Father directory workflow
    # ═══════════════════════════════════════════════════════════════════════

    def _open_dataset_directory(self) -> None:
        father = filedialog.askdirectory(title="Select Dataset Directory")
        if not father:
            return

        channel_paths: dict[int, str] = {}
        for slot, dirname in enumerate(CHANNEL_DIRS):
            candidate = os.path.join(father, dirname)
            if os.path.isdir(candidate):
                channel_paths[slot] = candidate

        if not channel_paths:
            messagebox.showerror(
                "No channels found",
                f"None of the expected subdirectories were found in:\n{father}\n\n"
                f"Expected: {', '.join(CHANNEL_DIRS)}",
            )
            return

        frame_counts: dict[int, int] = {
            slot: sum(
                1 for f in os.listdir(path)
                if f.lower().endswith(IMG_EXTS)
            )
            for slot, path in channel_paths.items()
        }

        if len(set(frame_counts.values())) > 1:
            detail = "\n".join(
                f"  {CHANNEL_DIRS[slot]}: {count} frame(s)"
                for slot, count in sorted(frame_counts.items())
            )
            messagebox.showerror(
                "Frame count mismatch",
                f"Image counts differ across channels — loading aborted.\n\n"
                f"{detail}\n\n"
                "All active channels must contain the same number of images.",
            )
            return

        self.father_dir = father

        classes_txt = os.path.join(father, "classes.txt")
        if os.path.isfile(classes_txt):
            names = self._read_classes_txt(classes_txt)
            self._finish_loading(channel_paths, names)
        else:
            self._ask_class_names(father, channel_paths)

    def _read_classes_txt(self, path: str) -> list[str]:
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _ask_class_names(self, father: str,
                         channel_paths: dict[int, str]) -> None:
        settings = self._load_settings()
        dialog = ClassSelectionDialog(self.root,
                                      settings["class_configurations"])
        self.root.wait_window(dialog)

        if dialog.result is None:
            self.status_var.set("Dataset load cancelled.")
            return

        names = dialog.result

        classes_txt = os.path.join(father, "classes.txt")
        with open(classes_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")

        settings["class_configurations"] = self._push_class_config(
            settings["class_configurations"], names
        )
        self._save_settings(settings)

        self._finish_loading(channel_paths, names)

    def _finish_loading(self, channel_paths: dict[int, str],
                        class_names: list[str]) -> None:
        # Auto-save any annotations from a previously loaded dataset
        for view in self.views:
            if view.loaded:
                self._save_view(view, silent=True)

        self.class_names = class_names
        self._update_class_selector()

        # Honour whatever resize resolution is currently in the entries.
        self._cur_res = self._resize_dims()
        self.resize_w_var.set(str(self._cur_res[0]))
        self.resize_h_var.set(str(self._cur_res[1]))

        # Reset all view state + per-session touch labels
        self.touch_labels = {}
        self._bg_pre = {}
        for view in self.views:
            view.image_paths = []
            view.current_index = 0
            view.annotations.clear()
            view.suggestions = []
            view.thermal = None
            view.current_pil = None
            view.photo_image = None
            if view.canvas:
                view.canvas.delete("all")

        for slot, dir_path in channel_paths.items():
            view = self.views[slot]
            view.image_paths = sorted(
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.lower().endswith(IMG_EXTS)
            )
            view.current_index = 0
            view.thermal = self._load_thermal(slot)

        self._build_backgrounds(list(channel_paths))

        self._refresh_canvas_layout()
        for slot in channel_paths:
            self._load_view_image(self.views[slot])

        first = next((v for v in self.views if v.loaded), None)
        if first:
            self._set_active_slot(first.slot)

        n_frames = len(first.image_paths) if first else 0
        fname = os.path.basename(self.father_dir or "")
        self.dataset_var.set(f"{fname}  ({len(channel_paths)} ch × {n_frames} frames)")
        self.load_btn.config(text=f"✓  {fname}", bg="#2a5e2a")
        if not _HAVE_PIPELINE:
            sug = "Suggest: fire only (person pipeline unavailable)."
        elif self._bg_pre:
            sug = f"Suggest: fire + person (bg: {self._bg_source})."
        else:
            sug = "Suggest: fire only (no empty_room background found)."
        self.status_var.set(
            f"Loaded: {len(channel_paths)} channel(s), {n_frames} frame(s) each.\n"
            f"Classes: {', '.join(class_names)}\n{sug}"
        )
        self._update_touch_button()

    def _load_thermal(self, slot: int) -> np.ndarray | None:
        """Load chX_raw_data.npz ('frames') for the recommender, if present."""
        if not self.father_dir:
            return None
        npz_path = os.path.join(self.father_dir, NPZ_NAMES[slot])
        if not os.path.isfile(npz_path):
            return None
        try:
            with np.load(npz_path) as z:
                return np.asarray(z["frames"], dtype=np.float32)
        except Exception as exc:
            print(f"[annotator] Warning – could not load {npz_path}: {exc}")
            return None

    def _build_backgrounds(self, slots: list[int]) -> None:
        """
        Fit a Tateno background subtractor per channel from a sibling
        empty-room recording (datasets root / BACKGROUND_SESSIONS / chN_*.npz).
        Person suggestions need this; if no background session is found they
        are disabled (fire suggestions still work — they don't need it).
        """
        self._bg_pre = {}
        if not (_HAVE_PIPELINE and self.father_dir):
            return
        root = os.path.dirname(self.father_dir.rstrip("/\\"))
        for bg_name in BACKGROUND_SESSIONS:
            bg_dir = os.path.join(root, bg_name)
            if not os.path.isdir(bg_dir):
                continue
            built = 0
            for slot in slots:
                npz_path = os.path.join(bg_dir, NPZ_NAMES[slot])
                if not os.path.isfile(npz_path):
                    continue
                try:
                    with np.load(npz_path) as z:
                        bg_frames = np.asarray(z["frames"], dtype=np.float32)
                    pre = TatenoPipeline(WAVESHARE_26984)
                    pre.fit(bg_frames)
                    self._bg_pre[slot] = pre
                    built += 1
                except Exception as exc:
                    print(f"[annotator] Warning – background fit failed "
                          f"({bg_name} ch{slot}): {exc}")
            if built:
                self._bg_source = bg_name
                return
        self._bg_source = None

    # ═══════════════════════════════════════════════════════════════════════
    # IMAGE LOADING PER VIEW
    # ═══════════════════════════════════════════════════════════════════════

    def _resize_dims(self) -> tuple[int, int]:
        """Parse the resize W×H entries; fall back to the applied resolution."""
        try:
            w = int(float(self.resize_w_var.get()))
            h = int(float(self.resize_h_var.get()))
            if w < 8 or h < 8 or w > 4096 or h > 4096:
                raise ValueError
            return w, h
        except (ValueError, tk.TclError):
            return self._cur_res

    def _sigma_value(self) -> float:
        """Current Gaussian sigma from the entry (falls back to the default)."""
        try:
            s = float(self.gaussian_sigma_var.get())
            return s if s > 0 else DEFAULT_GAUSSIAN_SIGMA
        except (ValueError, tk.TclError):
            return DEFAULT_GAUSSIAN_SIGMA

    def _processed_frame(self, view: ViewState, frame: np.ndarray) -> np.ndarray:
        """Apply the selected PROCESSING mode to the native (H, W) thermal frame.

        'none' -> raw; 'gaussian' -> Gaussian blur; 'tateno' -> Tateno residual
        (|smoothed - background|) using this channel's empty-room background.
        Tateno silently falls back to raw if no background is available.
        """
        mode = self.display_mode_var.get()
        if mode == "gaussian":
            return _gaussian_blur(frame, self._sigma_value())
        if mode == "tateno":
            pre = self._bg_pre.get(view.slot)
            if pre is None:
                return frame
            try:
                return np.asarray(
                    pre.predict(_ThermalFrame(
                        data=frame.astype(np.float32),
                        timestamp=0.0, camera_id=view.slot)).data,
                    dtype=np.float32,
                )
            except Exception as exc:
                print(f"[annotator] tateno display failed (ch{view.slot}): {exc}")
                return frame
        return frame

    def _make_display_image(self, view: ViewState) -> Image.Image:
        """Render this view's current frame for display.

        Renders from the thermal .npz: apply the PROCESSING mode, then resize +
        inferno. Falls back to the preview PNG only if the thermal is missing.
        """
        frame = view.thermal_frame()
        if frame is not None:
            w, h = self._cur_res
            return render_thermal_frame(self._processed_frame(view, frame), w, h)
        return Image.open(view.current_path).convert("RGB")

    def _load_view_image(self, view: ViewState) -> None:
        if not view.loaded:
            return
        path = view.current_path
        view.current_pil = self._make_display_image(view)
        view.orig_w, view.orig_h = view.current_pil.size

        if path not in view.annotations:
            self._load_annotations_from_disk(view, path)

        self._recompute_suggestions(view)

        idx_str = f"{view.current_index + 1}/{len(view.image_paths)}"
        temp = ""
        raw = view.thermal_frame()
        if raw is not None:
            temp = f"  │  {float(raw.min()):.1f} - {float(raw.max()):.1f} °C"
        view.header_var.set(
            f"  {CHAN_NAMES[view.slot]}  {idx_str}  │  {view.current_filename}{temp}")

        self._redraw_view(view)

    def _rerender_loaded(self) -> None:
        """Regenerate the display image for every loaded view and redraw."""
        for view in self.views:
            if view.loaded:
                view.current_pil = self._make_display_image(view)
                view.orig_w, view.orig_h = view.current_pil.size
                self._redraw_view(view)

    def _on_display_mode_change(self) -> None:
        """Processing-mode / sigma change — visual only, no coordinate change."""
        if self.display_mode_var.get() == "tateno" and not self._bg_pre:
            self.status_var.set(
                "Tateno needs an empty_room background — none found; showing raw.")
        self._rerender_loaded()

    def _on_resolution_change(self) -> None:
        """Apply a new resize resolution.

        The annotation pixel space is tied to the display resolution, so any
        existing in-memory boxes are rescaled to the new dimensions. YOLO .txt
        on disk is normalised, so saved labels are unaffected either way.
        """
        new_w, new_h = self._resize_dims()
        old_w, old_h = self._cur_res
        if (new_w, new_h) == (old_w, old_h):
            return
        fx, fy = new_w / old_w, new_h / old_h
        for view in self.views:
            for boxes in view.annotations.values():
                for i, (cid, x1, y1, x2, y2) in enumerate(boxes):
                    boxes[i] = (cid, int(x1 * fx), int(y1 * fy),
                                int(x2 * fx), int(y2 * fy))
        self._cur_res = (new_w, new_h)
        self.resize_w_var.set(str(new_w))
        self.resize_h_var.set(str(new_h))
        self._rerender_loaded()
        self._refresh_listbox()
        self.status_var.set(f"Resolution set to {new_w}×{new_h}.")

    def _load_annotations_from_disk(self, view: ViewState,
                                    image_path: str) -> None:
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        boxes: list[tuple] = []

        if os.path.isfile(txt_path):
            try:
                with open(txt_path, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cid = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])
                        x1 = int((xc - bw / 2) * view.orig_w)
                        y1 = int((yc - bh / 2) * view.orig_h)
                        x2 = int((xc + bw / 2) * view.orig_w)
                        y2 = int((yc + bh / 2) * view.orig_h)
                        boxes.append((cid, x1, y1, x2, y2))
            except Exception as exc:
                print(f"[annotator] Warning – could not read {txt_path}: {exc}")

        view.annotations[image_path] = boxes

    # ═══════════════════════════════════════════════════════════════════════
    # RECOMMENDER
    # ═══════════════════════════════════════════════════════════════════════

    def _person_boxes(self, view: ViewState, frame: np.ndarray) -> list[tuple]:
        """Run Tateno background subtraction + adaptive-threshold detector.

        Returns person bounding boxes as (x1, y1, x2, y2) in thermal pixels.
        Empty if the pipeline or this channel's background is unavailable.
        """
        if self._human_detector is None or view.slot not in self._bg_pre:
            return []
        pre = self._bg_pre[view.slot]
        try:
            residual = pre.predict(
                _ThermalFrame(data=frame, timestamp=0.0, camera_id=view.slot))
            dets = self._human_detector.predict(residual)
        except Exception as exc:
            print(f"[annotator] person suggest failed (ch{view.slot}): {exc}")
            return []
        boxes = []
        for d in dets:
            x, y, w, h = d.bbox
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
        return boxes

    def _recompute_suggestions(self, view: ViewState) -> None:
        """Fill view.suggestions (image px) from the thermal frame, if Suggest on."""
        view.suggestions = []
        if not self.suggest_var.get():
            return
        frame = view.thermal_frame()
        if frame is None:
            return
        th_h, th_w = frame.shape
        sx = view.orig_w / th_w
        sy = view.orig_h / th_h

        def scale(box, cid):
            x1, y1, x2, y2 = box
            return (cid, int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

        for box in suggest_fire(frame):                       # raw, no background
            view.suggestions.append(scale(box, FIRE_CLASS_ID))
        for box in self._person_boxes(view, frame):           # residual pipeline
            view.suggestions.append(scale(box, PERSON_CLASS_ID))

    def _update_touch_hint(self) -> None:
        """Show the precision-first touch=0 hint when confident.

        Touch needs >= 2 people; if every view's person detector finds <= 1
        person we can confidently hint 'no touch'. Never hints touch=1.
        """
        if not self.suggest_var.get():
            self.touch_hint_var.set("")
            return
        counts = []
        for v in self.views:
            if not v.loaded:
                continue
            f = v.thermal_frame()
            if f is None:
                continue
            counts.append(len(self._person_boxes(v, f)))
        if counts and all(c <= 1 for c in counts):
            self.touch_hint_var.set("suggest: no touch")
        else:
            self.touch_hint_var.set("")

    def accept_suggestions(self) -> None:
        """Commit the active view's ghost boxes into real annotations."""
        view = self.views[self.active_slot]
        if not view.loaded or not view.suggestions:
            return
        anns = view.annotations.setdefault(view.current_path, [])
        existing = set(anns)
        added = 0
        for s in view.suggestions:
            if s not in existing:
                anns.append(s)
                added += 1
        view.suggestions = []
        self._redraw_view(view)
        self._refresh_listbox()
        self.status_var.set(
            f"[{CHAN_NAMES[view.slot]}] Accepted {added} suggestion(s).")

    def _on_suggest_toggle(self) -> None:
        for view in self.views:
            if view.loaded:
                self._recompute_suggestions(view)
        self._redraw_all()
        self._update_touch_hint()

    def _toggle_suggest_key(self) -> None:
        self.suggest_var.set(not self.suggest_var.get())
        self._on_suggest_toggle()

    # ═══════════════════════════════════════════════════════════════════════
    # HOMOGRAPHY CALIBRATION
    # ═══════════════════════════════════════════════════════════════════════

    def open_homography_calibration(self) -> None:
        """Open the manual homography calibration dialog for the loaded views."""
        if not _HAVE_HOMOGRAPHY:
            messagebox.showerror(
                "Calibration unavailable",
                "The homography solver could not be imported "
                "(thermal_algorithms not on the path).")
            return
        loaded = [v for v in self.views if v.loaded and v.thermal is not None]
        if not loaded:
            messagebox.showerror(
                "No thermal data",
                "Load a dataset with chN_raw_data.npz files before calibrating.")
            return
        HomographyCalibrationDialog(self.root, self)

    # ═══════════════════════════════════════════════════════════════════════
    # TOUCH (single per-frame label across all three views)
    # ═══════════════════════════════════════════════════════════════════════

    def _master_index(self) -> int | None:
        first = next((v for v in self.views if v.loaded), None)
        return first.current_index if first else None

    def toggle_touch(self) -> None:
        idx = self._master_index()
        if idx is None:
            return
        self.touch_labels[idx] = 0 if self.touch_labels.get(idx, 0) else 1
        self._update_touch_button()

    def _update_touch_button(self) -> None:
        idx = self._master_index()
        val = self.touch_labels.get(idx, 0) if idx is not None else 0
        if val:
            self.touch_btn.config(text="✋  Touch:  1", bg="#2d7d2d",
                                  activebackground="#3a9a3a")
        else:
            self.touch_btn.config(text="✋  Touch:  0", bg="#3c3c3c",
                                  activebackground="#555")
        self._update_touch_hint()

    # ═══════════════════════════════════════════════════════════════════════
    # RENDERING
    # ═══════════════════════════════════════════════════════════════════════

    def _redraw_view(self, view: ViewState) -> None:
        if view.current_pil is None or view.canvas is None:
            return

        view.canvas.update_idletasks()
        cw = view.canvas.winfo_width()
        ch = view.canvas.winfo_height()
        if cw < 2 or ch < 2:
            self.root.after(40, lambda v=view: self._redraw_view(v))
            return

        if self.scale_to_fit_var.get():
            view.scale_factor = min(cw / view.orig_w, ch / view.orig_h, 1.0)
            disp_w = int(view.orig_w * view.scale_factor)
            disp_h = int(view.orig_h * view.scale_factor)
            view.offset_x = (cw - disp_w) // 2
            view.offset_y = (ch - disp_h) // 2
            display_img = view.current_pil.resize(
                (disp_w, disp_h), Image.Resampling.BICUBIC)
        else:
            view.scale_factor = 1.0
            view.offset_x = (cw - view.orig_w) // 2
            view.offset_y = (ch - view.orig_h) // 2
            display_img = view.current_pil

        view.photo_image = ImageTk.PhotoImage(display_img)
        view.canvas.delete("all")
        view.canvas.create_image(view.offset_x, view.offset_y,
                                 anchor=tk.NW, image=view.photo_image)

        # Ghost suggestions first (so committed boxes draw on top)
        for ann in view.suggestions:
            self._paint_box_on(view, ann, ghost=True)
        for ann in view.annotations.get(view.current_path, []):
            self._paint_box_on(view, ann)

    def _paint_box_on(self, view: ViewState, ann: tuple, *, ghost: bool = False) -> None:
        cid, x1, y1, x2, y2 = ann
        color = CLASS_COLORS[cid % len(CLASS_COLORS)]

        cx1 = x1 * view.scale_factor + view.offset_x
        cy1 = y1 * view.scale_factor + view.offset_y
        cx2 = x2 * view.scale_factor + view.offset_x
        cy2 = y2 * view.scale_factor + view.offset_y

        label = (self.class_names[cid]
                 if 0 <= cid < len(self.class_names) else str(cid))

        if ghost:
            # Dashed, dimmed proposal; no filled badge — clearly distinct.
            view.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                         outline=color, width=1, dash=(3, 2),
                                         tags="box")
            view.canvas.create_text(cx1 + 2, cy1 + 6, anchor=tk.W,
                                     text=f"?{label}", fill=color,
                                     font=("Segoe UI", 7, "bold"), tags="box")
            return

        view.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                     outline=color, width=2, tags="box")
        badge_w = max(18, len(label) * 7 + 8)
        view.canvas.create_rectangle(cx1, cy1, cx1 + badge_w, cy1 + 14,
                                     fill=color, outline="", tags="box")
        view.canvas.create_text(cx1 + badge_w / 2, cy1 + 7, text=label,
                                fill="white", font=("Segoe UI", 7, "bold"),
                                tags="box")

    def _redraw_all(self) -> None:
        for view in self.views:
            if view.loaded:
                self._redraw_view(view)

    # ═══════════════════════════════════════════════════════════════════════
    # MOUSE DRAWING
    # ═══════════════════════════════════════════════════════════════════════

    def _on_press(self, event: tk.Event, view: ViewState) -> None:
        if not view.loaded:
            return
        self._set_active_slot(view.slot)
        view.drag_start_x = event.x
        view.drag_start_y = event.y
        view.rubber_band = view.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="white", width=1, dash=(4, 3),
        )

    def _on_drag(self, event: tk.Event, view: ViewState) -> None:
        if view.rubber_band is not None:
            view.canvas.coords(view.rubber_band,
                               view.drag_start_x, view.drag_start_y,
                               event.x, event.y)

    def _on_release(self, event: tk.Event, view: ViewState) -> None:
        if view.rubber_band is None or not view.loaded:
            return

        view.canvas.delete(view.rubber_band)
        view.rubber_band = None

        ex, ey = event.x, event.y
        if abs(ex - view.drag_start_x) < 5 or abs(ey - view.drag_start_y) < 5:
            return

        class_name = self.class_name_var.get()
        if class_name in self.class_names:
            cid = self.class_names.index(class_name)
        elif self.class_names:
            cid = 0
        else:
            messagebox.showerror("No classes",
                                 "Load a dataset before drawing annotations.")
            return

        x1i, y1i = self._canvas_to_image(view, view.drag_start_x, view.drag_start_y)
        x2i, y2i = self._canvas_to_image(view, ex, ey)
        x1i, x2i = sorted([x1i, x2i])
        y1i, y2i = sorted([y1i, y2i])

        view.annotations.setdefault(view.current_path, []).append(
            (cid, int(x1i), int(y1i), int(x2i), int(y2i))
        )

        self._redraw_view(view)
        self._refresh_listbox()

    def _canvas_to_image(self, view: ViewState,
                          cx: float, cy: float) -> tuple[float, float]:
        ix = (cx - view.offset_x) / view.scale_factor
        iy = (cy - view.offset_y) / view.scale_factor
        return (max(0.0, min(ix, float(view.orig_w))),
                max(0.0, min(iy, float(view.orig_h))))

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE / EXPORT
    # ═══════════════════════════════════════════════════════════════════════

    def _save_view(self, view: ViewState, *, silent: bool = False) -> None:
        if not view.loaded or view.current_pil is None:
            return

        path     = view.current_path
        boxes    = view.annotations.get(path, [])
        txt_path = os.path.splitext(path)[0] + ".txt"

        lines: list[str] = []
        for cid, x1, y1, x2, y2 in boxes:
            abs_xc = (x1 + x2) / 2.0
            abs_yc = (y1 + y2) / 2.0
            abs_bw = float(x2 - x1)
            abs_bh = float(y2 - y1)
            norm_xc = abs_xc / view.orig_w
            norm_yc = abs_yc / view.orig_h
            norm_bw = abs_bw / view.orig_w
            norm_bh = abs_bh / view.orig_h
            lines.append(
                f"{cid} {norm_xc:.6f} {norm_yc:.6f} {norm_bw:.6f} {norm_bh:.6f}"
            )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        if not silent:
            self.status_var.set(
                f"[{CHAN_NAMES[view.slot]}] Saved {len(lines)} box(es) → "
                f"{os.path.basename(txt_path)}"
            )

    def _save_all(self) -> None:
        count = sum(1 for v in self.views if v.loaded)
        for view in self.views:
            if view.loaded:
                self._save_view(view, silent=True)
        if count:
            self.status_var.set(f"Saved annotations for {count} channel(s).")

    # ── Auto-generated xlsx + contact_labels.csv ────────────────────────────

    def _count_boxes_on_disk(self, view: ViewState, frame_idx: int) -> tuple[int, int]:
        """Return (person_count, fire_present) from the saved .txt for a frame."""
        if frame_idx >= len(view.image_paths):
            return 0, 0
        txt = os.path.splitext(view.image_paths[frame_idx])[0] + ".txt"
        persons = fire = 0
        if os.path.isfile(txt):
            with open(txt, encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    if cid == PERSON_CLASS_ID:
                        persons += 1
                    elif cid == FIRE_CLASS_ID:
                        fire = 1
        return persons, fire

    def export_labels(self) -> None:
        if not self.father_dir:
            messagebox.showerror("No dataset", "Load a dataset first.")
            return
        if openpyxl is None:
            messagebox.showerror(
                "openpyxl missing",
                "Export needs openpyxl.\n\n    pip install openpyxl")
            return

        # Make sure the YOLO .txt files reflect the latest boxes.
        self._save_all()

        loaded = [v for v in self.views if v.loaded]
        if not loaded:
            return
        n_frames = len(loaded[0].image_paths)
        scene = os.path.basename(self.father_dir.rstrip("/\\"))

        # ── contact_labels.csv (one row per frame) ──────────────────────────
        csv_path = os.path.join(self.father_dir, "contact_labels.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "contact"])
            for i in range(n_frames):
                w.writerow([i, int(self.touch_labels.get(i, 0))])

        # ── labels.xlsx (sheet per scene, one row per channel/frame) ────────
        xlsx_path = os.path.join(self.father_dir, "labels.xlsx")
        if os.path.isfile(xlsx_path):
            wb = openpyxl.load_workbook(xlsx_path)
        else:
            wb = openpyxl.Workbook()
            wb.remove(wb.active)

        if scene in wb.sheetnames:
            del wb[scene]
        ws = wb.create_sheet(title=scene[:31])  # Excel sheet-name limit
        ws.append(["channel", "frame", "single human", "two humans",
                   "three humans", "four human", "fire", "touch"])

        for view in loaded:
            for i in range(n_frames):
                persons, fire = self._count_boxes_on_disk(view, i)
                onehot = [0, 0, 0, 0]
                if persons >= 1:
                    onehot[min(persons, 4) - 1] = 1
                touch = int(self.touch_labels.get(i, 0))
                ws.append([view.slot, i, *onehot, fire, touch])

        wb.save(xlsx_path)
        self.status_var.set(
            f"Exported labels.xlsx (sheet '{scene}') + contact_labels.csv "
            f"({n_frames} frames)."
        )

    # ═══════════════════════════════════════════════════════════════════════
    # SYNCHRONIZED NAVIGATION
    # ═══════════════════════════════════════════════════════════════════════

    def next_image(self) -> None:
        self._navigate(+1)

    def prev_image(self) -> None:
        self._navigate(-1)

    def _navigate(self, delta: int) -> None:
        loaded = [v for v in self.views if v.loaded]
        if not loaded:
            return

        for view in loaded:
            self._save_view(view, silent=True)

        master = loaded[0]
        master.current_index = (
            master.current_index + delta) % len(master.image_paths)
        target_filename = master.current_filename
        self._load_view_image(master)

        for view in loaded[1:]:
            matched = next(
                (i for i, p in enumerate(view.image_paths)
                 if os.path.basename(p) == target_filename),
                None,
            )
            view.current_index = (
                matched if matched is not None
                else min(master.current_index, len(view.image_paths) - 1)
            )
            self._load_view_image(view)

        self._update_counter()
        self._refresh_listbox()
        self._update_touch_button()

    # ═══════════════════════════════════════════════════════════════════════
    # ANNOTATION LIST
    # ═══════════════════════════════════════════════════════════════════════

    def delete_selected(self) -> None:
        sel = self.ann_listbox.curselection()
        if not sel:
            return
        view = self.views[self.active_slot]
        if not view.loaded:
            return
        anns = view.annotations.get(view.current_path, [])
        idx = sel[0]
        if idx < len(anns):
            del anns[idx]
            self._redraw_view(view)
            self._refresh_listbox()

    def _refresh_listbox(self) -> None:
        self.ann_listbox.delete(0, tk.END)
        view = self.views[self.active_slot]
        if not view.loaded:
            return
        for i, (cid, x1, y1, x2, y2) in enumerate(
            view.annotations.get(view.current_path, [])
        ):
            color  = CLASS_COLORS[cid % len(CLASS_COLORS)]
            label  = (self.class_names[cid]
                      if 0 <= cid < len(self.class_names) else str(cid))
            self.ann_listbox.insert(
                tk.END,
                f"[{i}] {label:<12}  ({x1},{y1})→({x2},{y2})",
            )
            self.ann_listbox.itemconfig(i, fg=color)

    # ═══════════════════════════════════════════════════════════════════════
    # CLASS SELECTOR HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _update_class_selector(self) -> None:
        menu = self.class_selector["menu"]
        menu.delete(0, "end")
        for name in self.class_names:
            menu.add_command(
                label=name,
                command=lambda n=name: self.class_name_var.set(n),
            )
        if self.class_names:
            self.class_name_var.set(self.class_names[0])

    def _update_swatch(self, *_) -> None:
        name = self.class_name_var.get()
        cid  = (self.class_names.index(name)
                if name in self.class_names else 0)
        color = CLASS_COLORS[cid % len(CLASS_COLORS)]
        self.swatch.config(bg=color, text=f"  {name or '—'}  ")

    # ═══════════════════════════════════════════════════════════════════════
    # PERSISTENT SETTINGS  (~/.annotator_settings.json)
    # ═══════════════════════════════════════════════════════════════════════

    def _load_settings(self) -> dict:
        if SETTINGS_PATH.exists():
            try:
                with open(SETTINGS_PATH, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"class_configurations": []}

    def _save_settings(self, settings: dict) -> None:
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except Exception as exc:
            print(f"[annotator] Warning – could not save settings: {exc}")

    def _push_class_config(self, configs: list[list[str]],
                           new_cfg: list[str]) -> list[list[str]]:
        deduped = [c for c in configs if c != new_cfg]
        return ([new_cfg] + deduped)[:MAX_CLASS_HISTORY]

    # ═══════════════════════════════════════════════════════════════════════
    # MISC HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _set_active_slot(self, slot: int) -> None:
        self.active_slot = slot
        self.active_label_var.set(f"ANNOTATIONS  ({CHAN_NAMES[slot]})")
        self._update_counter()
        self._refresh_listbox()
        for view in self.views:
            if view.canvas is None:
                continue
            if view.slot == slot:
                view.canvas.config(highlightthickness=2,
                                   highlightbackground=CAM_ACCENT[slot])
            else:
                view.canvas.config(highlightthickness=1,
                                   highlightbackground="#333")

    def _update_counter(self) -> None:
        view = self.views[self.active_slot]
        if view.loaded:
            self.counter_var.set(
                f"{CHAN_NAMES[view.slot]}:  "
                f"{view.current_index + 1} / {len(view.image_paths)}\n"
                f"{view.current_filename}"
            )
        else:
            n = sum(1 for v in self.views if v.loaded)
            self.counter_var.set(
                f"{n} channel(s) loaded" if n else "No channels loaded")

    def _on_scale_toggle(self) -> None:
        self._redraw_all()

    def _on_window_resize(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(60, self._redraw_all)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = ImageAnnotator(root)
    root.mainloop()
