# CLAUDE.md

Guidance for Claude Code when working in this folder (the rebuilt image
annotator for the **Waveshare** thermal dataset).

## What this is

A single-file Tkinter tool (`annotator.py`, one class `ImageAnnotator`) for
drawing YOLO bounding boxes across the three thermal camera channels of the
Smart Thermal System dataset. It is a rebuild of the original MLX90640-era
annotator, kept visually identical, fitted to the reorganized Waveshare layout
and with four additions (see below).

It expects the on-disk layout produced by
`../scripts/reorganize_waveshare.py`:

```
<session>/
  ch0_frames/frame_00000.png ...   (+ frame_00000.txt written by this tool)
  ch1_frames/ ...   ch2_frames/ ...
  ch0_raw_data.npz  (key 'frames', (N, 62, 80) degC) — drives the recommender
  ch1_raw_data.npz  ch2_raw_data.npz
  classes.txt       ("fire\nperson\n")
  meta.json
```

## Commands

```bash
# Run from source (recommended — the recommender imports the project package)
python annotator.py

# Dependencies
pip install Pillow numpy openpyxl opencv-python
```

The recommender reuses the project's own detection code
(`thermal_algorithms.preprocessing.TatenoPipeline` +
`thermal_algorithms.human_detection.AdaptiveThresholdDetector`). The repo root
is added to `sys.path` at startup (`Path(__file__).parent.parent`). If those
imports fail (e.g. opencv missing), **person** suggestions disable gracefully;
**fire** suggestions and everything else keep working.

```bash
# Standalone exe (optional). Because the recommender imports the project
# package, the spec bundles thermal_algorithms + cv2 as hiddenimports. Run from
# the repo root so the package is discoverable at build time:
pyinstaller image_annotator/ImageAnnotator.spec
```

## Architecture

### Display is rendered from the thermal `.npz` (not the PNGs)

`_make_display_image` / `render_thermal_frame` turn the raw `(62, 80)` °C frame
into the on-screen RGB image:

```
thermal frame  →  [optional Gaussian blur, native grid]  →  bicubic resize to
WxH  →  per-frame min-max normalise  →  inferno colormap (baked LUT)
```

- **Resize W×H** — sidebar entries + `Apply` (`_on_resolution_change`). Default
  320×240 (matches the old previews). This is also the annotation pixel space,
  so changing it rescales in-memory boxes; on-disk YOLO is normalised and
  unaffected.
- **Gaussian (σ)** — checkbox + sigma entry (`_on_gaussian_change`). Applied on
  the native 62×80 grid **before** resizing. Display-only — it does NOT affect
  saved labels or the recommender.
- **Colormap** — matplotlib `inferno`, baked as a 256×3 LUT constant
  (`INFERNO_LUT`), so display needs neither matplotlib nor cv2 (PIL + NumPy
  only). The `chX_frames/*.png` previews are now used only for the `.txt` save
  location, frame count, and filenames.

### Coordinate spaces

| Space | Description | Where used |
|---|---|---|
| **Canvas** | Pixel on the Tk `Canvas` | mouse events |
| **Image** | Pixel in the rendered display image (= current resize W×H) | `annotations` storage |
| **YOLO-normalised** | [0, 1] of image dims | `.txt` files on disk |
| **Thermal** | Pixel in the `(62, 80)` °C `.npz` frame | rendering + recommender |

Recommender suggestion boxes are computed in thermal space then scaled to image
space (`sx = orig_w/80`, `sy = orig_h/62`).

### The four additions over the original

1. **Single per-frame TOUCH button** (`toggle_touch`, hotkey `T`). One binary
   contact label for all three views at once, stored in `self.touch_labels`
   `{frame_index: 0/1}`. Not a YOLO box.
2. **Recommender** (`Suggest` toggle, hotkey `S`; `Accept Suggestions`, hotkey
   `A`). Ghost boxes (dashed, dimmed) the user commits explicitly.
   - **Fire** — `suggest_fire()`: connected pixels `> T_FIRE_C` (50 °C). No
     human/empty frame in the dataset reaches 50 °C, so this is ~zero-FP even
     at `FIRE_AREA_MIN = 1` (catches cigarettes).
   - **Person** — `_person_boxes()`: Tateno background subtraction (fitted on a
     sibling `empty_room`/`calibrate_room` recording, per `BACKGROUND_SESSIONS`)
     → `AdaptiveThresholdDetector` on the residual. Local thresholding +
     background removal is what stops warm static equipment from reading as a
     person. Tuned (`PERSON_DET_KWARGS`) so empty rooms yield 0 detections.
     Splits/merges happen (1 person → up to ~2 boxes) — fine for reviewed ghosts.
3. **Auto xlsx** (`export_labels`, `📊 Export Labels`). Writes a per-session
   sheet to `labels.xlsx` with columns
   `channel, frame, single human, two humans, three humans, four human, fire, touch`.
   Human one-hot is **derived from the count of class-1 boxes** per channel/frame;
   `fire` from any class-0 box; `touch` replicated across the 3 channel rows.
4. **Auto contact CSV** — `export_labels` also writes per-session
   `contact_labels.csv` (`frame_idx,contact`) read by the training stack.

### Recommender tuning

All thresholds are module-level constants near the top of `annotator.py`
(`T_FIRE_C`, `FIRE_AREA_MIN`, `BACKGROUND_SESSIONS`, `PERSON_DET_KWARGS`). They
were tuned against the reorganized Waveshare sessions (empty/calibrate → 0
detections; every occupied scene detects). Re-tune here as more data arrives.

### Annotation lifecycle (unchanged)

Drawn → stored in `view.annotations[path]` (image px) → painted via
`_paint_box_on` → saved as YOLO `.txt` (auto on Prev/Next, or `Ctrl+S`) →
reloaded by `_load_annotations_from_disk` on first visit.
