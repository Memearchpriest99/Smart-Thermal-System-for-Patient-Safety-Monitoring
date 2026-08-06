"""
Thermal Monitoring - Multi-View Temporal Annotation Tool
Backend: FastAPI server with HDF5 + NPZ thermal array + video file support.
"""

import io
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
import sys         # <--- Add this
import asyncio     # <--- Add this
# --- ADD THIS BLOCK RIGHT AFTER IMPORTS ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# ------------------------------------------
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel



# ── HDF5 support (Zstd filter registered by hdf5plugin on import) ──────
try:
    import h5py
    import hdf5plugin          # registers Zstd filter ID 32015 globally
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("[WARNING] h5py / hdf5plugin not installed — HDF5 sessions will not load.")
    print("          Install with: pip install h5py hdf5plugin")

# ── Load config (Single Source of Truth) ──────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

DATASET_ROOT = Path(CONFIG["system"]["dataset_root_path"])
CSV_PATH = Path(CONFIG["system"]["export_csv_path"])
SERVER_PORT = CONFIG["system"]["server_port"]
VIDEO_EXT = set(CONFIG.get("video", {}).get("supported_extensions", [".mp4"]))
NPZ_EXT = {".npz"}
H5_EXT  = {".h5", ".hdf5"}
SOURCE_FPS = CONFIG.get("video", {}).get("source_fps", 25)
DISPLAY_FPS = CONFIG.get("video", {}).get("display_fps", 8)

# Mutable dataset root — can be changed at runtime via /api/set_dataset_path
_dataset_root = DATASET_ROOT

def get_dataset_root() -> Path:
    return _dataset_root

# Colormap config: which matplotlib colormap to use for thermal rendering
COLORMAP_NAME = CONFIG.get("video", {}).get("thermal_colormap", "inferno")

# ── Ensure CSV exists with correct schema ─────────────────────────────
CSV_COLUMNS = ["Room_ID", "Date", "Start_Time", "End_Time", "Event_Class", "Event_Class_ID", "Cameras", "Timestamp"]

def _ensure_csv():
    """Create CSV if missing, or migrate an existing CSV to the current schema."""
    if not CSV_PATH.exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)
        return

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        backup = CSV_PATH.with_suffix(".csv.bak")
        CSV_PATH.rename(backup)
        print(f"[WARNING] Could not read {CSV_PATH.name} — backed up to {backup.name} and created fresh file.")
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)
        return

    changed = False
    defaults = {
        "Room_ID":        "",
        "Date":           "",
        "Start_Time":     "00:00:00.000",
        "End_Time":       "00:00:00.000",
        "Event_Class":    "",
        "Event_Class_ID": 0,
        "Cameras":        "",
        "Timestamp":      "",
    }
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = defaults.get(col, "")
            changed = True
            print(f"[MIGRATION] Added missing column '{col}' to {CSV_PATH.name}")

    # Drop columns no longer in schema (e.g. Source_FPS)
    extra_cols = [c for c in df.columns if c not in CSV_COLUMNS]
    if extra_cols:
        df = df.drop(columns=extra_cols)
        changed = True
        print(f"[MIGRATION] Removed obsolete columns {extra_cols} from {CSV_PATH.name}")

    df = df[CSV_COLUMNS]

    if changed:
        df.to_csv(CSV_PATH, index=False)
        print(f"[MIGRATION] {CSV_PATH.name} updated to current schema.")

_ensure_csv()

# ── Colormap generation (no matplotlib dependency) ────────────────────
# Pre-baked inferno colormap LUT (256 entries, RGB)
# Generated from matplotlib's inferno — works without matplotlib installed
def _generate_inferno_lut():
    """Generate a 256-entry inferno colormap lookup table."""
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(COLORMAP_NAME)
        return np.array([cmap(i / 255.0)[:3] for i in range(256)], dtype=np.float32)
    except ImportError:
        # Fallback: simple black-body approximation (black → red → yellow → white)
        lut = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            t = i / 255.0
            lut[i] = [
                min(1.0, t * 2.5),
                min(1.0, max(0, (t - 0.4) * 2.5)),
                min(1.0, max(0, (t - 0.7) * 3.3)),
            ]
        return lut

COLORMAP_LUT = _generate_inferno_lut()


def thermal_to_png(array: np.ndarray, temp_min: float = None, temp_max: float = None,
                   gamma: float = 1.0, colormap: str = None) -> bytes:
    """Convert a 2D thermal array to a colorized PNG image.

    Args:
        array: 2D numpy array of temperature values
        temp_min: Override min temp for normalization (clip below)
        temp_max: Override max temp for normalization (clip above)
        gamma: Gamma correction (>1 = darken midtones, <1 = brighten)
        colormap: Override colormap name (None = use default)

    Returns:
        PNG image bytes
    """
    arr = array.astype(np.float32)

    # Temperature clipping
    vmin = temp_min if temp_min is not None else float(np.nanmin(arr))
    vmax = temp_max if temp_max is not None else float(np.nanmax(arr))
    if vmax - vmin < 1e-6:
        normalized = np.zeros_like(arr, dtype=np.float32)
    else:
        normalized = np.clip((arr - vmin) / (vmax - vmin), 0, 1)

    # Gamma correction
    if gamma != 1.0 and gamma > 0:
        normalized = np.power(normalized, 1.0 / gamma)

    norm_u8 = (normalized * 255).astype(np.uint8)

    # Colormap selection
    if colormap and colormap != COLORMAP_NAME:
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap)
            lut = np.array([cmap(i / 255.0)[:3] for i in range(256)], dtype=np.float32)
        except Exception:
            lut = COLORMAP_LUT
    else:
        lut = COLORMAP_LUT

    rgb = (lut[norm_u8] * 255).astype(np.uint8)

    # Encode to PNG using raw Python (no PIL dependency)
    try:
        from PIL import Image
        img = Image.fromarray(rgb, mode="RGB")
        # Upscale for visibility (80x62 is tiny)
        scale = max(1, 480 // max(rgb.shape[0], rgb.shape[1]))
        if scale > 1:
            img = img.resize(
                (rgb.shape[1] * scale, rgb.shape[0] * scale),
                Image.NEAREST  # preserve pixel boundaries
            )
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        return buf.getvalue()
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Pillow is required for NPZ rendering. Install: pip install Pillow"
        )


# ── NPZ Session Index ────────────────────────────────────────────────
# For NPZ sessions, we need to build an index: camera_name → [frame0, frame1, ...]
# Each "camera" is a subdirectory or a prefix pattern in the session folder.

@lru_cache(maxsize=32)
def scan_npz_session(session_path_str: str) -> dict:
    """Scans session, detects 3D arrays, and builds a global-to-local frame map."""
    session_path = Path(session_path_str)
    raw_files = {}

    # Check for subdirectory layout
    subdirs = sorted([d for d in session_path.iterdir() if d.is_dir() and not d.name.startswith(".")])

    if subdirs:
        for subdir in subdirs:
            cam_name = subdir.name
            npz_files = sorted([f for f in subdir.iterdir() if f.suffix.lower() == ".npz"])
            if npz_files:
                raw_files[cam_name] = npz_files
    else:
        # Flat layout
        all_npz = sorted([f for f in session_path.iterdir() if f.suffix.lower() == ".npz"])
        if not all_npz:
            return {"cameras": [], "frame_count": 0, "frames": {}, "duration": 0}

        first_name = all_npz[0].stem
        if "_" in first_name:
            for f in all_npz:
                cam_name = f.stem.split("_", 1)[0]
                if cam_name not in raw_files:
                    raw_files[cam_name] = []
                raw_files[cam_name].append(f)
        else:
            raw_files["cam1"] = all_npz

    # Build the Frame Map (Global Frame -> (Filename, Local Slice Index))
    frames_by_cam = {}
    for cam_name, files in raw_files.items():
        frame_map = []
        for f in files:
            try:
                data = np.load(str(f))
                arr = data[list(data.keys())[0]]
                if len(arr.shape) == 3:  # It's a 3D block of frames
                    num_frames = arr.shape[0]
                    for i in range(num_frames):
                        rel_path = f.name if f.parent == session_path else f"{f.parent.name}/{f.name}"
                        frame_map.append((rel_path, i))
                else:  # It's a 2D single frame
                    rel_path = f.name if f.parent == session_path else f"{f.parent.name}/{f.name}"
                    frame_map.append((rel_path, 0))
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
        frames_by_cam[cam_name] = frame_map

    max_frames = max((len(v) for v in frames_by_cam.values()), default=0)
    duration = max_frames / SOURCE_FPS if SOURCE_FPS > 0 else 0

    return {
        "cameras": sorted(frames_by_cam.keys()),
        "frame_count": max_frames,
        "frames": frames_by_cam,
        "duration": round(duration, 3),
    }


# ── NPZ frame cache ──────────────────────────────────────────────────
@lru_cache(maxsize=256)
def load_npz_array(file_path: str) -> np.ndarray:
    """Load and return the thermal array from an NPZ file (cached)."""
    p = Path(file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"NPZ file not found: {file_path}")

    data = np.load(str(p))
    # NPZ files can have multiple arrays — take the first one
    keys = list(data.keys())
    if not keys:
        raise HTTPException(status_code=400, detail=f"NPZ file is empty: {file_path}")
    return data[keys[0]]


# ── HDF5 Session Scanner ──────────────────────────────────────────────
# File layout written by storage_stage.cpp:
#   <session_path>/cam_<N>/<HHMMSS>_seq<start>_seq<end>.h5
# Each .h5 file has:
#   /raw_metrics  uint16  [N_frames, CAM_H, CAM_W]   (62 × 80)
#   /timestamps   uint64  [N_frames]
#   /seqs         uint64  [N_frames]
#   attrs on raw_metrics: raw_to_celsius_scale, kelvin_offset, fps, camera_id, room_id

@lru_cache(maxsize=32)
def scan_h5_session(session_path_str: str) -> dict:
    """Scan a session directory for HDF5 files.

    Mirrors scan_npz_session() exactly — returns the same dict shape so
    the rest of the API can treat H5 and NPZ sessions identically.

    Returns:
        {
            "cameras":     [cam_name, ...],
            "frame_count": <total source frames>,
            "frames":      { cam_name: [(rel_path, local_frame_idx), ...] },
            "duration":    <seconds>,
            "source_fps":  <fps from file attributes or config>,
        }
    """
    if not HDF5_AVAILABLE:
        return {"cameras": [], "frame_count": 0, "frames": {}, "duration": 0, "source_fps": SOURCE_FPS}

    session_path = Path(session_path_str)
    frames_by_cam: dict = {}
    base_time = 0.0

    # Storage stage writes one cam_N/ subdir per camera
    subdirs = sorted([d for d in session_path.iterdir() if d.is_dir() and not d.name.startswith(".")])

    if subdirs:
        for subdir in subdirs:
            cam_name = subdir.name   # e.g. "cam_0"
            h5_files = sorted([f for f in subdir.iterdir() if f.suffix.lower() in H5_EXT])
            if h5_files:
                frames_by_cam[cam_name] = h5_files
    else:
        # Flat layout fallback (non-standard but handle gracefully)
        all_h5 = sorted([f for f in session_path.iterdir() if f.suffix.lower() in H5_EXT])
        if all_h5:
            frames_by_cam["cam_0"] = all_h5

    if not frames_by_cam:
        return {"cameras": [], "frame_count": 0, "frames": {}, "duration": 0, "source_fps": SOURCE_FPS}

    # Build frame map: cam_name → [(rel_path_str, local_frame_idx), ...]
    # Also read fps attribute from the first file we can open
    file_fps = SOURCE_FPS
    frame_map: dict = {}

    for cam_name, files in frames_by_cam.items():
        cam_frames = []
        for f in files:
            if base_time == 0.0:
                m = re.match(r'^(\d{2})(\d{2})(\d{2})_', f.name)
                if m:
                    base_time = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            try:
                with h5py.File(str(f), "r") as hf:
                    ds = hf["frames"]
                    n_frames = ds.shape[0]
                    # Read fps attribute once (same across all files in a session)
                    if "fps" in ds.attrs:
                        file_fps = int(ds.attrs["fps"])
                    rel = f"{f.parent.name}/{f.name}"
                    for i in range(n_frames):
                        cam_frames.append((rel, i))
            except Exception as e:
                print(f"[H5] Error scanning {f}: {e}")
        frame_map[cam_name] = cam_frames

    max_frames = max((len(v) for v in frame_map.values()), default=0)
    duration = max_frames / file_fps if file_fps > 0 else 0

    return {
        "cameras":     sorted(frame_map.keys()),
        "frame_count": max_frames,
        "frames":      frame_map,
        "duration":    round(duration, 3),
        "source_fps":  file_fps,
        "start_time_offset": base_time    # <--- ADD THIS
    }


# H5 conversion params cache: file_path → (scale, offset)
@lru_cache(maxsize=64)
def _h5_conversion_params(file_path: str) -> tuple:
    """Read raw_to_celsius_scale and kelvin_offset from HDF5 file attributes."""
    try:
        with h5py.File(file_path, "r") as hf:
            ds = hf["frames"]
            scale  = float(ds.attrs.get("raw_to_celsius_scale", 0.1))
            offset = float(ds.attrs.get("kelvin_offset", 273.15))
            return scale, offset
    except Exception:
        return 0.1, 273.15  # safe defaults matching hdf5_writer.hpp


@lru_cache(maxsize=10)
def load_h5_file(file_path: str) -> np.ndarray:
    """Load raw_metrics from an HDF5 file and return as float32 Celsius.

    Conversion: temp_c = raw_uint16 * scale - offset
    (scale=0.1, offset=273.15 by default, stored as attributes)
    This is lossless — the uint16 raw values are the sensor's native format;
    Celsius is recovered exactly with no rounding beyond float32 precision.
    """
    p = Path(file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"HDF5 file not found: {file_path}")

    scale, offset = _h5_conversion_params(file_path)

    try:
        with h5py.File(file_path, "r") as hf:
            raw = hf["frames"][:]          # uint16 [N, H, W]
        celsius = raw.astype(np.float32) * np.float32(scale) - np.float32(offset)
        return celsius                           # float32 [N, H, W]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read HDF5 file: {e}")


# ── FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(title="Thermal Annotation Tool")


# ── Pydantic Models ───────────────────────────────────────────────────
class LabelPayload(BaseModel):
    room_id: str
    date: str
    start_time: float
    end_time: float
    event_class: str
    cameras: list[str]


class DeletePayload(BaseModel):
    index: int
    room_id: str
    date: str


# ── API Routes ────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config():
    return JSONResponse(content=CONFIG)


# ── Browse / Dataset Path ─────────────────────────────────────────────

class SetPathPayload(BaseModel):
    path: str

@app.get("/api/browse_dirs")
async def browse_dirs(path: str = ""):
    """List directories at the given path for the folder browser.
    Returns parent path and list of child directories.
    """
    try:
        if not path:
            # Start from the current dataset root's parent, or home
            p = get_dataset_root().parent.resolve()
        else:
            p = Path(path).resolve()

        if not p.exists() or not p.is_dir():
            p = Path.home()

        children = sorted([
            {"name": d.name, "path": str(d)}
            for d in p.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ], key=lambda x: x["name"].lower())

        return JSONResponse(content={
            "current": str(p),
            "parent": str(p.parent) if p != p.parent else None,
            "dirs": children,
        })
    except PermissionError:
        return JSONResponse(content={
            "current": str(Path.home()),
            "parent": str(Path.home().parent),
            "dirs": [],
            "error": "Permission denied"
        })


@app.post("/api/set_dataset_path")
async def set_dataset_path(payload: SetPathPayload):
    """Change the dataset root directory at runtime.
    Also updates config.json so it persists across restarts.
    """
    global _dataset_root
    new_path = Path(payload.path).resolve()

    if not new_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {payload.path}")
    if not new_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {payload.path}")

    _dataset_root = new_path

    # Update config.json on disk
    CONFIG["system"]["dataset_root_path"] = str(new_path)
    with open(CONFIG_PATH, "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Remount static files for the new path
    for route in app.routes:
        if hasattr(route, 'name') and route.name == 'dataset':
            app.routes.remove(route)
            break
    app.mount("/Dataset_Root", StaticFiles(directory=str(new_path)), name="dataset")

    # Clear NPZ caches since path changed
    scan_npz_session.cache_clear()
    load_npz_file.cache_clear()
    scan_h5_session.cache_clear()
    load_h5_file.cache_clear()
    _h5_conversion_params.cache_clear()

    return JSONResponse(content={
        "status": "ok",
        "path": str(new_path),
    })


@app.get("/api/dataset_path")
async def get_dataset_path():
    """Return the current dataset root path."""
    return JSONResponse(content={"path": str(get_dataset_root().resolve())})


# ── Filter Presets ────────────────────────────────────────────────────

FILTER_PRESETS_PATH = Path(__file__).parent / "filter_presets.json"

@app.get("/api/filter_presets")
async def get_filter_presets():
    """Load saved filter presets."""
    if FILTER_PRESETS_PATH.exists():
        with open(FILTER_PRESETS_PATH) as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(content={"presets": {}})

class FilterPresetPayload(BaseModel):
    name: str
    settings: dict

@app.post("/api/filter_presets")
async def save_filter_preset(payload: FilterPresetPayload):
    """Save a named filter preset."""
    presets = {}
    if FILTER_PRESETS_PATH.exists():
        with open(FILTER_PRESETS_PATH) as f:
            presets = json.load(f).get("presets", {})
    presets[payload.name] = payload.settings
    with open(FILTER_PRESETS_PATH, "w") as f:
        json.dump({"presets": presets}, f, indent=2)
    return JSONResponse(content={"status": "saved", "presets": presets})

@app.delete("/api/filter_presets/{name}")
async def delete_filter_preset(name: str):
    """Delete a named filter preset."""
    presets = {}
    if FILTER_PRESETS_PATH.exists():
        with open(FILTER_PRESETS_PATH) as f:
            presets = json.load(f).get("presets", {})
    presets.pop(name, None)
    with open(FILTER_PRESETS_PATH, "w") as f:
        json.dump({"presets": presets}, f, indent=2)
    return JSONResponse(content={"status": "deleted", "presets": presets})


@app.get("/api/rooms")
async def get_rooms():
    if not get_dataset_root().exists():
        return JSONResponse(content={"rooms": []})
    rooms = sorted([
        d.name for d in get_dataset_root().iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    return JSONResponse(content={"rooms": rooms})


@app.get("/api/dates/{room_id}")
async def get_dates(room_id: str):
    room_path = get_dataset_root() / room_id
    if not room_path.exists():
        raise HTTPException(status_code=404, detail=f"Room '{room_id}' not found")
    dates = sorted([
        d.name for d in room_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    return JSONResponse(content={"dates": dates})


@app.get("/api/session/{room_id}/{date}")
async def get_session(room_id: str, date: str):
    session_path = get_dataset_root() / room_id / date
    if not session_path.exists():
        raise HTTPException(status_code=404, detail=f"Session not found: {room_id}/{date}")

    video_files = sorted([f.name for f in session_path.iterdir()
                          if f.is_file() and f.suffix.lower() in VIDEO_EXT])
    has_h5  = HDF5_AVAILABLE and any(f.suffix.lower() in H5_EXT
                                     for f in session_path.rglob("*") if f.is_file())
    has_npz = any(f.suffix.lower() == ".npz"
                  for f in session_path.rglob("*") if f.is_file())

    if video_files:
        return JSONResponse(content={
            "type": "video",
            "cameras": [f.rsplit(".", 1)[0] for f in video_files],
            "files": video_files,
            "frame_count": None,
            "duration": None,
            "source_fps": SOURCE_FPS,
        })
    elif has_h5:
        # HDF5 is preferred over NPZ — it is the primary recording format
        info = scan_h5_session(str(session_path))
        return JSONResponse(content={
            "type":        "h5",
            "cameras":     info["cameras"],
            "files":       info["frames"],
            "frame_count": info["frame_count"],
            "duration":    info["duration"],
            "source_fps":  info["source_fps"],
            "start_time_offset": info.get("start_time_offset", 0.0),
        })
    elif has_npz:
        npz_info = scan_npz_session(str(session_path))
        return JSONResponse(content={
            "type":        "npz",
            "cameras":     npz_info["cameras"],
            "files":       npz_info["frames"],
            "frame_count": npz_info["frame_count"],
            "duration":    npz_info["duration"],
            "source_fps":  SOURCE_FPS,
        })
    else:
        return JSONResponse(content={
            "type": "empty", "cameras": [], "files": [],
            "frame_count": 0, "duration": 0, "source_fps": SOURCE_FPS,
        })


# Keep backward compat
@app.get("/api/videos/{room_id}/{date}")
async def get_videos(room_id: str, date: str):
    """Legacy endpoint — redirects to session info (video files only)."""
    session_path = get_dataset_root() / room_id / date
    if not session_path.exists():
        raise HTTPException(status_code=404, detail=f"Session not found: {room_id}/{date}")
    videos = sorted([
        f.name for f in session_path.iterdir()
        if f.suffix.lower() in VIDEO_EXT
    ])
    return JSONResponse(content={"videos": videos})

@lru_cache(maxsize=10)  # Reduced to 10 because 3D arrays take up much more RAM
def load_npz_file(file_path: str) -> np.ndarray:
    """Load and return the entire thermal array from an NPZ file (cached)."""
    p = Path(file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"NPZ file not found: {file_path}")

    data = np.load(str(p))
    keys = list(data.keys())
    if not keys:
        raise HTTPException(status_code=400, detail=f"NPZ file is empty: {file_path}")
    return data[keys[0]]

@app.get("/api/npz_frame/{room_id}/{date}/{cam_name}/{frame_idx}")
async def get_npz_frame(
    room_id: str, date: str, cam_name: str, frame_idx: int,
    temp_min: float = None, temp_max: float = None,
    gamma: float = 1.0, colormap: str = None,
):
    """Serve a single NPZ frame as a colorized PNG.
    
    Optional filter query params:
        temp_min, temp_max: clip temperature range for normalization
        gamma: gamma correction (default 1.0)
        colormap: override colormap name
    """
    session_path = get_dataset_root() / room_id / date
    npz_info = scan_npz_session(str(session_path))

    if cam_name not in npz_info["frames"]:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_name}' not found")

    file_list = npz_info["frames"][cam_name]
    if frame_idx < 0 or frame_idx >= len(file_list):
        raise HTTPException(status_code=404, detail=f"Frame out of range")

    file_rel_path, local_idx = file_list[frame_idx]
    full_path = session_path / file_rel_path

    arr = load_npz_file(str(full_path))
    if len(arr.shape) == 3:
        frame_data = arr[local_idx]
    else:
        frame_data = arr

    png_bytes = thermal_to_png(
        frame_data,
        temp_min=temp_min,
        temp_max=temp_max,
        gamma=gamma,
        colormap=colormap,
    )

    # No cache when filters are active (params change the output)
    cache_header = "no-cache" if any([temp_min, temp_max, gamma != 1.0, colormap]) else "public, max-age=3600"

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Cache-Control": cache_header},
    )


@app.get("/api/h5_frame/{room_id}/{date}/{cam_name}/{frame_idx}")
async def get_h5_frame(
    room_id: str, date: str, cam_name: str, frame_idx: int,
    temp_min: float = None, temp_max: float = None,
    colormap: str = None,
):
    """Serve a single HDF5 frame as a colorized PNG.

    Reads uint16 raw sensor values from the .h5 file, converts to float32
    Celsius using the attributes stored by hdf5_writer.cpp, then renders
    via thermal_to_png().

    frame_idx is a SOURCE frame index (not a display frame index).
    """
    if not HDF5_AVAILABLE:
        raise HTTPException(status_code=503,
                            detail="HDF5 support not available — install h5py and hdf5plugin")

    session_path = get_dataset_root() / room_id / date
    h5_info = scan_h5_session(str(session_path))

    if cam_name not in h5_info["frames"]:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_name}' not found in H5 session")

    file_list = h5_info["frames"][cam_name]
    if frame_idx < 0 or frame_idx >= len(file_list):
        raise HTTPException(status_code=404,
                            detail=f"Frame {frame_idx} out of range (session has {len(file_list)} frames)")

    file_rel_path, local_idx = file_list[frame_idx]
    full_path = session_path / file_rel_path

    # load_h5_file returns float32 Celsius [N, H, W]
    arr = load_h5_file(str(full_path))
    frame_data = arr[local_idx]   # shape [H, W] = [62, 80]

    png_bytes = thermal_to_png(
        frame_data,
        temp_min=temp_min,
        temp_max=temp_max,
        colormap=colormap,
    )

    cache_header = "no-cache" if any([temp_min, temp_max, colormap]) else "public, max-age=3600"

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Cache-Control": cache_header},
    )


def _seconds_to_hhmmss(t: float) -> str:
    """Convert a float seconds value to HH:MM:SS.mmm string."""
    t = max(0.0, t)
    h = int(t) // 3600
    m = (int(t) % 3600) // 60
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


@app.post("/api/save_label")
async def save_label(payload: LabelPayload):
    if payload.start_time >= payload.end_time:
        raise HTTPException(status_code=400, detail="Start_Time must be less than End_Time")

    duration = payload.end_time - payload.start_time
    if duration < 1.0:
        raise HTTPException(status_code=400, detail=f"Label too short: {duration:.3f}s (minimum 1.0s)")

    # Resolve event class name and numeric ID from config
    event_classes = CONFIG.get("annotation", {}).get("event_classes", {})
    class_id_str = str(payload.event_class)
    class_name = event_classes.get(class_id_str, class_id_str)
    try:
        class_id = int(class_id_str)
    except ValueError:
        # payload sent the name directly — find the matching ID
        class_id = next((int(k) for k, v in event_classes.items() if v == class_id_str), 0)

    # Strip "cam_" prefix so cameras are stored as "0|1|2"
    cam_nums = [c.replace("cam_", "") for c in payload.cameras]

    now_utc = datetime.now(timezone.utc)
    new_row = pd.DataFrame([{
        "Room_ID":        payload.room_id,
        "Date":           payload.date,
        "Start_Time":     _seconds_to_hhmmss(payload.start_time),
        "End_Time":       _seconds_to_hhmmss(payload.end_time),
        "Event_Class":    class_name,
        "Event_Class_ID": class_id,
        "Cameras":        "|".join(cam_nums),
        "Timestamp":      now_utc.isoformat(),
    }])

    new_row.to_csv(CSV_PATH, mode="a", header=False, index=False)
    return await get_labels(payload.room_id, payload.date)


@app.get("/api/labels/{room_id}/{date}")
async def get_labels(room_id: str, date: str):
    df = pd.read_csv(CSV_PATH)
    session_df = df[(df["Room_ID"] == room_id) & (df["Date"] == date)]
    labels = session_df.to_dict(orient="records")
    return JSONResponse(content={"labels": labels})


@app.post("/api/delete_label")
async def delete_label(payload: DeletePayload):
    """Delete a label by its local session index.

    The frontend sends the index within the filtered session view (0, 1, 2...).
    We map it to the actual global CSV row index before dropping, so labels
    from other sessions are never touched.
    """
    df = pd.read_csv(CSV_PATH)

    # Filter to the specific session — preserves original global indices
    session_mask = (df["Room_ID"] == payload.room_id) & (df["Date"] == payload.date)
    session_indices = df[session_mask].index.tolist()

    if payload.index < 0 or payload.index >= len(session_indices):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid index {payload.index} for session "
                   f"{payload.room_id}/{payload.date} ({len(session_indices)} labels)"
        )

    # Map local session index → global CSV row index
    global_index = session_indices[payload.index]
    df = df.drop(index=global_index).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    return await get_labels(payload.room_id, payload.date)


# ── Serve static files ────────────────────────────────────────────────
if get_dataset_root().exists():
    app.mount("/Dataset_Root", StaticFiles(directory=str(get_dataset_root())), name="dataset")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend not built yet.</h1>")
    return HTMLResponse(index_path.read_text(encoding="utf-8")) # <--- Added encoding


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"\n{'='*60}")
    print(f"  Thermal Annotation Tool")
    print(f"  Server: http://localhost:{SERVER_PORT}")
    print(f"  Dataset: {get_dataset_root().resolve()}")
    print(f"  Labels:  {CSV_PATH.resolve()}")
    print(f"  Formats: Video {VIDEO_EXT} + NPZ thermal arrays")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="127.0.0.1", port=SERVER_PORT)
