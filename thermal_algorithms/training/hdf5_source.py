"""HDF5-backed session reader for the synthetic (``synth_room_*``) and real
``room-1`` recordings.

These sessions store raw thermal frames as chunked ``.h5`` files (one per
~5-minute window, per camera) under ``<date>/cam_{0,1,2}/*.h5`` — not the
``ch{N}_raw_data.npz`` layout ``DatasetIndex`` expects.

Two schemas have been observed on disk:

* **Synthetic** (``synth_room_*``): keys ``frames`` (uint16), ``timestamps``
  (float64, absolute Unix-epoch seconds), ``sequence_numbers`` (uint64,
  monotonic), plus a ``metadata`` group with attrs (``cam_id``, ``fps``,
  coarse whole-chunk ``event_class``, ``room_id``, ``scene_seed``,
  ``synthetic``, ``pipeline_version``). Verified clean: inter-frame spacing
  is exactly ``1/24`` s, matching ``fps=24.0``.
* **Real hardware** (``room-1``): keys ``frames`` (uint16), ``seqs``
  (uint64), ``timestamps`` (uint64) — no ``metadata`` group. **Confirmed
  broken for the whole session, not just some chunks**: checked the first,
  one middle, and the last chunk of the ``2026-06-30`` session. The first and
  middle chunks have ``seqs``/``timestamps`` entirely zero. The *last* chunk
  looked populated (non-zero, internally monotonic) but turned out to be
  bogus too — concatenated with the other (filename-derived) chunks it lands
  on the year 2319, so whatever clock produced it is not Unix-epoch seconds.
  Frame pixel data itself is unaffected throughout.

Per user decision, any chunk whose ``timestamps`` don't land in a plausible
epoch range (see ``_PLAUSIBLE_EPOCH_RANGE`` — this catches both the all-zero
chunks and the bogus-large-value chunk uniformly) falls back to a wall-clock
time derived from the chunk filename (``HHMMSS_seq<start>_seq<end>.h5`` — the
leading ``HHMMSS`` is each chunk's real capture start time; e.g. the very
first chunk of ``room-1``'s ``2026-06-30`` session, ``121624_...``, starts at
exactly the same ``12:16:24.000`` timestamp as that session's first
``labels.csv`` row — confirmed) plus ``local_frame_idx / fallback_fps``.

**Important refinement**: ``HDF5CameraSession`` does NOT trust every broken
chunk's own filename independently. The session's trailing (typically
short/partial) chunk was found to have a filename-declared start time
inconsistent with the otherwise-exact, regular spacing of every other chunk —
sequence numbers show it genuinely is the next contiguous chunk, so its
filename time is simply wrong, not a real gap. Trusting it per-chunk made the
reconstructed timeline jump backwards at that boundary. Instead, each broken
chunk chains off the *previous* chunk's own end time (whether that previous
chunk was itself valid or filename/chain-derived); a chunk's own filename is
only used as the anchor when there is no prior chunk to chain from (normally
just the session's first chunk, if it happens to be broken).

``fallback_fps`` is NOT hardcoded: the synthetic data's ``file_chunk_s=300``
config implies 7200 frames/chunk = 24 fps, but this does not hold for real
hardware — room-1's chunk filenames are consistently exactly 600s (10 min)
apart while each full chunk still holds 7200 frames, implying 12 fps, not
24 (confirmed across all 18 full-chunk gaps in the ``2026-06-30`` session; a
naive 24fps assumption made the reconstructed timeline non-monotonic at
every chunk boundary). ``HDF5CameraSession`` therefore infers fps per
session from consecutive chunk filenames' wall-clock spacing
(``infer_fps_from_chunk_spacing``) whenever a fallback is needed and no
explicit ``fallback_fps`` is given — do not assume a fixed rate holds across
sessions. This is still an approximation, not frame-exact ground truth;
``used_fallback`` / ``HDF5CameraSession.used_fallback`` surfaces where it was
applied so downstream code (and the Phase 1 data-quality write-up) can flag
it rather than silently presenting it as measured.

Room-level event ground truth lives separately in a sibling ``labels.csv``
(``Room_ID,Date,Start_Time,End_Time,Event_Class,Event_Class_ID,Cameras,
Timestamp``) — see ``label_join.py`` for turning that into per-frame labels.
There is no bounding-box ground truth anywhere in this source (only
``waveshare_work`` has real YOLO boxes).

Scale note: a full-day synthetic camera recording is ~1.45M frames — loading
an entire session's frames into one in-memory array (~27 GiB at 62x80
float32) is not viable. ``HDF5CameraSession`` therefore loads lazily,
one chunk at a time, with a small LRU cache of decoded chunks — not the
whole session at once. Use ``load_frame(i)`` / iterate; reserve
``load_frames()`` for sessions you know are small (e.g. room-1's ~130k
frames/day is still ~2.4 GiB per camera — manageable but not free).
"""

from __future__ import annotations

import bisect
import datetime as _dt
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from thermal_algorithms.core.types import Frame

RAW_TO_CELSIUS_SCALE: float = 100.0
"""uint16 raw sensor count -> degrees Celsius: temp_C = raw / 100.

Verified two ways: (1) a sample chunk's frame-0 values (~2953-3032) map to
~29.5-30.3 degC, a physically plausible room temperature; (2) the generating
simulator's own ``config.json`` clips its thermal-preview rendering to
``thermal_clip_c: [15.0, 45.0]``, which the same /100 conversion falls
comfortably inside.
"""

DEFAULT_FALLBACK_FPS: float = 24.0
"""Assumed fps for filename-derived fallback timing (see module docstring)."""

DEFAULT_CHUNK_CACHE_SIZE: int = 2
"""How many decoded chunks HDF5CameraSession keeps in memory at once."""

_PLAUSIBLE_EPOCH_RANGE: tuple[float, float] = (946684800.0, 4102444800.0)
"""Unix-epoch seconds for [2000-01-01, 2100-01-01) — timestamps outside this
range are treated as unusable (catches both all-zero chunks and the
bogus-large-value chunk described in the module docstring), not as real
capture times, regardless of which failure mode produced them."""

_CHUNK_NAME_RE = re.compile(r"^(\d{6})_seq(\d+)_seq(\d+)\.h5$")


def _chunk_sort_key(path: Path) -> str:
    """Chunks are named ``HHMMSS_seq<start>_seq<end>.h5``; the leading HHMMSS
    is monotonic within one ``<date>/cam_N/`` directory."""
    m = _CHUNK_NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected h5 chunk filename: {path.name!r}")
    return m.group(1)


def list_chunks(cam_dir: str | Path) -> list[Path]:
    """Sorted list of ``.h5`` chunk files in one ``cam_N/`` directory."""
    cam_dir = Path(cam_dir)
    return sorted(cam_dir.glob("*.h5"), key=_chunk_sort_key)


def _chunk_wallclock_start(path: Path, session_date: str) -> _dt.datetime:
    """Wall-clock start time of a chunk, parsed from its filename + session date.

    Args:
        path: chunk file, named ``HHMMSS_seq<start>_seq<end>.h5``.
        session_date: the session's date directory name, ``"YYYY-MM-DD"``.
    """
    m = _CHUNK_NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected h5 chunk filename: {path.name!r}")
    hhmmss = m.group(1)
    h, mi, s = int(hhmmss[0:2]), int(hhmmss[2:4]), int(hhmmss[4:6])
    y, mo, d = (int(x) for x in session_date.split("-"))
    return _dt.datetime(y, mo, d, h, mi, s)


def _timestamps_look_valid(timestamps: np.ndarray) -> bool:
    """True if the majority of timestamps fall in a plausible Unix-epoch
    range. Catches both the all-zero chunks and the single bogus-large-value
    chunk uniformly (see module docstring) — "not zero" alone is not enough."""
    if timestamps.size == 0:
        return True
    lo, hi = _PLAUSIBLE_EPOCH_RANGE
    in_range = (timestamps >= lo) & (timestamps < hi)
    return np.count_nonzero(in_range) > 0.5 * timestamps.size


def infer_fps_from_chunk_spacing(
    chunk_paths: "tuple[Path, ...] | list[Path]",
    chunk_lengths: "tuple[int, ...] | list[int]",
    session_date: str,
) -> Optional[float]:
    """Infer the real capture fps from consecutive chunk filenames' wall-clock
    gaps, rather than assuming a fixed value.

    Motivation: the synthetic data's ``file_chunk_s=300`` config (7200 frames
    per chunk => 24 fps) does NOT hold for real hardware — room-1's chunk
    filenames are consistently exactly 600s (10 min) apart while each full
    chunk still holds 7200 frames, implying 7200/600 = 12 fps, not 24. Rather
    than hardcode either number, compute it per session from the filenames
    themselves: ``chunk_i.n_frames / gap_to_next_chunk_start_seconds``,
    median across all consecutive pairs (robust to the one irregular
    short/partial trailing chunk most sessions end with).

    Returns None if there are fewer than 2 chunks or no gap is usable (e.g.
    all chunks share the same start time).
    """
    if len(chunk_paths) < 2:
        return None
    starts = [_chunk_wallclock_start(p, session_date) for p in chunk_paths]
    implied: list[float] = []
    for i in range(len(chunk_paths) - 1):
        gap_s = (starts[i + 1] - starts[i]).total_seconds()
        if gap_s > 0:
            implied.append(chunk_lengths[i] / gap_s)
    if not implied:
        return None
    return float(np.median(implied))


def _peek_chunk_header(path: str | Path) -> tuple[int, np.ndarray]:
    """Cheaply read a chunk's frame count and timestamps, without touching
    the (large) pixel data. Used to build the session's frame index."""
    import h5py
    import hdf5plugin  # noqa: F401  registers the Zstd filter these chunks use

    with h5py.File(path, "r") as f:
        n = int(f["frames"].shape[0])
        timestamps = f["timestamps"][...].astype(np.float64)
    return n, timestamps


def read_chunk(
    path: str | Path,
    *,
    session_date: Optional[str] = None,
    fallback_fps: float = DEFAULT_FALLBACK_FPS,
    chunk_start_override: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, dict, bool]:
    """Read one ``.h5`` chunk, pixel data included.

    Args:
        path: chunk file path.
        session_date: the session's date directory name (``"YYYY-MM-DD"``),
            used to derive a filename-based fallback start time if this chunk
            needs one and ``chunk_start_override`` isn't given.
        fallback_fps: fps assumed for the fallback.
        chunk_start_override: if given, use this Unix-epoch start time for the
            fallback instead of deriving one from the filename. Chaining from
            the previous chunk's end (rather than trusting each chunk's own
            filename independently) is what ``HDF5CameraSession`` actually
            uses in practice — see module docstring: a session's *own*
            trailing chunk can have a filename-declared start time that is
            inconsistent with the otherwise-perfectly-regular spacing of every
            other chunk, so filename-derived starts are only trustworthy for
            a chunk with no prior chunk to chain from (typically the first).

    Returns:
        frames_degC: (N, H, W) float32, converted via ``RAW_TO_CELSIUS_SCALE``.
        timestamps: (N,) float64, absolute Unix-epoch seconds.
        meta: chunk-level metadata group attrs, if any (coarse ``event_class``
            etc — NOT frame-accurate; join against ``labels.csv`` instead).
        used_fallback: True if this chunk's real timestamps were unusable and
            the fallback (override- or filename-derived) was used instead.
    """
    import h5py
    import hdf5plugin  # noqa: F401  registers the Zstd filter these chunks use

    path = Path(path)
    with h5py.File(path, "r") as f:
        raw = f["frames"][...]
        timestamps = f["timestamps"][...].astype(np.float64)
        meta = dict(f["metadata"].attrs) if "metadata" in f else {}

    frames = raw.astype(np.float32) / RAW_TO_CELSIUS_SCALE
    n = frames.shape[0]

    if _timestamps_look_valid(timestamps):
        return frames, timestamps, meta, False

    if chunk_start_override is not None:
        chunk_start_epoch = chunk_start_override
    elif session_date is not None:
        chunk_start_epoch = _chunk_wallclock_start(path, session_date).timestamp()
    else:
        raise ValueError(
            f"{path}: timestamps are unusable (outside a plausible epoch "
            "range — the known room-1 acquisition bug) and neither "
            "chunk_start_override nor session_date was given to derive a "
            "fallback start time."
        )
    timestamps = chunk_start_epoch + np.arange(n, dtype=np.float64) / fallback_fps
    return frames, timestamps, meta, True


class HDF5CameraSession:
    """One camera's recording for one ``<date>`` session, indexed across all
    its ``.h5`` chunks but loaded lazily — at most
    ``DEFAULT_CHUNK_CACHE_SIZE`` decoded chunks are held in memory at once.

    A full synthetic camera-day is ~1.45M frames (~27 GiB as float32); this
    class is built around random single-frame access (``load_frame``), not
    materializing a whole session. Mirrors enough of ``SessionMetadata``'s
    read surface (``n_frames``, ``load_frame``) to be a drop-in frame source,
    but is not a ``SessionMetadata`` itself — there's no ``ch{N}_raw_data.npz``
    here.
    """

    def __init__(
        self,
        cam_dir: str | Path,
        cam_id: int,
        *,
        session_date: Optional[str] = None,
        fallback_fps: Optional[float] = None,
        cache_chunks: int = DEFAULT_CHUNK_CACHE_SIZE,
    ) -> None:
        """
        Args:
            fallback_fps: fps to assume for filename-derived fallback timing.
                If None (default) and a fallback is actually needed, it's
                inferred per-session from chunk filename spacing (see
                ``infer_fps_from_chunk_spacing``) rather than assumed —
                real-hardware sessions have been observed to run at a
                different rate (12 fps) than the synthetic data's nominal
                24 fps, so a single hardcoded default is not safe to apply
                to both.
        """
        self.cam_dir = Path(cam_dir)
        self.cam_id = cam_id
        self.session_date = session_date
        self._cache_chunks = max(1, cache_chunks)

        self._chunk_paths: tuple[Path, ...] = tuple(list_chunks(self.cam_dir))
        if not self._chunk_paths:
            raise ValueError(f"No .h5 chunks found in {self.cam_dir}")

        lengths: list[int] = []
        chunk_timestamps: list[np.ndarray] = []
        used_fallback_per_chunk: list[bool] = []
        for p in self._chunk_paths:
            n, timestamps = _peek_chunk_header(p)
            lengths.append(n)
            chunk_timestamps.append(timestamps)
            used_fallback_per_chunk.append(not _timestamps_look_valid(timestamps))
            if used_fallback_per_chunk[-1] and self.session_date is None:
                raise ValueError(
                    f"{p}: timestamps are unusable and no session_date was "
                    "given to derive a filename-based fallback."
                )
        self._chunk_lengths: tuple[int, ...] = tuple(lengths)
        self._chunk_offsets: np.ndarray = np.concatenate(
            [[0], np.cumsum(self._chunk_lengths)]
        )
        self._used_fallback = any(used_fallback_per_chunk)

        if fallback_fps is not None:
            self.fallback_fps = fallback_fps
        elif self._used_fallback and self.session_date is not None:
            inferred = infer_fps_from_chunk_spacing(
                self._chunk_paths, self._chunk_lengths, self.session_date
            )
            self.fallback_fps = inferred if inferred is not None else DEFAULT_FALLBACK_FPS
        else:
            self.fallback_fps = DEFAULT_FALLBACK_FPS

        # Chain fallback chunks off the previous chunk's own end time rather
        # than trusting each chunk's own filename independently — a session's
        # own trailing chunk has been observed to have a filename-declared
        # start inconsistent with the otherwise-perfectly-regular spacing of
        # every other chunk (see module docstring). Filename-derived starts
        # are only used when there's no prior chunk/timestamp to chain from.
        overrides: list[Optional[float]] = []
        next_expected: Optional[float] = None
        for p, n, timestamps, broken in zip(
            self._chunk_paths, lengths, chunk_timestamps, used_fallback_per_chunk
        ):
            if not broken:
                overrides.append(None)  # read_chunk uses the chunk's own valid timestamps
                next_expected = float(timestamps[-1]) + 1.0 / self.fallback_fps
                continue
            if next_expected is not None:
                start = next_expected
            else:
                # No prior chunk/timestamp to chain from (e.g. the session's
                # very first chunk is itself broken) — filename is the only
                # anchor available. self.session_date is guaranteed non-None
                # here (checked above whenever any chunk is broken).
                start = _chunk_wallclock_start(p, self.session_date).timestamp()  # type: ignore[arg-type]
            overrides.append(start)
            next_expected = start + n / self.fallback_fps
        self._chunk_start_overrides: tuple[Optional[float], ...] = tuple(overrides)

        # LRU cache of decoded (frames, timestamps) per chunk index.
        self._cache: "OrderedDict[int, tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    @property
    def n_frames(self) -> int:
        return int(self._chunk_offsets[-1])

    @property
    def n_chunks(self) -> int:
        return len(self._chunk_paths)

    @property
    def used_fallback(self) -> bool:
        """True if any chunk in this session needed filename-derived timing
        (the room-1 acquisition bug) rather than its own recorded timestamps."""
        return self._used_fallback

    def _chunk_for_frame(self, frame_idx: int) -> tuple[int, int]:
        if not 0 <= frame_idx < self.n_frames:
            raise IndexError(f"frame_idx {frame_idx} out of range [0, {self.n_frames})")
        chunk_i = bisect.bisect_right(self._chunk_offsets, frame_idx) - 1
        local_idx = frame_idx - int(self._chunk_offsets[chunk_i])
        return chunk_i, local_idx

    def _get_chunk(self, chunk_i: int) -> tuple[np.ndarray, np.ndarray]:
        if chunk_i in self._cache:
            self._cache.move_to_end(chunk_i)
            return self._cache[chunk_i]

        frames, timestamps, _meta, _used_fallback = read_chunk(
            self._chunk_paths[chunk_i],
            session_date=self.session_date,
            fallback_fps=self.fallback_fps,
            chunk_start_override=self._chunk_start_overrides[chunk_i],
        )
        self._cache[chunk_i] = (frames, timestamps)
        if len(self._cache) > self._cache_chunks:
            self._cache.popitem(last=False)
        return frames, timestamps

    def load_frame(self, frame_idx: int) -> Frame:
        chunk_i, local_idx = self._chunk_for_frame(frame_idx)
        frames, timestamps = self._get_chunk(chunk_i)
        return Frame(
            data=frames[local_idx],
            timestamp=float(timestamps[local_idx]),
            camera_id=self.cam_id,
        )

    def iter_frames(self, start: int = 0, stop: Optional[int] = None):
        """Iterate ``Frame``s in order, decoding each chunk once regardless
        of the cache size — the efficient way to walk a whole session."""
        stop = self.n_frames if stop is None else stop
        if not (0 <= start <= stop <= self.n_frames):
            raise IndexError(f"invalid range [{start}, {stop}) for {self.n_frames} frames")
        if start == stop:
            return
        start_chunk, _ = self._chunk_for_frame(start)
        end_chunk, _ = self._chunk_for_frame(stop - 1)
        idx = start
        for chunk_i in range(start_chunk, end_chunk + 1):
            frames, timestamps = self._get_chunk(chunk_i)
            chunk_start = int(self._chunk_offsets[chunk_i])
            chunk_end = int(self._chunk_offsets[chunk_i + 1])
            lo = max(start, chunk_start) - chunk_start
            hi = min(stop, chunk_end) - chunk_start
            for local_idx in range(lo, hi):
                yield Frame(
                    data=frames[local_idx],
                    timestamp=float(timestamps[local_idx]),
                    camera_id=self.cam_id,
                )
                idx += 1

    def load_frames(self) -> np.ndarray:
        """All frames for this camera, concatenated across every chunk, in
        degC. Only use this for sessions you know are small (e.g. room-1's
        scale, a few GiB) — for a full synthetic camera-day this will try to
        allocate tens of GiB. Prefer ``load_frame`` / ``iter_frames``."""
        return np.concatenate([self._get_chunk(i)[0] for i in range(self.n_chunks)], axis=0)

    def load_timestamps(self) -> np.ndarray:
        return np.concatenate([self._get_chunk(i)[1] for i in range(self.n_chunks)], axis=0)

    def clear_cache(self) -> None:
        """Drop cached decoded chunks to free memory."""
        self._cache.clear()
