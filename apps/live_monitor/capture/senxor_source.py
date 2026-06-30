"""Raspberry Pi hardware backend — Waveshare 80x62 module (Senxor/MI48).

Uses the manufacturer library ``pysenxor-lite`` (``pip install pysenxor-lite``),
imported lazily so the rest of the app runs on machines without it.

    from senxor import connect, list_senxor
    dev = connect(<address>)
    header, frame = dev.read()      # frame: 2-D float32 °C

On connect we enable the MI48's **on-chip denoising filters** — the
"non-optional on-board filter" required for this project. The MI48 exposes four
filter blocks (all default to 0 / OFF):

    FILTER_CONTROL      (addr 208)  + FILTER_SETTING_1_0/1 (209/210)  temporal
    MEDIAN_CTRL         (addr  48)                                    median
    STARK_CTRL          (addr  32)                                    STARK (edge-preserving)
    MMS_CTRL            (addr  37)                                    min/max stabilization

The exact recommended values vary by firmware; the defaults below follow the
Senxor reference flow (temporal + median + STARK + MMS all enabled). They are
named constants so they can be pinned against the on-device SDK example /
MI48 datasheet without touching call sites. ``frame`` is returned already
filtered by the chip.

NOTE: This backend can only be exercised on the Pi with hardware attached; on a
dev machine it raises a clear error at ``start()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import Frame
from apps.live_monitor.capture.base import FrameSource


# MI48 filter register addresses (see module docstring).
REG_FILTER_CONTROL = 208
REG_FILTER_SETTING_1_0 = 209
REG_FILTER_SETTING_1_1 = 210
REG_STARK_CTRL = 32
REG_MEDIAN_CTRL = 48
REG_MMS_CTRL = 37


@dataclass
class Mi48FilterConfig:
    """Recommended MI48 on-chip filter configuration.

    Values follow the Senxor reference streaming flow. Adjust against the
    MI48 datasheet / on-device SDK if the vendor recommends different bits.
    """

    temporal_enable: bool = True
    temporal_control: int = 0x03            # FILTER_CONTROL enable bits
    temporal_setting_lsb: int = 0x20        # FILTER_SETTING_1_0
    temporal_setting_msb: int = 0x00        # FILTER_SETTING_1_1
    median_enable: bool = True
    median_control: int = 0x03              # MEDIAN_CTRL (3x3 median)
    stark_enable: bool = True
    stark_control: int = 0x03               # STARK_CTRL
    mms_enable: bool = True
    mms_control: int = 0x01                 # MMS_CTRL (min/max stabilization)

    def as_register_writes(self) -> list[tuple[int, int, str]]:
        """(address, value, human-name) tuples in the order they should write."""
        writes: list[tuple[int, int, str]] = []
        if self.temporal_enable:
            writes.append((REG_FILTER_SETTING_1_0, self.temporal_setting_lsb, "FILTER_SETTING_1_0"))
            writes.append((REG_FILTER_SETTING_1_1, self.temporal_setting_msb, "FILTER_SETTING_1_1"))
            writes.append((REG_FILTER_CONTROL, self.temporal_control, "FILTER_CONTROL"))
        if self.median_enable:
            writes.append((REG_MEDIAN_CTRL, self.median_control, "MEDIAN_CTRL"))
        if self.stark_enable:
            writes.append((REG_STARK_CTRL, self.stark_control, "STARK_CTRL"))
        if self.mms_enable:
            writes.append((REG_MMS_CTRL, self.mms_control, "MMS_CTRL"))
        return writes


class SenxorSource(FrameSource):
    """Live Waveshare/MI48 camera over the pysenxor-lite driver.

    Args:
        camera_id: 0/1/2.
        profile: sensor geometry (should be WAVESHARE_26984).
        device_address: explicit device address/handle to pass to ``connect``.
            If None, the camera_id-th entry from ``list_senxor()`` is used.
        filters: on-chip filter configuration (always applied on connect).
        emissivity: optional target emissivity (skin ≈ 0.98) if the driver
            exposes it; ignored gracefully otherwise.
    """

    def __init__(
        self,
        camera_id: int,
        profile: SensorProfile,
        *,
        device_address: Optional[Any] = None,
        filters: Optional[Mi48FilterConfig] = None,
        emissivity: Optional[float] = None,
    ) -> None:
        super().__init__(camera_id, profile)
        self._device_address = device_address
        self._filters = filters or Mi48FilterConfig()
        self._emissivity = emissivity
        self._dev: Any = None

    # ---- Lifecycle ------------------------------------------------------

    def _open(self) -> None:
        try:
            from senxor import connect, list_senxor  # type: ignore
        except Exception as exc:  # pragma: no cover - hardware-only path
            raise RuntimeError(
                "pysenxor-lite is not installed. Install it on the Raspberry Pi "
                "with `pip install pysenxor-lite` (this backend is hardware-only)."
            ) from exc

        address = self._device_address
        if address is None:
            devices = list(list_senxor())
            if len(devices) <= self._camera_id:
                raise RuntimeError(
                    f"Camera {self._camera_id} requested but list_senxor() found "
                    f"only {len(devices)} SenXor device(s): {devices!r}."
                )
            address = devices[self._camera_id]

        self._dev = connect(address)
        self._health.extra["device_address"] = str(address)
        self._apply_filters()
        self._apply_emissivity()
        self._health.connected = True

    def _close(self) -> None:
        dev, self._dev = self._dev, None
        if dev is None:
            return
        for meth in ("stop", "close", "disconnect"):
            fn = getattr(dev, meth, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break

    def _read_raw(self) -> Optional[Frame]:
        if self._dev is None:
            return None
        header, frame = self._dev.read()
        if frame is None:
            return None
        data = np.asarray(frame, dtype=np.float32)
        if data.ndim != 2:
            data = data.reshape(self._profile.height, self._profile.width)
        meta = {"header": header} if header is not None else {}
        return Frame(data=data, timestamp=time.monotonic(), camera_id=self._camera_id, metadata=meta)

    # ---- MI48 configuration --------------------------------------------

    def _apply_filters(self) -> None:
        """Write the on-chip filter registers. The chip then returns filtered
        frames. We record exactly which writes succeeded for the debug HUD."""
        applied: dict[str, int] = {}
        for addr, value, name in self._filters.as_register_writes():
            if self._write_register(addr, value, name):
                applied[name] = value
        self._health.extra["filters_applied"] = applied
        if not applied:
            self._health.last_error = (
                "Could not write any MI48 filter register via the driver API; "
                "frames may be unfiltered. Check pysenxor-lite version."
            )

    def _apply_emissivity(self) -> None:
        if self._emissivity is None:
            return
        for setter in ("set_emissivity",):
            fn = getattr(self._dev, setter, None)
            if callable(fn):
                try:
                    fn(self._emissivity)
                    self._health.extra["emissivity"] = self._emissivity
                except Exception:
                    pass
                return

    def _write_register(self, addr: int, value: int, name: str) -> bool:
        """Best-effort register write across the pysenxor API surface.

        The lite driver exposes registers/fields differently across versions;
        try the documented field API, then a regs mapping, then setter methods.
        """
        dev = self._dev
        # 1) Named field API: dev.fields.<NAME>.set(value)
        fields = getattr(dev, "fields", None)
        if fields is not None:
            field = getattr(fields, name, None)
            setter = getattr(field, "set", None) if field is not None else None
            if callable(setter):
                try:
                    setter(value)
                    return True
                except Exception:
                    pass
        # 2) Register mapping: dev.regs[addr] = value
        regs = getattr(dev, "regs", None)
        if regs is not None:
            try:
                regs[addr] = value
                return True
            except Exception:
                pass
        # 3) Setter methods: dev.set_reg(addr, value) / dev.write_register(...)
        for meth in ("set_reg", "write_register", "regwrite", "set_register"):
            fn = getattr(dev, meth, None)
            if callable(fn):
                try:
                    fn(addr, value)
                    return True
                except Exception:
                    pass
        return False
