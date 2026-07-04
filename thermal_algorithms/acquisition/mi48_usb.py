"""USB acquisition from the Waveshare Thermal Camera Module (MI48, USB-C).

For modules connected over USB (CDC serial) rather than the SPI/I2C HAT
pins — the module enumerates as a virtual COM port (VID 0x0416, PID
0xB002 family). Control commands and frame data both travel over the
serial link, mirroring pysenxor's ``stream_usb.py`` / ``connect_senxor``:

    Serial(dev) -> senxor.interfaces.USB_Interface -> MI48([usb, usb])

Multi-camera identity: ports are sorted by USB *location* (the physical
bus/port topology), so ``camera_id`` maps to a physical USB socket and
stays stable across reboots — not by enumeration order, which does not.

No GPIO, no raspi-config, no DATA_READY pin: ``mi48.read()`` blocks on
the serial stream. Hardware imports (``serial``, ``senxor``) are lazy,
so this module is importable and testable on any dev machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from thermal_algorithms.core.types import Frame
from thermal_algorithms.acquisition.mi48_camera import (
    AcquisitionError,
    mi48_data_to_array,
)

MI_VID = 0x0416
MI_PIDS = (0xB002, 0xB020)   # EVK / XPro families (senxor.interfaces.MI_PIDs)


def list_mi48_ports() -> list[str]:
    """Device paths of all connected MI48 USB cameras, in stable order.

    Sorted by USB location (physical socket topology), falling back to the
    device name. An empty list means no camera is connected.
    """
    from serial.tools import list_ports

    cams = [p for p in list_ports.comports()
            if p.vid == MI_VID and p.pid in MI_PIDS]
    cams.sort(key=lambda p: (p.location or "", p.device))
    return [p.device for p in cams]


@dataclass(frozen=True)
class MI48USBCameraConfig:
    """One USB-connected MI48 camera.

    Args:
        camera_id: Channel index in {0, 1, 2}.
        port: Serial device (e.g. '/dev/ttyACM0', 'COM7'). None = resolve
            by ``camera_id`` from `list_mi48_ports()` (topology order).
        fps: Stream rate; 8.0 is the project standard (module max 25).
        hflip: Horizontal flip for forward-looking mounts.
    """

    camera_id: int = 0
    port: Optional[str] = None
    fps: float = 8.0
    with_header: bool = True
    hflip: bool = False


class MI48USBCamera:
    """Same surface as `MI48Camera`, but over the module's USB-C port."""

    def __init__(
        self,
        config: MI48USBCameraConfig = MI48USBCameraConfig(),
        *,
        _mi48: Any = None,
    ) -> None:
        self.config = config
        self._mi48 = _mi48
        self._serial = None
        self._started = False

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "MI48USBCamera":
        if self._mi48 is not None:   # injected (tests)
            return self

        from serial import Serial
        from senxor.mi48 import MI48
        from senxor.interfaces import USB_Interface

        cfg = self.config
        port = cfg.port
        if port is None:
            ports = list_mi48_ports()
            if cfg.camera_id >= len(ports):
                raise AcquisitionError(
                    f"camera {cfg.camera_id}: only {len(ports)} MI48 USB "
                    f"device(s) found ({ports})"
                )
            port = ports[cfg.camera_id]

        self._serial = Serial(port)
        usb = USB_Interface(self._serial)
        # Control and data share the one USB link (per connect_senxor).
        self._mi48 = MI48([usb, usb], name=f"cam{cfg.camera_id}@{port}",
                          read_raw=False)

        self._mi48.get_camera_info()
        self._mi48.set_fps(cfg.fps)
        # Vendor USB-demo defaults (stream_usb.py): temporal filter only.
        self._mi48.disable_filter(f1=True, f2=True, f3=True)
        self._mi48.set_filter_1(85)
        self._mi48.enable_filter(f1=True, f2=False, f3=False, f3_ks_5=False)
        self._mi48.set_offset_corr(0.0)
        self._mi48.set_sens_factor(100)
        return self

    def start(self) -> None:
        if self._mi48 is None:
            self.open()
        self._mi48.start(stream=True, with_header=self.config.with_header)
        self._started = True

    def stop(self) -> None:
        if self._mi48 is not None and self._started:
            self._mi48.stop()
            self._started = False

    def close(self) -> None:
        self.stop()
        try:
            if self._serial is not None:
                self._serial.close()
        except Exception:
            pass

    def __enter__(self) -> "MI48USBCamera":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- frame readout ------------------------------------------------------

    @property
    def fpa_shape(self) -> tuple[int, int]:
        return tuple(self._mi48.fpa_shape)

    def read_frame(self) -> Frame:
        """Block until the next full frame arrives over USB."""
        data, header = self._mi48.read()
        if data is None:
            raise AcquisitionError(
                f"camera {self.config.camera_id}: MI48 returned no data "
                f"(expected GFRA)"
            )
        arr = mi48_data_to_array(data, self.fpa_shape, hflip=self.config.hflip)
        metadata: dict[str, Any] = {}
        if header is not None:
            metadata["mi48_header"] = header
        return Frame(
            data=arr,
            timestamp=time.time(),
            camera_id=self.config.camera_id,
            metadata=metadata,
        )
