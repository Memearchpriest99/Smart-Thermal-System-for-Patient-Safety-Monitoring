"""Live frame acquisition from the Waveshare Thermal Camera Module (MI48).

Implements the acquisition pipeline documented in the Waveshare wiki
(https://www.waveshare.com/wiki/Thermal-Camera-Module):

  - **I2C** configures the MI48 camera registers (FPS, filters, offset).
    Default address 0x40; 0x41 selectable with the on-board 0R resistor.
  - **SPI** carries the full-frame temperature data (mode 0, MSB first,
    read as 8-bit transfers and reassembled into 16-bit words).
  - **GPIO** DATA_READY (BCM24) gates every frame read; nRESET (BCM23)
    provides a software-driven hardware reset; chip-select (BCM7) is
    driven manually because spidev on kernel 5.x+ no longer handles CS
    (the wiki's Bookworm setup adds ``dtoverlay=spi0-0cs`` for this).

The register map, CRC check and deci-Kelvin → °C conversion are handled by
Meridian Innovation's ``senxor`` package (pysenxor), which ships inside the
Waveshare demo archive (``Thermal_Camera_Hat.zip``) and must be installed on
the Raspberry Pi::

    wget https://files.waveshare.com/wiki/Thermal_Camera_Module/Thermal_Camera_Hat.zip
    unzip Thermal_Camera_Hat.zip && cd pysenxor-master && pip install -e ./

This module has **no hard dependency** on the hardware stack: ``senxor``,
``smbus``, ``spidev`` and ``gpiozero`` are imported lazily inside
:meth:`MI48Camera.open`, so it imports (and is testable) on any dev machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from thermal_algorithms.core.types import Frame


class AcquisitionError(RuntimeError):
    """Raised when the MI48 returns no / malformed frame data."""


@dataclass(frozen=True)
class MI48CameraConfig:
    """Wiring and streaming configuration for one MI48 camera.

    Defaults match the Waveshare Thermal Camera HAT wiring from the wiki
    (SDA/SCL on I2C-1, SPI0 with manual CS on BCM7, DATA_READY on BCM24,
    nRESET on BCM23).
    """

    camera_id: int = 0
    i2c_channel: int = 1
    i2c_address: int = 0x40          # 0x41 optional via 0R resistor
    spi_bus: int = 0
    spi_device: int = 0
    spi_max_speed_hz: int = 31_200_000
    spi_xfer_size: int = 160         # bytes; one 80-px row at 2 B/px
    spi_cs_pin: str = "BCM7"         # SS is routed to GPIO7 on the HAT
    spi_cs_delay_s: float = 1e-4     # CS assert/deassert settling delay
    data_ready_pin: str = "BCM24"
    reset_pin: str = "BCM23"
    fps: float = 8.0                 # project standard (WAVESHARE_26984)
    with_header: bool = True
    hflip: bool = False


def mi48_data_to_array(
    data: np.ndarray, fpa_shape: tuple[int, int], hflip: bool = False
) -> np.ndarray:
    """Reshape the MI48's 1-D column-major readout into an (H, W) array.

    Mirrors ``senxor.utils.data_to_frame`` (Fortran-order reshape of the
    (cols, rows) FPA, then transpose) but returns float32 for the repo's
    `Frame` convention. For the 80x62 module this yields shape (62, 80).
    """
    arr = np.asarray(data).reshape(fpa_shape, order="F").T
    if hflip:
        arr = np.flip(arr, 1)
    return np.ascontiguousarray(arr, dtype=np.float32)


class MI48Camera:
    """One Waveshare Thermal Camera Module, streamed per the wiki pipeline.

    Usage on the Pi::

        with MI48Camera(MI48CameraConfig(camera_id=0)) as cam:
            cam.start()
            frame = cam.read_frame()   # -> core.types.Frame, °C float32

    For testing, the hardware objects can be injected via the private
    constructor arguments, bypassing all hardware imports.
    """

    def __init__(
        self,
        config: MI48CameraConfig = MI48CameraConfig(),
        *,
        _mi48: Any = None,
        _cs_pin: Any = None,
        _data_ready: Any = None,
    ) -> None:
        self.config = config
        self._mi48 = _mi48
        self._cs_pin = _cs_pin
        self._data_ready = _data_ready
        self._i2c = None
        self._spi = None
        self._reset_pin = None
        self._started = False

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "MI48Camera":
        """Open I2C/SPI/GPIO and configure the MI48 registers."""
        if self._mi48 is not None:  # injected (tests) — nothing to open
            return self

        # Hardware-only imports, kept local so the module imports anywhere.
        from smbus import SMBus
        from spidev import SpiDev
        from gpiozero import DigitalInputDevice, DigitalOutputDevice
        from senxor.mi48 import MI48
        from senxor.interfaces import I2C_Interface, SPI_Interface

        cfg = self.config

        self._i2c = I2C_Interface(SMBus(cfg.i2c_channel), cfg.i2c_address)

        spi = SPI_Interface(
            SpiDev(cfg.spi_bus, cfg.spi_device), xfer_size=cfg.spi_xfer_size
        )
        spi.device.mode = 0b00
        spi.device.max_speed_hz = cfg.spi_max_speed_hz
        spi.device.bits_per_word = 8
        try:
            # Kernel 5.x+ leaves CS to the device; we drive it via GPIO.
            spi.device.no_cs = True
        except (AttributeError, OSError):
            pass
        self._spi = spi

        self._cs_pin = DigitalOutputDevice(
            cfg.spi_cs_pin, active_high=False, initial_value=False
        )
        self._data_ready = DigitalInputDevice(cfg.data_ready_pin, pull_up=False)
        self._reset_pin = DigitalOutputDevice(
            cfg.reset_pin, active_high=False, initial_value=True
        )

        self._mi48 = MI48(
            [self._i2c, self._spi],
            data_ready=self._data_ready,
            reset_handler=_MI48Reset(self._reset_pin),
        )

        self._mi48.get_camera_info()
        self._mi48.set_fps(cfg.fps)
        if int(self._mi48.fw_version[0]) >= 2:
            # Wiki demo defaults: temporal + rolling-average filters on,
            # median off, factory per-pixel calibration (no offset).
            self._mi48.enable_filter(f1=True, f2=True, f3=False)
            self._mi48.set_offset_corr(0.0)
        return self

    def start(self) -> None:
        """Begin continuous frame streaming."""
        if self._mi48 is None:
            self.open()
        self._mi48.start(stream=True, with_header=self.config.with_header)
        self._started = True

    def stop(self) -> None:
        if self._mi48 is not None and self._started:
            self._mi48.stop(stop_timeout=0.5)
            self._started = False

    def close(self) -> None:
        self.stop()
        for iface in (self._spi, self._i2c):
            try:
                if iface is not None:
                    iface.close()
            except Exception:
                pass

    def __enter__(self) -> "MI48Camera":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- frame readout ------------------------------------------------------

    @property
    def fpa_shape(self) -> tuple[int, int]:
        """(cols, rows) of the focal-plane array, e.g. (80, 62)."""
        return tuple(self._mi48.fpa_shape)

    def read_frame(self) -> Frame:
        """Block until DATA_READY, then read one full frame over SPI.

        Returns a `Frame` with °C float32 data of shape (rows, cols)
        — (62, 80) for the 80x62 module — stamped with ``time.time()``.
        """
        cfg = self.config

        # Wait for the DATA_READY pin (or poll the STATUS register on
        # firmware that predates the pin).
        if self._data_ready is not None:
            self._data_ready.wait_for_active()
        else:
            from senxor.mi48 import DATA_READY

            while not (self._mi48.get_status() & DATA_READY):
                time.sleep(0.01)

        # Manual chip-select around the SPI burst, per the wiki demo.
        if self._cs_pin is not None:
            self._cs_pin.on()
            time.sleep(cfg.spi_cs_delay_s)
        try:
            data, header = self._mi48.read()
        finally:
            if self._cs_pin is not None:
                time.sleep(cfg.spi_cs_delay_s)
                self._cs_pin.off()

        if data is None:
            raise AcquisitionError(
                f"camera {cfg.camera_id}: MI48 returned no data (expected GFRA)"
            )

        arr = mi48_data_to_array(data, self.fpa_shape, hflip=cfg.hflip)
        metadata: dict[str, Any] = {}
        if header is not None:
            metadata["mi48_header"] = header
        return Frame(
            data=arr,
            timestamp=time.time(),
            camera_id=cfg.camera_id,
            metadata=metadata,
        )


class _MI48Reset:
    """Hardware reset via the nRESET line (active-low pulse)."""

    def __init__(
        self,
        pin: Any,
        assert_seconds: float = 0.000035,
        deassert_seconds: float = 0.050,
    ) -> None:
        self.pin = pin
        self.assert_time = assert_seconds
        self.deassert_time = deassert_seconds

    def __call__(self) -> None:
        self.pin.on()
        time.sleep(self.assert_time)
        self.pin.off()
        time.sleep(self.deassert_time)
