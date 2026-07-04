"""Live data acquisition from the Waveshare Thermal Camera Module (MI48).

Replaces the DataRecorder acquisition path (§ 5.1.2) with the SPI/I2C
pipeline from the Waveshare wiki: I2C register configuration, SPI
full-frame readout gated on the DATA_READY GPIO, via the vendor's
``senxor`` (pysenxor) package. Hardware libraries are imported lazily,
so this package is importable on machines without the sensor attached.
"""

from thermal_algorithms.acquisition.mi48_camera import (
    AcquisitionError,
    MI48Camera,
    MI48CameraConfig,
    mi48_data_to_array,
)
from thermal_algorithms.acquisition.mi48_usb import (
    MI48USBCamera,
    MI48USBCameraConfig,
    list_mi48_ports,
)
from thermal_algorithms.acquisition.recorder import SessionRecorder

__all__ = [
    "AcquisitionError",
    "MI48Camera",
    "MI48CameraConfig",
    "MI48USBCamera",
    "MI48USBCameraConfig",
    "list_mi48_ports",
    "mi48_data_to_array",
    "SessionRecorder",
]
