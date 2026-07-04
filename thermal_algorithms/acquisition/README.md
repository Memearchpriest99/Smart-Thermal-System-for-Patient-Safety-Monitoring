# acquisition — live capture from the Waveshare Thermal Camera Module

Two transports, one `Frame` contract (float32 °C, 62×80):

**USB-C (`MI48USBCamera`, the deployed wiring)** — each module's USB-C port
to a Pi USB-A socket; enumerates as a CDC serial device (VID 0x0416).
Control and frame data share the serial link (pysenxor `stream_usb.py`
pattern). Multi-camera identity is resolved by USB *topology* order
(`list_mi48_ports()`), so `camera_id` = physical socket, stable across
reboots. Needs only `pyserial` + `crcmod` + the vendor `senxor` package,
plus membership in the `dialout` group. No GPIO, no raspi-config.

**SPI/I2C HAT (`MI48Camera`)** — the wiki's 40-pin pipeline, kept as an
alternative:

| Bus  | Role | Wiring (HAT default) |
|------|------|----------------------|
| I2C  | MI48 register config (FPS, filters, offset) | channel 1, addr `0x40` (`0x41` via 0R resistor) |
| SPI  | Full-frame temperature readout, mode 0, MSB first | SPI0.0 @ 31.2 MHz, 160-byte transfers |
| GPIO | `DATA_READY` gates each read; `nRESET` hardware reset; manual chip-select | BCM24 / BCM23 / BCM7 |

## Components

- **`MI48Camera`** (`mi48_camera.py`) — one camera. `open()` configures the
  registers via the vendor's `senxor` (pysenxor) package, `read_frame()`
  blocks on DATA_READY, asserts CS, reads the frame over SPI and returns a
  `core.types.Frame` (float32 °C, shape (62, 80), `time.time()` timestamp,
  MI48 header in `metadata`).
- **`SessionRecorder`** (`recorder.py`) — reads N cameras in lockstep and
  writes the repo's raw capture layout, directly consumable by
  `scripts/reorganize_waveshare.py` and the annotator:

  ```
  <root>/<scene>/<YYYYMMDD_HHMMSS>/
      ch0_thermal.npz          # key 'frames', (N, 62, 80) float32 °C
      ch0/frame_00000.png ...  # 320x240 colormapped previews (optional)
      ch1_* / ch2_* ...
  ```

  Lockstep capture guarantees equal frame counts across channels — the tail
  misalignment the reorganize script had to repair cannot occur.

## Raspberry Pi setup (once, per the wiki Bookworm tutorial)

```bash
sudo raspi-config          # Interface Options → enable SPI and I2C
# /boot/firmware/config.txt: add `dtoverlay=spi0-0cs` below `dtparam=spi=on`
wget https://files.waveshare.com/wiki/Thermal_Camera_Module/Thermal_Camera_Hat.zip
unzip Thermal_Camera_Hat.zip && cd pysenxor-master && pip install -e ./
pip install gpiozero smbus spidev crcmod
```

## Recording

```bash
python scripts/acquire_mi48.py 2ppl_walk --duration 60          # single camera
python scripts/acquire_mi48.py 2ppl_walk --frames 480 --config cams.json  # multi-cam
```

`senxor`, `smbus`, `spidev` and `gpiozero` are imported lazily inside
`MI48Camera.open()`, so this package imports and tests fine on dev machines
(`tests/test_acquisition.py` runs with the hardware fully mocked).
