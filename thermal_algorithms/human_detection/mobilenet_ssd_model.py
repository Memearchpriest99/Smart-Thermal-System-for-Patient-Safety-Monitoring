"""MicroMobileNet-SSD - the nn.Module architecture for the deep-learning
human detector (Section 4.4.2.3).

Adapted-resolution design
-------------------------
Rather than upsampling 32x24 (MLX) or 80x62 (Waveshare) thermal frames to
300x300 like the original MobileNet-SSD spec, this architecture operates at
the native sensor resolution. Two backbone configurations are provided, one
per sensor profile; both end at roughly the same feature-map sizes so the
SSD head is identical across profiles.

Backbone (MLX, input 32x24x1)
    Stem  conv 3x3 s1                32x24x16
    DW block s1                       32x24x32
    DW block s2  -> F1 feature map    16x12x64
    DW block s1                       16x12x64
    DW block s2  -> F2 feature map     8x 6x128
    DW block s1                        8x 6x128

Backbone (Waveshare, input 80x62x1)
    Stem  conv 3x3 s2                40x31x16
    DW block s1                      40x31x32
    DW block s2  -> F1 feature map   20x15x64
    DW block s1                      20x15x64
    DW block s2  -> F2 feature map   10x 7x128
    DW block s1                      10x 7x128

SSD head (shared across profiles)
    For each cell of F1, F2: predict K anchors with (4 box offsets + 2 class
    logits). With K=3 aspect ratios:
        MLX:        16x12*3 + 8x6*3 =  720 anchors
        Waveshare:  20x15*3 + 10x7*3 = 1110 anchors

The architecture intentionally keeps the same K, anchor scales, and feature
map count across profiles - only the input/output resolutions differ. That
lets the matching/loss code (which lives in mobilenet_ssd_anchors.py) be
profile-agnostic.

This module declares the model architecture and anchor generation only.
Training, inference, and the public detector API live in mobilenet_ssd.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Architecture config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackboneConfig:
    """Per-profile backbone configuration.

    Attributes
    ----------
    stem_stride : Initial conv stride. 1 for MLX (preserve resolution),
        2 for Waveshare (downsample once to get F1 ~16x12).
    channels    : Number of channels at each stage [stem, stage1, F1, F2].
    """
    stem_stride: int = 1
    channels: tuple[int, int, int, int] = (16, 32, 64, 128)


@dataclass(frozen=True)
class SSDConfig:
    """Per-profile SSD-head configuration.

    Anchor sizes are specified in input-pixel units. With three aspect ratios
    each (1:1, 1:2, 1:3 - person-like tall-skinny boxes), each feature map
    cell produces K=3 anchors.

    Attributes
    ----------
    num_classes      : 2 (background + person). Background is class 0.
    anchor_widths    : Anchor width in input pixels at each feature map (F1, F2).
    aspect_ratios    : (w / h) values defining anchor shapes.
                       Defaults: (1.0, 0.5, 0.33) = squares, 2x-tall, 3x-tall.
    """
    num_classes: int = 2
    anchor_widths: tuple[float, float] = (4.0, 6.0)
    aspect_ratios: tuple[float, ...] = (1.0, 0.5, 0.33)

    @property
    def num_anchors_per_cell(self) -> int:
        return len(self.aspect_ratios)


# Pre-defined configs for the two supported sensor profiles.
MLX90640_BACKBONE = BackboneConfig(stem_stride=1, channels=(16, 32, 64, 128))
WAVESHARE_BACKBONE = BackboneConfig(stem_stride=2, channels=(16, 32, 64, 128))

MLX90640_SSD = SSDConfig(anchor_widths=(4.0, 6.0))
WAVESHARE_SSD = SSDConfig(anchor_widths=(10.0, 16.0))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DepthwiseSeparableBlock(nn.Module):
    """A standard MobileNet depthwise-separable conv block.

        Depthwise 3x3 (stride s) -> BN -> ReLU
        Pointwise 1x1            -> BN -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


# ---------------------------------------------------------------------------
# MicroMobileNet backbone
# ---------------------------------------------------------------------------

class MicroMobileNet(nn.Module):
    """MobileNet-style backbone scaled to native thermal sensor resolution.

    Outputs two feature maps F1, F2 at roughly comparable spatial sizes across
    the MLX and Waveshare profiles (the difference is absorbed by the stem
    stride). F1 is used for smaller targets, F2 for larger.
    """

    def __init__(self, config: BackboneConfig, in_channels: int = 1) -> None:
        super().__init__()
        c_stem, c_s1, c_f1, c_f2 = config.channels

        # Stem: regular conv (depthwise-separable doesn't make sense with a
        # single input channel).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_stem, kernel_size=3, stride=config.stem_stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(c_stem),
            nn.ReLU(inplace=True),
        )
        # Stage 1: same resolution as stem output, expand channels.
        self.stage1 = DepthwiseSeparableBlock(c_stem, c_s1, stride=1)

        # F1: downsample once.
        self.to_f1 = DepthwiseSeparableBlock(c_s1, c_f1, stride=2)
        self.f1_refine = DepthwiseSeparableBlock(c_f1, c_f1, stride=1)

        # F2: downsample again.
        self.to_f2 = DepthwiseSeparableBlock(c_f1, c_f2, stride=2)
        self.f2_refine = DepthwiseSeparableBlock(c_f2, c_f2, stride=1)

        self._out_channels: tuple[int, int] = (c_f1, c_f2)

    @property
    def out_channels(self) -> tuple[int, int]:
        """Channel count for (F1, F2) outputs."""
        return self._out_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        f1 = self.f1_refine(self.to_f1(x))
        f2 = self.f2_refine(self.to_f2(f1))
        return f1, f2


# ---------------------------------------------------------------------------
# SSD head
# ---------------------------------------------------------------------------

class SSDHead(nn.Module):
    """SSD prediction head.

    For each cell of the input feature map, produces K * (4 + num_classes)
    outputs:
        - First 4*K: box regression offsets (Δcx, Δcy, Δlog_w, Δlog_h) per anchor.
        - Remaining num_classes*K: class logits per anchor.
    """

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.box_head = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (box_offsets, class_logits).

        box_offsets : (B, H*W*K, 4)
        class_logits: (B, H*W*K, num_classes)
        """
        B = x.shape[0]
        box = self.box_head(x)                                # (B, K*4, H, W)
        cls = self.cls_head(x)                                # (B, K*C, H, W)
        # (B, K*4, H, W) -> (B, H, W, K, 4) -> (B, H*W*K, 4)
        box = box.permute(0, 2, 3, 1).reshape(B, -1, 4)
        cls = cls.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
        return box, cls


# ---------------------------------------------------------------------------
# Anchor generation
# ---------------------------------------------------------------------------

def generate_anchors(
    feature_map_size: tuple[int, int],
    input_size: tuple[int, int],
    anchor_width: float,
    aspect_ratios: tuple[float, ...],
) -> torch.Tensor:
    """Generate anchor boxes for one feature map level.

    Each cell (i, j) of the H x W feature map gets `len(aspect_ratios)` anchors,
    centered at the corresponding input-pixel location. Anchor sizes are
    derived from anchor_width and aspect_ratio = w/h:
        anchor_w = anchor_width
        anchor_h = anchor_width / aspect_ratio

    Returns
    -------
    anchors : (H*W*K, 4) tensor of [cx, cy, w, h] in input-pixel coordinates.
    """
    fh, fw = feature_map_size
    ih, iw = input_size
    stride_y = ih / fh
    stride_x = iw / fw

    # Cell centers in input pixel coordinates.
    ys = (torch.arange(fh, dtype=torch.float32) + 0.5) * stride_y
    xs = (torch.arange(fw, dtype=torch.float32) + 0.5) * stride_x
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")     # (H, W)

    anchors = []
    for ar in aspect_ratios:
        w = torch.full_like(xx, float(anchor_width))
        h = w / float(ar)
        # (H, W, 4) for this aspect ratio
        a = torch.stack((xx, yy, w, h), dim=-1)
        anchors.append(a)
    # (H, W, K, 4) -> (H*W*K, 4)
    out = torch.stack(anchors, dim=2).reshape(-1, 4)
    return out


# ---------------------------------------------------------------------------
# Full detector module
# ---------------------------------------------------------------------------

class MicroMobileNetSSD(nn.Module):
    """End-to-end SSD detector at native thermal resolution.

    Forward returns:
        box_preds : (B, N_anchors, 4)  raw offsets (NOT decoded to absolute boxes)
        cls_preds : (B, N_anchors, num_classes)

    The corresponding anchor grid is accessible as `self.anchors` - a buffer
    of shape (N_anchors, 4) in [cx, cy, w, h] input-pixel coordinates.

    Decoding to absolute boxes, NMS, and loss computation happen in
    mobilenet_ssd_anchors.py.
    """

    def __init__(
        self,
        input_size: tuple[int, int],          # (H, W) of the thermal frame
        backbone_config: BackboneConfig,
        ssd_config: SSDConfig,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.backbone_config = backbone_config
        self.ssd_config = ssd_config

        self.backbone = MicroMobileNet(backbone_config, in_channels=1)
        c_f1, c_f2 = self.backbone.out_channels

        K = ssd_config.num_anchors_per_cell
        self.head_f1 = SSDHead(c_f1, num_anchors=K, num_classes=ssd_config.num_classes)
        self.head_f2 = SSDHead(c_f2, num_anchors=K, num_classes=ssd_config.num_classes)

        # Generate anchor grid lazily on the first forward (we need to know F1/F2
        # spatial sizes for the chosen input_size). Store as a non-persistent buffer.
        self._anchors: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _compute_anchors(self, f1_shape: tuple[int, int], f2_shape: tuple[int, int]) -> torch.Tensor:
        a1 = generate_anchors(
            f1_shape, self.input_size,
            anchor_width=self.ssd_config.anchor_widths[0],
            aspect_ratios=self.ssd_config.aspect_ratios,
        )
        a2 = generate_anchors(
            f2_shape, self.input_size,
            anchor_width=self.ssd_config.anchor_widths[1],
            aspect_ratios=self.ssd_config.aspect_ratios,
        )
        return torch.cat([a1, a2], dim=0)

    @property
    def anchors(self) -> torch.Tensor:
        """All anchors stacked, shape (N_anchors, 4) [cx, cy, w, h]. Computed
        lazily on the first forward pass."""
        if self._anchors is None:
            raise RuntimeError(
                "Anchors not yet computed. Call forward() once or invoke "
                "`build_anchors()` explicitly."
            )
        return self._anchors

    def build_anchors(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Materialize the anchor grid by running a dummy forward pass through
        the backbone (cheap; no SSD heads). Returns the anchor tensor."""
        device = device or next(self.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *self.input_size, device=device)
            f1, f2 = self.backbone(dummy)
            anchors = self._compute_anchors(f1.shape[-2:], f2.shape[-2:])
            self._anchors = anchors.to(device)
        return self._anchors

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)   # accept (B, H, W) as well
        f1, f2 = self.backbone(x)

        # First forward computes the anchor grid.
        if self._anchors is None:
            anchors = self._compute_anchors(f1.shape[-2:], f2.shape[-2:])
            self._anchors = anchors.to(x.device)

        box1, cls1 = self.head_f1(f1)
        box2, cls2 = self.head_f2(f2)
        box = torch.cat([box1, box2], dim=1)            # (B, N, 4)
        cls = torch.cat([cls1, cls2], dim=1)            # (B, N, num_classes)
        return box, cls
