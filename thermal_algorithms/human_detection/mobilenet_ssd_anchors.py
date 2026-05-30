"""Anchor matching, box encoding/decoding, and SSD losses.

These utilities are profile-agnostic - they consume the anchor grid produced
by MicroMobileNetSSD.build_anchors() and operate on tensors in input-pixel
coordinates throughout.

Box representations used in this module
---------------------------------------
* (cx, cy, w, h) - center + size, the canonical representation for anchors
                   and ground-truth boxes.
* (x1, y1, x2, y2) - corner form, only used internally for IoU computation.
* (dx, dy, dlog_w, dlog_h) - offsets relative to an anchor, the regression
                              target. Multiplied/divided by the standard SSD
                              variance scales {0.1, 0.1, 0.2, 0.2}.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# Standard SSD variance scales (Liu et al. 2016).
BBOX_VARIANCE_XY = 0.1
BBOX_VARIANCE_WH = 0.2


# ---------------------------------------------------------------------------
# Box geometry helpers
# ---------------------------------------------------------------------------

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(N, 4) cx,cy,w,h -> (N, 4) x1,y1,x2,y2."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """(N, 4) x1,y1,x2,y2 -> (N, 4) cx,cy,w,h."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)


def pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """IoU between every pair of boxes.

    Args
    ----
    boxes_a : (A, 4) in cx,cy,w,h.
    boxes_b : (B, 4) in cx,cy,w,h.

    Returns
    -------
    iou : (A, B) tensor.
    """
    a = cxcywh_to_xyxy(boxes_a)                                # (A, 4)
    b = cxcywh_to_xyxy(boxes_b)                                # (B, 4)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])         # (A,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])         # (B,)

    # Pairwise intersection rectangles.
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])         # (A, B, 2)
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])         # (A, B, 2)
    wh = (rb - lt).clamp(min=0)                                 # (A, B, 2)
    inter = wh[..., 0] * wh[..., 1]                            # (A, B)

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Encode / decode offsets
# ---------------------------------------------------------------------------

def encode_boxes(boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Encode (cx,cy,w,h) boxes as offsets relative to anchors.

    Standard SSD encoding:
        dx     = (cx_gt - cx_a) / w_a / variance_xy
        dy     = (cy_gt - cy_a) / h_a / variance_xy
        dlog_w = log(w_gt / w_a) / variance_wh
        dlog_h = log(h_gt / h_a) / variance_wh

    Args
    ----
    boxes   : (N, 4) ground-truth boxes in cxcywh.
    anchors : (N, 4) matching anchors in cxcywh (broadcasts okay).
    """
    cx_gt, cy_gt, w_gt, h_gt = boxes.unbind(-1)
    cx_a, cy_a, w_a, h_a = anchors.unbind(-1)
    dx = (cx_gt - cx_a) / w_a / BBOX_VARIANCE_XY
    dy = (cy_gt - cy_a) / h_a / BBOX_VARIANCE_XY
    dlog_w = torch.log(w_gt / w_a) / BBOX_VARIANCE_WH
    dlog_h = torch.log(h_gt / h_a) / BBOX_VARIANCE_WH
    return torch.stack((dx, dy, dlog_w, dlog_h), dim=-1)


def decode_boxes(offsets: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Inverse of encode_boxes - turn predicted offsets into absolute cxcywh."""
    dx, dy, dlog_w, dlog_h = offsets.unbind(-1)
    cx_a, cy_a, w_a, h_a = anchors.unbind(-1)
    cx = dx * BBOX_VARIANCE_XY * w_a + cx_a
    cy = dy * BBOX_VARIANCE_XY * h_a + cy_a
    w = torch.exp(dlog_w * BBOX_VARIANCE_WH) * w_a
    h = torch.exp(dlog_h * BBOX_VARIANCE_WH) * h_a
    return torch.stack((cx, cy, w, h), dim=-1)


# ---------------------------------------------------------------------------
# Anchor matching
# ---------------------------------------------------------------------------

# Sentinel target index for "no GT assigned" (ignore).
NO_GT = -1
NEG_GT = -2   # explicit background marker


def match_anchors_to_targets(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_pos_threshold: float = 0.5,
    iou_neg_threshold: float = 0.4,
) -> torch.Tensor:
    """For each anchor, assign a GT index (or -2 for negative, -1 for ignore).

    Strategy (standard SSD):
        1. Compute pairwise IoU(anchor, gt).
        2. Every GT is assigned to the anchor with the largest IoU (forced
           positive match - guarantees every GT has at least one positive
           anchor, even if max IoU < pos_threshold).
        3. Every anchor with max-over-GT IoU >= pos_threshold is also positive
           and assigned to its argmax GT.
        4. Every anchor with max-over-GT IoU < neg_threshold is negative.
        5. Anchors in between are ignored.

    Args
    ----
    anchors  : (A, 4) cxcywh in input-pixel coords.
    gt_boxes : (G, 4) cxcywh in input-pixel coords. May be empty (G=0).

    Returns
    -------
    matches : (A,) int64 tensor.
              >= 0 = index into gt_boxes (positive).
               -2 = negative (background).
               -1 = ignore.
    """
    A = anchors.shape[0]
    if gt_boxes.numel() == 0:
        return torch.full((A,), NEG_GT, dtype=torch.long, device=anchors.device)

    iou = pairwise_iou(anchors, gt_boxes)                       # (A, G)
    max_iou, max_idx = iou.max(dim=1)                           # (A,)

    matches = torch.full((A,), NO_GT, dtype=torch.long, device=anchors.device)
    matches[max_iou < iou_neg_threshold] = NEG_GT
    pos_mask = max_iou >= iou_pos_threshold
    matches[pos_mask] = max_idx[pos_mask]

    # Force each GT to have its best-IoU anchor assigned, even if below threshold.
    best_anchor_per_gt = iou.argmax(dim=0)                      # (G,)
    matches[best_anchor_per_gt] = torch.arange(
        gt_boxes.shape[0], device=anchors.device, dtype=torch.long,
    )

    return matches


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------

def hard_negative_mining(
    cls_logits: torch.Tensor,
    matches: torch.Tensor,
    neg_pos_ratio: float = 3.0,
) -> torch.Tensor:
    """Select the hardest N_pos * ratio negatives per sample.

    "Hard" means: the negative that the classifier confidently believes is a
    person (low score for class 0 background). We sort negatives by their
    background score (ascending) and keep the lowest.

    Args
    ----
    cls_logits : (A, num_classes) - classification logits for one sample.
    matches    : (A,) - anchor assignments from `match_anchors_to_targets`.
    neg_pos_ratio : Ratio of negatives to positives to keep.

    Returns
    -------
    keep : (A,) bool mask - True for anchors that contribute to the
           classification loss.
    """
    pos_mask = matches >= 0
    n_pos = int(pos_mask.sum().item())
    if n_pos == 0:
        # No positives - keep a small fixed budget of hard negatives.
        n_neg_keep = min(8, int((matches == NEG_GT).sum().item()))
    else:
        n_neg_keep = min(int(neg_pos_ratio * n_pos), int((matches == NEG_GT).sum().item()))

    # Background = class 0. A confident classification of background has high
    # score for class 0; a HARD negative has low score for class 0.
    bg_score = F.log_softmax(cls_logits, dim=-1)[:, 0]            # (A,)
    # Mask out non-negatives so they don't get chosen.
    masked = bg_score.clone()
    masked[matches != NEG_GT] = float("inf")

    # Lowest bg score = hardest negative.
    n_neg_keep = max(n_neg_keep, 0)
    if n_neg_keep == 0:
        keep_neg = torch.zeros_like(matches, dtype=torch.bool)
    else:
        _, idx = masked.topk(n_neg_keep, largest=False)
        keep_neg = torch.zeros_like(matches, dtype=torch.bool)
        keep_neg[idx] = True

    return pos_mask | keep_neg


# ---------------------------------------------------------------------------
# SSD loss
# ---------------------------------------------------------------------------

def ssd_loss(
    box_preds: torch.Tensor,                  # (B, A, 4) predicted offsets
    cls_preds: torch.Tensor,                  # (B, A, num_classes)
    anchors: torch.Tensor,                    # (A, 4) cxcywh
    targets: list[torch.Tensor],              # list of (G_b, 4) GT boxes per sample
    *,
    iou_pos_threshold: float = 0.5,
    iou_neg_threshold: float = 0.4,
    neg_pos_ratio: float = 3.0,
    loc_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the standard SSD loss = classification + localization.

    Returns
    -------
    dict with keys 'loss', 'cls_loss', 'loc_loss', 'n_pos' (for logging).
    """
    B = box_preds.shape[0]
    device = box_preds.device

    total_cls = box_preds.new_zeros(())
    total_loc = box_preds.new_zeros(())
    total_pos = 0

    for b in range(B):
        gt = targets[b].to(device)
        matches = match_anchors_to_targets(
            anchors, gt, iou_pos_threshold, iou_neg_threshold,
        )                                                      # (A,)

        pos_mask = matches >= 0
        n_pos = int(pos_mask.sum().item())
        total_pos += n_pos

        # Localization loss: smooth L1 on encoded offsets, positives only.
        if n_pos > 0:
            matched_gt = gt[matches[pos_mask]]                 # (N_pos, 4)
            matched_anchors = anchors[pos_mask]                # (N_pos, 4)
            encoded_target = encode_boxes(matched_gt, matched_anchors)
            loc = F.smooth_l1_loss(
                box_preds[b][pos_mask], encoded_target, reduction="sum",
            )
            total_loc = total_loc + loc

        # Classification loss: positives + hard negatives only.
        keep = hard_negative_mining(cls_preds[b], matches, neg_pos_ratio)
        if keep.any():
            cls_targets = torch.where(
                matches >= 0,
                torch.ones_like(matches),     # class 1 = person
                torch.zeros_like(matches),    # class 0 = background
            )
            cls = F.cross_entropy(
                cls_preds[b][keep], cls_targets[keep], reduction="sum",
            )
            total_cls = total_cls + cls

    # Normalize by number of positives across the batch (standard SSD recipe).
    norm = max(total_pos, 1)
    cls_loss = total_cls / norm
    loc_loss = total_loc / norm
    loss = cls_loss + loc_loss_weight * loc_loss

    return {
        "loss": loss,
        "cls_loss": cls_loss.detach(),
        "loc_loss": loc_loss.detach(),
        "n_pos": torch.tensor(total_pos, device=device),
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def decode_predictions(
    box_preds: torch.Tensor,                  # (A, 4) offsets - SINGLE sample
    cls_preds: torch.Tensor,                  # (A, num_classes)
    anchors: torch.Tensor,                    # (A, 4) cxcywh
    score_threshold: float = 0.5,
    class_id: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode raw SSD predictions for ONE image to (boxes_xyxy, scores).

    Returns boxes that pass the score threshold for the target class.
    The caller is expected to run NMS afterwards.
    """
    decoded = decode_boxes(box_preds, anchors)                  # (A, 4) cxcywh
    boxes_xyxy = cxcywh_to_xyxy(decoded)
    probs = F.softmax(cls_preds, dim=-1)                        # (A, num_classes)
    scores = probs[:, class_id]
    keep = scores > score_threshold
    return boxes_xyxy[keep], scores[keep]


def nms_xyxy(
    boxes_xyxy: torch.Tensor,                 # (N, 4)
    scores: torch.Tensor,                     # (N,)
    iou_threshold: float = 0.3,
) -> torch.Tensor:
    """Greedy NMS. Returns indices to keep."""
    if boxes_xyxy.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)

    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep_indices: list[int] = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep_indices.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-6)
        order = rest[iou < iou_threshold]

    return torch.tensor(keep_indices, dtype=torch.long, device=boxes_xyxy.device)
