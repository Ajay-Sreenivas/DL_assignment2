"""Custom IoU loss
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Boxes are expected in (x_center, y_center, width, height) format in pixel
    space.  The loss is defined as  L = 1 - IoU  so it lies in [0, 1]:
      * IoU = 1  (perfect overlap)  →  loss = 0
      * IoU = 0  (no overlap)       →  loss = 1

    A small epsilon is added to the denominator for numerical stability and to
    keep gradients well-defined even when boxes do not overlap.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output:
                       'mean' | 'sum' | 'none'.
        """
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes:   [B, 4] predicted boxes  (x_center, y_center, w, h) in pixel space.
            target_boxes: [B, 4] target boxes      (x_center, y_center, w, h) in pixel space.

        Returns:
            Scalar loss (if reduction='mean'/'sum') or per-sample tensor [B] (if 'none').
        """
        # Convert (cx, cy, w, h) → (x1, y1, x2, y2)
        pred_x1   = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1   = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2   = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2   = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        tgt_x1    = target_boxes[:, 0] - target_boxes[:, 2] / 2
        tgt_y1    = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tgt_x2    = target_boxes[:, 0] + target_boxes[:, 2] / 2
        tgt_y2    = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # Intersection
        inter_x1  = torch.max(pred_x1, tgt_x1)
        inter_y1  = torch.max(pred_y1, tgt_y1)
        inter_x2  = torch.min(pred_x2, tgt_x2)
        inter_y2  = torch.min(pred_y2, tgt_y2)

        inter_w   = (inter_x2 - inter_x1).clamp(min=0)
        inter_h   = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        pred_area  = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        tgt_area   = (tgt_x2  - tgt_x1).clamp(min=0)  * (tgt_y2  - tgt_y1).clamp(min=0)
        union_area  = pred_area + tgt_area - inter_area + self.eps

        iou        = inter_area / union_area            # [B], in [0, 1]
        loss       = 1.0 - iou                          # [B], in [0, 1]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss