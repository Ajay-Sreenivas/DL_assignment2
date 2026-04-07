"""Custom IoU loss and combined localisation loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Boxes are expected in (x_center, y_center, width, height) format.
    Works for both pixel-space and normalised [0, 1] coordinates.

    The loss is defined as  L = 1 - IoU  so it lies in [0, 1]:
      * IoU = 1  (perfect overlap)  →  loss = 0
      * IoU = 0  (no overlap)       →  loss = 1

    A small epsilon is added to the denominator for numerical stability.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialise the IoULoss module.

        Args:
            eps:       Small value to avoid division by zero.
            reduction: 'mean' | 'sum' | 'none'.
        """
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes:   [B, 4] predicted boxes  (cx, cy, w, h).
            target_boxes: [B, 4] target boxes      (cx, cy, w, h).

        Returns:
            Scalar loss (reduction='mean'/'sum') or per-sample [B] (reduction='none').
        """
        # Convert (cx, cy, w, h) → (x1, y1, x2, y2)
        pred_x1  = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1  = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2  = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2  = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        tgt_x1   = target_boxes[:, 0] - target_boxes[:, 2] / 2
        tgt_y1   = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tgt_x2   = target_boxes[:, 0] + target_boxes[:, 2] / 2
        tgt_y2   = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w  = (inter_x2 - inter_x1).clamp(min=0)
        inter_h  = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        tgt_area  = (tgt_x2 - tgt_x1).clamp(min=0)  * (tgt_y2 - tgt_y1).clamp(min=0)
        union_area = pred_area + tgt_area - inter_area + self.eps

        iou  = inter_area / union_area   # [B]
        loss = 1.0 - iou                 # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedLocLoss(nn.Module):
    """Combined IoU + L1 localisation loss.  (Fix 4)

    Pure IoU loss has two failure modes:
      1. When predicted and target boxes do not overlap at all, the IoU
         gradient is exactly zero — the model receives no signal on *how far
         off* it is.
      2. IoU is scale-invariant, so small boxes and large boxes contribute
         equally, which can destabilise early training.

    Adding an L1 (smooth-L1 / Huber) term fixes both issues:
      * It provides a non-zero gradient even for non-overlapping boxes.
      * It acts as a direct coordinate regression signal that kick-starts
        training before boxes start overlapping.

    Loss = IoU_loss + λ * L1_loss
    Default λ = 0.5 balances the two terms without heavy tuning.

    Both losses expect boxes in normalised [0, 1] space so their magnitudes
    are comparable.
    """

    def __init__(self, lambda_l1: float = 0.5, eps: float = 1e-6):
        """
        Initialise the combined loss.

        Args:
            lambda_l1: Weight for the L1 term (default 0.5).
            eps:       Epsilon forwarded to IoULoss for numerical stability.
        """
        super().__init__()
        self.iou_loss  = IoULoss(eps=eps, reduction="mean")
        self.lambda_l1 = lambda_l1

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute combined IoU + L1 loss.

        Args:
            pred_boxes:   [B, 4] predicted boxes  (cx, cy, w, h) ∈ [0, 1].
            target_boxes: [B, 4] target boxes      (cx, cy, w, h) ∈ [0, 1].

        Returns:
            Scalar combined loss.
        """
        iou = self.iou_loss(pred_boxes, target_boxes)
        l1  = F.smooth_l1_loss(pred_boxes, target_boxes)
        return iou + self.lambda_l1 * l1