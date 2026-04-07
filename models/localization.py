"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based single-object localizer.

    Uses the VGG11 convolutional backbone as a partially fine-tuned feature
    extractor and attaches a spatial regression head to predict the bounding
    box as (x_center, y_center, width, height) in *normalised* [0, 1] space
    (i.e. divided by image width/height).

    Fine-tuning strategy: We freeze the early encoder blocks (block1–block3)
    which capture generic low-level features (edges, textures, colours) that
    transfer well from classification.  We unfreeze block4 and block5 so the
    deeper spatial features can adapt to the WHERE-is-the-object task.
    Rationale:
      1. Early layers are task-agnostic; freezing them reduces over-fitting on
         the small Oxford-IIIT Pet dataset.
      2. Later layers encode high-level spatial structure (head, body) that
         must shift from class-discriminative to location-discriminative
         representations — these need to be fine-tuned.
      3. Partial fine-tuning typically gains 10–20 IoU points over a fully
         frozen encoder on this dataset.

    Head design: Instead of flattening 7×7×512 directly into an MLP we first
    apply two convolutional layers to let the head reason spatially before
    collapsing with GlobalAvgPool.  This retains structural information longer
    and reduces the parameter count of the FC stage.

    Output activation: Sigmoid maps all four coordinates to [0, 1].  Training
    targets must also be normalised to [0, 1] (divide cx, cy by W; w, h by W
    and H respectively, or normalise uniformly by the shorter side).  Sigmoid
    stabilises regression training compared with unbounded ReLU activations.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialise the VGG11Localizer model.

        Args:
            in_channels: Number of input channels (default 3 for RGB).
            dropout_p:   Dropout probability in the regression head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ------------------------------------------------------------------
        # Spatial conv head  (Fix 3)
        # Retains spatial structure longer than an immediate flatten.
        # 512 → 256 → 128 channels, then GlobalAvgPool collapses H×W to 1×1.
        # ------------------------------------------------------------------
        self.conv_head = nn.Sequential(
            # 7×7×512 → 7×7×256
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 7×7×256 → 7×7×128
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 7×7×128 → 1×1×128
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # ------------------------------------------------------------------
        # FC regression head
        # Sigmoid final activation keeps output in [0, 1]  (Fix 2)
        # ------------------------------------------------------------------
        self.reg_head = nn.Sequential(
            nn.Flatten(),                      # [B, 128]
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(256, 4),
            nn.Sigmoid(),                      # → [B, 4] in [0, 1]
        )

    def freeze_encoder_partial(self) -> None:
        """Freeze blocks 1–3; leave blocks 4–5 trainable.  (Fix 1)

        Call this after loading pretrained weights and before creating the
        optimiser so that only the unfrozen parameters are passed to the
        optimiser.
        """
        # Freeze everything first
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Unfreeze the two deepest blocks so they adapt to localisation
        for name, p in self.encoder.named_parameters():
            if "block4" in name or "block5" in name:
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the localisation model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box tensor [B, 4] in (x_center, y_center, width, height)
            format, **normalised to [0, 1]** (divide by image side length to
            recover pixel coordinates).
        """
        features = self.encoder(x)       # [B, 512, 7, 7]
        features = self.conv_head(features)  # [B, 128, 1, 1]
        return self.reg_head(features)   # [B, 4]  ∈ [0, 1]