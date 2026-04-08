"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

IMAGE_SIZE = 224  # VGG11 input size


class VGG11Localizer(nn.Module):
    """VGG11-based single-object localizer.

    Encoder strategy: Fully frozen.
    The entire VGG11 encoder is frozen after loading pretrained classification
    weights.  Justification:
      1. Oxford-IIIT Pet is a small dataset (~3,600 training images).  Fine-tuning
         the encoder with only 30 epochs does not give the backbone enough updates
         to converge to a better representation — it instead degrades the well-
         trained classification features before the head can compensate.
      2. VGG11 features trained on 37-class pet classification already encode
         strong spatial signals (head shape, ears, snout) that transfer directly
         to localisation.  Freezing preserves these signals intact.
      3. A frozen backbone lets the optimiser focus all capacity on the regression
         head, which is randomly initialised and needs the most updates.

    Regression head: Deep MLP.
    Flattening 512×7×7 = 25,088 frozen features and passing them through a
    MLP gives the head sufficient capacity to learn the bbox mapping.  Five FC
    layers (2048 → 1024 → 512 → 256 → 4) with BatchNorm and Dropout at every
    hidden layer (including the 256 layer) provide strong regularisation.
    First layer is reduced from 4096 → 2048 to cut the parameter count and
    reduce memorisation risk.

    Output activation: ReLU.
    Bounding-box coordinates are non-negative pixel values in [0, IMAGE_SIZE].
    ReLU enforces non-negativity while keeping the output unbounded upward,
    matching the target space exactly.  Sigmoid would compress output to [0, 1]
    and introduce saturation that slows gradient flow.
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
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ------------------------------------------------------------------
        # Deep MLP regression head
        # Input:  512 × 7 × 7 = 25,088 flattened features
        # Output: 4 non-negative values (cx, cy, w, h) in pixel space
        # ------------------------------------------------------------------
        self.reg_head = nn.Sequential(
            nn.Flatten(),                        # [B, 25088]

            nn.Linear(512 * 7 * 7, 2048),        # reduced from 4096 → less memorisation
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),          # added — regularise pre-output layer

            nn.Linear(256, 4),
            nn.ReLU(),                           # non-negative pixel coordinates
        )

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters.

        Called explicitly in train.py after loading pretrained weights so that
        the optimiser only updates the regression head.
        """
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the localisation model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box tensor [B, 4] in (x_center, y_center, width, height)
            format in pixel space — values in [0, IMAGE_SIZE].
        """
        features = self.encoder(x)           # [B, 512, 7, 7]
        features = self.avgpool(features)    # [B, 512, 7, 7]
        return self.reg_head(features)       # [B, 4]  pixel space