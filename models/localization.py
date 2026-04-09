"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

IMAGE_SIZE = 224  # VGG11 input size


class VGG11Localizer(nn.Module):
    """VGG11-based single-object localizer.

    Encoder strategy: Freeze all, then selectively unfreeze block5.
    The entire VGG11 encoder is first frozen after loading pretrained
    classification weights.  Then block5 (the last conv block) is
    unfrozen and fine-tuned with a small LR so the last-level features
    can specialise for spatial regression rather than classification.

    Regression head: Compact MLP.
    25,088 -> 1024 -> 256 -> 4 with a single Dropout(0.2) after the
    first FC layer.  Fewer layers = fewer things to overfit.
    BatchNorm at each hidden layer provides sufficient regularisation.

    Output activation: Sigmoid + Scaling.
    Bounding-box coordinates are mapped to [0, 1] via Sigmoid, then
    scaled by IMAGE_SIZE to strictly bind them to [0, 224] pixel space.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.2):
        """
        Args:
            in_channels: Number of input channels (default 3 for RGB).
            dropout_p:   Dropout probability after the first FC layer (default 0.2).
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ------------------------------------------------------------------
        # Compact MLP regression head
        # Input:  512 x 7 x 7 = 25,088 flattened features
        # Output: 4 constrained values (cx, cy, w, h) in pixel space
        # ------------------------------------------------------------------
        self.reg_head = nn.Sequential(
            nn.Flatten(),                        # [B, 25088]

            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),          # single dropout - light regularisation

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),
            nn.Sigmoid(),                        # Bounds output strictly to [0.0, 1.0]
        )

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self) -> None:
        """Unfreeze encoder.block5 so the last conv features can adapt
        to localisation with a small learning rate.
        """
        for p in self.encoder.block5.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Returns:
            Bounding box tensor [B, 4] in (cx, cy, w, h) pixel space.
        """
        features = self.encoder(x)           # [B, 512, 7, 7]
        features = self.avgpool(features)    # [B, 512, 7, 7]
        out = self.reg_head(features)        # [B, 4] in [0, 1]
        
        # Scale to [0, 224] so the loss function and metrics interact with proper pixel coordinates
        return out * IMAGE_SIZE