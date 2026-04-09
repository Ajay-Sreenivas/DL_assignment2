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
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.2):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.reg_head = nn.Sequential(
            nn.Flatten(),                        # [B, 25088]

            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),          

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),
            nn.Sigmoid(),                        # Ensures output is [0, 1]
        )

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self) -> None:
        """Unfreeze encoder.block5."""
        for p in self.encoder.block5.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns bounding box tensor [B, 4] in (cx, cy, w, h) pixel space."""
        features = self.encoder(x)           
        features = self.avgpool(features)    
        # Scale the Sigmoid [0, 1] output to strictly [0, 224] pixel space
        return self.reg_head(features) * IMAGE_SIZE