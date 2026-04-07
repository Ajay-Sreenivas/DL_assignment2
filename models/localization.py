"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based single-object localizer.

    Uses the pre-trained VGG11 convolutional backbone as a frozen feature
    extractor and attaches a lightweight regression head to predict the
    bounding box as (x_center, y_center, width, height) in pixel space.

    Freezing strategy: We freeze the entire encoder and fine-tune only the
    regression head.  This is justified because:
    1. The Oxford-IIIT Pet dataset is small enough that full fine-tuning risks
       over-fitting the backbone to localisation noise.
    2. Features learned for breed classification (textures, shapes, edges) are
       highly transferable to head localisation — the same convolutional
       patterns that discriminate breeds also locate the head region.
    3. A frozen backbone dramatically reduces training time and GPU memory while
       still yielding competitive IoU scores.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # [B, 512, 7, 7]

        # Regression head: outputs [xcenter, ycenter, width, height] in pixel space.
        # ReLU on the output keeps all values ≥ 0 (coordinates/dims are non-negative).
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(256, 4),
            nn.ReLU(),  # bbox coordinates are non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            format in original image pixel space (not normalised values).
        """
        features = self.encoder(x)        # [B, 512, 7, 7]
        features = self.avgpool(features) # [B, 512, 7, 7]
        return self.reg_head(features)    # [B, 4]
