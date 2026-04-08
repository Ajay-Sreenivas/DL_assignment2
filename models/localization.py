"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

IMAGE_SIZE = 224  # VGG11 input size


class VGG11Localizer(nn.Module):
    """VGG11-based single-object localizer.

    Encoder strategy: Partial fine-tuning.
    - block1, block2, block3 FROZEN  — generic low-level features that transfer well.
    - block4, block5        UNFROZEN — object-specific spatial features that must
                                       adapt from classification → localisation.

    Regression head: FC-based with Sigmoid output.
    - Sigmoid constrains outputs to [0, 1].
    - forward() multiplies by IMAGE_SIZE to return pixel-space coordinates.
    - FC head converges faster than Conv head from a random init, which is
      critical since only the head weights start from scratch.

    Output format: [x_center, y_center, width, height] in pixel space [0, 224].
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p:   Dropout probability for the regression head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # [B, 512, 7, 7]

        # FC regression head with Sigmoid output
        # 512*7*7 = 25088 → 1024 → 256 → 4
        self.reg_head = nn.Sequential(
            nn.Flatten(),                        # [B, 25088]
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(256, 4),
            nn.Sigmoid(),                        # [B, 4] in [0, 1]
        )

        # Apply partial freeze immediately (re-applied after weight loading in train.py)
        self._freeze_early_blocks()

    def _freeze_early_blocks(self):
        """Freeze block1/2/3, unfreeze block4/5 of the encoder."""
        for name, param in self.encoder.named_parameters():
            if any(b in name for b in ["block1", "block2", "block3",
                                        "pool1",  "pool2",  "pool3"]):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.

        Args:
            x: Input tensor [B, in_channels, H, W] — ImageNet-normalised.

        Returns:
            [B, 4] bounding box in (x_center, y_center, width, height)
            pixel coordinates — values in [0, IMAGE_SIZE].
        """
        features = self.encoder(x)           # [B, 512, 7, 7]
        features = self.avgpool(features)    # [B, 512, 7, 7]
        norm_out = self.reg_head(features)   # [B, 4]  in [0, 1]
        return norm_out * IMAGE_SIZE         # [B, 4]  in pixel space