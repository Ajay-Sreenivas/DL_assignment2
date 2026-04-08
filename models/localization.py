"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

IMAGE_SIZE = 224  # VGG11 input size — used to convert normalised output to pixel space


class VGG11Localizer(nn.Module):
    """VGG11-based single-object localizer.

    Encoder strategy: Partial fine-tuning.
    - block1, block2, block3 are FROZEN — they capture generic low-level features
      (edges, textures, colours) that transfer well from classification.
    - block4, block5 are UNFROZEN — these deeper layers detect object-specific
      spatial features (head shape, ears, snout) that must adapt for localisation.
    - This avoids overfitting the early layers while allowing the model to learn
      WHERE the object is rather than just WHAT it is.

    Regression head: Conv layers before FC to retain spatial structure.
    - Instead of flattening 7x7x512 immediately, two Conv layers reduce channels
      while keeping spatial reasoning alive.
    - GlobalAvgPool then collapses to a vector before the final FC.
    - Sigmoid output constrains predictions to [0,1]; forward() scales back to
      pixel space by multiplying by IMAGE_SIZE, satisfying the assignment requirement
      that outputs are in pixel coordinates.
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

        # Partial freeze: block1-3 frozen, block4-5 trainable
        # Applied after loading pretrained weights in train.py
        self._freeze_early_blocks()

        # Conv-based regression head (retains spatial structure longer)
        self.reg_head = nn.Sequential(
            # Spatial reasoning via conv layers on 7x7 feature map
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # [B, 256, 7, 7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # [B, 128, 7, 7]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),                   # [B, 128, 1, 1]
            nn.Flatten(),                                   # [B, 128]
            CustomDropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid(),   # outputs in [0, 1]; scaled to pixels in forward()
        )

    def _freeze_early_blocks(self):
        """Freeze block1, block2, block3. Unfreeze block4, block5."""
        for name, param in self.encoder.named_parameters():
            if "block1" in name or "block2" in name or "block3" in name or \
               "pool1" in name or "pool2" in name or "pool3" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            format in pixel space (values in [0, IMAGE_SIZE]).
        """
        features = self.encoder(x)      # [B, 512, 7, 7]
        norm_out  = self.reg_head(features)  # [B, 4] in [0, 1]
        # Scale normalised output back to pixel coordinates
        return norm_out * IMAGE_SIZE    # [B, 4] in pixel space