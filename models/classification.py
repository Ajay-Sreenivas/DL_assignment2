"""Classification components
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead.

    The classification head follows the original VGG paper:
        AdaptiveAvgPool -> Flatten -> FC(4096) -> BN -> ReLU -> Dropout
                       -> FC(4096) -> BN -> ReLU -> Dropout -> FC(num_classes)

    BatchNorm is placed *before* the activation in every FC block so that the
    pre-activation distribution is normalised before the ReLU cuts it; this
    stabilises training and allows higher learning rates.

    CustomDropout is placed *after* the activation so it operates on
    already-rectified, normalised features — empirically this leads to a
    cleaner regularisation signal than dropping before activation.
    """

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Adaptive pool collapses any spatial size to 7x7 (same as VGG input 224x224)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # [B, 512, 7, 7]

        self.classifier = nn.Sequential(
            nn.Flatten(),                          # [B, 512*7*7 = 25088]
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x)           # [B, 512, 7, 7]
        features = self.avgpool(features)    # [B, 512, 7, 7]
        return self.classifier(features)     # [B, num_classes]
