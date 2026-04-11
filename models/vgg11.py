"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style convolutional encoder with BatchNorm.

    Architecture (from the original VGG paper, Table 1, column A):
      Block 1:  Conv(64)  -> BN -> ReLU -> MaxPool
      Block 2:  Conv(128) -> BN -> ReLU -> MaxPool
      Block 3:  Conv(256) -> BN -> ReLU -> Conv(256) -> BN -> ReLU -> MaxPool
      Block 4:  Conv(512) -> BN -> ReLU -> Conv(512) -> BN -> ReLU -> MaxPool
      Block 5:  Conv(512) -> BN -> ReLU -> Conv(512) -> BN -> ReLU -> MaxPool

    Input assumed to be 224x224 (per VGG paper).
    After all 5 max-pools the spatial size is 7x7, giving a 512x7x7 bottleneck.

    Skip-connection feature maps (for U-Net decoder) are saved *before* each
    MaxPool so their spatial dimensions match the encoder stage output:
        'block1': [B, 64,  112, 112]  -- after block1 conv, before pool
        'block2': [B, 128,  56,  56]
        'block3': [B, 256,  28,  28]
        'block4': [B, 512,  14,  14]
        'block5': [B, 512,   7,   7]  (bottleneck, after pool)
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        def conv_bn_relu(in_c, out_c, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        # --- convolutional blocks (without pooling layers so we can grab skip maps) ---
        self.block1 = conv_bn_relu(in_channels, 64)   
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.block2 = conv_bn_relu(64, 128)            
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.block3 = nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
        )                                              
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.block4 = nn.Sequential(
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
        )                                              
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.block5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
        )                                              
        self.pool5  = nn.MaxPool2d(kernel_size=2, stride=2)  

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, 224, 224].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor [B, 512, 7, 7].
            - if return_features=True: (bottleneck, feature_dict) where feature_dict
              contains skip-connection tensors before each pooling step.
        """
        s1 = self.block1(x)          
        x  = self.pool1(s1)          

        s2 = self.block2(x)          
        x  = self.pool2(s2)          

        s3 = self.block3(x)          
        x  = self.pool3(s3)          

        s4 = self.block4(x)          
        x  = self.pool4(s4)          

        s5 = self.block5(x)          
        bottleneck = self.pool5(s5)  

        if return_features:
            features = {
                "block1": s1,        
                "block2": s2,        
                "block3": s3,        
                "block4": s4,        
                "block5": s5,        
            }
            return bottleneck, features

        return bottleneck