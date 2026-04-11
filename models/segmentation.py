"""Segmentation model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _double_conv(in_c: int, out_c: int) -> nn.Sequential:
    """Two Conv-BN-ReLU blocks (standard U-Net decoder pattern)."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """U-Net style semantic segmentation network with VGG11 encoder.

    The contracting path is the VGG11 encoder (with skip connections).
    The expansive path mirrors the encoder using transposed convolutions for
    upsampling (bilinear interpolation is not used as per the spec).

    Skip connections concatenate encoder feature maps onto decoder maps at
    each corresponding spatial resolution.

    Loss choice: Combined Dice + Cross-Entropy loss.
      - Cross-Entropy provides a strong, pixel-wise gradient signal that trains
        quickly and handles multi-class output naturally.
      - Dice Loss directly optimises the F1-like overlap metric, making it
        robust to class imbalance (background dominates in trimaps).
      Combining both yields faster convergence (CE) with better boundary
      accuracy and imbalance handling (Dice).

    Decoder path (skip channel arithmetic):
        d5: ConvTranspose 512 → 512, cat(s5=[B,512,14,14]) → 1024, doubleconv → 512
        d4: ConvTranspose 512 → 256, cat(s4=[B,512,28,28]) →  768, doubleconv → 256
        d3: ConvTranspose 256 → 128, cat(s3=[B,256,56,56]) →  384, doubleconv → 128
        d2: ConvTranspose 128 →  64, cat(s2=[B,128,112,112]) → 192, doubleconv →  64
        d1: ConvTranspose  64 →  64, cat(s1=[B, 64,224,224]) → 128, doubleconv →  64
        out: Conv 64 → num_classes
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability (applied in decoder bottleneck).
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Bottleneck dropout for regularisation
        self.bottleneck_drop = CustomDropout(p=dropout_p)

        # Decoder: transposed convolutions for learned upsampling
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)   
        self.dec5 = _double_conv(512 + 512, 512)   

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   
        self.dec4 = _double_conv(256 + 512, 256)   

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   
        self.dec3 = _double_conv(128 + 256, 128)   

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    
        self.dec2 = _double_conv(64 + 128, 64)    

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)     
        self.dec1 = _double_conv(64 + 64, 64)     

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, skips = self.encoder(x, return_features=True)

        b = self.bottleneck_drop(bottleneck)  

        d = self.up5(b)                                        
        d = self.dec5(torch.cat([d, skips["block5"]], dim=1))  

        d = self.up4(d)                                        
        d = self.dec4(torch.cat([d, skips["block4"]], dim=1))  

        d = self.up3(d)                                        
        d = self.dec3(torch.cat([d, skips["block3"]], dim=1))  

        d = self.up2(d)                                        
        d = self.dec2(torch.cat([d, skips["block2"]], dim=1))  

        d = self.up1(d)                                        
        d = self.dec1(torch.cat([d, skips["block1"]], dim=1))  

        return self.out_conv(d)                                
    