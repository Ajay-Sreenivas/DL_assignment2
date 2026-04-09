"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
import gdown
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

##
# Drive IDs — fill these in after uploading your checkpoints to Google Drive
CLASSIFIER_DRIVE_ID = "128xX5UlMk5k_jzx5HQFzc9VopEl8DhCE"
LOCALIZER_DRIVE_ID  = "1p-Ns0vBfOG5Mh0Oux0BpfTNXm5YTta_U"
UNET_DRIVE_ID       = "1KD1DcLiMNEjrp9mZnQG_avIwnxY1pHUE"


def _double_conv(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    Architecture:
        shared VGG11Encoder -> 3 task heads:
            1. ClassificationHead  -> [B, num_breeds]
            2. LocalizationHead    -> [B, 4]
            3. SegmentationDecoder -> [B, seg_classes, H, W]

    Checkpoints are downloaded from Google Drive automatically if not present.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        """
        Initialize the shared backbone/heads using trained weights.
        Args:
            num_breeds:       Number of output classes for classification head.
            seg_classes:      Number of output classes for segmentation head.
            in_channels:      Number of input channels.
            classifier_path:  Path to trained classifier weights.
            localizer_path:   Path to trained localizer weights.
            unet_path:        Path to trained unet weights.
        """
        super().__init__()

        # --- Download checkpoints from Google Drive ---
        os.makedirs("checkpoints", exist_ok=True)
        gdown.download(id=CLASSIFIER_DRIVE_ID, output=classifier_path, quiet=False)
        gdown.download(id=LOCALIZER_DRIVE_ID,  output=localizer_path,  quiet=False)
        gdown.download(id=UNET_DRIVE_ID,       output=unet_path,       quiet=False)

        # --- Shared backbone ---
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # --- Classification head ---
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, num_breeds),
        )

        # --- Localisation head (FC-based, matches VGG11Localizer.reg_head architecture) ---
        # Architecture: 25088 → 2048 → 1024 → 512 → 256 → 4
        # Dropout at every hidden layer (including pre-output 256) for regularisation.
        # Output: (cx, cy, w, h) in pixel space via ReLU (matches localizer exactly)
        self.loc_head = nn.Sequential(
            nn.Flatten(),                        # [B, 25088]
            nn.Linear(512 * 7 * 7, 2048),        # reduced from 4096 → less memorisation
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),               # added — regularise pre-output layer
            nn.Linear(256, 4),
            nn.ReLU(),                           # non-negative pixel coordinates
        )

        # --- Segmentation decoder ---
        self.bottleneck_drop = CustomDropout(p=0.5)

        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _double_conv(512 + 512, 512)

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _double_conv(256 + 512, 256)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _double_conv(128 + 256, 128)

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _double_conv(64 + 128, 64)

        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = _double_conv(64 + 64, 64)

        self.seg_out = nn.Conv2d(64, seg_classes, kernel_size=1)

        # --- Load pre-trained weights ---
        self._load_pretrained(classifier_path, localizer_path, unet_path)

    # ------------------------------------------------------------------
    # Weight loading helpers
    # ------------------------------------------------------------------

    def _load_ckpt(self, path: str) -> dict:
        """Load a checkpoint, resolving relative paths from project root."""
        if not os.path.isabs(path):
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base, path)
        if not os.path.exists(path):
            print(f"[MultiTask] WARNING: checkpoint not found at {path}, skipping.")
            return {}
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt

    def _load_pretrained(self, classifier_path, localizer_path, unet_path):
        """Transfer encoder + head weights from the three single-task checkpoints."""

        # classifier -> encoder + cls_head
        cls_sd = self._load_ckpt(classifier_path)
        if cls_sd:
            enc_w  = {k[len("encoder."):]: v   for k, v in cls_sd.items() if k.startswith("encoder.")}
            head_w = {k[len("classifier."):]: v for k, v in cls_sd.items() if k.startswith("classifier.")}
            self.encoder.load_state_dict(enc_w,   strict=False)
            self.cls_head.load_state_dict(head_w, strict=False)
            print("[MultiTask] Loaded classifier weights.")

        # localizer -> loc_head
        loc_sd = self._load_ckpt(localizer_path)
        if loc_sd:
            head_w = {k[len("reg_head."):]: v for k, v in loc_sd.items() if k.startswith("reg_head.")}
            self.loc_head.load_state_dict(head_w, strict=False)
            print("[MultiTask] Loaded localizer weights.")

        # unet -> segmentation decoder
        unet_sd = self._load_ckpt(unet_path)
        if unet_sd:
            seg_layers = ["up5", "dec5", "up4", "dec4", "up3", "dec3",
                          "up2", "dec2", "up1", "dec1", "seg_out", "bottleneck_drop"]
            for layer in seg_layers:
                module = getattr(self, layer)
                lw = {k[len(layer)+1:]: v for k, v in unet_sd.items() if k.startswith(layer + ".")}
                if lw:
                    module.load_state_dict(lw, strict=False)
            print("[MultiTask] Loaded UNet weights.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization':   [B, 4] bounding box tensor.
            - 'segmentation':   [B, seg_classes, H, W] segmentation logits tensor.
        """
        bottleneck, skips = self.encoder(x, return_features=True)

        pooled   = self.avgpool(bottleneck)      # [B, 512, 7, 7]
        cls_out  = self.cls_head(pooled)         # [B, num_breeds]
        loc_out  = self.loc_head(pooled)         # [B, 4] pixel space (cx,cy,w,h)

        b = self.bottleneck_drop(bottleneck)
        d = self.up5(b);  d = self.dec5(torch.cat([d, skips["block5"]], dim=1))
        d = self.up4(d);  d = self.dec4(torch.cat([d, skips["block4"]], dim=1))
        d = self.up3(d);  d = self.dec3(torch.cat([d, skips["block3"]], dim=1))
        d = self.up2(d);  d = self.dec2(torch.cat([d, skips["block2"]], dim=1))
        d = self.up1(d);  d = self.dec1(torch.cat([d, skips["block1"]], dim=1))
        seg_out = self.seg_out(d)

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }