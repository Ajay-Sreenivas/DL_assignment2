"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
import gdown
from .vgg11 import VGG11Encoder
from .layers import CustomDropout



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

    Weight loading order:
        1. classifier.pth  -> encoder (base) + cls_head
        2. localizer.pth   -> loc_head
        3. unet.pth        -> decoder layers + encoder (override)

    The encoder is loaded TWICE deliberately:
        First from classifier.pth  (good classification features as base)
        Then from unet.pth         (segmentation-adapted features as override)

    This is critical because the UNet decoder was trained against the
    segmentation-adapted encoder features from unet.pth, not the pure
    classification encoder from classifier.pth.  Loading decoder weights
    without also loading the matching encoder produces a feature distribution
    mismatch that collapses Dice from ~0.87 to ~0.25.

    The UNet encoder started from classification weights and was fine-tuned
    only slightly, so classification performance is preserved (~0.86 F1).
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
        super().__init__()

        # --- Download checkpoints from Google Drive (skip if already present) ---
        os.makedirs("checkpoints", exist_ok=True)
        for drive_id, out_path in [
            (CLASSIFIER_DRIVE_ID, classifier_path),
            (LOCALIZER_DRIVE_ID,  localizer_path),
            (UNET_DRIVE_ID,       unet_path),
        ]:
            if os.path.exists(out_path):
                print(f"[MultiTask] Checkpoint already exists, skipping download: {out_path}")
            else:
                print(f"[MultiTask] Downloading checkpoint to {out_path} ...")
                gdown.download(id=drive_id, output=out_path, quiet=False)

        # --- Shared backbone (classification + localisation) ---
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # --- Segmentation encoder (separate from shared encoder) ---
        # The UNet decoder was trained with an encoder that was fine-tuned
        # for segmentation (not frozen during train_unet).  Its weights
        # diverged from classifier.pth, so the decoder skip-connection
        # features only match the UNet-adapted encoder.
        # Solution: keep a dedicated seg_encoder loaded from unet.pth so
        # the decoder always sees the features it was trained against,
        # while self.encoder (from classifier.pth) serves cls + loc.
        self.seg_encoder = VGG11Encoder(in_channels=in_channels)

        # --- Localization encoder (separate from shared encoder) ---
        # train_localizer unfreezes block5 and fine-tunes it, so localizer.pth
        # block5 differs from classifier.pth block5.  reg_head was trained
        # against the fine-tuned block5 features — if we feed it classifier
        # block5 features instead, it outputs garbage (Acc@IoU = 0.0%).
        # A dedicated loc_encoder loaded from localizer.pth fixes this.
        self.loc_encoder = VGG11Encoder(in_channels=in_channels)

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

        # --- Localisation head ---
        # Matches VGG11Localizer.reg_head in localization.py exactly:
        #   25088 → 1024 → 256 → 4, dropout_p=0.2
        self.loc_head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.2),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 4),
            nn.ReLU(),
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
        """Transfer encoder + head weights from the three single-task checkpoints.

        Loading order is intentional:
          1. classifier.pth  sets encoder + cls_head
          2. localizer.pth   sets loc_head
          3. unet.pth        sets decoder AND overrides encoder with
                             segmentation-adapted weights so the decoder
                             sees the same feature distribution it was
                             trained against.
        """

        # --- Step 1: classifier → encoder (base) + cls_head ---
        cls_sd = self._load_ckpt(classifier_path)
        if cls_sd:
            enc_w  = {k[len("encoder."):]: v   for k, v in cls_sd.items() if k.startswith("encoder.")}
            head_w = {k[len("classifier."):]: v for k, v in cls_sd.items() if k.startswith("classifier.")}
            self.encoder.load_state_dict(enc_w,   strict=False)
            self.cls_head.load_state_dict(head_w, strict=False)
            print("[MultiTask] Loaded classifier weights.")

        # --- Step 2: localizer → loc_encoder + loc_head ---
        # loc_encoder is loaded with localizer.pth encoder weights so that
        # loc_head receives the same block5 features it was trained against.
        loc_sd = self._load_ckpt(localizer_path)
        if loc_sd:
            # 2a. Localizer encoder (contains fine-tuned block5)
            loc_enc_w = {k[len("encoder."):]: v
                         for k, v in loc_sd.items() if k.startswith("encoder.")}
            if loc_enc_w:
                self.loc_encoder.load_state_dict(loc_enc_w, strict=False)

            # 2b. Regression head
            head_w = {k[len("reg_head."):]: v
                      for k, v in loc_sd.items() if k.startswith("reg_head.")}
            missing, unexpected = self.loc_head.load_state_dict(head_w, strict=False)
            if missing:
                print(f"[MultiTask] WARNING: localizer missing keys ({len(missing)}): {missing[:2]}")
            else:
                print("[MultiTask] Loaded localizer weights.")

        # --- Step 3: unet → seg_encoder + decoder ---
        # seg_encoder is a dedicated VGG11Encoder for the segmentation path.
        # It is loaded with UNet-adapted weights so the decoder skip connections
        # see exactly the same feature distribution they were trained against.
        # self.encoder (classification encoder) is never touched here,
        # preserving cls_head performance.
        unet_sd = self._load_ckpt(unet_path)
        if unet_sd:
            # 3a. Load UNet encoder into dedicated seg_encoder
            unet_enc_w = {k[len("encoder."):]: v
                          for k, v in unet_sd.items() if k.startswith("encoder.")}
            if unet_enc_w:
                self.seg_encoder.load_state_dict(unet_enc_w, strict=False)

            # 3b. Decoder layers
            seg_layers = ["up5", "dec5", "up4", "dec4", "up3", "dec3",
                          "up2", "dec2", "up1", "dec1", "bottleneck_drop"]
            for layer in seg_layers:
                module = getattr(self, layer)
                lw = {k[len(layer) + 1:]: v
                      for k, v in unet_sd.items() if k.startswith(layer + ".")}
                if lw:
                    module.load_state_dict(lw, strict=False)

            # 3c. Final projection — accept both "seg_out.*" (new) and "out_conv.*" (old)
            final_w = {k[len("seg_out."):]: v
                       for k, v in unet_sd.items() if k.startswith("seg_out.")}
            if not final_w:
                final_w = {k[len("out_conv."):]: v
                           for k, v in unet_sd.items() if k.startswith("out_conv.")}
                if final_w:
                    print("[MultiTask] INFO: unet.pth uses 'out_conv' key — loading into seg_out.")
            if final_w:
                self.seg_out.load_state_dict(final_w, strict=True)
                print("[MultiTask] Loaded UNet weights.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """Single forward pass yielding all three task outputs.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            dict with keys:
              'classification' — [B, num_breeds] logits
              'localization'   — [B, 4] (cx, cy, w, h) in pixel space
              'segmentation'   — [B, seg_classes, H, W] logits
        """
        # Classification uses the shared (classifier) encoder
        bottleneck, _ = self.encoder(x, return_features=True)
        pooled  = self.avgpool(bottleneck)
        cls_out = self.cls_head(pooled)          # [B, num_breeds]

        # Localization uses the dedicated loc_encoder (fine-tuned block5)
        # so reg_head sees the same feature distribution it was trained against
        loc_bottleneck = self.loc_encoder(x)
        loc_pooled = self.avgpool(loc_bottleneck)
        loc_out = self.loc_head(loc_pooled)      # [B, 4] pixel space

        # Segmentation uses the dedicated seg_encoder (UNet-adapted weights)
        # so the decoder skip connections see the feature distribution they
        # were trained against
        seg_bottleneck, skips = self.seg_encoder(x, return_features=True)
        b = self.bottleneck_drop(seg_bottleneck)
        d = self.up5(b);  d = self.dec5(torch.cat([d, skips["block5"]], dim=1))
        d = self.up4(d);  d = self.dec4(torch.cat([d, skips["block4"]], dim=1))
        d = self.up3(d);  d = self.dec3(torch.cat([d, skips["block3"]], dim=1))
        d = self.up2(d);  d = self.dec2(torch.cat([d, skips["block2"]], dim=1))
        d = self.up1(d);  d = self.dec1(torch.cat([d, skips["block1"]], dim=1))
        seg_out = self.seg_out(d)                # [B, seg_classes, H, W]

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }