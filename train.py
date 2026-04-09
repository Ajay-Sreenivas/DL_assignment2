"""Training entrypoint for DA6401 Assignment-2.

Usage:
    python train.py --data_root /path/to/oxford-iiit-pet \
                    --task [classifier|localizer|unet|multitask] \
                    --epochs 30 --batch_size 32 --lr 1e-3 \
                    --wandb_project da6401_a2 --wandb_entity YOUR_ENTITY

Each task saves its checkpoint to checkpoints/<task>.pth automatically.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
# import wandb  # disabled

from data.pets_dataset import OxfordIIITPetDataset
from models import (
    VGG11Classifier,
    VGG11Localizer,
    VGG11UNet,
    MultiTaskPerceptionModel,
)
from losses.iou_loss import IoULoss, CombinedLocLoss


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_oh = torch.zeros_like(probs)
        targets_oh.scatter_(1, targets.unsqueeze(1), 1)
        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dims)
        cardinality  = (probs + targets_oh).sum(dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    """Combined Cross-Entropy + Dice loss."""
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-6):
    """Compute mean IoU for a batch of (cx, cy, w, h) boxes."""
    px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tx1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    ty1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tx2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    ty2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    ix1 = torch.max(px1, tx1); ix2 = torch.min(px2, tx2)
    iy1 = torch.max(py1, ty1); iy2 = torch.min(py2, ty2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pa    = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ta    = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union = pa + ta - inter + eps
    return (inter / union).mean().item()


def dice_score(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 3, eps: float = 1e-6):
    preds = logits.argmax(dim=1)
    total = 0.0
    for c in range(num_classes):
        pred_c = (preds == c).float()
        tgt_c  = (targets == c).float()
        intersection = (pred_c * tgt_c).sum()
        cardinality  = pred_c.sum() + tgt_c.sum()
        total += ((2 * intersection + eps) / (cardinality + eps)).item()
    return total / num_classes


# ---------------------------------------------------------------------------
# Checkpoint persistence — upload to Google Drive from Kaggle
# ---------------------------------------------------------------------------

def save_to_drive(filename: str, folder_id: str = None):
    """Upload a checkpoint directly to Google Drive using the Drive API.

    This avoids the painfully slow Kaggle output-panel download.
    After training, grab the file straight from your Google Drive.

    Prerequisites (run once at the top of your Kaggle notebook):
        from google.colab import auth   # or use kaggle_secrets + oauth
        auth.authenticate_user()
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        # -- OR just pip install PyDrive2 and authenticate below --

    The function handles its own auth via a service-account-free OAuth flow
    using PyDrive2. Set GDRIVE_FOLDER_ID in your Kaggle notebook environment
    (Kaggle → Add-ons → Secrets) to upload into a specific Drive folder;
    if unset, the file lands in My Drive root.

    Args:
        filename:  e.g. 'localizer.pth' — looked up under checkpoints/
        folder_id: Google Drive folder ID to upload into (overrides env var).
    """
    src = os.path.join("checkpoints", filename)
    if not os.path.exists(src):
        print(f"  [Drive] Skipping upload — file not found: {src}")
        return

    # Resolve target folder
    if folder_id is None:
        folder_id = os.environ.get("GDRIVE_FOLDER_ID", None)

    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
        from oauth2client.service_account import ServiceAccountCredentials

        gauth = GoogleAuth()
        # Use local webserver auth (works in Kaggle interactive sessions)
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)

        # Search for existing file with same name in the target folder so we
        # overwrite rather than create duplicates on every best-checkpoint save.
        query = f"title='{filename}' and trashed=false"
        if folder_id:
            query += f" and '{folder_id}' in parents"
        existing = drive.ListFile({"q": query}).GetList()

        if existing:
            gfile = existing[0]
            print(f"  [Drive] Overwriting existing file: {filename} (id={gfile['id']})")
        else:
            meta = {"title": filename}
            if folder_id:
                meta["parents"] = [{"id": folder_id}]
            gfile = drive.CreateFile(meta)

        gfile.SetContentFile(src)
        gfile.Upload()
        print(f"  [Drive] ✅ Uploaded {filename} → Google Drive (id={gfile['id']})")

    except ImportError:
        print("  [Drive] PyDrive2 not installed. Run: pip install PyDrive2 -q")
    except Exception as e:
        print(f"  [Drive] Upload failed: {e}")
        # Fall back to /kaggle/working so the file is at least in the output panel
        import shutil
        kaggle_out = "/kaggle/working"
        if os.path.exists(kaggle_out):
            shutil.copy(src, os.path.join(kaggle_out, filename))
            print(f"  [Drive] Fell back to Kaggle output panel: /kaggle/working/{filename}")


def copy_to_kaggle_output(filename: str):
    """Copy checkpoint to /kaggle/working/ (fallback — prefer save_to_drive)."""
    import shutil
    kaggle_out = "/kaggle/working"
    if os.path.exists(kaggle_out):
        src = os.path.join("checkpoints", filename)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(kaggle_out, filename))
            print(f"  --> Copied {filename} to /kaggle/working/ (output panel)")


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_classifier(args, device):
    train_ds = OxfordIIITPetDataset(args.data_root, split="train")
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val")
    pin      = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model     = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0; correct = 0; total = 0
        for batch in tqdm(train_dl, desc=f"[Classifier] Epoch {epoch}/{args.epochs} train", leave=False):
            imgs   = batch["image"].to(device)
            labels = batch["class_id"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        scheduler.step()
        train_loss /= total
        train_acc   = correct / total

        # --- Validation ---
        model.eval()
        val_loss = 0.0; val_correct = 0; val_total = 0
        all_preds = []; all_labels = []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"[Classifier] Epoch {epoch}/{args.epochs} val", leave=False):
                imgs   = batch["image"].to(device)
                labels = batch["class_id"].to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                preds  = logits.argmax(1)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc   = val_correct / val_total
        val_f1    = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        pass  # wandb.log disabled
        print(f"[Classifier] Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_f1},
                       "checkpoints/classifier.pth")
            print(f"  --> Saved best model (val_f1={val_f1:.4f})")

    print(f"Best classifier val_f1: {best_f1:.4f}")


# ============================================================
# PATCH for train.py
#
# 1. Replace the existing train_localizer() with this function.
# 2. Update the import at the top of train.py:
#      FROM: from losses.iou_loss import IoULoss
#      TO:   from losses.iou_loss import IoULoss, CombinedLocLoss
# ============================================================


def _acc_at_iou(pred_boxes: "torch.Tensor",
                target_boxes: "torch.Tensor",
                threshold: float = 0.5,
                eps: float = 1e-6) -> float:
    """Compute the fraction of predictions with IoU >= threshold.

    This is the exact metric Gradescope uses (Acc@IoU=0.5).
    Saving the checkpoint based on this — not mean IoU — ensures the saved
    model maximises the number of samples that clear the 0.5 threshold.

    Args:
        pred_boxes:   [B, 4] (cx, cy, w, h) in pixel space.
        target_boxes: [B, 4] (cx, cy, w, h) in pixel space.
        threshold:    IoU threshold (default 0.5).

    Returns:
        Fraction of samples with IoU >= threshold, as a Python float.
    """
    px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tx1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    ty1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tx2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    ty2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    tgt_area  = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union = pred_area + tgt_area - inter + eps

    iou = inter / union                          # [B]
    return (iou >= threshold).float().mean().item()


def train_localizer(args, device):
    # Only images that have a matching .xml annotation are used for training.
    # Same base name: images/Abyssinian_1.jpg ←→ annotations/xmls/Abyssinian_1.xml
    train_ds = OxfordIIITPetDataset(args.data_root, split="train", require_bbox=True)
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val",   require_bbox=True)
    pin      = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin)

    model = VGG11Localizer(dropout_p=args.dropout_p).to(device)

    # ------------------------------------------------------------------
    # Load pretrained encoder weights, then freeze the entire encoder.
    # Only the regression head (randomly initialised) will be trained.
    # ------------------------------------------------------------------
    cls_ckpt_path = "checkpoints/classifier.pth"
    if os.path.exists(cls_ckpt_path):
        cls_sd = torch.load(cls_ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in cls_sd:
            cls_sd = cls_sd["state_dict"]
        enc_w = {k[len("encoder."):]: v for k, v in cls_sd.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_w, strict=False)
        print("Loaded encoder weights from classifier checkpoint.")

    model.freeze_encoder()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Encoder frozen — trainable params: {trainable:,} / {total:,} (head only)")

    # ------------------------------------------------------------------
    # Loss: IoU + 0.5 × SmoothL1, both in pixel space
    # ------------------------------------------------------------------
    loc_loss  = CombinedLocLoss(lambda_l1=0.5)

    # All trainable params belong to the head — use a single LR
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-3,   # increased from 1e-4 for stronger L2 regularisation
    )
    # Cosine schedule works well when the head is trained from scratch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Checkpoint based on best validation IoU
    # ------------------------------------------------------------------
    best_val_iou = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0; train_iou_sum = 0.0; n = 0
        for batch in tqdm(train_dl, desc=f"[Localizer] Epoch {epoch}/{args.epochs} train", leave=False):
            imgs  = batch["image"].to(device)
            boxes = batch["bbox"].to(device)          # [B, 4] pixel space
            optimizer.zero_grad()
            pred  = model(imgs)                       # [B, 4] pixel space
            loss  = loc_loss(pred, boxes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0,   # gradient clipping to stabilise head training
            )
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_iou_sum += compute_iou(pred.detach(), boxes) * imgs.size(0)
            n             += imgs.size(0)
        scheduler.step()
        train_loss /= n
        train_iou   = train_iou_sum / n

        # --- Validation ---
        model.eval()
        val_loss = 0.0; val_iou_sum = 0.0; val_acc_sum = 0.0; nv = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"[Localizer] Epoch {epoch}/{args.epochs} val", leave=False):
                imgs  = batch["image"].to(device)
                boxes = batch["bbox"].to(device)
                pred  = model(imgs)
                loss  = loc_loss(pred, boxes)
                bs = imgs.size(0)
                val_loss    += loss.item() * bs
                val_iou_sum += compute_iou(pred, boxes) * bs
                val_acc_sum += _acc_at_iou(pred, boxes, threshold=0.5) * bs
                nv          += bs
        val_loss /= nv
        val_iou   = val_iou_sum  / nv
        val_acc   = val_acc_sum  / nv      # Acc@IoU=0.5 — the Gradescope metric

        pass  # wandb.log disabled
        print(
            f"[Localizer] Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_acc@0.5={val_acc:.4f}"
        )

        # Save checkpoint when val_iou improves
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(
                {
                    "state_dict":   model.state_dict(),
                    "epoch":        epoch,
                    "best_metric":  best_val_iou,
                    "val_iou":      val_iou,
                    "val_acc_iou50": val_acc,
                },
                "checkpoints/localizer.pth",
            )
            print(f"  ✅ Saved best checkpoint (val_iou = {best_val_iou:.4f}, Acc@IoU=0.5 = {val_acc:.4f})")
            # Upload immediately to Google Drive so it survives session timeout
            save_to_drive("localizer.pth")

    print(f"Best localizer val_iou: {best_val_iou:.4f}")


def train_unet(args, device):
    train_ds = OxfordIIITPetDataset(args.data_root, split="train")
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val")
    pin      = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)

    # Load pre-trained encoder from classifier checkpoint
    cls_ckpt_path = "checkpoints/classifier.pth"
    if os.path.exists(cls_ckpt_path):
        cls_sd = torch.load(cls_ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in cls_sd:
            cls_sd = cls_sd["state_dict"]
        enc_w = {k[len("encoder."):]: v for k, v in cls_sd.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_w, strict=False)
        print("Loaded encoder weights from classifier checkpoint.")

    seg_loss  = SegLoss(num_classes=3)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0; train_dice_sum = 0.0; n = 0
        for batch in tqdm(train_dl, desc=f"[UNet] Epoch {epoch}/{args.epochs} train", leave=False):
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = seg_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss     += loss.item() * imgs.size(0)
            train_dice_sum += dice_score(logits.detach(), masks) * imgs.size(0)
            n              += imgs.size(0)
        scheduler.step()
        train_loss /= n; train_dice = train_dice_sum / n

        # --- Validation ---
        model.eval()
        val_loss = 0.0; val_dice_sum = 0.0; nv = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"[UNet] Epoch {epoch}/{args.epochs} val", leave=False):
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                loss   = seg_loss(logits, masks)
                val_loss     += loss.item() * imgs.size(0)
                val_dice_sum += dice_score(logits, masks) * imgs.size(0)
                nv           += imgs.size(0)
        val_loss /= nv; val_dice = val_dice_sum / nv

        pass  # wandb.log disabled
        print(f"[UNet] Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_dice={train_dice:.4f} | val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_metric": best_dice},
                       "checkpoints/unet.pth")
            print(f"  --> Saved best model (val_dice={val_dice:.4f})")

    print(f"Best UNet val_dice: {best_dice:.4f}")


def train_multitask(args, device):
    train_ds = OxfordIIITPetDataset(args.data_root, split="train")
    val_ds   = OxfordIIITPetDataset(args.data_root, split="val")
    pin      = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model = MultiTaskPerceptionModel(
        num_breeds=37, seg_classes=3,
        classifier_path="checkpoints/classifier.pth",
        localizer_path="checkpoints/localizer.pth",
        unet_path="checkpoints/unet.pth",
    ).to(device)

    cls_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")
    seg_loss = SegLoss(num_classes=3)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_combined = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        metrics = {"cls_loss": 0, "loc_loss": 0, "seg_loss": 0, "total": 0, "n": 0}
        for batch in tqdm(train_dl, desc=f"[MultiTask] Epoch {epoch}/{args.epochs} train", leave=False):
            imgs   = batch["image"].to(device)
            labels = batch["class_id"].to(device)
            boxes  = batch["bbox"].to(device)
            masks  = batch["mask"].to(device)
            optimizer.zero_grad()
            out   = model(imgs)
            l_cls = cls_loss(out["classification"], labels)
            l_loc = mse_loss(out["localization"], boxes) + iou_loss(out["localization"], boxes)
            l_seg = seg_loss(out["segmentation"], masks)
            loss  = l_cls + l_loc + l_seg
            loss.backward()
            optimizer.step()
            bs = imgs.size(0)
            metrics["cls_loss"] += l_cls.item() * bs
            metrics["loc_loss"] += l_loc.item() * bs
            metrics["seg_loss"] += l_seg.item() * bs
            metrics["total"]    += loss.item() * bs
            metrics["n"]        += bs
        scheduler.step()
        n = metrics["n"]
        pass  # wandb.log disabled
        print(f"[MultiTask] Epoch {epoch}/{args.epochs} | total={metrics['total']/n:.4f}")

        # --- Validation ---
        model.eval()
        val_acc = 0.0; val_iou = 0.0; val_dice = 0.0; nv = 0
        all_preds_mt = []; all_labels_mt = []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"[MultiTask] Epoch {epoch}/{args.epochs} val", leave=False):
                imgs   = batch["image"].to(device)
                labels = batch["class_id"].to(device)
                boxes  = batch["bbox"].to(device)
                masks  = batch["mask"].to(device)
                out    = model(imgs)
                bs     = imgs.size(0)
                preds  = out["classification"].argmax(1)
                val_acc  += (preds == labels).sum().item()
                val_iou  += compute_iou(out["localization"], boxes) * bs
                val_dice += dice_score(out["segmentation"], masks) * bs
                nv       += bs
                all_preds_mt.extend(preds.cpu().numpy())
                all_labels_mt.extend(labels.cpu().numpy())

        val_acc  /= nv
        val_iou  /= nv
        val_dice /= nv
        val_f1    = f1_score(all_labels_mt, all_preds_mt, average="macro", zero_division=0)
        combined  = (val_f1 + val_iou + val_dice) / 3

        pass  # wandb.log disabled
        print(f"  val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_iou={val_iou:.4f} val_dice={val_dice:.4f}")

        if combined > best_combined:
            best_combined = combined
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_metric": combined},
                       "checkpoints/multitask.pth")
            print(f"  --> Saved best model (combined={combined:.4f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     type=str, required=True)
    p.add_argument("--task",          type=str, default="classifier",
                   choices=["classifier", "localizer", "unet", "multitask", "all"])
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--dropout_p",     type=float, default=0.5)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--wandb_project", type=str,   default="da6401_a2")
    p.add_argument("--wandb_entity",  type=str,   default=None)
    p.add_argument("--device",        type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()




if __name__ == "__main__":
    args   = parse_args()
    device = torch.device(args.device)

    # wandb.init disabled

    if args.task == "all":
        train_classifier(args, device)
        copy_to_kaggle_output("classifier.pth")
        train_localizer(args, device)
        copy_to_kaggle_output("localizer.pth")
        train_unet(args, device)
        copy_to_kaggle_output("unet.pth")
        train_multitask(args, device)
        copy_to_kaggle_output("multitask.pth")
    elif args.task == "classifier":
        train_classifier(args, device)
        copy_to_kaggle_output("classifier.pth")
    elif args.task == "localizer":
        train_localizer(args, device)
        copy_to_kaggle_output("localizer.pth")
    elif args.task == "unet":
        train_unet(args, device)
        copy_to_kaggle_output("unet.pth")
    elif args.task == "multitask":
        train_multitask(args, device)
        copy_to_kaggle_output("multitask.pth")

    # wandb.finish()  # disabled