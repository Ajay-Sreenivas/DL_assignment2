"""Inference and evaluation script for DA6401 Assignment-2.

Usage:
    python inference.py --data_root /path/to/oxford-iiit-pet \
                        --split val \
                        --device cuda
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


def compute_iou_batch(pred_boxes, target_boxes, eps=1e-6):
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
    pa = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ta = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    return inter / (pa + ta - inter + eps)


def dice_coeff(preds, targets, num_classes=3, eps=1e-6):
    total = 0.0
    for c in range(num_classes):
        p = (preds == c).float(); t = (targets == c).float()
        total += (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return (total / num_classes).item()


def evaluate(args):
    device = torch.device(args.device)
    ds = OxfordIIITPetDataset(args.data_root, split=args.split)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MultiTaskPerceptionModel(
        classifier_path="checkpoints/classifier.pth",
        localizer_path="checkpoints/localizer.pth",
        unet_path="checkpoints/unet.pth",
    ).to(device)
    model.eval()

    all_preds = []; all_labels = []
    iou_list  = []; dice_list  = []

    with torch.no_grad():
        for batch in dl:
            imgs   = batch["image"].to(device)
            labels = batch["class_id"].to(device)
            boxes  = batch["bbox"].to(device)
            masks  = batch["mask"].to(device)
            out    = model(imgs)

            all_preds.extend(out["classification"].argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            ious = compute_iou_batch(out["localization"], boxes)
            iou_list.extend(ious.cpu().numpy())

            seg_preds = out["segmentation"].argmax(1)
            for i in range(imgs.size(0)):
                dice_list.append(dice_coeff(seg_preds[i], masks[i]))

    f1  = f1_score(all_labels, all_preds, average="macro")
    mAP = float(np.mean(iou_list))   # using mean IoU as proxy for mAP
    dice = float(np.mean(dice_list))

    print(f"=== Evaluation on '{args.split}' split ===")
    print(f"  Classification Macro-F1 : {f1:.4f}")
    print(f"  Detection mean IoU       : {mAP:.4f}")
    print(f"  Segmentation Dice        : {dice:.4f}")
    return {"f1": f1, "mAP": mAP, "dice": dice}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  type=str, required=True)
    p.add_argument("--split",      type=str, default="val")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    evaluate(args)
