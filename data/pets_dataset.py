"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalisation (standard for VGG-family networks)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224  # VGG11 input size


def get_transforms(split: str = "train"):
    """Return albumentations transform pipeline for a given split."""
    if split == "train":
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="albumentations",   # [x_min, y_min, x_max, y_max] normalised
                label_fields=["bbox_labels"],
                min_visibility=0.1,
            ),
        )
    else:  # val / test
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="albumentations",
                label_fields=["bbox_labels"],
                min_visibility=0.1,
            ),
        )


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Directory layout expected:
        root/
            images/           *.jpg
            annotations/
                list.txt      (filename, class_id, species, breed_id)
                xmls/         *.xml  (head bounding boxes)
                trimaps/      *.png  (segmentation masks)

    Bounding boxes in the XML files use absolute pixel coordinates
    (xmin, ymin, xmax, ymax). We convert to (x_center, y_center, width, height)
    in pixel space for the localisation target, as required by the assignment.

    Segmentation trimaps use pixel values: 1=foreground, 2=background, 3=border.
    We map these to class indices 0/1/2 for CrossEntropy.
    """

    def __init__(self, root: str, split: str = "train", val_fraction: float = 0.15, seed: int = 42):
        """
        Args:
            root:          Path to the dataset root directory.
            split:         'train' | 'val' | 'test'
            val_fraction:  Fraction of training data to hold out as validation.
            seed:          Random seed for reproducible splits.
        """
        super().__init__()
        self.root   = root
        self.split  = split
        self.transform = get_transforms(split)

        self.image_dir  = os.path.join(root, "images")
        self.xml_dir    = os.path.join(root, "annotations", "xmls")
        self.trimap_dir = os.path.join(root, "annotations", "trimaps")

        # Parse list.txt
        list_path = os.path.join(root, "annotations", "list.txt")
        self.samples = self._parse_list(list_path)

        # Reproducible train / val split
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)
        n_val = int(len(indices) * val_fraction)

        if split == "train":
            self.samples = [self.samples[i] for i in indices[n_val:]]
        elif split == "val":
            self.samples = [self.samples[i] for i in indices[:n_val]]
        # "test" keeps all samples (used by the autograder with its own test images)

    # ------------------------------------------------------------------

    def _parse_list(self, list_path: str):
        """Parse annotations/list.txt into a list of dicts."""
        samples = []
        with open(list_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                name     = parts[0]          # e.g. "Abyssinian_1"
                class_id = int(parts[1]) - 1  # 1-indexed → 0-indexed
                samples.append({"name": name, "class_id": class_id})
        return samples

    def _parse_xml_bbox(self, xml_path: str):
        """Return (xmin, ymin, xmax, ymax) from a VOC-style XML file."""
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find(".//bndbox")
        if bndbox is None:
            return None
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        return xmin, ymin, xmax, ymax

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        name     = sample["name"]
        class_id = sample["class_id"]

        # --- Image ---
        img_path = os.path.join(self.image_dir, name + ".jpg")
        image    = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        # --- Bounding box ---
        xml_path = os.path.join(self.xml_dir, name + ".xml")
        bbox_abs = self._parse_xml_bbox(xml_path) if os.path.exists(xml_path) else None

        if bbox_abs is not None:
            xmin, ymin, xmax, ymax = bbox_abs
            # Clip to image bounds
            xmin = max(0.0, min(xmin, orig_w))
            xmax = max(0.0, min(xmax, orig_w))
            ymin = max(0.0, min(ymin, orig_h))
            ymax = max(0.0, min(ymax, orig_h))
            # Normalise to [0,1] for albumentations "albumentations" format
            bbox_norm = [xmin / orig_w, ymin / orig_h, xmax / orig_w, ymax / orig_h]
            bboxes = [bbox_norm]
        else:
            bboxes = [[0.0, 0.0, 1.0, 1.0]]  # fallback: whole image

        # --- Segmentation mask ---
        trimap_path = os.path.join(self.trimap_dir, name + ".png")
        if os.path.exists(trimap_path):
            trimap = np.array(Image.open(trimap_path))  # values in {1, 2, 3}
        else:
            trimap = np.ones((orig_h, orig_w), dtype=np.uint8)

        # trimap: 1=fg, 2=bg, 3=border → 0-indexed class labels
        mask = (trimap - 1).astype(np.uint8)  # {0, 1, 2}

        # --- Apply transforms ---
        transformed = self.transform(
            image=image,
            mask=mask,
            bboxes=bboxes,
            bbox_labels=[0],
        )
        image_t = transformed["image"]          # [3, H, W] float
        mask = transformed["mask"]
        mask_t = mask.long() if hasattr(mask, "long") else torch.from_numpy(mask).long() 

        # Recover bbox in pixel space after resize to IMAGE_SIZE x IMAGE_SIZE
        if transformed["bboxes"]:
            bx1, by1, bx2, by2 = transformed["bboxes"][0]
            # Convert normalised → pixel coords at IMAGE_SIZE resolution
            bx1 *= IMAGE_SIZE; bx2 *= IMAGE_SIZE
            by1 *= IMAGE_SIZE; by2 *= IMAGE_SIZE
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            bw = bx2 - bx1
            bh = by2 - by1
        else:
            cx, cy, bw, bh = IMAGE_SIZE / 2, IMAGE_SIZE / 2, IMAGE_SIZE, IMAGE_SIZE

        bbox_t = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

        return {
            "image":      image_t,            # [3, 224, 224]
            "class_id":   torch.tensor(class_id, dtype=torch.long),
            "bbox":       bbox_t,             # [4]  (cx, cy, w, h) in pixel space
            "mask":       mask_t,             # [224, 224]  class indices 0/1/2
            "name":       name,
        }
