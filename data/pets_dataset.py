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
    """Return albumentations transform pipeline for a given split.

    Train pipeline uses aggressive spatial + colour augmentations to fight
    overfitting on the small Oxford-IIIT Pet dataset (~3 600 training images).
    All spatial transforms are applied BEFORE the final Resize so that
    albumentations automatically updates the bounding-box coordinates.

    Augmentation choices:
      ShiftScaleRotate  — random translation (±10 %), scale (±20 %), and
                          rotation (±15°) with zero-padding.  This is the
                          single most useful transform for bbox regression
                          because it creates diverse object positions and
                          sizes the model has never seen verbatim.
      HorizontalFlip    — cheap mirror augmentation; valid for pets.
      RandomBrightnessContrast / HueSaturationValue
                        — colour jitter that is harder to overfit than a
                          fixed ColorJitter, covering brightness, contrast,
                          hue, and saturation independently.
      GaussNoise        — adds small Gaussian noise to simulate sensor noise
                          and reduce pixel-level memorisation.
      CoarseDropout     — randomly blacks-out 1-4 small patches (Cutout),
                          forcing the model to rely on context rather than
                          specific texture cues.  max_holes=4, max_height/
                          width=32 px (≈14 % of 224 px side).
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),

                # --- Geometric (bbox-aware) ---
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    border_mode=0,          # zero-padding
                    p=0.7,
                ),
                A.HorizontalFlip(p=0.5),

                # --- Colour ---
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.6,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5,
                ),

                # --- Noise / occlusion ---
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.CoarseDropout(
                    max_holes=4,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    fill_value=0,
                    p=0.3,
                ),

                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="albumentations",   # [x_min, y_min, x_max, y_max] normalised
                label_fields=["bbox_labels"],
                min_visibility=0.2,        # raised from 0.1 — drop heavily cropped boxes
            ),
        )
    else:  # val / test — deterministic, no augmentation
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

    Directory layout expected::

        root/
            images/           *.jpg
            annotations/
                list.txt      (filename, class_id, species, breed_id)
                xmls/         *.xml  (head bounding boxes — ONLY ~50 % of images)
                trimaps/      *.png  (segmentation masks)

    .. important::
        The Oxford-IIIT Pet dataset ships bounding-box XMLs only for the
        *trainval* VOC split (~3 686 images out of 7 349 total).  The
        remaining ~3 663 images have **no XML file at all**.

        The old fallback ``bboxes = [[0, 0, 1, 1]]`` (whole image) silently
        poisoned ~50 % of localiser training samples with incorrect labels,
        causing the model to partially learn "predict image centre" and
        severely hurting generalisation on the held-out test set.

        Set ``require_bbox=True`` (default for localiser) to drop every
        sample that has no corresponding XML file **before** the train/val
        split is applied.  This gives a clean ~3 686-image subset where
        every label is real.

    Bounding boxes in the XML files use absolute pixel coordinates
    (xmin, ymin, xmax, ymax). We convert to (x_center, y_center, width, height)
    in pixel space for the localisation target, as required by the assignment.

    Segmentation trimaps use pixel values: 1=foreground, 2=background, 3=border.
    We map these to class indices 0/1/2 for CrossEntropy.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.15,
        seed: int = 42,
        require_bbox: bool = False,
    ):
        """
        Args:
            root:          Path to the dataset root directory.
            split:         'train' | 'val' | 'test'
            val_fraction:  Fraction of training data to hold out as validation.
            seed:          Random seed for reproducible splits.
            require_bbox:  If True, discard every sample that has no XML
                           bounding-box annotation **before** splitting into
                           train/val.  Always pass ``require_bbox=True`` when
                           training or evaluating the localiser so that no
                           fake whole-image fallback boxes pollute the labels.
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

        # ------------------------------------------------------------------
        # KEY FIX: drop samples with no bounding-box XML before splitting.
        # Without this, ~50 % of samples get a fake [0,0,1,1] whole-image
        # bbox that trains the model to predict the image centre.
        # ------------------------------------------------------------------
        if require_bbox:
            before = len(self.samples)
            self.samples = [
                s for s in self.samples
                if os.path.exists(os.path.join(self.xml_dir, s["name"] + ".xml"))
            ]
            after = len(self.samples)
            print(
                f"[Dataset] require_bbox=True: kept {after}/{before} samples "
                f"that have XML annotations ({before - after} dropped)."
            )

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
            # No XML annotation: use whole-image box as a safe fallback for
            # classification / segmentation tasks that don't use the bbox.
            # The localiser task should never reach here because OxfordIIITPetDataset
            # is constructed with require_bbox=True, which drops these samples.
            bboxes = [[0.0, 0.0, 1.0, 1.0]]

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