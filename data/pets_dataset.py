"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224  


def get_transforms(split: str = "train"):
    if split == "train":
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    border_mode=0,          
                    p=0.5,                  
                ),
                A.HorizontalFlip(p=0.5),
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
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="albumentations",   
                label_fields=["bbox_labels"],
                min_visibility=0.05,       # REDUCED: Avoids dropping cropped boxes entirely
            ),
        )
    else:  
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="albumentations",
                label_fields=["bbox_labels"],
                min_visibility=0.01,       # Validation should ideally never drop boxes
            ),
        )


class OxfordIIITPetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.15,
        seed: int = 42,
        require_bbox: bool = False,
    ):
        super().__init__()
        self.root   = root
        self.split  = split
        self.transform = get_transforms(split)

        self.image_dir  = os.path.join(root, "images")
        self.xml_dir    = os.path.join(root, "annotations", "xmls")
        self.trimap_dir = os.path.join(root, "annotations", "trimaps")

        list_path = os.path.join(root, "annotations", "list.txt")
        self.samples = self._parse_list(list_path)

        if require_bbox:
            before = len(self.samples)
            self.samples = [
                s for s in self.samples
                if os.path.exists(os.path.join(self.image_dir, s["name"] + ".jpg"))
                and os.path.exists(os.path.join(self.xml_dir,   s["name"] + ".xml"))
            ]

        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.samples))
        rng.shuffle(indices)
        n_val = int(len(indices) * val_fraction)

        if split == "train":
            self.samples = [self.samples[i] for i in indices[n_val:]]
        elif split == "val":
            self.samples = [self.samples[i] for i in indices[:n_val]]

    def _parse_list(self, list_path: str):
        samples = []
        with open(list_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                parts = line.split()
                if len(parts) < 2: continue
                name     = parts[0]          
                class_id = int(parts[1]) - 1  
                samples.append({"name": name, "class_id": class_id})
        return samples

    def _parse_xml_bbox(self, xml_path: str):
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find(".//bndbox")
        if bndbox is None: return None
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        return xmin, ymin, xmax, ymax

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        name     = sample["name"]
        class_id = sample["class_id"]

        img_path = os.path.join(self.image_dir, name + ".jpg")
        image    = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        xml_path = os.path.join(self.xml_dir, name + ".xml")
        bbox_abs = self._parse_xml_bbox(xml_path) if os.path.exists(xml_path) else None

        if bbox_abs is not None:
            xmin, ymin, xmax, ymax = bbox_abs
            xmin = max(0.0, min(xmin, orig_w))
            xmax = max(0.0, min(xmax, orig_w))
            ymin = max(0.0, min(ymin, orig_h))
            ymax = max(0.0, min(ymax, orig_h))
            bbox_norm = [xmin / orig_w, ymin / orig_h, xmax / orig_w, ymax / orig_h]
            bboxes = [bbox_norm]
        else:
            bboxes = [[0.0, 0.0, 1.0, 1.0]]

        trimap_path = os.path.join(self.trimap_dir, name + ".png")
        if os.path.exists(trimap_path):
            trimap = np.array(Image.open(trimap_path))  
        else:
            trimap = np.ones((orig_h, orig_w), dtype=np.uint8)

        mask = (trimap - 1).astype(np.uint8)  

        transformed = self.transform(
            image=image,
            mask=mask,
            bboxes=bboxes,
            bbox_labels=[0],
        )
        image_t = transformed["image"]          
        mask = transformed["mask"]
        mask_t = mask.long() if hasattr(mask, "long") else torch.from_numpy(mask).long() 

        if transformed["bboxes"]:
            bx1, by1, bx2, by2 = transformed["bboxes"][0]
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
            "image":      image_t,            
            "class_id":   torch.tensor(class_id, dtype=torch.long),
            "bbox":       bbox_t,             
            "mask":       mask_t,             
            "name":       name,
        }