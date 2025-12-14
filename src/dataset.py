import os
import random
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        return torchvision.transforms.functional.to_tensor(image), target


class ColorJitter:
    def __init__(self, brightness=0.25, contrast=0.25, saturation=0.2, hue=0.08):
        self.t = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, image, target):
        return self.t(image), target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = torchvision.transforms.functional.hflip(image)
            width, _ = image.size
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class Resize:
    def __init__(self, min_size=None, max_size=None):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        if self.min_size is None:
            return image, target

        orig_w, orig_h = image.size
        scale = self.min_size / min(orig_h, orig_w)
        if self.max_size is not None and max(orig_h, orig_w) * scale > self.max_size:
            scale = self.max_size / max(orig_h, orig_w)

        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        if new_w == orig_w and new_h == orig_h:
            return image, target

        image = torchvision.transforms.functional.resize(image, [new_h, new_w])

        boxes = target["boxes"]
        if boxes.numel() > 0:
            boxes = boxes * torch.tensor([scale, scale, scale, scale], dtype=boxes.dtype)
            target["boxes"] = boxes
            target["area"] = target["area"] * (scale * scale)

        return image, target


class RandomResize:
    def __init__(self, base_min, max_size=None, scale_range=(0.6, 1.4)):
        self.base_min = base_min
        self.max_size = max_size
        self.scale_range = scale_range

    def __call__(self, image, target):
        if self.base_min is None:
            return image, target
        s = random.uniform(self.scale_range[0], self.scale_range[1])
        jittered_min = int(max(1, round(self.base_min * s)))
        return Resize(jittered_min, self.max_size)(image, target)


class HelmetDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Map COCO category IDs -> contiguous labels 1..K (0 is background)
        cat_ids = sorted(self.coco.cats.keys())
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
        self.label_to_cat_id = {v: k for k, v in self.cat_id_to_label.items()}

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_anns = coco.loadAnns(ann_ids)

        info = coco.loadImgs(img_id)[0]
        path = info["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        img_w, img_h = img.size

        boxes, labels, areas, iscrowd = [], [], [], []

        for ann in coco_anns:
            x, y, w, h = ann["bbox"]
            x0 = max(0.0, float(x))
            y0 = max(0.0, float(y))
            x1 = min(float(img_w), float(x) + float(w))
            y1 = min(float(img_h), float(y) + float(h))

            if x1 <= x0 or y1 <= y0:
                continue

            boxes.append([x0, y0, x1, y1])
            labels.append(self.cat_id_to_label[int(ann["category_id"])])
            areas.append((x1 - x0) * (y1 - y0))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        # ✅ CRITICAL FIX: use len(boxes), not len(coco_anns)
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def get_transform(train: bool, resize_min=800, resize_max=1333, resize_jitter=None):
    t = []
    if train:
        t.append(RandomResize(resize_min, resize_max, scale_range=(0.6, 1.4) if resize_jitter is None else (1 - resize_jitter, 1 + resize_jitter)))
        t.append(RandomHorizontalFlip(0.5))
        t.append(ColorJitter())
    else:
        t.append(Resize(resize_min, resize_max))

    t.append(ToTensor())
    return Compose(t)
