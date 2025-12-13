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
        if self.max_size is not None:
            if max(orig_h, orig_w) * scale > self.max_size:
                scale = self.max_size / max(orig_h, orig_w)

        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        if new_w == orig_w and new_h == orig_h:
            return image, target

        image = torchvision.transforms.functional.resize(image, [new_h, new_w])
        boxes = target["boxes"]
        if boxes.numel() > 0:
            scale_tensor = torch.tensor([scale, scale, scale, scale], dtype=boxes.dtype)
            boxes = boxes * scale_tensor
            target["boxes"] = boxes
            target["area"] = target["area"] * (scale * scale)
        return image, target

class HelmetDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        # Map original category ids to contiguous labels starting from 1 (0 is background)
        cat_ids = sorted(self.coco.cats.keys())
        self.cat_id_to_label = {cat_id: idx + 1 for idx, cat_id in enumerate(cat_ids)}
        self.label_to_cat_id = {v: k for k, v in self.cat_id_to_label.items()}

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # Path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # Open image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img_w, img_h = img.size

        # Number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            w = coco_annotation[i]['bbox'][2]
            h = coco_annotation[i]['bbox'][3]

            # Convert to [xmin, ymin, xmax, ymax] and clamp to image bounds
            x0 = max(0, xmin)
            y0 = max(0, ymin)
            x1 = min(img_w, xmin + w)
            y1 = min(img_h, ymin + h)

            # Skip invalid/zero-area boxes to avoid exploding losses
            if x1 <= x0 or y1 <= y0:
                continue

            boxes.append([x0, y0, x1, y1])
            labels.append(self.cat_id_to_label[coco_annotation[i]['category_id']])
            areas.append((x1 - x0) * (y1 - y0))
            iscrowd.append(coco_annotation[i]['iscrowd'])

        if num_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Handle images with no objects
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        img_id = torch.tensor([img_id])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform(train, resize_min=None, resize_max=None):
    transforms = []
    # Resize first so boxes stay aligned
    transforms.append(Resize(resize_min, resize_max))
    if train:
        # Simple horizontal flip that also updates boxes
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(ToTensor())
    return Compose(transforms)
