import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import kagglehub


def parse_annotations(ann_dir: Path):
    """Parse all VOC XMLs to collect category names and file list."""
    ann_paths = sorted(ann_dir.glob("*.xml"))
    categories = set()
    for ann in ann_paths:
        root = ET.parse(ann).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name:
                categories.add(name.strip())
    categories = sorted(categories)
    cat_to_id = {c: i + 1 for i, c in enumerate(categories)}  # COCO categories start at 1
    return ann_paths, categories, cat_to_id


def convert_split(ann_paths, cat_to_id, images_dir: Path, split_dir: Path, start_image_id: int = 0):
    """Convert a list of VOC XMLs into COCO JSON and copy images."""
    split_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 0

    for idx, ann_path in enumerate(ann_paths):
        root = ET.parse(ann_path).getroot()
        filename = root.findtext("filename")

        w_txt = root.findtext("size/width")
        h_txt = root.findtext("size/height")
        if w_txt is None or h_txt is None:
            # Skip broken XML
            continue

        width = int(float(w_txt))
        height = int(float(h_txt))
        image_id = start_image_id + idx

        src_img = images_dir / filename
        if not src_img.exists():
            # Some datasets have mismatched names; skip if missing
            continue

        dst_img = split_dir / filename
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)

        images.append(
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": filename,
            }
        )

        for obj in root.findall("object"):
            name = obj.findtext("name")
            if not name:
                continue

            name = name.strip()
            if name not in cat_to_id:
                continue
            cat_id = cat_to_id[name]

            bbox = obj.find("bndbox")
            if bbox is None:
                continue

            x0 = float(bbox.findtext("xmin", "0"))
            y0 = float(bbox.findtext("ymin", "0"))
            x1 = float(bbox.findtext("xmax", "0"))
            y1 = float(bbox.findtext("ymax", "0"))

            # Clamp to bounds
            x0 = max(0.0, min(x0, float(width)))
            y0 = max(0.0, min(y0, float(height)))
            x1 = max(0.0, min(x1, float(width)))
            y1 = max(0.0, min(y1, float(height)))

            if x1 <= x0 or y1 <= y0:
                continue

            bw = x1 - x0
            bh = y1 - y0

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x0, y0, bw, bh],  # COCO xywh
                    "area": float(bw * bh),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

    return images, annotations


def save_coco_json(path: Path, images, annotations, categories):
    data = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(categories)
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def setup_data(train_split: float = 0.8, seed: int = 42, overwrite: bool = True):
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if overwrite:
        for p in (train_dir, test_dir):
            if p.exists():
                shutil.rmtree(p)

    print("Downloading dataset andrewmvd/helmet-detection from Kaggle...")
    ds_path = Path(kagglehub.dataset_download("andrewmvd/helmet-detection"))
    images_dir = ds_path / "images"
    ann_dir = ds_path / "annotations"
    if not images_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError(f"Expected images/ and annotations/ inside: {ds_path}")

    ann_paths, categories, cat_to_id = parse_annotations(ann_dir)

    random.seed(seed)
    random.shuffle(ann_paths)
    split_idx = int(len(ann_paths) * train_split)

    train_anns = ann_paths[:split_idx]
    test_anns = ann_paths[split_idx:]

    print(f"Total XMLs: {len(ann_paths)} | Train: {len(train_anns)} | Test: {len(test_anns)}")
    print(f"Categories: {categories}")

    train_images, train_annotations = convert_split(train_anns, cat_to_id, images_dir, train_dir, start_image_id=0)
    test_images, test_annotations = convert_split(
        test_anns, cat_to_id, images_dir, test_dir, start_image_id=len(train_images)
    )

    save_coco_json(train_dir / "_annotations.coco.json", train_images, train_annotations, categories)
    save_coco_json(test_dir / "_annotations.coco.json", test_images, test_annotations, categories)

    print("✅ Dataset setup complete.")
    print(f"Train folder: {train_dir}")
    print(f"Test folder : {test_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle helmet dataset and convert to COCO")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_overwrite", action="store_true")
    args = parser.parse_args()

    setup_data(train_split=args.train_split, seed=args.seed, overwrite=not args.no_overwrite)
