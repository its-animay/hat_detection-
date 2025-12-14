import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

from src.dataset import get_transform  
from src.model import get_model  


def load_label_maps(annotation_file: Path):
    with open(annotation_file, "r") as f:
        data = json.load(f)
    categories = data["categories"]
    cat_ids = sorted(cat["id"] for cat in categories)
    cat_id_to_label = {cid: idx + 1 for idx, cid in enumerate(cat_ids)}
    label_to_cat_id = {v: k for k, v in cat_id_to_label.items()}
    label_to_name = {idx + 1: next(cat["name"] for cat in categories if cat["id"] == cid) for idx, cid in enumerate(cat_ids)}
    return cat_id_to_label, label_to_cat_id, label_to_name


def prepare_image(image_path: Path, transform):
    image = Image.open(image_path).convert("RGB")
    image, _ = transform(image, {})  
    return image


def draw_boxes(image: Image.Image, boxes, labels, scores, label_to_name, score_thresh):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        name = label_to_name.get(int(label), str(label))
        text = f"{name}: {score:.2f}"
        draw.text((x0 + 2, y0 + 2), text, fill="red")
    return image


def main(args):
    project_root = Path(__file__).resolve().parent.parent

    device = torch.device("cuda" if args.device == "cuda" else "mps" if args.device == "mps" else "cpu") if args.device != "auto" else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    ann_path = project_root / args.ann
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    cat_id_to_label, label_to_cat_id, label_to_name = load_label_maps(ann_path)
    num_classes = len(cat_id_to_label) + 1

    model = get_model(num_classes, backbone=args.model)
    model.load_state_dict(torch.load(project_root / args.weights, map_location=device))
    model.to(device)
    model.eval()

    transform = get_transform(train=False, resize_min=args.resize, resize_max=args.resize_max)

    images_dir = project_root / args.images_dir
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    output_dir = project_root / args.output_dir if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_json = output_dir / "predictions.json"
    else:
        pred_json = None

    exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])
    print(f"Found {len(image_files)} images in {images_dir}")

    all_results = []
    with torch.no_grad():
        for img_path in image_files:
            image_tensor = prepare_image(img_path, transform).to(device)
            outputs = model([image_tensor])[0]

            boxes = outputs["boxes"].cpu().numpy()
            scores = outputs["scores"].cpu().numpy()
            labels = outputs["labels"].cpu().numpy()

            # Save prediction entries in COCO-style
            for box, score, label in zip(boxes, scores, labels):
                if score < args.score_thresh:
                    continue
                x0, y0, x1, y1 = box
                all_results.append(
                    {
                        "image_id": img_path.name,
                        "category_id": int(label_to_cat_id.get(int(label), int(label))),
                        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                        "score": float(score),
                        "label_name": label_to_name.get(int(label), str(label)),
                    }
                )

            if output_dir:
                img_vis = Image.open(img_path).convert("RGB")
                img_vis = draw_boxes(img_vis, boxes, labels, scores, label_to_name, args.score_thresh)
                img_vis.save(output_dir / img_path.name)

    if pred_json:
        with open(pred_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved predictions JSON to {pred_json}")
    print("Inference completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for helmet detector")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory with images to run inference on")
    parser.add_argument("--ann", type=str, default="data/test/_annotations.coco.json", help="COCO annotations to derive label mapping")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device")
    parser.add_argument("--model", type=str, default="mobilenet_320", choices=["resnet50", "mobilenet", "mobilenet_320"], help="Backbone choice")
    parser.add_argument("--score_thresh", type=float, default=0.5, help="Confidence threshold for outputs")
    parser.add_argument("--output_dir", type=str, default="output/inference", help="Where to save visualized predictions and JSON")
    parser.add_argument("--resize", type=int, default=640, help="Resize shorter side to this (keeps aspect ratio)")
    parser.add_argument("--resize_max", type=int, default=640, help="Clamp longer side to this after resize")

    args = parser.parse_args()
    main(args)
