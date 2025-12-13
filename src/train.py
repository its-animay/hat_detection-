import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval

# Allow running both as a module (python -m src.train) and as a script (python src/train.py)
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

from src.dataset import HelmetDataset, get_transform
from src.model import get_model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} Training")
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())

    return total_loss / len(data_loader)

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    dataset = data_loader.dataset
    coco = getattr(dataset, "coco", None)
    if coco is None and hasattr(dataset, "dataset"):
        coco = getattr(dataset.dataset, "coco", None)
    if coco is None:
        raise ValueError("COCO annotations not found in dataset; cannot evaluate.")

    # Map model label -> original category id for correct COCO eval
    label_to_cat = getattr(dataset, "label_to_cat_id", None)
    if label_to_cat is None and hasattr(dataset, "dataset"):
        label_to_cat = getattr(dataset.dataset, "label_to_cat_id", None)
    
    results = []
    
    pbar = tqdm(data_loader, desc="Evaluating")
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        
        # Get predictions
        outputs = model(images)
        
        # Convert to COCO format for evaluation
        for i, output in enumerate(outputs):
            image_id = targets[i]["image_id"].item()
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                cat_id = int(label_to_cat[label]) if label_to_cat is not None else int(label)
                # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                
                results.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "score": float(score)
                })

    if not results:
        print("No detections found!")
        return 0.0

    # Load results into COCO object
    coco_dt = coco.loadRes(results)
    
    # Run COCO evaluation
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Return mAP@0.5:0.95 (first metric)
    return coco_eval.stats[0]

def main(args):
    project_root = Path(__file__).resolve().parent.parent
    if args.device == "auto":
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Data loading
    print("Loading data...")
    # Train dataset
    train_data_dir = project_root / 'data' / 'train'
    train_dataset = HelmetDataset(
        str(train_data_dir),
        str(train_data_dir / '_annotations.coco.json'),
        get_transform(train=True, resize_min=args.resize, resize_max=args.resize_max),
    )
    if args.max_train_images is not None:
        indices = list(range(min(args.max_train_images, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn
    )
    
    # Test/Validation dataset
    # Check if test data exists, otherwise split train
    test_ann_path = project_root / 'data' / 'test' / '_annotations.coco.json'
    if test_ann_path.exists():
        test_dataset = HelmetDataset(
            str(test_ann_path.parent),
            str(test_ann_path),
            get_transform(train=False, resize_min=args.resize, resize_max=args.resize_max),
        )
        if args.max_val_images is not None:
            indices = list(range(min(args.max_val_images, len(test_dataset))))
            test_dataset = torch.utils.data.Subset(test_dataset, indices)
    else:
        print("Test dataset not found, using subset of train for validation (not recommended for final eval)")
        indices = torch.randperm(len(train_dataset)).tolist()
        subset_size = args.max_val_images if args.max_val_images is not None else 50
        test_dataset = torch.utils.data.Subset(train_dataset, indices[:subset_size])
        
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
    )

    # Model
    base_train_dataset = train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset
    num_classes = len(base_train_dataset.coco.cats) + 1

    print(f"Num classes: {num_classes}")
    model = get_model(num_classes, backbone=args.model)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_map = 0.0
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        print(f"Average Training Loss: {avg_loss:.4f}")
        
        lr_scheduler.step()
        
        # Validation
        print("Running validation...")
        map_score = evaluate(model, test_loader, device)
        print(f"Validation mAP: {map_score:.4f}")
        
        # Save checkpoint
        if args.output_dir:
            output_dir = project_root / args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            # Save latest
            torch.save(model.state_dict(), os.path.join(output_dir, "model_last.pth"))
            
            # Save best
            if map_score > best_map:
                best_map = map_score
                print(f"New best model! (mAP: {best_map:.4f})")
                torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pth"))

    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN Helmet Detector")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (reduce if memory/heat is an issue)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device")
    parser.add_argument("--model", type=str, default="mobilenet_320", choices=["resnet50", "mobilenet", "mobilenet_320"], help="Backbone choice; mobilenet_320 is lightest/fastest")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--max_train_images", type=int, default=None, help="Limit number of training images for quick/light runs")
    parser.add_argument("--max_val_images", type=int, default=None, help="Limit number of validation images for quick/light runs")
    parser.add_argument("--resize", type=int, default=640, help="Resize shorter side to this (keeps aspect ratio); lower is faster")
    parser.add_argument("--resize_max", type=int, default=640, help="Clamp longer side to this after resize")
    
    args = parser.parse_args()
    main(args)
