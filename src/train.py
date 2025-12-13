import argparse
import os
import sys
import time
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

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


# -----------------------------
# Utils
# -----------------------------
def collate_fn(batch):
    return tuple(zip(*batch))


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_base_dataset(ds):
    return ds.dataset if hasattr(ds, "dataset") else ds


def to_device(images, targets, device):
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return images, targets


def make_loader(dataset, batch_size, shuffle, num_workers):
    pin = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


# -----------------------------
# Better subset selection
# -----------------------------
def choose_best_subset(base_ds: HelmetDataset, fraction: float, seed: int = 42) -> List[int]:
    """
    Picks a "better" subset than random:
    - prioritizes images that actually have boxes
    - roughly balances classes by oversampling rare-class images into the selection pool
    """
    assert 0.0 < fraction <= 1.0
    rng = np.random.default_rng(seed)

    # Precompute counts per image (fast using COCO annotations)
    img_ids = base_ds.ids
    per_img = []
    class_freq = {}

    for idx, img_id in enumerate(img_ids):
        ann_ids = base_ds.coco.getAnnIds(imgIds=img_id)
        anns = base_ds.coco.loadAnns(ann_ids)

        labels = []
        for a in anns:
            cat = a["category_id"]
            # mapped label (1..K)
            lbl = base_ds.cat_id_to_label[cat]
            labels.append(lbl)

        # image "value": more boxes -> more signal
        num_boxes = len(labels)

        # update class freq
        for lbl in labels:
            class_freq[lbl] = class_freq.get(lbl, 0) + 1

        per_img.append((idx, num_boxes, labels))

    # Compute weights: rare classes get higher weight
    if class_freq:
        maxf = max(class_freq.values())
        class_weight = {c: (maxf / f) for c, f in class_freq.items()}
    else:
        class_weight = {}

    # Score each image
    scores = []
    for idx, num_boxes, labels in per_img:
        if num_boxes == 0:
            score = 0.05  # keep a few negatives but not too many
        else:
            # base score from number of boxes + rare-class boost
            rare_boost = sum(class_weight.get(l, 1.0) for l in set(labels))
            score = (1.0 + np.log1p(num_boxes)) * rare_boost
        scores.append(score)

    scores = np.array(scores, dtype=np.float32)
    scores = scores / (scores.sum() + 1e-12)

    k = int(round(len(img_ids) * fraction))
    k = max(1, k)

    chosen = rng.choice(len(img_ids), size=k, replace=False, p=scores)
    chosen = sorted(chosen.tolist())
    return chosen


# -----------------------------
# Optim / Scheduler (stable for detection fine-tuning)
# -----------------------------
def build_optimizer(model, lr, weight_decay, momentum):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)


def build_scheduler(optimizer, num_epochs: int, steps_per_epoch: int, warmup_epochs: int = 1):
    total_steps = max(1, num_epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # keep a floor so LR doesn't die completely
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -----------------------------
# Train / Eval
# -----------------------------
def freeze_backbone(model, freeze: bool = True):
    for p in model.backbone.parameters():
        p.requires_grad = not freeze


def train_one_epoch(
    model,
    optimizer,
    scheduler,
    data_loader,
    device,
    epoch: int,
    use_amp: bool,
    grad_clip_norm: float = 0.0,
    log_every: int = 50,
):
    model.train()
    running = 0.0
    n = 0

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} Training", dynamic_ncols=True)
    for step, (images, targets) in enumerate(pbar):
        images, targets = to_device(images, targets, device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        # ✅ Scheduler step AFTER optimizer.step (fixes your warning)
        if scheduler is not None:
            scheduler.step()

        running += float(loss.item())
        n += 1

        if step % log_every == 0:
            pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

    return running / max(1, n)


@torch.no_grad()
def evaluate_coco(model, data_loader, device, score_thresh: float = 0.02, max_dets: int = 200):
    model.eval()
    base = get_base_dataset(data_loader.dataset)

    coco = getattr(base, "coco", None)
    if coco is None:
        raise ValueError("COCO annotations not found in dataset; cannot evaluate.")
    label_to_cat = getattr(base, "label_to_cat_id", None)

    results = []
    pbar = tqdm(data_loader, desc="Evaluating", dynamic_ncols=True)

    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        for i, out in enumerate(outputs):
            image_id = int(targets[i]["image_id"].item())
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()

            keep = scores >= score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if len(scores) > max_dets:
                order = np.argsort(-scores)[:max_dets]
                boxes, scores, labels = boxes[order], scores[order], labels[order]

            for box, score, label in zip(boxes, scores, labels):
                cat_id = int(label_to_cat[int(label)]) if label_to_cat is not None else int(label)
                x1, y1, x2, y2 = box
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                results.append(
                    dict(
                        image_id=image_id,
                        category_id=cat_id,
                        bbox=[float(x1), float(y1), w, h],
                        score=float(score),
                    )
                )

    if not results:
        print("No detections found (after filtering).")
        return 0.0

    coco_dt = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])


def save_checkpoint(model, optimizer, path: Path, epoch: int, best_map: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_map": best_map,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        str(path),
    )


def sanity_print_dataset(ds, name="DATASET"):
    base = get_base_dataset(ds)
    cats = base.coco.cats
    print(f"\n[{name}] images={len(base.ids)} anns={len(base.coco.anns)}")
    print(f"[{name}] categories:")
    for k, v in cats.items():
        print(" ", k, "->", v.get("name"))

    # quick label sanity from 3 samples
    for i in [0, min(1, len(base)-1), min(2, len(base)-1)]:
        img, tgt = base[i]
        if tgt["labels"].numel():
            print(f"[{name}] sample {i} labels unique:", torch.unique(tgt["labels"]).tolist())
        else:
            print(f"[{name}] sample {i} has NO boxes")


# -----------------------------
# Main
# -----------------------------
def main(args):
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent.parent

    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # ------------------ Data ------------------
    print("Loading data...")
    train_dir = project_root / "data" / "train"
    train_base = HelmetDataset(
        str(train_dir),
        str(train_dir / "_annotations.coco.json"),
        get_transform(train=True, resize_min=args.resize, resize_max=args.resize_max),
    )

    test_ann_path = project_root / "data" / "test" / "_annotations.coco.json"
    if test_ann_path.exists():
        test_base = HelmetDataset(
            str(test_ann_path.parent),
            str(test_ann_path),
            get_transform(train=False, resize_min=args.resize, resize_max=args.resize_max),
        )
    else:
        raise FileNotFoundError("data/test/_annotations.coco.json not found. Please provide a real test/val split.")

    # Optional: best subset of training data
    if args.train_fraction < 1.0:
        chosen = choose_best_subset(train_base, fraction=args.train_fraction, seed=args.seed)
        train_ds = torch.utils.data.Subset(train_base, chosen)
        print(f"Using BEST subset: train_fraction={args.train_fraction} -> {len(chosen)} images")
    else:
        train_ds = train_base

    # Optional caps
    if args.max_train_images is not None:
        train_ds = torch.utils.data.Subset(train_ds, list(range(min(args.max_train_images, len(train_ds)))))
    if args.max_val_images is not None:
        test_ds = torch.utils.data.Subset(test_base, list(range(min(args.max_val_images, len(test_base)))))
    else:
        test_ds = test_base

    sanity_print_dataset(train_ds, "TRAIN")
    sanity_print_dataset(test_ds, "VAL")

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    base_train = get_base_dataset(train_ds)
    num_classes = len(base_train.coco.cats) + 1
    print(f"\nNum classes: {num_classes}")

    # ------------------ Model ------------------
    model = get_model(num_classes, backbone=args.model)
    model.to(device)

    # Backbone freezing: for custom datasets, often better to not freeze long
    if args.freeze_backbone_epochs > 0:
        freeze_backbone(model, True)
        print(f"Backbone frozen for first {args.freeze_backbone_epochs} epoch(s).")

    # ------------------ Optim/Sched ------------------
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = build_scheduler(optimizer, num_epochs=args.epochs, steps_per_epoch=len(train_loader), warmup_epochs=args.warmup_epochs)

    use_amp = (device.type == "cuda") and args.amp

    out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    best_map = -1.0

    # ------------------ Train Loop ------------------
    for epoch in range(args.epochs):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs:
            freeze_backbone(model, False)
            print("Unfroze backbone.")

        t0 = time.time()
        avg_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            grad_clip_norm=args.grad_clip_norm,
        )
        print(f"Average Training Loss: {avg_loss:.4f} | epoch_time: {time.time()-t0:.1f}s")

        save_checkpoint(model, optimizer, out_dir / "ckpt_last.pt", epoch=epoch, best_map=best_map)

        if ((epoch + 1) % args.eval_every == 0) or (epoch == args.epochs - 1):
            print("Running validation...")
            map_score = evaluate_coco(
                model=model,
                data_loader=test_loader,
                device=device,
                score_thresh=args.eval_score_thresh,
                max_dets=args.eval_max_dets,
            )
            print(f"Validation mAP (AP@[.50:.95]): {map_score:.4f}")

            if map_score > best_map:
                best_map = map_score
                print(f"New best model! (mAP: {best_map:.4f})")
                save_checkpoint(model, optimizer, out_dir / "ckpt_best.pt", epoch=epoch, best_map=best_map)

    print("Training completed.")
    print(f"Best mAP: {best_map:.4f}")
    print(f"Saved checkpoints in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN Helmet Detector (Accuracy-Optimized)")

    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=2)

    # IMPORTANT: for small datasets, lower LR helps
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "mobilenet", "mobilenet_320"])
    parser.add_argument("--output_dir", type=str, default="models")

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_images", type=int, default=None)
    parser.add_argument("--max_val_images", type=int, default=None)

    # Resize: resnet50 usually works well at 640
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--resize_max", type=int, default=640)

    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)

    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_score_thresh", type=float, default=0.02)
    parser.add_argument("--eval_max_dets", type=int, default=200)

    # ✅ Train on best subset
    parser.add_argument("--train_fraction", type=float, default=1.0, help="Use best subset of training data (e.g., 0.5)")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
