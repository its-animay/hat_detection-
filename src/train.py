import argparse
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
    # handles torch.utils.data.Subset and raw dataset
    return ds.dataset if hasattr(ds, "dataset") else ds


def to_device(images, targets, device):
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return images, targets


def make_loader(dataset, batch_size, shuffle, num_workers):
    # pin_memory helps on CUDA, harmless otherwise
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
# LR Scheduling
# -----------------------------
def build_optimizer(model, lr, weight_decay, momentum):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)


def build_scheduler(optimizer, num_epochs: int, steps_per_epoch: int, warmup_epochs: int = 1):
    """
    Warmup for a few epochs then cosine decay.
    This is a strong default for fine-tuning detection models.
    """
    total_steps = max(1, num_epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        # cosine from 1 -> 0.05 (keeps LR from collapsing to 0)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(
    model,
    optimizer,
    scheduler,
    data_loader,
    device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip_norm: float = 0.0,
    log_every: int = 20,
):
    model.train()
    running = 0.0
    n = 0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} Training", dynamic_ncols=True)
    for step, (images, targets) in enumerate(pbar):
        images, targets = to_device(images, targets, device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
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

        # scheduler is step-based (per iteration)
        if scheduler is not None:
            scheduler.step()

        running += float(loss.item())
        n += 1

        if step % log_every == 0:
            pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

    return running / max(1, n)


@torch.no_grad()
def evaluate_coco(model, data_loader, device, score_thresh: float = 0.05, max_dets: int = 100):
    model.eval()

    ds = data_loader.dataset
    base = get_base_dataset(ds)

    coco = getattr(base, "coco", None)
    if coco is None:
        raise ValueError("COCO annotations not found in dataset; cannot evaluate.")

    label_to_cat = getattr(base, "label_to_cat_id", None)

    results = []
    pbar = tqdm(data_loader, desc="Evaluating", dynamic_ncols=True)

    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
            image_id = int(targets[i]["image_id"].item())

            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()

            # Filter low-confidence detections (stabilizes COCOeval early)
            keep = scores >= score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # Optional: cap detections
            if len(scores) > max_dets:
                order = np.argsort(-scores)[:max_dets]
                boxes, scores, labels = boxes[order], scores[order], labels[order]

            for box, score, label in zip(boxes, scores, labels):
                cat_id = int(label_to_cat[int(label)]) if label_to_cat is not None else int(label)

                x1, y1, x2, y2 = box
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))

                results.append(
                    {
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [float(x1), float(y1), w, h],
                        "score": float(score),
                    }
                )

    if not results:
        print("No detections found (after filtering).")
        return 0.0

    coco_dt = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return float(coco_eval.stats[0])  # AP@[0.5:0.95]


def freeze_backbone(model, freeze: bool = True):
    """
    Freezing backbone for first 1-2 epochs can help stabilize fine-tuning and improve accuracy.
    (Especially on small datasets.)
    """
    for name, param in model.backbone.named_parameters():
        param.requires_grad = not freeze


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

    # Use more CPU threads for dataloading/transforms
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # ------------------ Data ------------------
    print("Loading data...")
    train_dir = project_root / "data" / "train"
    train_ds = HelmetDataset(
        str(train_dir),
        str(train_dir / "_annotations.coco.json"),
        get_transform(train=True, resize_min=args.resize, resize_max=args.resize_max),
    )
    if args.max_train_images is not None:
        idx = list(range(min(args.max_train_images, len(train_ds))))
        train_ds = torch.utils.data.Subset(train_ds, idx)

    test_ann_path = project_root / "data" / "test" / "_annotations.coco.json"
    if test_ann_path.exists():
        test_ds = HelmetDataset(
            str(test_ann_path.parent),
            str(test_ann_path),
            get_transform(train=False, resize_min=args.resize, resize_max=args.resize_max),
        )
        if args.max_val_images is not None:
            idx = list(range(min(args.max_val_images, len(test_ds))))
            test_ds = torch.utils.data.Subset(test_ds, idx)
    else:
        print("Test dataset not found, using subset of train for validation (not recommended for final eval).")
        perm = torch.randperm(len(train_ds)).tolist()
        subset_size = args.max_val_images if args.max_val_images is not None else 50
        test_ds = torch.utils.data.Subset(train_ds, perm[:subset_size])

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    base_train = get_base_dataset(train_ds)
    num_classes = len(base_train.coco.cats) + 1
    print(f"Num classes: {num_classes}")

    # ------------------ Model ------------------
    model = get_model(num_classes, backbone=args.model)
    model.to(device)

    # Optional: freeze backbone for first epochs
    if args.freeze_backbone_epochs > 0:
        freeze_backbone(model, True)
        print(f"Backbone frozen for first {args.freeze_backbone_epochs} epoch(s).")

    # ------------------ Optim/Sched ------------------
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    steps_per_epoch = len(train_loader)

    scheduler = build_scheduler(
        optimizer,
        num_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )

    # AMP only on CUDA
    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ------------------ Train Loop ------------------
    best_map = -1.0
    patience_left = args.early_stop_patience

    out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
            scaler=scaler if use_amp else None,
            grad_clip_norm=args.grad_clip_norm,
        )
        dt = time.time() - t0
        print(f"Average Training Loss: {avg_loss:.4f}  | epoch_time: {dt:.1f}s")

        # Save last each epoch
        save_checkpoint(model, optimizer, out_dir / "ckpt_last.pt", epoch=epoch, best_map=best_map)

        # Evaluate every N epochs
        do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == args.epochs - 1)
        if do_eval:
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
                patience_left = args.early_stop_patience
                print(f"New best model! (mAP: {best_map:.4f})")
                save_checkpoint(model, optimizer, out_dir / "ckpt_best.pt", epoch=epoch, best_map=best_map)
            else:
                if args.early_stop_patience > 0:
                    patience_left -= 1
                    print(f"No improvement. Early-stop patience left: {patience_left}")
                    if patience_left <= 0:
                        print("Early stopping triggered.")
                        break

    print("Training completed.")
    print(f"Best mAP: {best_map:.4f}")
    print(f"Saved checkpoints in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN Helmet Detector (Optimized)")

    # Core
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--model", type=str, default="mobilenet_320", choices=["resnet50", "mobilenet", "mobilenet_320"])
    parser.add_argument("--output_dir", type=str, default="models")

    # DataLoader
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_images", type=int, default=None)
    parser.add_argument("--max_val_images", type=int, default=None)

    # Resize
    parser.add_argument("--resize", type=int, default=320, help="Set to 320 for mobilenet_320; 640 for stronger accuracy but slower")
    parser.add_argument("--resize_max", type=int, default=320)

    # Optimization knobs
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)

    # Eval knobs
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_score_thresh", type=float, default=0.05)
    parser.add_argument("--eval_max_dets", type=int, default=100)

    # Repro/Early stop
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=0, help="0 disables early stopping")

    args = parser.parse_args()
    main(args)
