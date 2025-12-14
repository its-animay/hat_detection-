import argparse
import sys
import time
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Allow running as script or module
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_base_dataset(ds):
    return ds.dataset if hasattr(ds, "dataset") else ds


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


def random_subset(dataset, n: int, seed: int):
    """Return a random Subset of size n (or full dataset if n is None)."""
    if n is None:
        return dataset
    n = min(n, len(dataset))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    return torch.utils.data.Subset(dataset, idx)


def print_label_distribution(ds, name="DATASET", max_items=300):
    """Quick sanity check: how many labels appear in the subset."""
    base = get_base_dataset(ds)
    cnt = Counter()
    k = min(max_items, len(ds))
    for i in range(k):
        _, t = ds[i]
        cnt.update(t["labels"].tolist())
    print(f"\n[{name}] Label distribution (first {k} samples): {dict(cnt)}")


# -----------------------------
# Optimizer + Scheduler
# -----------------------------
def build_optimizer(model, lr, momentum, weight_decay, kind: str):
    params = [p for p in model.parameters() if p.requires_grad]
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
    )


def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int, kind: str):
    """
    - cosine: warmup then cosine decay (epoch-level)
    - plateau: ReduceLROnPlateau on validation loss (step after validation)
    """
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
        )

    warmup_epochs = max(0, min(warmup_epochs, num_epochs))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, num_epochs), eta_min=1e-5
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=1e-5
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp, grad_clip_norm):
    model.train()
    running = 0.0
    n = 0

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} Training", dynamic_ncols=True)
    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", enabled=True):
                loss_dict = model(images, targets)
                loss = sum(v for v in loss_dict.values())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running += float(loss.item())
        n += 1
        pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

    return running / max(1, n)


@torch.no_grad()
def evaluate_coco(model, data_loader, device, score_thresh=0.05, max_dets=100):
    model.eval()

    base = get_base_dataset(data_loader.dataset)
    coco = getattr(base, "coco", None)
    if coco is None:
        raise ValueError("COCO annotations not found; cannot evaluate.")

    label_to_cat = getattr(base, "label_to_cat_id", None)

    results = []
    pbar = tqdm(data_loader, desc="Evaluating", dynamic_ncols=True)

    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        for i, out in enumerate(outputs):
            image_id = int(targets[i]["image_id"].item())

            # Original image size (COCO expects original coords)
            info = coco.loadImgs(image_id)[0]
            orig_w, orig_h = float(info["width"]), float(info["height"])

            _, resized_h, resized_w = images[i].shape
            sx = orig_w / float(resized_w)
            sy = orig_h / float(resized_h)

            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()

            keep = scores >= score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if len(scores) > max_dets:
                order = np.argsort(-scores)[:max_dets]
                boxes, scores, labels = boxes[order], scores[order], labels[order]

            # Scale boxes back to original image coords
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

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
        print("No detections found.")
        return 0.0

    coco_dt = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0])


@torch.no_grad()
def evaluate_torchmetrics(model, data_loader, device, score_thresh=0.05, max_dets=100):
    """COCO-free evaluation using torchmetrics MeanAveragePrecision."""
    metric = MeanAveragePrecision(iou_type="bbox")
    model.eval()

    pbar = tqdm(data_loader, desc="Evaluating", dynamic_ncols=True)
    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        preds = []
        gts = []

        for i, out in enumerate(outputs):
            boxes = out["boxes"].detach().cpu()
            scores = out["scores"].detach().cpu()
            labels = out["labels"].detach().cpu()

            keep = scores >= score_thresh
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if len(scores) > max_dets:
                order = torch.argsort(scores, descending=True)[:max_dets]
                boxes, scores, labels = boxes[order], scores[order], labels[order]

            preds.append({"boxes": boxes, "scores": scores, "labels": labels})

            gt = {
                "boxes": targets[i]["boxes"].cpu(),
                "labels": targets[i]["labels"].cpu(),
            }
            gts.append(gt)

        metric.update(preds, gts)

    res = metric.compute()
    return float(res["map_50"].item()), res


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

    # ------------------ Data ------------------
    print("Loading data...")

    # Resolve paths
    train_dir = project_root / args.train_dir
    train_ann = train_dir / args.train_ann
    val_dir = project_root / args.val_dir
    val_ann = val_dir / args.val_ann

    for p in [train_dir, train_ann, val_dir, val_ann]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run `python -m src.data_setup` to download/prepare the dataset.")

    train_full = HelmetDataset(
        str(train_dir),
        str(train_ann),
        get_transform(train=True, resize_min=args.resize, resize_max=args.resize_max),
    )

    val_full = HelmetDataset(
        str(val_dir),
        str(val_ann),
        get_transform(train=False, resize_min=args.resize, resize_max=args.resize_max),
    )

    # Subset 
    train_ds = random_subset(train_full, args.max_train_images, seed=args.seed)
    val_ds = random_subset(val_full, args.max_val_images, seed=args.seed + 999)

    if args.max_train_images is not None:
        print(f"Training on subset: {len(train_ds)} images")
    if args.max_val_images is not None:
        print(f"Validating on subset: {len(val_ds)} images")

    if args.print_dist:
        print_label_distribution(train_ds, "TRAIN_SUBSET")
        print_label_distribution(val_ds, "VAL_SUBSET")

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    base_train = get_base_dataset(train_ds)
    num_classes = len(base_train.coco.cats) + 1
    print(f"Num classes: {num_classes}")

    # ------------------ Model ------------------
    model = get_model(num_classes, backbone=args.model)
    model.to(device)

    # ------------------ Optim/Sched ------------------
    optimizer = build_optimizer(model, args.lr, args.momentum, args.weight_decay, kind=args.optimizer)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, kind=args.scheduler)
    use_amp = (device.type == "cuda") and args.amp

    out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ Train Loop ------------------
    best_map = -1.0
    stale_epochs = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            grad_clip_norm=args.grad_clip_norm,
        )
        if args.scheduler != "plateau":
            scheduler.step()  # epoch-level scheduler step

        print(f"Epoch {epoch+1}/{args.epochs} | loss={loss:.4f} | time={time.time()-t0:.1f}s")

        if (epoch + 1) % args.eval_every == 0 or (epoch == args.epochs - 1):
            print("Running validation...")
            if args.eval_type == "coco":
                mAP = evaluate_coco(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                    score_thresh=args.eval_score_thresh,
                    max_dets=args.eval_max_dets,
                )
                metrics_to_print = {"mAP_coco": mAP}
                val_score = mAP
            else:
                mAP50, full = evaluate_torchmetrics(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                    score_thresh=args.eval_score_thresh,
                    max_dets=args.eval_max_dets,
                )
                metrics_to_print = {
                    "mAP@0.5": mAP50,
                    "mAP@0.5:0.95": float(full["map"].item()),
                    "precision": float(full["map_per_class"].mean().item()) if full["map_per_class"].numel() > 0 else None,
                    "recall": float(full["mar"].item()),
                }
                val_score = mAP50

            print(f"Validation metrics: {metrics_to_print}")

            if args.scheduler == "plateau":
                scheduler.step(1.0 - val_score)

            if val_score > best_map + 1e-4:
                best_map = val_score
                stale_epochs = 0
                torch.save(model.state_dict(), out_dir / "model_best.pth")
                print(f"Saved best model (score={best_map:.4f})")
            else:
                stale_epochs += 1

            if args.early_stop_patience is not None and stale_epochs >= args.early_stop_patience:
                print(f"No improvement for {stale_epochs} evals. Early stopping.")
                break

    print("Done.")
    print(f"Best mAP: {best_map:.4f}")
    print(f"Saved in: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Helmet Detection")

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--model", default="mobilenet_320", choices=["resnet50", "mobilenet", "mobilenet_320"])

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    p.add_argument("--scheduler", type=str, default="plateau", choices=["cosine", "plateau"])

    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--resize", type=int, default=640)
    p.add_argument("--resize_max", type=int, default=640)

    p.add_argument("--num_workers", type=int, default=2)

    # Data paths
    p.add_argument("--train_dir", type=str, default="data/train")
    p.add_argument("--train_ann", type=str, default="_annotations.coco.json")
    p.add_argument("--val_dir", type=str, default="data/test")
    p.add_argument("--val_ann", type=str, default="_annotations.coco.json")

    #subset knobs
    p.add_argument("--max_train_images", type=int, default=None, help="If set, randomly subsample this many train images")
    p.add_argument("--max_val_images", type=int, default=None, help="If set, randomly subsample this many val images")

    # Eval knobs
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--eval_score_thresh", type=float, default=0.05)
    p.add_argument("--eval_max_dets", type=int, default=100)
    p.add_argument("--eval_type", type=str, default="torchmetrics", choices=["torchmetrics", "coco"], help="Use COCO mAP or torchmetrics mAP (no pycocotools needed)")
    p.add_argument("--early_stop_patience", type=int, default=5, help="Stop if no improvement after this many evals; set None to disable")

    p.add_argument("--output_dir", type=str, default="models")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_dist", action="store_true", help="Print label distribution for sanity check")

    args = p.parse_args()
    main(args)
