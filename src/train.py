import argparse
import random
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval

from src.dataset import HelmetDataset, get_transform
from src.model import get_model


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


def build_optimizer(model, lr, momentum, weight_decay, kind: str):
    params = [p for p in model.parameters() if p.requires_grad]
    if kind == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)


def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int, kind: str):
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
        )

    warmup_epochs = max(0, min(warmup_epochs, num_epochs))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs), eta_min=1e-5)

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=1e-5
    )
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp, grad_clip_norm):
    model.train()
    running = 0.0
    n = 0

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1} Train", dynamic_ncols=True)
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
    label_to_cat = getattr(base, "label_to_cat_id", None)

    results = []
    pbar = tqdm(data_loader, desc="Eval (COCO)", dynamic_ncols=True)

    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)

        for i, out in enumerate(outputs):
            image_id = int(targets[i]["image_id"].item())

            # COCO expects ORIGINAL coords
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
        return 0.0

    coco_dt = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[1])  # stats[1] is often AP@0.5 (depending on COCOeval version)


def main(args):
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = project_root / args.train_dir
    val_dir = project_root / args.val_dir
    train_ann = train_dir / args.train_ann
    val_ann = val_dir / args.val_ann

    for p in [train_dir, train_ann, val_dir, val_ann]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run: python -m src.data_setup")

    train_ds = HelmetDataset(str(train_dir), str(train_ann), get_transform(True, args.resize, args.resize_max))
    val_ds = HelmetDataset(str(val_dir), str(val_ann), get_transform(False, args.resize, args.resize_max))

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(train_ds.coco.cats) + 1
    print(f"Num classes (incl background): {num_classes}")

    model = get_model(num_classes, backbone=args.model).to(device)

    optimizer = build_optimizer(model, args.lr, args.momentum, args.weight_decay, args.optimizer)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, args.scheduler)

    use_amp = (device.type == "cuda") and args.amp
    out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    best = -1.0
    stale = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch, use_amp, args.grad_clip_norm)
        if args.scheduler != "plateau":
            scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | loss={loss:.4f} | time={time.time()-t0:.1f}s")

        print("Validating...")
        ap50 = evaluate_coco(model, val_loader, device, args.eval_score_thresh, args.eval_max_dets)
        print(f"Val AP@0.5: {ap50:.4f}")

        if args.scheduler == "plateau":
            scheduler.step(1.0 - ap50)

        if ap50 > best + 1e-4:
            best = ap50
            stale = 0
            torch.save(model.state_dict(), out_dir / "model_best.pth")
            print(f"✅ Saved best model: {best:.4f}")
        else:
            stale += 1

        if args.early_stop_patience is not None and stale >= args.early_stop_patience:
            print("Early stopping.")
            break

    print(f"Done. Best AP@0.5: {best:.4f}")
    print(f"Saved at: {out_dir / 'model_best.pth'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Helmet Detection (COCO)")
    p.add_argument("--model", default="resnet50", choices=["resnet50", "mobilenet", "mobilenet_320"])

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"])

    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--resize", type=int, default=800)
    p.add_argument("--resize_max", type=int, default=1333)

    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--train_dir", type=str, default="data/train")
    p.add_argument("--train_ann", type=str, default="_annotations.coco.json")
    p.add_argument("--val_dir", type=str, default="data/test")
    p.add_argument("--val_ann", type=str, default="_annotations.coco.json")

    p.add_argument("--eval_score_thresh", type=float, default=0.05)
    p.add_argument("--eval_max_dets", type=int, default=100)

    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="models")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    main(args)
