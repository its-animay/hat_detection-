import io
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset import get_transform  # noqa: E402
from src.model import get_model  # noqa: E402


def load_label_maps(annotation_file: Path):
    import json

    with open(annotation_file, "r") as f:
        data = json.load(f)
    categories = data["categories"]
    cat_ids = sorted(cat["id"] for cat in categories)
    cat_id_to_label = {cid: idx + 1 for idx, cid in enumerate(cat_ids)}
    label_to_cat_id = {v: k for k, v in cat_id_to_label.items()}
    label_to_name = {idx + 1: next(cat["name"] for cat in categories if cat["id"] == cid) for idx, cid in enumerate(cat_ids)}
    return cat_id_to_label, label_to_cat_id, label_to_name


def prepare_image(img: Image.Image, transform):
    img = img.convert("RGB")
    dummy_target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "area": torch.zeros((0,), dtype=torch.float32),
        "iscrowd": torch.zeros((0,), dtype=torch.int64),
    }
    tensor, _ = transform(img, dummy_target)
    return tensor


def load_model(
    weights_path: Path,
    ann_path: Path,
    backbone: str,
    device: torch.device,
    resize_min: Optional[int],
    resize_max: Optional[int],
):
    cat_id_to_label, label_to_cat_id, label_to_name = load_label_maps(ann_path)
    num_classes = len(cat_id_to_label) + 1

    model = get_model(num_classes, backbone=backbone)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    transform = get_transform(train=False, resize_min=resize_min, resize_max=resize_max)
    return model, transform, label_to_cat_id, label_to_name


MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "model_best.pth"))
ANN_PATH = Path(os.getenv("ANN_PATH", PROJECT_ROOT / "data" / "test" / "_annotations.coco.json"))
BACKBONE = os.getenv("MODEL_BACKBONE", "resnet50")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESIZE_MIN = int(os.getenv("INFER_RESIZE", "800"))
RESIZE_MAX = int(os.getenv("INFER_RESIZE_MAX", "1024"))
SCORE_THRESH = float(os.getenv("SCORE_THRESH", "0.4"))
MAX_DETS = int(os.getenv("MAX_DETECTIONS", "50"))

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
if not ANN_PATH.exists():
    raise FileNotFoundError(f"ANN_PATH not found: {ANN_PATH}")

model, transform, label_to_cat_id, label_to_name = load_model(
    MODEL_PATH, ANN_PATH, BACKBONE, DEVICE, RESIZE_MIN, RESIZE_MAX
)

app = FastAPI(title="Helmet Detector", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
@torch.no_grad()
def predict(file: UploadFile = File(...)):
    try:
        image_bytes = file.file.read()
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    tensor = prepare_image(img, transform).to(DEVICE)
    outputs = model([tensor])[0]

    boxes = outputs["boxes"].cpu()
    scores = outputs["scores"].cpu()
    labels = outputs["labels"].cpu()

    keep = scores >= SCORE_THRESH
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if len(scores) > MAX_DETS:
        order = torch.argsort(scores, descending=True)[:MAX_DETS]
        boxes, scores, labels = boxes[order], scores[order], labels[order]

    detections: List[dict] = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
                "category_id": int(label_to_cat_id.get(int(label), int(label))),
                "label": label_to_name.get(int(label), str(label)),
            }
        )

    return {"detections": detections, "count": len(detections)}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "backbone": BACKBONE}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
