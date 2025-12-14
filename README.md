# Helmet Detection Project

End-to-end helmet/person detector built with PyTorch Faster R-CNN, trained on the Kaggle “Helmet Detection” dataset and served via FastAPI (with optional Docker). Includes data download/conversion, training, evaluation, and inference APIs.

## Contents
- `src/data_setup.py`: downloads the Kaggle dataset (`andrewmvd/helmet-detection`), converts VOC XML to COCO JSON, and creates `data/train` + `data/test`.
- `src/dataset.py`: custom dataset + transforms (resize, flip, color jitter) with COCO-style targets.
- `src/model.py`: Faster R-CNN with configurable backbone (`resnet50`, `mobilenet`, `mobilenet_320`).
- `src/train.py`: training loop with AdamW/SGD, cosine or plateau scheduler, optional early stopping, COCO or torchmetrics eval.
- `src/inference.py`: batch/offline inference over an image folder; saves JSON + visualizations.
- `src/app.py`: FastAPI server for single-image inference (`/predict`), plus `/health`.
- `Dockerfile`: container for the inference API.
- `data/`: created by `data_setup.py` (COCO JSON + images).
- `models/`: saved checkpoints (`model_best.pth`, `model_last.pth`).
- `requirements.txt`: dependencies.

## Setup (local)
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data
The default pipeline uses the Kaggle “Helmet Detection” dataset. You need Kaggle credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`) available as env vars.
```bash
python -m src.data_setup
# Result: data/train, data/test, each with images + _annotations.coco.json
```
If you already have data, set `--train_dir/--val_dir/--train_ann/--val_ann` when training.

## Training
Run from the project root:
```bash
python -m src.train \
  --device cuda \
  --model resnet50 \
  --batch_size 2 \
  --epochs 20 \
  --optimizer adamw \
  --scheduler plateau \
  --eval_type torchmetrics \
  --resize 800 --resize_max 1024 \
  --early_stop_patience 5
```
Key flags:
- `--model`: `resnet50` (best accuracy), `mobilenet`, `mobilenet_320` (lightest).
- `--device`: `cuda`/`mps`/`cpu`/`auto`.
- `--optimizer`: `adamw` (default) or `sgd`; `--scheduler`: `plateau` (val-driven) or `cosine`.
- `--eval_type`: `torchmetrics` (COCO-free) or `coco`.
- `--max_train_images/--max_val_images`: subsample for quick runs.
- Paths: `--train_dir`, `--train_ann`, `--val_dir`, `--val_ann`.
Checkpoints are saved to `models/` by default.

## Offline Inference (folder → JSON/visuals)
```bash
python -m src.inference \
  --weights models/model_best.pth \
  --images_dir data/test \
  --ann data/test/_annotations.coco.json \
  --device cuda \
  --model resnet50 \
  --score_thresh 0.5 \
  --output_dir output/inference \
  --resize 800 --resize_max 1024
```
Outputs: annotated images and `predictions.json` in `output/inference`.

## FastAPI Inference Server (local)
Start the server:
```bash
uvicorn app:app --reload --port 8090
```
Env vars (optional):
- `MODEL_PATH` (default `models/model_best.pth`)
- `ANN_PATH` (default `data/test/_annotations.coco.json`)
- `MODEL_BACKBONE` (`resnet50`/`mobilenet`/`mobilenet_320`)
- `INFER_RESIZE`, `INFER_RESIZE_MAX`, `SCORE_THRESH`, `MAX_DETECTIONS`

Test with curl:
```bash
curl -X POST http://127.0.0.1:8090/predict \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```
Response: JSON with `detections` (bbox, score, category_id, label).

Health check:
```bash
curl http://127.0.0.1:8090/health
```

## Dockerized Inference
```bash
docker build -t helmet-infer .
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/models/model_best.pth \
  -e ANN_PATH=/app/data/test/_annotations.coco.json \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  helmet-infer
```
Then POST to `http://localhost:8000/predict` as above.
For GPU, run with `--gpus all` and a CUDA-capable base/host.

## Notes & Tips
- Defaults prioritize accuracy (ResNet50, 800px resize). For speed/thermals, use `mobilenet_320` and lower resize/batch size.
- If COCO eval errors, use `--eval_type torchmetrics`.
- Always ensure `data_setup.py` has been run (or provide correct paths) before training/inference.


## Results / Model Accuracy

**Model:** Faster R-CNN (ResNet-50 + FPN)  
**Task:** 2-class object detection — `helmet` vs `no_helmet` (background handled internally)  
**Device:** CUDA (GPU)  
**Epochs:** Early stopped at **20** (max planned: 30)  
**Best Checkpoint:** `models/model_best.pth`

### Key Validation Metrics (COCO Evaluation)

- **Best AP@0.50 (IoU=0.50): `0.7745` (~77.45%)**
- **AP@[0.50:0.95]: `0.454`** (stricter metric averaged across IoU thresholds)
- **AP@0.75: `~0.480`**
- **AR@[0.50:0.95] (maxDets=100): `~0.591`**

### Notes

- Performance improved quickly from **AP@0.50 = 0.431 (Epoch 1)** → **0.708 (Epoch 2)**, then stabilized around **0.75–0.77**.
- **Early stopping** was applied when validation AP@0.50 stopped improving, to reduce overfitting and save compute.
