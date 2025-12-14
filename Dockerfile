FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src ./src

# Default paths (override via env)
ENV MODEL_PATH=/app/models/model_best.pth \
    ANN_PATH=/app/data/test/_annotations.coco.json \
    MODEL_BACKBONE=resnet50 \
    INFER_RESIZE=800 \
    INFER_RESIZE_MAX=1024 \
    SCORE_THRESH=0.4 \
    MAX_DETECTIONS=50

# Expose API port
EXPOSE 8000

# Start the inference server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
