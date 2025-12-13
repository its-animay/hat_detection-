import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def _replace_head(model, num_classes: int):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_model(num_classes: int, backbone: str = "resnet50"):
    """
    Create a Faster R-CNN model with a configurable backbone.

    backbone:
      - \"resnet50\" (default): strongest but heavier
      - \"mobilenet\": lighter/faster, good for low compute
      - \"mobilenet_320\": lightest (320px), fastest, lower accuracy
    """
    backbone = backbone.lower()
    if backbone == "mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    elif backbone == "mobilenet_320":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    return _replace_head(model, num_classes)
