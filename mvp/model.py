import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models, transforms


def load_meta(meta_path: Path) -> dict[str, Any]:
    return json.loads(meta_path.read_text(encoding="utf-8"))


def build_preprocess(meta: dict[str, Any]) -> transforms.Compose:
    imgsz = int(meta["imgsz"])
    mean = meta["normalize_mean"]
    std = meta["normalize_std"]
    # Match validation geometry used in training
    return transforms.Compose([
        transforms.Resize(imgsz + 64),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_densenet121_from_state_dict(
    weights_path: Path,
    meta: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    num_classes = int(meta["num_classes"])
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    state = torch.load(str(weights_path), map_location=device)  # state_dict
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_softmax(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # returns probs [C] on CPU
    with torch.no_grad():
        logits = model(x)  # [1, C]
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    return probs