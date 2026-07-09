import json
from pathlib import Path
from typing import Any, cast

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_DIR = Path("models/anti_spoof_lcc")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224


class AntiSpoofModel:
    def __init__(self) -> None:
        with open(MODEL_DIR / "class_names.json", "r", encoding="utf-8") as f:
            self.class_names: list[str] = json.load(f)

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE))
        self.model = model.to(DEVICE)
        self.model.eval()
        self.device = DEVICE

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, frame_bgr: Any) -> tuple[str, float]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        transformed = self.transform(image)
        tensor = cast(torch.Tensor, transformed)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        pred_index = int(pred_idx.item())
        confidence = float(conf.item())
        label = self.class_names[pred_index]

        return label, confidence