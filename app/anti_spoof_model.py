from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


IMAGE_SIZE = 224


def resource_path(relative_path: str) -> Path:
    if getattr(sys, "frozen", False):
        return Path(cast(str, getattr(sys, "_MEIPASS"))) / relative_path
    return Path(__file__).resolve().parent.parent / relative_path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = resource_path("models/anti_spoof_lcc")


class AntiSpoofModel:
    def __init__(self) -> None:
        class_names_path = MODEL_DIR / "class_names.json"
        weights_path = MODEL_DIR / "best_model.pth"

        if not class_names_path.exists():
            raise FileNotFoundError(f"Не найден файл классов: {class_names_path}")

        if not weights_path.exists():
            raise FileNotFoundError(f"Не найден файл весов: {weights_path}")

        with class_names_path.open("r", encoding="utf-8") as f:
            self.class_names: list[str] = json.load(f)

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

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
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

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

    @staticmethod
    def is_live_label(label: str) -> bool:
        normalized = label.strip().lower()

        live_keywords = {
            "live", "real", "genuine", "bonafide",
            "bona_fide", "authentic", "natural",
        }

        spoof_keywords = {
            "spoof", "fake", "attack", "photo",
            "print", "screen", "replay", "video",
        }

        if normalized in live_keywords:
            return True

        if normalized in spoof_keywords:
            return False

        if "live" in normalized or "real" in normalized or "genuine" in normalized:
            return True

        if (
            "spoof" in normalized
            or "fake" in normalized
            or "photo" in normalized
            or "print" in normalized
            or "screen" in normalized
            or "attack" in normalized
            or "replay" in normalized
            or "video" in normalized
        ):
            return False

        return False