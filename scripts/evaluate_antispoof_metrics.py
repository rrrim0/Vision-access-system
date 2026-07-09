from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


TEST_DIR = Path("data") / "lcc_fasd_raw" / "LCC_FASD_evaluation"
MODEL_PATH = Path("models") / "antispoof_model.pth"
OUT_DIR = Path("debug_output")


def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка: {TEST_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдена модель: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Классы датасета:", test_dataset.classes)
    if set(test_dataset.classes) != {"real", "spoof"}:
        raise RuntimeError("Ожидаются классы real и spoof.")

    class_to_idx = test_dataset.class_to_idx
    idx_real = class_to_idx["real"]
    idx_spoof = class_to_idx["spoof"]

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[idx_real, idx_spoof])

    # Матрица:
    # [[real->real, real->spoof],
    #  [spoof->real, spoof->spoof]]
    tn_real = cm[0, 0]   # real -> real
    fp_real = cm[0, 1]   # real -> spoof
    fn_spoof = cm[1, 0]  # spoof -> real
    tp_spoof = cm[1, 1]  # spoof -> spoof

    total_real = cm[0].sum()
    total_spoof = cm[1].sum()

    # FRR-подобный: подлинный пользователь отвергнут
    frr_like = fp_real / total_real if total_real > 0 else 0.0

    # FAR-подобный: атака принята как подлинный пользователь
    far_like = fn_spoof / total_spoof if total_spoof > 0 else 0.0

    print("\n==============================")
    print("АНАЛИЗ АНТИСПУФИНГА")
    print("==============================")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Balanced accuracy  : {bal_acc:.4f}")
    print(f"FRR-like (real→spoof) : {frr_like:.4f}")
    print(f"FAR-like (spoof→real): {far_like:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["real", "spoof"],
            zero_division=0,
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # График матрицы ошибок
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["real", "spoof"]
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    plt.title("Матрица ошибок антиспуфинг-модели")
    plt.tight_layout()

    out_path = OUT_DIR / "antispoof_confusion_matrix.png"
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"\nГрафик сохранён: {out_path}")


if __name__ == "__main__":
    main()