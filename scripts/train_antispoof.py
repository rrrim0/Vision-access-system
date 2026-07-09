from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix


TRAIN_DIR = Path("data") / "lcc_fasd_raw" / "LCC_FASD_training"
VAL_DIR = Path("data") / "lcc_fasd_raw" / "LCC_FASD_development"
TEST_DIR = Path("data") / "lcc_fasd_raw" / "LCC_FASD_evaluation"

MODEL_PATH = Path("models") / "antispoof_model.pth"


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    acc = correct / max(total, 1)
    return acc, all_true, all_pred


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка: {TRAIN_DIR}")
    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка: {VAL_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка: {TEST_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(VAL_DIR), transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=eval_transform)

    print("Классы:", train_dataset.classes)
    if set(train_dataset.classes) != {"real", "spoof"}:
        raise RuntimeError("Ожидаются папки классов real/ и spoof/.")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Замораживаем backbone, обучаем классификатор
    for p in model.features.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    epochs = 8
    best_val_acc = 0.0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(len(train_dataset), 1)
        val_acc, _, _ = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                },
                MODEL_PATH,
            )
            print(f"Сохранена лучшая модель: {MODEL_PATH} (val_acc={best_val_acc:.4f})")

    # Финальная оценка на evaluation
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    test_acc, y_true, y_pred = evaluate(model, test_loader, device)

    print("\n==============================")
    print("ФИНАЛЬНАЯ ОЦЕНКА АНТИСПУФИНГА")
    print("==============================")
    print(f"Test accuracy: {test_acc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=train_dataset.classes,
            zero_division=0,
        )
    )

    print(f"\nИтоговая модель сохранена: {MODEL_PATH}")


if __name__ == "__main__":
    main()