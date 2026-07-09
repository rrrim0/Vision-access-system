import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

DATASET_ROOT = Path("data/lcc_fasd_raw")
TRAIN_DIR = DATASET_ROOT / "LCC_FASD_training"
VAL_DIR = DATASET_ROOT / "LCC_FASD_development"
TEST_DIR = DATASET_ROOT / "LCC_FASD_evaluation"

MODEL_DIR = Path("models/anti_spoof_lcc")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def build_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    targets = dataset.targets
    class_counts = Counter(targets)

    print("Распределение классов в train:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"  class {class_idx}: {count}")

    class_weights = {
        class_idx: 1.0 / count
        for class_idx, count in class_counts.items()
    }

    sample_weights = [float(class_weights[target]) for target in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_items += labels.size(0)

    avg_loss = total_loss / total_items
    avg_acc = total_correct / total_items
    return avg_loss, avg_acc


def main() -> None:
    print("🔹 Загрузка датасетов...")

    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR),
        transform=make_transforms(train=True),
    )
    val_dataset = datasets.ImageFolder(
        root=str(VAL_DIR),
        transform=make_transforms(train=False),
    )
    test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=make_transforms(train=False),
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    class_names = train_dataset.classes
    print("Классы:", class_names)

    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print("\nСоздание sampler...")
    train_sampler = build_weighted_sampler(train_dataset)

    print("Создание DataLoader...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("🔹 Инициализация модели...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Старт обучения...\n")

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n===== Эпоха {epoch + 1}/{EPOCHS} =====")

        model.train()
        running_loss = 0.0
        running_correct = 0
        total_items = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total_items += labels.size(0)

            # ЛОГ КАЖДЫЕ 50 БАТЧЕЙ
            if batch_idx % 50 == 0:
                print(
                    f"[Epoch {epoch+1}] Batch {batch_idx} | "
                    f"loss={loss.item():.4f}"
                )

        train_loss = running_loss / total_items
        train_acc = running_correct / total_items

        print("🔹 Валидация...")
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pth")
            print("Сохранена лучшая модель.")

    print(f"\nЛучшая val_acc = {best_val_acc:.4f}")

    print("Финальная проверка (evaluation)...")
    model.load_state_dict(torch.load(MODEL_DIR / "best_model.pth", map_location=DEVICE))
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"Evaluation | test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

if __name__ == "__main__":
    main()