from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from app.face_embedder import FaceEmbedder


DATASET_DIR = Path("data") / "id_test_images" / "twins"
LABELS = ["twinA", "twinB"]
NUM_REFERENCE_IMAGES = 3

CONF_MATRIX_PNG = Path("twins_facenet_confusion_matrix.png")
SIMILARITY_PNG = Path("twins_facenet_similarity.png")


def load_image_paths(person_dir):
    valid_suffixes = {".jpg", ".jpeg", ".png"}
    return [
        p for p in sorted(person_dir.glob("*"))
        if p.is_file() and p.suffix.lower() in valid_suffixes
    ]


def read_image(path):
    return cv2.imread(str(path))


def cosine_similarity(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mean_embedding(embeddings):
    arr = np.stack(embeddings, axis=0)
    mean_emb = arr.mean(axis=0).astype(np.float32)

    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm

    return mean_emb


def build_reference_embeddings(embedder, dataset_dir, labels, num_reference_images):
    references = {}

    print("\n=== Формирование эталонных эмбеддингов ===")

    for label in labels:
        person_dir = dataset_dir / label
        if not person_dir.exists():
            raise FileNotFoundError(f"Не найдена папка класса: {person_dir}")

        image_paths = load_image_paths(person_dir)
        if len(image_paths) < num_reference_images + 1:
            raise ValueError(
                f"Для класса {label} недостаточно изображений. "
                f"Нужно хотя бы {num_reference_images + 1}, найдено {len(image_paths)}."
            )

        ref_paths = image_paths[:num_reference_images]
        ref_embeddings = []

        print(f"\n{label}: эталонные изображения:")
        for path in ref_paths:
            print(f"  - {path.name}")

            img = read_image(path)
            if img is None:
                print(f"    [SKIP] Не удалось прочитать: {path}")
                continue

            emb, _ = embedder.get_embedding(img)
            if emb is None:
                print(f"    [SKIP] Лицо не найдено: {path.name}")
                continue

            ref_embeddings.append(emb)

        if not ref_embeddings:
            raise RuntimeError(
                f"Не удалось получить ни одного эталонного эмбеддинга для {label}"
            )

        references[label] = mean_embedding(ref_embeddings)
        print(f"  Итог: получено эталонных эмбеддингов = {len(ref_embeddings)}")

    return references


def evaluate(embedder, dataset_dir, labels, references, num_reference_images):
    y_true = []
    y_pred = []

    total_images = 0
    skipped = 0

    same_scores = []
    other_scores = []
    details = []

    print("\n=== Тестирование на остальных изображениях ===")

    for true_label in labels:
        person_dir = dataset_dir / true_label
        image_paths = load_image_paths(person_dir)
        test_paths = image_paths[num_reference_images:]

        print(f"\n{true_label}: тестовых изображений = {len(test_paths)}")

        for path in test_paths:
            total_images += 1

            img = read_image(path)
            if img is None:
                skipped += 1
                print(f"[SKIP] Не удалось прочитать: {path}")
                continue

            emb, _ = embedder.get_embedding(img)
            if emb is None:
                skipped += 1
                print(f"[SKIP] Лицо не найдено: {path.name}")
                continue

            sim_a = float(cosine_similarity(emb, references[LABELS[0]]))
            sim_b = float(cosine_similarity(emb, references[LABELS[1]]))

            if sim_a >= sim_b:
                pred_label = LABELS[0]
            else:
                pred_label = LABELS[1]

            y_true.append(true_label)
            y_pred.append(pred_label)

            if true_label == LABELS[0]:
                true_score = sim_a
                other_score = sim_b
            else:
                true_score = sim_b
                other_score = sim_a

            same_scores.append(true_score)
            other_scores.append(other_score)

            details.append({
                "file": path.name,
                "true": true_label,
                "pred": pred_label,
                "same_score": true_score,
                "other_score": other_score,
                "margin": true_score - other_score,
            })

    return y_true, y_pred, total_images, skipped, same_scores, other_scores, details


def save_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Матрица ошибок FaceNet на датасете близнецов")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_similarity_plot(same_scores, other_scores, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = 20
    ax.hist(same_scores, bins=bins, alpha=0.7, label="Сходство с истинным классом")
    ax.hist(other_scores, bins=bins, alpha=0.7, label="Сходство с другим близнецом")

    ax.set_title("Распределение cosine similarity для FaceNet")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Количество изображений")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_hard_cases(details, top_k=10):
    print("\n=== Самые сложные случаи ===")
    sorted_cases = sorted(details, key=lambda x: x["margin"])

    for row in sorted_cases[:top_k]:
        print(
            f"{row['file']}: true={row['true']}, pred={row['pred']}, "
            f"same={row['same_score']:.4f}, "
            f"other={row['other_score']:.4f}, "
            f"margin={row['margin']:.4f}"
        )


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка датасета: {DATASET_DIR}")

    embedder = FaceEmbedder()

    references = build_reference_embeddings(
        embedder,
        DATASET_DIR,
        LABELS,
        NUM_REFERENCE_IMAGES,
    )

    y_true, y_pred, total_images, skipped, same_scores, other_scores, details = evaluate(
        embedder,
        DATASET_DIR,
        LABELS,
        references,
        NUM_REFERENCE_IMAGES,
    )

    print("\n==============================")
    print("РЕЗУЛЬТАТЫ ТЕСТА FACENET НА БЛИЗНЕЦАХ")
    print("==============================")
    print(f"Всего тестовых изображений: {total_images}")
    print(f"Пропущено: {skipped}")
    print(f"Оценено: {len(y_true)}")

    if len(y_true) == 0:
        print("Нет данных для оценки.")
        return

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    print(f"\nСреднее сходство с истинным классом: {np.mean(same_scores):.4f}")
    print(f"Среднее сходство с другим близнецом: {np.mean(other_scores):.4f}")
    print(f"Средний отрыв (margin): {(np.mean(same_scores) - np.mean(other_scores)):.4f}")

    print_hard_cases(details, top_k=10)

    save_confusion_matrix(y_true, y_pred, LABELS, CONF_MATRIX_PNG)
    save_similarity_plot(same_scores, other_scores, SIMILARITY_PNG)

    print("\nСохранено:")
    print(f"  - {CONF_MATRIX_PNG.resolve()}")
    print(f"  - {SIMILARITY_PNG.resolve()}")


if __name__ == "__main__":
    main()