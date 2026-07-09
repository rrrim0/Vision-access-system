from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

from sklearn.manifold import TSNE

# =====================================================
# ПОДКЛЮЧЕНИЕ КОРНЯ ПРОЕКТА
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.face_embedder import FaceEmbedder  # noqa: E402


# =====================================================
# НАСТРОЙКИ
# =====================================================

DATASET_ROOT = PROJECT_ROOT / "data" / "evaluation_dataset_matched"

REAL_DIR = DATASET_ROOT / "real"
SPOOF_DIR = DATASET_ROOT / "spoof"

RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_CSV = RESULTS_DIR / "final_metrics_report.csv"
SUMMARY_TXT = RESULTS_DIR / "summary.txt"

CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
SIMILARITY_HIST_PATH = RESULTS_DIR / "similarity_distribution.png"
ROC_PATH = RESULTS_DIR / "roc_curve.png"
TSNE_PATH = RESULTS_DIR / "tsne_embeddings.png"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Сколько изображений из real-кластера использовать как эталон
REFERENCE_IMAGES_PER_CLUSTER = 2

# Порог cosine similarity для распознавания лица
FACE_MATCH_THRESHOLD = 0.60

# Для t-SNE лучше ограничить количество точек, чтобы не было слишком тяжело
MAX_TSNE_SAMPLES = 800


# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================

def collect_images(folder: Path) -> list[Path]:
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]


def read_image(path: Path):
    return cv2.imread(str(path))


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)

    if norm > 0:
        embedding = embedding / norm

    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_embedding(a)
    b = normalize_embedding(b)
    return float(np.dot(a, b))


def get_embedding_safe(embedder: FaceEmbedder, image_path: Path):
    image = read_image(image_path)

    if image is None:
        return None, None, 0.0

    start_time = time.perf_counter()

    try:
        embedding, bbox = embedder.get_embedding(image)
    except Exception:
        embedding, bbox = None, None

    elapsed = time.perf_counter() - start_time

    if embedding is None or bbox is None:
        return None, None, elapsed

    return normalize_embedding(embedding), bbox, elapsed


# =====================================================
# ФОРМИРОВАНИЕ ЭТАЛОНОВ ПО REAL-КЛАСТЕРАМ
# =====================================================

def build_reference_database(embedder: FaceEmbedder):
    references = {}
    test_real_images = []

    cluster_dirs = sorted(
        p for p in REAL_DIR.iterdir()
        if p.is_dir() and p.name.startswith("cluster_")
    )

    for cluster_dir in cluster_dirs:
        images = collect_images(cluster_dir)

        if len(images) <= REFERENCE_IMAGES_PER_CLUSTER:
            continue

        reference_paths = images[:REFERENCE_IMAGES_PER_CLUSTER]
        test_paths = images[REFERENCE_IMAGES_PER_CLUSTER:]

        embeddings = []

        for image_path in reference_paths:
            embedding, bbox, _ = get_embedding_safe(embedder, image_path)

            if embedding is not None:
                embeddings.append(embedding)

        if not embeddings:
            continue

        centroid = np.mean(np.vstack(embeddings), axis=0)
        references[cluster_dir.name] = normalize_embedding(centroid)

        for path in test_paths:
            test_real_images.append((path, cluster_dir.name))

    return references, test_real_images


# =====================================================
# РАСПОЗНАВАНИЕ
# =====================================================

def recognize(embedding: np.ndarray, references: dict[str, np.ndarray]):
    best_cluster = None
    best_score = -1.0

    for cluster_name, ref_embedding in references.items():
        score = cosine_similarity(embedding, ref_embedding)

        if score > best_score:
            best_score = score
            best_cluster = cluster_name

    if best_score < FACE_MATCH_THRESHOLD:
        return "unknown", best_score

    return best_cluster, best_score


# =====================================================
# ОСНОВНАЯ ОЦЕНКА
# =====================================================

def evaluate():
    print("===================================")
    print("ОЦЕНКА ИТОГОВОЙ СИСТЕМЫ")
    print("===================================")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Не найден датасет: {DATASET_ROOT}")

    embedder = FaceEmbedder()

    print("\nФормирование эталонной базы real-кластеров...")
    references, test_real_images = build_reference_database(embedder)

    print(f"Эталонных пользователей/кластеров: {len(references)}")
    print(f"Тестовых real-изображений: {len(test_real_images)}")

    y_true = []
    y_pred = []

    binary_true = []
    binary_scores = []

    similarity_scores_real = []
    similarity_scores_spoof = []

    all_embeddings = []
    all_embedding_labels = []

    processing_times = []

    total_real = 0
    correct_real = 0
    false_rejects = 0

    total_spoof = 0
    blocked_spoof = 0
    false_accepts = 0

    report_rows = []

    # =====================================================
    # REAL TEST
    # =====================================================

    print("\nТестирование real-изображений...")

    for image_path, true_cluster in test_real_images:
        total_real += 1

        embedding, bbox, elapsed = get_embedding_safe(embedder, image_path)
        processing_times.append(elapsed)

        if embedding is None:
            predicted_cluster = "unknown"
            best_score = 0.0
            false_rejects += 1
        else:
            predicted_cluster, best_score = recognize(embedding, references)

            similarity_scores_real.append(best_score)
            binary_true.append(1)
            binary_scores.append(best_score)

            all_embeddings.append(embedding)
            all_embedding_labels.append(true_cluster)

            if predicted_cluster == true_cluster:
                correct_real += 1
            else:
                false_rejects += 1

        y_true.append(true_cluster)
        y_pred.append(predicted_cluster)

        report_rows.append({
            "path": str(image_path),
            "type": "real",
            "true_cluster": true_cluster,
            "predicted_cluster": predicted_cluster,
            "similarity": round(float(best_score), 6),
            "processing_time_sec": round(float(elapsed), 6),
            "result": "correct" if predicted_cluster == true_cluster else "error",
        })

    # =====================================================
    # SPOOF TEST
    # =====================================================

    print("\nТестирование spoof-изображений...")

    spoof_cluster_dirs = sorted(
        p for p in SPOOF_DIR.iterdir()
        if p.is_dir() and p.name.startswith("cluster_")
    )

    for spoof_cluster_dir in spoof_cluster_dirs:
        true_cluster = spoof_cluster_dir.name
        spoof_images = collect_images(spoof_cluster_dir)

        for image_path in spoof_images:
            total_spoof += 1

            embedding, bbox, elapsed = get_embedding_safe(embedder, image_path)
            processing_times.append(elapsed)

            if embedding is None:
                predicted_cluster = "blocked"
                best_score = 0.0
                blocked_spoof += 1
            else:
                predicted_cluster, best_score = recognize(embedding, references)

                similarity_scores_spoof.append(best_score)
                binary_true.append(0)
                binary_scores.append(best_score)

                all_embeddings.append(embedding)
                all_embedding_labels.append("spoof")

                if predicted_cluster == "unknown":
                    blocked_spoof += 1
                    predicted_cluster = "blocked"
                else:
                    false_accepts += 1

            report_rows.append({
                "path": str(image_path),
                "type": "spoof",
                "true_cluster": true_cluster,
                "predicted_cluster": predicted_cluster,
                "similarity": round(float(best_score), 6),
                "processing_time_sec": round(float(elapsed), 6),
                "result": "blocked" if predicted_cluster == "blocked" else "false_accept",
            })

    # =====================================================
    # МЕТРИКИ
    # =====================================================

    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    ) if y_true else 0.0

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    ) if y_true else 0.0

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    ) if y_true else 0.0

    far = false_accepts / total_spoof if total_spoof > 0 else 0.0
    frr = false_rejects / total_real if total_real > 0 else 0.0

    successful_pass_rate = correct_real / total_real if total_real > 0 else 0.0
    spoof_detection_rate = blocked_spoof / total_spoof if total_spoof > 0 else 0.0

    avg_time = float(np.mean(processing_times)) if processing_times else 0.0

    # =====================================================
    # СОХРАНЕНИЕ CSV
    # =====================================================

    with REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "path",
                "type",
                "true_cluster",
                "predicted_cluster",
                "similarity",
                "processing_time_sec",
                "result",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    if y_true:
        labels = sorted(list(set(y_true + y_pred)))

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig, ax = plt.subplots(figsize=(12, 10))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels,
        )

        disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix: Real Face Recognition")
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
        plt.close()

    # =====================================================
    # HISTOGRAM SIMILARITY
    # =====================================================

    plt.figure(figsize=(9, 5))

    if similarity_scores_real:
        plt.hist(
            similarity_scores_real,
            bins=30,
            alpha=0.6,
            label="real",
        )

    if similarity_scores_spoof:
        plt.hist(
            similarity_scores_spoof,
            bins=30,
            alpha=0.6,
            label="spoof",
        )

    plt.axvline(FACE_MATCH_THRESHOLD, linestyle="--", label="threshold")
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SIMILARITY_HIST_PATH, dpi=300)
    plt.close()

    # =====================================================
    # ROC CURVE
    # =====================================================

    if len(set(binary_true)) == 2:
        fpr, tpr, _ = roc_curve(binary_true, binary_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve: Real vs Spoof Acceptance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ROC_PATH, dpi=300)
        plt.close()
    else:
        roc_auc = 0.0

    # =====================================================
    # t-SNE
    # =====================================================

    if len(all_embeddings) >= 5:
        embeddings_array = np.vstack(all_embeddings)

        if len(embeddings_array) > MAX_TSNE_SAMPLES:
            indices = np.random.choice(
                len(embeddings_array),
                size=MAX_TSNE_SAMPLES,
                replace=False,
            )
            embeddings_array = embeddings_array[indices]
            labels_for_tsne = [all_embedding_labels[i] for i in indices]
        else:
            labels_for_tsne = all_embedding_labels

        perplexity = min(30, max(2, len(embeddings_array) // 3))

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="pca",
            learning_rate="auto",
        )

        tsne_result = tsne.fit_transform(embeddings_array)

        plt.figure(figsize=(9, 7))

        unique_labels = sorted(set(labels_for_tsne))

        for label in unique_labels:
            indices = [i for i, item in enumerate(labels_for_tsne) if item == label]
            points = tsne_result[indices]

            if label == "spoof":
                plt.scatter(points[:, 0], points[:, 1], s=12, marker="x", label=label)
            else:
                plt.scatter(points[:, 0], points[:, 1], s=12, label=label)

        plt.title("t-SNE Embedding Visualization")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

        if len(unique_labels) <= 15:
            plt.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(TSNE_PATH, dpi=300)
        plt.close()

    # =====================================================
    # SUMMARY
    # =====================================================

    summary = f"""
ИТОГОВАЯ ОЦЕНКА СИСТЕМЫ РАСПОЗНАВАНИЯ ЛИЦ

Датасет:
{DATASET_ROOT}

Количество real-кластеров: {len(references)}
Количество real-тестов: {total_real}
Количество spoof-тестов: {total_spoof}

Accuracy: {accuracy:.4f}
Precision weighted: {precision:.4f}
Recall weighted: {recall:.4f}
Weighted F1: {weighted_f1:.4f}

FAR: {far:.4f}
FRR: {frr:.4f}

Процент успешных проходов: {successful_pass_rate * 100:.2f}%
Процент заблокированных spoof-атак: {spoof_detection_rate * 100:.2f}%

Среднее время обработки одного изображения: {avg_time:.4f} сек
ROC AUC: {roc_auc:.4f}

False Accepts: {false_accepts}
False Rejects: {false_rejects}

Порог cosine similarity: {FACE_MATCH_THRESHOLD}
"""

    print("\n===================================")
    print(summary)
    print("===================================")

    SUMMARY_TXT.write_text(summary, encoding="utf-8")

    print("\nФайлы сохранены:")
    print(f"- {REPORT_CSV}")
    print(f"- {SUMMARY_TXT}")
    print(f"- {CONFUSION_MATRIX_PATH}")
    print(f"- {SIMILARITY_HIST_PATH}")
    print(f"- {ROC_PATH}")
    print(f"- {TSNE_PATH}")


if __name__ == "__main__":
    evaluate()