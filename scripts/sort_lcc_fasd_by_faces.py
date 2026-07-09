from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# =====================================================
# ДОБАВЛЯЕМ КОРЕНЬ ПРОЕКТА В PYTHON PATH
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.face_embedder import FaceEmbedder  # noqa: E402


# =====================================================
# НАСТРОЙКИ
# =====================================================

INPUT_ROOT = PROJECT_ROOT / "data" / "lcc_fasd_raw" / "LCC_FASD_evaluation"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "evaluation_dataset_sorted"

REPORT_PATH = OUTPUT_ROOT / "sorting_report.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# DBSCAN работает по cosine distance:
# меньше eps = строже разделяет лица
# больше eps = сильнее объединяет похожие лица
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 2

COPY_IMAGES = True


# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================

def collect_images(folder: Path) -> list[Path]:
    return [
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]


def safe_read_image(path: Path):
    image = cv2.imread(str(path))
    return image


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)

    if norm > 0:
        embedding = embedding / norm

    return embedding


def make_unique_output_path(out_dir: Path, image_path: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = out_dir / image_path.name

    if not candidate.exists():
        return candidate

    stem = image_path.stem
    suffix = image_path.suffix

    counter = 1
    while True:
        candidate = out_dir / f"{stem}_{counter:04d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def cluster_folder(split_name: str, embedder: FaceEmbedder) -> dict:
    input_dir = INPUT_ROOT / split_name
    output_dir = OUTPUT_ROOT / split_name
    unknown_dir = output_dir / "unknown"

    image_paths = collect_images(input_dir)

    records = []
    embeddings = []
    valid_image_paths = []

    unknown_count = 0

    print(f"\nОбработка папки: {split_name}")
    print(f"Найдено изображений: {len(image_paths)}")

    for image_path in image_paths:
        image = safe_read_image(image_path)

        if image is None:
            out_path = make_unique_output_path(unknown_dir, image_path)

            if COPY_IMAGES:
                shutil.copy2(image_path, out_path)

            records.append({
                "original_path": str(image_path),
                "output_path": str(out_path),
                "split": split_name,
                "cluster_id": "unknown",
                "face_detected": False,
                "similarity_or_distance": "",
            })

            unknown_count += 1
            continue

        try:
            embedding, bbox = embedder.get_embedding(image)
        except Exception as error:
            print(f"[ERROR] {image_path}: {error}")
            embedding = None
            bbox = None

        if embedding is None or bbox is None:
            out_path = make_unique_output_path(unknown_dir, image_path)

            if COPY_IMAGES:
                shutil.copy2(image_path, out_path)

            records.append({
                "original_path": str(image_path),
                "output_path": str(out_path),
                "split": split_name,
                "cluster_id": "unknown",
                "face_detected": False,
                "similarity_or_distance": "",
            })

            unknown_count += 1
            continue

        embedding = normalize_embedding(embedding)

        embeddings.append(embedding)
        valid_image_paths.append(image_path)

    if not embeddings:
        return {
            "split": split_name,
            "total_images": len(image_paths),
            "clusters_count": 0,
            "unknown_count": unknown_count,
            "avg_cluster_size": 0,
            "records": records,
        }

    embeddings_array = np.vstack(embeddings)

    clustering = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="cosine"
    )

    labels = clustering.fit_predict(embeddings_array)

    unique_clusters = sorted(label for label in set(labels) if label != -1)

    cluster_name_map = {
        label: f"cluster_{index + 1:03d}"
        for index, label in enumerate(unique_clusters)
    }

    cluster_sizes = {}

    for idx, label in enumerate(labels):
        image_path = valid_image_paths[idx]
        embedding = embeddings_array[idx]

        if label == -1:
            cluster_id = "unknown"
            target_dir = unknown_dir
            distance_value = ""
            unknown_count += 1
        else:
            cluster_id = cluster_name_map[label]
            target_dir = output_dir / cluster_id

            cluster_indices = np.where(labels == label)[0]
            cluster_embeddings = embeddings_array[cluster_indices]

            centroid = cluster_embeddings.mean(axis=0)
            centroid = normalize_embedding(centroid)

            cosine_similarity = float(np.dot(embedding, centroid))
            cosine_distance = 1.0 - cosine_similarity
            distance_value = round(cosine_distance, 6)

            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

        out_path = make_unique_output_path(target_dir, image_path)

        if COPY_IMAGES:
            shutil.copy2(image_path, out_path)

        records.append({
            "original_path": str(image_path),
            "output_path": str(out_path),
            "split": split_name,
            "cluster_id": cluster_id,
            "face_detected": True,
            "similarity_or_distance": distance_value,
        })

    clusters_count = len(cluster_sizes)

    avg_cluster_size = (
        sum(cluster_sizes.values()) / clusters_count
        if clusters_count > 0 else 0
    )

    return {
        "split": split_name,
        "total_images": len(image_paths),
        "clusters_count": clusters_count,
        "unknown_count": unknown_count,
        "avg_cluster_size": avg_cluster_size,
        "records": records,
    }


def save_report(all_records: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "original_path",
        "output_path",
        "split",
        "cluster_id",
        "face_detected",
        "similarity_or_distance",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)


def main():
    print("===================================")
    print("СОРТИРОВКА LCC_FASD ПО ЛИЦАМ")
    print("===================================")

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Не найдена папка датасета: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    embedder = FaceEmbedder()

    all_records = []
    summaries = []

    for split_name in ["real", "spoof"]:
        split_dir = INPUT_ROOT / split_name

        if not split_dir.exists():
            print(f"[SKIP] Папка не найдена: {split_dir}")
            continue

        summary = cluster_folder(split_name, embedder)

        summaries.append(summary)
        all_records.extend(summary["records"])

    save_report(all_records)

    print("\n===================================")
    print("ИТОГ")
    print("===================================")

    total_images = sum(item["total_images"] for item in summaries)
    total_clusters = sum(item["clusters_count"] for item in summaries)
    total_unknown = sum(item["unknown_count"] for item in summaries)

    for item in summaries:
        print(f"\nПапка: {item['split']}")
        print(f"Изображений обработано: {item['total_images']}")
        print(f"Кластеров найдено: {item['clusters_count']}")
        print(f"Попало в unknown: {item['unknown_count']}")
        print(f"Средний размер кластера: {item['avg_cluster_size']:.2f}")

    print("\nОбщее количество изображений:", total_images)
    print("Общее количество кластеров:", total_clusters)
    print("Общее количество unknown:", total_unknown)
    print(f"\nCSV-отчет сохранен: {REPORT_PATH}")
    print(f"Отсортированный датасет: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()