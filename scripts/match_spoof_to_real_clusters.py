from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.face_embedder import FaceEmbedder  # noqa: E402


# =====================================================
# НАСТРОЙКИ
# =====================================================

REAL_CLUSTERS_DIR = PROJECT_ROOT / "data" / "evaluation_dataset_sorted" / "real"
SPOOF_SOURCE_DIR = PROJECT_ROOT / "data" / "lcc_fasd_raw" / "LCC_FASD_evaluation" / "spoof"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "evaluation_dataset_matched"
OUTPUT_REAL_DIR = OUTPUT_ROOT / "real"
OUTPUT_SPOOF_DIR = OUTPUT_ROOT / "spoof"

REPORT_PATH = OUTPUT_ROOT / "spoof_matching_report.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Чем выше threshold, тем строже совпадение.
# Если много spoof уходит в unmatched — снизь до 0.45.
# Если разные люди смешиваются — подними до 0.60.
MATCH_THRESHOLD = 0.50


# =====================================================
# ФУНКЦИИ
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


def make_unique_path(folder: Path, source_path: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)

    candidate = folder / source_path.name

    if not candidate.exists():
        return candidate

    stem = source_path.stem
    suffix = source_path.suffix

    counter = 1
    while True:
        candidate = folder / f"{stem}_{counter:04d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def build_real_cluster_centroids(embedder: FaceEmbedder) -> dict[str, np.ndarray]:
    centroids = {}

    cluster_dirs = sorted(
        p for p in REAL_CLUSTERS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("cluster_")
    )

    for cluster_dir in cluster_dirs:
        embeddings = []

        for image_path in collect_images(cluster_dir):
            image = read_image(image_path)

            if image is None:
                continue

            try:
                embedding, bbox = embedder.get_embedding(image)
            except Exception:
                embedding, bbox = None, None

            if embedding is None or bbox is None:
                continue

            embeddings.append(normalize_embedding(embedding))

        if embeddings:
            centroid = np.mean(np.vstack(embeddings), axis=0)
            centroids[cluster_dir.name] = normalize_embedding(centroid)

    return centroids


def copy_real_clusters():
    if OUTPUT_REAL_DIR.exists():
        shutil.rmtree(OUTPUT_REAL_DIR)

    shutil.copytree(REAL_CLUSTERS_DIR, OUTPUT_REAL_DIR)


def match_spoof_images(embedder: FaceEmbedder, centroids: dict[str, np.ndarray]):
    records = []

    spoof_images = collect_images(SPOOF_SOURCE_DIR)

    matched_count = 0
    unmatched_count = 0
    no_face_count = 0

    unmatched_dir = OUTPUT_SPOOF_DIR / "unmatched"
    no_face_dir = OUTPUT_SPOOF_DIR / "unknown"

    for image_path in spoof_images:
        image = read_image(image_path)

        if image is None:
            target_path = make_unique_path(no_face_dir, image_path)
            shutil.copy2(image_path, target_path)

            records.append({
                "original_path": str(image_path),
                "output_path": str(target_path),
                "assigned_cluster": "unknown",
                "best_similarity": "",
                "matched": False,
                "reason": "image_read_error",
            })

            no_face_count += 1
            continue

        try:
            embedding, bbox = embedder.get_embedding(image)
        except Exception as error:
            print(f"[ERROR] {image_path}: {error}")
            embedding, bbox = None, None

        if embedding is None or bbox is None:
            target_path = make_unique_path(no_face_dir, image_path)
            shutil.copy2(image_path, target_path)

            records.append({
                "original_path": str(image_path),
                "output_path": str(target_path),
                "assigned_cluster": "unknown",
                "best_similarity": "",
                "matched": False,
                "reason": "face_not_detected",
            })

            no_face_count += 1
            continue

        embedding = normalize_embedding(embedding)

        best_cluster = None
        best_similarity = -1.0

        for cluster_name, centroid in centroids.items():
            similarity = cosine_similarity(embedding, centroid)

            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_name

        if best_cluster is not None and best_similarity >= MATCH_THRESHOLD:
            target_dir = OUTPUT_SPOOF_DIR / best_cluster
            target_path = make_unique_path(target_dir, image_path)
            shutil.copy2(image_path, target_path)

            matched = True
            reason = "matched"
            matched_count += 1
        else:
            target_path = make_unique_path(unmatched_dir, image_path)
            shutil.copy2(image_path, target_path)

            matched = False
            reason = "below_threshold"
            unmatched_count += 1

        records.append({
            "original_path": str(image_path),
            "output_path": str(target_path),
            "assigned_cluster": best_cluster if matched else "unmatched",
            "best_similarity": round(best_similarity, 6),
            "matched": matched,
            "reason": reason,
        })

    return records, matched_count, unmatched_count, no_face_count


def save_report(records: list[dict]):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "original_path",
            "output_path",
            "assigned_cluster",
            "best_similarity",
            "matched",
            "reason",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    print("===================================")
    print("ПРИВЯЗКА SPOOF К REAL-КЛАСТЕРАМ")
    print("===================================")

    if not REAL_CLUSTERS_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка real-кластеров: {REAL_CLUSTERS_DIR}")

    if not SPOOF_SOURCE_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка spoof: {SPOOF_SOURCE_DIR}")

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    embedder = FaceEmbedder()

    print("\nКопирование real-кластеров...")
    copy_real_clusters()

    print("\nФормирование центров real-кластеров...")
    centroids = build_real_cluster_centroids(embedder)

    print(f"Найдено real-кластеров: {len(centroids)}")

    print("\nСопоставление spoof-изображений с real-кластерами...")
    records, matched_count, unmatched_count, no_face_count = match_spoof_images(
        embedder,
        centroids,
    )

    save_report(records)

    spoof_cluster_dirs = [
        p for p in OUTPUT_SPOOF_DIR.iterdir()
        if p.is_dir() and p.name.startswith("cluster_")
    ] if OUTPUT_SPOOF_DIR.exists() else []

    print("\n===================================")
    print("ИТОГ")
    print("===================================")
    print(f"Всего spoof-изображений обработано: {len(records)}")
    print(f"Успешно привязано к real-кластерам: {matched_count}")
    print(f"Не подошли по threshold: {unmatched_count}")
    print(f"Не найдено лицо / ошибка чтения: {no_face_count}")
    print(f"Создано spoof-кластеров с совпадениями: {len(spoof_cluster_dirs)}")
    print(f"\nНовый датасет сохранен в: {OUTPUT_ROOT}")
    print(f"CSV-отчет сохранен в: {REPORT_PATH}")


if __name__ == "__main__":
    main()