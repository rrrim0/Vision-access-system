import os
import cv2
from pathlib import Path
from tqdm import tqdm

RAW_DATASET_DIR = Path("datasets/anti_spoof_raw")
OUTPUT_DIR = Path("data/anti_spoof_extracted")

REAL_CLASS = "real"
SPOOF_CLASSES = {"mask", "monitor", "outline", "print", "print_cut", "silicone"}

FRAME_STEP = 10  # брать каждый 10-й кадр
MAX_FRAMES_PER_VIDEO = 50  # ограничение, чтобы не было перекоса


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_frames_from_video(video_path: Path, output_dir: Path, frame_step: int, max_frames: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return 0

    saved = 0
    frame_index = 0
    video_name = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_step == 0:
            out_name = f"{video_name}_frame_{frame_index}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1

            if saved >= max_frames:
                break

        frame_index += 1

    cap.release()
    return saved


def main() -> None:
    ensure_dir(OUTPUT_DIR / "real")
    ensure_dir(OUTPUT_DIR / "spoof")

    total_saved = 0

    for class_dir in RAW_DATASET_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower()

        if class_name == REAL_CLASS:
            target_dir = OUTPUT_DIR / "real"
        elif class_name in SPOOF_CLASSES:
            target_dir = OUTPUT_DIR / "spoof"
        else:
            print(f"Пропускаю неизвестную папку: {class_name}")
            continue

        video_files = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mov"))

        for video_path in tqdm(video_files, desc=f"Обработка {class_name}"):
            saved = extract_frames_from_video(
                video_path=video_path,
                output_dir=target_dir,
                frame_step=FRAME_STEP,
                max_frames=MAX_FRAMES_PER_VIDEO
            )
            total_saved += saved

    print(f"Готово. Сохранено кадров: {total_saved}")


if __name__ == "__main__":
    main()