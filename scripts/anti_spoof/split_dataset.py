import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

SOURCE_DIR = Path("data/anti_spoof_extracted")
TARGET_DIR = Path("data/anti_spoof_frames")

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_files(files, target_dir: Path) -> None:
    ensure_dir(target_dir)
    for file_path in files:
        shutil.copy2(file_path, target_dir / file_path.name)


def split_class(class_name: str) -> None:
    files = list((SOURCE_DIR / class_name).glob("*.jpg"))
    if not files:
        print(f"Нет файлов для класса: {class_name}")
        return

    train_files, temp_files = train_test_split(files, test_size=(1 - TRAIN_SIZE), random_state=42)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        random_state=42
    )

    copy_files(train_files, TARGET_DIR / "train" / class_name)
    copy_files(val_files, TARGET_DIR / "val" / class_name)
    copy_files(test_files, TARGET_DIR / "test" / class_name)

    print(f"{class_name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")


def main() -> None:
    for split in ["train", "val", "test"]:
        for class_name in ["real", "spoof"]:
            ensure_dir(TARGET_DIR / split / class_name)

    split_class("real")
    split_class("spoof")
    print("Разбиение завершено.")


if __name__ == "__main__":
    main()