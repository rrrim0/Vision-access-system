import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


INPUT_FILE = Path("Вставленный текст.txt")
OUTPUT_FILE = Path("model_metrics_comparison.png")


def parse_accuracy(text: str) -> Optional[float]:
    match = re.search(r"Accuracy:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def parse_avg_row(text: str, row_name: str) -> Optional[dict[str, float]]:
    pattern = rf"{re.escape(row_name)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+"
    match = re.search(pattern, text)
    if not match:
        return None

    return {
        "precision": float(match.group(1)),
        "recall": float(match.group(2)),
        "f1": float(match.group(3)),
    }


def build_plot(
    accuracy: Optional[float],
    macro: Optional[dict[str, float]],
    weighted: Optional[dict[str, float]],
    output_path: Path,
) -> None:
    metric_names = [
        "Accuracy",
        "Macro\nPrecision",
        "Macro\nRecall",
        "Macro\nF1",
        "Weighted\nPrecision",
        "Weighted\nRecall",
        "Weighted\nF1",
    ]

    metric_values = [
        accuracy if accuracy is not None else 0.0,
        macro["precision"] if macro is not None else 0.0,
        macro["recall"] if macro is not None else 0.0,
        macro["f1"] if macro is not None else 0.0,
        weighted["precision"] if weighted is not None else 0.0,
        weighted["recall"] if weighted is not None else 0.0,
        weighted["f1"] if weighted is not None else 0.0,
    ]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(metric_names, metric_values)

    plt.ylim(0, 1.05)
    plt.ylabel("Значение метрики", fontsize=12)
    plt.title(
        "Сравнение агрегированных метрик модели распознавания лиц",
        fontsize=14,
        pad=16,
    )

    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    if accuracy is not None and macro is not None:
        diff = accuracy - macro["f1"]
        plt.text(
            0.02,
            0.96,
            (
                f"Разница между Accuracy и Macro F1: {diff:.3f}\n"
                "Большой разрыв указывает на неравномерное качество\n"
                "и слабую работу модели на малочисленных классах."
            ),
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
        )

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Файл не найден: {INPUT_FILE.resolve()}\n"
            "Положи 'Вставленный текст.txt' рядом со скриптом."
        )

    text = INPUT_FILE.read_text(encoding="utf-8", errors="ignore")

    accuracy = parse_accuracy(text)
    macro = parse_avg_row(text, "macro avg")
    weighted = parse_avg_row(text, "weighted avg")

    if accuracy is None and macro is None and weighted is None:
        raise ValueError(
            "Не удалось найти Accuracy / macro avg / weighted avg в файле."
        )

    build_plot(accuracy, macro, weighted, OUTPUT_FILE)
    print(f"Готово. PNG сохранён: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()