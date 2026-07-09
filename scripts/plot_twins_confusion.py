import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# твои реальные результаты
y_true = []
y_pred = []

# twinA: 148 изображений
y_true += ["twinA"] * 148
y_pred += ["twinA"] * 137 + ["twinB"] * 10 + ["other"] * 1  # пример

# twinB: 83 изображений
y_true += ["twinB"] * 83
y_pred += ["twinB"] * 63 + ["other"] * 20  # пример

labels = ["twinA", "twinB"]

cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Предсказанный класс")
plt.ylabel("Реальный класс")
plt.title("Матрица ошибок распознавания близнецов")

plt.tight_layout()

plt.savefig("twins_confusion_matrix.png", dpi=300)

plt.show()