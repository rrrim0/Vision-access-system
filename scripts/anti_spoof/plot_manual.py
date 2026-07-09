import matplotlib.pyplot as plt

# Вставляем твои данные
epochs = [1, 2, 3, 4]

train_acc = [0.9560, 0.9863, 0.9942, 0.9925]
val_acc = [0.9050, 0.9478, 0.9159, 0.9532]

train_loss = [0.1024, 0.0375, 0.0177, 0.0222]
val_loss = [0.2877, 0.1585, 0.2510, 0.1679]

# ===== Accuracy =====
plt.figure()
plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.grid()

# ===== Loss =====
plt.figure()
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.grid()

plt.show()