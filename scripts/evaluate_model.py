"""
scripts/evaluate_model.py

Avalia o modelo salvo no conjunto de teste.
Gera relatório de classificação e matriz de confusão.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

CLASSES   = ["metal", "organic", "paper", "plastic"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR   = Path("dataset/test")
MODEL_DIR  = Path("models/saved_model")
PLOTS_DIR  = Path("plots")


def main():
    print("=== Avaliação do Modelo ===")

    model = keras.models.load_model(str(MODEL_DIR))

    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASSES,
        shuffle=False,
    )

    # Coleta predições e labels reais
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()
    print(f"\nMatriz de confusão salva em {PLOTS_DIR}/confusion_matrix.png")

    # Destaca classes com mais erros
    errors = cm.sum(axis=1) - cm.diagonal()
    worst  = CLASSES[np.argmax(errors)]
    print(f"\nClasse com mais erros: {worst} ({errors.max()} erros)")


if __name__ == "__main__":
    main()
