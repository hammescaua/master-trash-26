"""
scripts/train_model.py

Treina um classificador de resíduos usando Transfer Learning com MobileNetV2.
Salva o modelo em models/saved_model/ e gera gráficos de treinamento.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# --- Configurações ---
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
CLASSES     = ["metal", "organic", "paper", "plastic"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR    = PROJECT_ROOT / "dataset" / "train"
VAL_DIR      = PROJECT_ROOT / "dataset" / "val"
SAVE_DIR     = PROJECT_ROOT / "models" / "saved_model"
PLOTS_DIR    = PROJECT_ROOT / "plots"


def build_data_augmentation():
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.15),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomContrast(0.15),
    ], name="augmentation")


def build_model(num_classes: int) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Congela pesos base no início

    inputs     = keras.Input(shape=(*IMAGE_SIZE, 3))
    augmented  = build_data_augmentation()(inputs, training=True)
    x          = keras.applications.mobilenet_v2.preprocess_input(augmented)
    x          = base(x, training=False)
    x          = keras.layers.GlobalAveragePooling2D()(x)
    x          = keras.layers.Dropout(0.3)(x)
    outputs    = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def load_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASSES,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASSES,
    )
    # Prefetch melhora performance
    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)


def save_plots(history):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"],     label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_curve.png")
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"],     label="Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curve.png")
    plt.close()

    print(f"Gráficos salvos em {PLOTS_DIR}/")


def main():
    print("=== Treinamento Master Trash 4.0 ===")
    print(f"GPU disponível: {len(tf.config.list_physical_devices('GPU')) > 0}")

    train_ds, val_ds = load_datasets()
    model = build_model(num_classes=len(CLASSES))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(SAVE_DIR))
    print(f"Modelo salvo em {SAVE_DIR}/")

    save_plots(history)


if __name__ == "__main__":
    main()
