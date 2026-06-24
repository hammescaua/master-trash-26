"""
scripts/export_tflite.py

Converte o modelo Keras salvo para TensorFlow Lite.
Gera model.tflite e labels.txt prontos para uso no Raspberry Pi.
"""

import tensorflow as tf
from pathlib import Path

CLASSES    = ["metal", "organic", "paper", "plastic"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR    = PROJECT_ROOT / "models" / "saved_model"
OUTPUT_DIR   = PROJECT_ROOT / "models"


def export_tflite():
    print("Carregando modelo...")
    model = tf.saved_model.load(str(MODEL_DIR))

    print("Convertendo para TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(MODEL_DIR))

    # Otimização padrão: reduz tamanho sem perda significativa de precisão
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    tflite_path = OUTPUT_DIR / "model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Modelo exportado: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")

    labels_path = OUTPUT_DIR / "labels.txt"
    with open(labels_path, "w") as f:
        for cls in CLASSES:
            f.write(cls + "\n")
    print(f"Labels exportados: {labels_path}")


def main():
    print("=== Exportação TFLite ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_tflite()
    print("Exportação concluída.")


if __name__ == "__main__":
    main()
