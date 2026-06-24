"""
scripts/test_camera_model.py

Teste em tempo real: captura com Picamera2 e executa inferência TFLite.
Exibe predição, confiança e FPS na tela. Pressione Q para sair.

Execute a partir de qualquer diretório:
    python scripts/test_camera_model.py
"""

import time
import numpy as np
import cv2
from pathlib import Path
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# Caminhos absolutos relativos ao projeto (funciona de qualquer diretório)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "model.tflite"
LABELS_PATH  = PROJECT_ROOT / "models" / "labels.txt"

PREVIEW_SIZE = (640, 480)   # Tamanho da janela de exibição
MODEL_SIZE   = (224, 224)   # Tamanho de entrada do modelo

# Cores HUD (BGR)
COLOR_OK   = (0, 255, 0)
COLOR_WARN = (0, 100, 255)
COLOR_INFO = (255, 255, 255)

# Limiar de confiança para destaque visual
CONFIDENCE_THRESHOLD = 0.70


def load_labels(path: Path) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_interpreter(model_path: Path) -> Interpreter:
    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    return interp


def predict(interpreter: Interpreter, image_rgb: np.ndarray):
    """
    Recebe imagem RGB já no tamanho do modelo (MODEL_SIZE).
    Retorna (índice_top, confiança).
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = image_rgb.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]["index"])[0]
    top    = int(np.argmax(scores))
    return top, float(scores[top])


def draw_hud(display, label, confidence, fps):
    h = display.shape[0]
    color = COLOR_OK if confidence >= CONFIDENCE_THRESHOLD else COLOR_WARN

    cv2.putText(display, f"Prediction:  {label.upper()}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(display, f"Confidence:  {confidence * 100:.1f}%",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(display, f"FPS:         {fps:.1f}",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_INFO, 2)

    if confidence < CONFIDENCE_THRESHOLD:
        cv2.putText(display, "BAIXA CONFIANCA",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WARN, 2)


def main():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print(f"Modelo ou labels nao encontrado.")
        print(f"  Esperado: {MODEL_PATH}")
        print(f"  Execute scripts/export_tflite.py primeiro.")
        return

    labels      = load_labels(LABELS_PATH)
    interpreter = load_interpreter(MODEL_PATH)
    print(f"Modelo carregado. Classes: {labels}")

    # Câmera em resolução de preview; captura já no tamanho do modelo
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": PREVIEW_SIZE, "format": "RGB888"},
        lores={"size": MODEL_SIZE,  "format": "RGB888"},
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    print("Pressione Q para sair.\n")

    prev_time = time.time()

    try:
        while True:
            # Stream principal para exibição, lores para inferência
            frame_display = picam2.capture_array("main")
            frame_model   = picam2.capture_array("lores")

            top_index, confidence = predict(interpreter, frame_model)
            label = labels[top_index]

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # RGB -> BGR para OpenCV exibir cores corretas
            display = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
            draw_hud(display, label, confidence, fps)

            cv2.imshow("Master Trash - Teste em Tempo Real", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("Encerrado.")


if __name__ == "__main__":
    main()
