"""
scripts/test_camera_model.py

Teste em tempo real: captura com Picamera2 e executa inferência TFLite.
Exibe predição, confiança e FPS. Pressione Q para sair.

Execute de qualquer diretório:
    python scripts/test_camera_model.py
"""

import time
import numpy as np
import cv2
from pathlib import Path
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "model.tflite"
LABELS_PATH  = PROJECT_ROOT / "models" / "labels.txt"

PREVIEW_SIZE = (640, 480)
MODEL_SIZE   = (224, 224)

COLOR_OK   = (0, 255, 0)
COLOR_WARN = (0, 100, 255)
COLOR_INFO = (255, 255, 255)

CONFIDENCE_THRESHOLD = 0.70


def load_labels(path: Path) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_interpreter(model_path: Path) -> Interpreter:
    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    return interp


def predict(interpreter: Interpreter, image_rgb: np.ndarray):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = image_rgb.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]["index"])[0]
    top    = int(np.argmax(scores))
    return top, float(scores[top])


def init_camera() -> Picamera2:
    picam2 = Picamera2()

    sensor_w, sensor_h = picam2.camera_properties["PixelArraySize"]
    print(f"[CAM] Sensor: {sensor_w}x{sensor_h}")

    config = picam2.create_preview_configuration(
        main={"size": PREVIEW_SIZE, "format": "BGR888"},   # exibição
        lores={"size": MODEL_SIZE,  "format": "RGB888"},   # inferência
        controls={"FrameRate": 30},
    )
    picam2.configure(config)
    picam2.start()

    time.sleep(1)
    picam2.set_controls({"ScalerCrop": [0, 0, sensor_w, sensor_h]})

    print("Aguardando AWB estabilizar...")
    time.sleep(2)
    print("Câmera pronta.\n")

    return picam2


def draw_hud(display, label, confidence, fps):
    h     = display.shape[0]
    color = COLOR_OK if confidence >= CONFIDENCE_THRESHOLD else COLOR_WARN

    cv2.putText(display, f"Prediction:  {label.upper()}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(display, f"Confidence:  {confidence * 100:.1f}%",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(display, f"FPS:         {fps:.1f}",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_INFO, 2)

    if confidence < CONFIDENCE_THRESHOLD:
        cv2.putText(display, "BAIXA CONFIANCA — reposicione o objeto",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WARN, 2)


def main():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print(f"Modelo ou labels nao encontrado.")
        print(f"  Esperado: {MODEL_PATH}")
        print(f"  Execute scripts/export_tflite.py primeiro.")
        return

    labels      = load_labels(LABELS_PATH)
    interpreter = load_interpreter(MODEL_PATH)
    print(f"Modelo carregado. Classes: {labels}")

    picam2    = init_camera()
    prev_time = time.time()

    print("Pressione Q para sair.")

    try:
        while True:
            # main=BGR888 para exibição, lores=RGB888 para inferência
            frame_bgr = picam2.capture_array("main")
            frame_rgb = picam2.capture_array("lores")

            top_index, confidence = predict(interpreter, frame_rgb)
            label = labels[top_index]

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(frame_bgr, label, confidence, fps)
            cv2.imshow("Master Trash - Teste em Tempo Real", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("Encerrado.")


if __name__ == "__main__":
    main()
