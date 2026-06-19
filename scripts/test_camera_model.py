"""
scripts/test_camera_model.py

Teste em tempo real: captura com Picamera2 e executa inferência TFLite.
Exibe predição, confiança e FPS na tela. Pressione Q para sair.
"""

import time
import numpy as np
import cv2
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
from pathlib import Path

MODEL_PATH  = Path("models/model.tflite")
LABELS_PATH = Path("models/labels.txt")
IMAGE_SIZE  = (224, 224)


def load_labels(path: Path) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_interpreter(model_path: Path) -> Interpreter:
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def predict(interpreter: Interpreter, image: np.ndarray) -> tuple[str, float]:
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = image.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]["index"])[0]
    top    = int(np.argmax(scores))
    return top, float(scores[top])


def main():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print("Modelo ou labels não encontrado. Execute export_tflite.py primeiro.")
        return

    labels      = load_labels(LABELS_PATH)
    interpreter = load_interpreter(MODEL_PATH)

    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": IMAGE_SIZE, "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    time.sleep(1)

    print("Pressione Q para sair.")

    prev_time = time.time()

    while True:
        frame = camera.capture_array()
        top_index, confidence = predict(interpreter, frame)
        label = labels[top_index]

        # Calcula FPS
        now  = time.time()
        fps  = 1.0 / (now - prev_time)
        prev_time = now

        # Desenha resultado
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(display, f"Prediction:  {label}",          (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Confidence:  {confidence*100:.1f}%", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"FPS:         {fps:.1f}",         (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Master Trash - Teste em Tempo Real", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
