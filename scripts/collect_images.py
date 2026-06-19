"""
scripts/collect_images.py

Coleta imagens para o dataset usando a Arducam via OpenCV.
Pressione a tecla correspondente à classe para salvar a imagem.

Teclas:
  P -> paper
  L -> plastic
  M -> metal
  O -> organic
  Q -> sair
"""

import cv2
import os
from pathlib import Path

# Mapeamento tecla -> pasta de destino
KEY_MAP = {
    ord("p"): "paper",
    ord("l"): "plastic",
    ord("m"): "metal",
    ord("o"): "organic",
}

BASE_DIR = Path("dataset/raw")


def count_existing(class_name: str) -> int:
    folder = BASE_DIR / class_name
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.jpg")))


def save_image(frame, class_name: str) -> str:
    folder = BASE_DIR / class_name
    folder.mkdir(parents=True, exist_ok=True)

    count = count_existing(class_name) + 1
    filename = folder / f"{class_name}_{count:03d}.jpg"
    cv2.imwrite(str(filename), frame)
    return str(filename)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    print("=== Coleta de Imagens ===")
    print("P=paper | L=plastic | M=metal | O=organic | Q=sair")

    selected_class = None
    saved_message  = ""
    message_timer  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar frame.")
            break

        display = frame.copy()

        # Contagem por classe
        counts = {name: count_existing(name) for name in ["paper", "plastic", "metal", "organic"]}

        # HUD
        y = 25
        for name, count in counts.items():
            cv2.putText(display, f"{name}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 22

        if selected_class:
            cv2.putText(display, f"Classe: {selected_class.upper()}", (10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if saved_message and message_timer > 0:
            cv2.putText(display, saved_message, (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            message_timer -= 1

        cv2.imshow("Master Trash - Coleta de Imagens", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key in KEY_MAP:
            class_name = KEY_MAP[key]
            selected_class = class_name
            path = save_image(frame, class_name)
            saved_message = f"Salvo: {os.path.basename(path)}"
            message_timer = 60  # Exibe por ~60 frames
            print(f"[OK] {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Coleta encerrada.")


if __name__ == "__main__":
    main()
