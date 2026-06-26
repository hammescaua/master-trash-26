"""
scripts/collect_images.py

Coleta imagens para o dataset da Master Trash usando
Raspberry Pi 5 + Arducam IMX519 16MP (Picamera2).

Teclas:
    1 ou P -> seleciona paper
    2 ou L -> seleciona plastic
    3 ou M -> seleciona metal
    4 ou O -> seleciona organic
    SPACE  -> salva imagem da classe selecionada
    Q      -> sair
"""

import cv2
import os
import time
from pathlib import Path
from picamera2 import Picamera2

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR     = PROJECT_ROOT / "dataset" / "raw"

CLASSES = ["paper", "plastic", "metal", "organic"]

SELECT_MAP = {
    ord("1"): "paper",
    ord("p"): "paper",
    ord("2"): "plastic",
    ord("l"): "plastic",
    ord("3"): "metal",
    ord("m"): "metal",
    ord("4"): "organic",
    ord("o"): "organic",
}

# Resolução binned do sensor (mesmo modo que rpicam-hello usa)
# Usa sensor completo e reduz 2x — sem crop, sem zoom, cores corretas
SENSOR_SIZE  = (2328, 1748)
DISPLAY_SIZE = (1280, 720)   # tamanho da janela OpenCV (resize em software)

COLOR_TITLE    = (0, 255, 255)
COLOR_WHITE    = (255, 255, 255)
COLOR_SELECTED = (0, 255, 0)
COLOR_SAVED    = (0, 200, 255)
COLOR_WARN     = (0, 100, 255)


# =============================================================================
# CÂMERA
# =============================================================================

def init_camera() -> Picamera2:
    """
    Captura em XBGR8888 no modo binned 2328x1748 (igual ao rpicam-hello).
    Isso evita crop do sensor (zoom) e entrega cores corretas.
    """
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": SENSOR_SIZE, "format": "XBGR8888"},
        controls={"FrameRate": 15},
    )
    picam2.configure(config)
    picam2.start()

    print("Aguardando AWB estabilizar...")
    time.sleep(3)
    print("Câmera pronta.\n")

    return picam2


def capture_frame(picam2: Picamera2):
    """
    Retorna frame BGR pronto para o OpenCV.
    XBGR8888 tem 4 canais: X, B, G, R → descartamos o X com [:, :, :3].
    Resultado: BGR sem nenhum cvtColor.
    """
    frame = picam2.capture_array()   # shape (1748, 2328, 4) — XBGR8888
    bgr   = frame[:, :, :3]          # descarta canal X → BGR (1748, 2328, 3)
    return bgr


# =============================================================================
# DATASET
# =============================================================================

def count_existing(class_name: str) -> int:
    folder = BASE_DIR / class_name
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.jpg")))


def save_image(frame_bgr, class_name: str) -> str:
    folder = BASE_DIR / class_name
    folder.mkdir(parents=True, exist_ok=True)
    count    = count_existing(class_name) + 1
    filename = folder / f"{class_name}_{count:04d}.jpg"
    cv2.imwrite(str(filename), frame_bgr)  # salva em resolução total (2328x1748)
    return str(filename)


# =============================================================================
# HUD
# =============================================================================

def draw_hud(display, selected_class, counts, saved_message, message_timer):
    h = display.shape[0]

    cv2.putText(display, "MASTER TRASH - DATASET COLLECTOR",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_TITLE, 2)

    y = 65
    for name in CLASSES:
        marker = ">>>" if name == selected_class else "   "
        color  = COLOR_SELECTED if name == selected_class else COLOR_WHITE
        cv2.putText(display, f"{marker} {name.upper()}: {counts[name]}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 28

    if selected_class:
        cv2.putText(display,
                    f"[SPACE] salvar como {selected_class.upper()}",
                    (10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_SELECTED, 2)
    else:
        cv2.putText(display,
                    "Selecione: 1=paper  2=plastic  3=metal  4=organic",
                    (10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WARN, 2)

    if saved_message and message_timer > 0:
        cv2.putText(display, saved_message,
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SAVED, 2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n=== MASTER TRASH - COLETA DE IMAGENS ===")
    print("1/P=PAPER  2/L=PLASTIC  3/M=METAL  4/O=ORGANIC")
    print("SPACE=SALVAR  Q=SAIR\n")

    picam2 = init_camera()

    selected_class = None
    saved_message  = ""
    message_timer  = 0

    try:
        while True:
            frame_bgr = capture_frame(picam2)

            # Reduz para exibição (só na tela, arquivo salvo em alta resolução)
            display = cv2.resize(frame_bgr, DISPLAY_SIZE)

            counts = {name: count_existing(name) for name in CLASSES}
            draw_hud(display, selected_class, counts, saved_message, message_timer)
            if message_timer > 0:
                message_timer -= 1

            cv2.imshow("Master Trash - Coleta de Imagens", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key in SELECT_MAP:
                selected_class = SELECT_MAP[key]
                print(f"[CLASSE] {selected_class.upper()} selecionada")
            elif key == ord(" "):
                if selected_class is None:
                    print("[AVISO] Selecione uma classe antes de salvar.")
                else:
                    path          = save_image(frame_bgr, selected_class)
                    saved_message = f"Salvo: {os.path.basename(path)}"
                    message_timer = 80
                    print(f"[OK] {path}")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("\n--- Resumo ---")
        for name in CLASSES:
            print(f"  {name}: {count_existing(name)} imagens")
        print("Coleta encerrada.\n")


if __name__ == "__main__":
    main()
