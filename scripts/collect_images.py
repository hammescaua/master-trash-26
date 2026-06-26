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

PREVIEW_SIZE = (1280, 720)

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
    Inicializa a Arducam IMX519 via Picamera2 de forma robusta.

    Estratégia:
    - Usa BGR888 diretamente → OpenCV exibe sem conversão de cor
    - Aplica ScalerCrop em chamada separada após start() (evita race condition)
    - Sleep escalonado: 1s geral + 1s extra para AWB estabilizar
    """
    picam2 = Picamera2()

    # Imprime propriedades para debug caso algo ainda não bata
    props = picam2.camera_properties
    sensor_w, sensor_h = props["PixelArraySize"]
    print(f"[CAM] Sensor: {sensor_w}x{sensor_h}")

    config = picam2.create_preview_configuration(
        main={"size": PREVIEW_SIZE, "format": "BGR888"},  # BGR nativo → sem conversão
        controls={"FrameRate": 30},
    )
    picam2.configure(config)
    picam2.start()

    # Pausa 1: sensor ligar
    time.sleep(1)

    # ScalerCrop em chamada isolada → usa sensor completo sem crop
    picam2.set_controls({"ScalerCrop": [0, 0, sensor_w, sensor_h]})

    # Pausa 2: AWB estabilizar (essencial para cor correta)
    print("Aguardando AWB estabilizar...")
    time.sleep(2)
    print("Câmera pronta.\n")

    return picam2, sensor_w, sensor_h


# =============================================================================
# DATASET
# =============================================================================

def count_existing(class_name: str) -> int:
    folder = BASE_DIR / class_name
    if not folder.exists():
        return 0
    return len(list(folder.glob("*.jpg")))


def save_image(frame, class_name: str) -> str:
    folder = BASE_DIR / class_name
    folder.mkdir(parents=True, exist_ok=True)
    count    = count_existing(class_name) + 1
    filename = folder / f"{class_name}_{count:04d}.jpg"
    cv2.imwrite(str(filename), frame)
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

    picam2, _, _ = init_camera()

    selected_class = None
    saved_message  = ""
    message_timer  = 0

    try:
        while True:
            # BGR888 → OpenCV usa direto, sem conversão
            frame   = picam2.capture_array()
            display = frame.copy()

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
                    path          = save_image(frame, selected_class)
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
