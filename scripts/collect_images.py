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

Fixes aplicados para Arducam IMX519:
    - AfMode Continuous  -> resolve foco
    - AwbMode Auto + 2s  -> resolve cor roxa/azul
    - ScalerCrop full    -> resolve zoom excessivo
"""

import cv2
import os
import time
from pathlib import Path
from picamera2 import Picamera2
from libcamera import controls

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

# Cores HUD (BGR)
COLOR_TITLE    = (0, 255, 255)
COLOR_WHITE    = (255, 255, 255)
COLOR_SELECTED = (0, 255, 0)
COLOR_SAVED    = (0, 200, 255)
COLOR_WARN     = (0, 100, 255)


# =============================================================================
# FUNÇÕES
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


def configure_camera(picam2: Picamera2) -> None:
    """
    Configura a câmera para a Arducam IMX519 no Raspberry Pi 5.
    Resolve os três problemas comuns: cor errada, foco e zoom.
    """
    config = picam2.create_preview_configuration(
        main={"size": PREVIEW_SIZE, "format": "RGB888"},
    )
    picam2.configure(config)
    picam2.start()

    # --- Fix 1: Foco contínuo (resolve imagem desfocada) ---
    picam2.set_controls({
        "AfMode":  controls.AfModeEnum.Continuous,
        "AfSpeed": controls.AfSpeedEnum.Fast,
    })

    # --- Fix 2: AWB explícito + tempo de estabilização (resolve cor roxa) ---
    picam2.set_controls({
        "AwbMode": controls.AwbModeEnum.Auto,
    })

    # --- Fix 3: ScalerCrop cobrindo o sensor completo (resolve zoom) ---
    sensor_w, sensor_h = picam2.camera_properties["PixelArraySize"]
    picam2.set_controls({
        "ScalerCrop": (0, 0, sensor_w, sensor_h),
    })

    # Aguarda AWB e AF estabilizarem — essencial para cor correta
    print("Aguardando câmera estabilizar (AWB + AF)...")
    time.sleep(3)
    print("Câmera pronta.")


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
                    "Selecione: 1=paper 2=plastic 3=metal 4=organic",
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

    picam2 = Picamera2()
    configure_camera(picam2)

    selected_class = None
    saved_message  = ""
    message_timer  = 0

    try:
        while True:
            frame     = picam2.capture_array()                      # RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)     # BGR para OpenCV
            display   = frame_bgr.copy()

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
