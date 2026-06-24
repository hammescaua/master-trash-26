"""
main.py

Ponto de entrada do Master Trash 4.0.
Orquestra a máquina de estados, câmera, classificador e Arduino.
"""

import time
import sys

import config
from camera.capture import CameraCapture
from ai.classifier import WasteClassifier
from serial_comm.arduino_controller import ArduinoController
from state_machine.states import StateMachine, State
from hardware.commands import CLASS_TO_COMMAND
from utils.logger import log, log_error, log_warning


def initialize_components():
    arduino = ArduinoController(config.SERIAL_PORT, config.BAUD_RATE)
    camera  = CameraCapture(image_size=config.IMAGE_SIZE)
    model   = WasteClassifier(config.MODEL_PATH, config.LABELS_PATH)

    if not arduino.connect():
        log_error("Não foi possível conectar ao Arduino. Encerrando.")
        sys.exit(1)

    if not camera.initialize():
        log_error("Não foi possível inicializar a câmera. Encerrando.")
        sys.exit(1)

    if not model.load_model():
        log_error("Não foi possível carregar o modelo. Encerrando.")
        sys.exit(1)

    return arduino, camera, model


def run():
    log("=== Master Trash 4.0 iniciado ===")

    arduino, camera, model = initialize_components()
    sm = StateMachine()

    try:
        while True:
            # --- IDLE: aguarda detecção ---
            if sm.is_in(State.IDLE):
                if arduino.is_object_detected():
                    sm.transition(State.OBJECT_DETECTED)

            # --- OBJECT_DETECTED: aguarda e captura ---
            elif sm.is_in(State.OBJECT_DETECTED):
                log("Objeto detectado.")
                time.sleep(config.CAPTURE_DELAY)
                sm.transition(State.CAPTURING_IMAGE)

            # --- CAPTURING_IMAGE ---
            elif sm.is_in(State.CAPTURING_IMAGE):
                image = camera.capture_image()
                if image is None:
                    log_error("Falha na captura da imagem.")
                    sm.transition(State.ERROR_STATE)
                    continue

                if config.SAVE_DEBUG_IMAGES:
                    camera.save_debug_image(image, config.DEBUG_IMAGE_PATH)

                sm.transition(State.CLASSIFYING)

            # --- CLASSIFYING ---
            elif sm.is_in(State.CLASSIFYING):
                label, confidence = model.get_top_prediction(image)

                if confidence < config.CONFIDENCE_THRESHOLD:
                    log_warning(
                        f"Confiança baixa ({confidence * 100:.1f}%). "
                        "Descartando predição."
                    )
                    sm.transition(State.ERROR_STATE)
                    continue

                sm.transition(State.SENDING_COMMAND)

            # --- SENDING_COMMAND ---
            elif sm.is_in(State.SENDING_COMMAND):
                command = CLASS_TO_COMMAND.get(label)
                if not command:
                    log_error(f"Classe desconhecida: {label}")
                    sm.transition(State.ERROR_STATE)
                    continue

                if not arduino.send_command(command):
                    sm.transition(State.ERROR_STATE)
                    continue

                sm.transition(State.WAITING_SERVO)

            # --- WAITING_SERVO ---
            elif sm.is_in(State.WAITING_SERVO):
                log(f"Aguardando servo por {config.SERVO_WAIT_TIME}s...")
                time.sleep(config.SERVO_WAIT_TIME)
                sm.transition(State.RETURN_TO_IDLE)

            # --- RETURN_TO_IDLE ---
            elif sm.is_in(State.RETURN_TO_IDLE):
                log("Ciclo completo. Retornando ao IDLE.")
                sm.transition(State.IDLE)

            # --- ERROR_STATE ---
            elif sm.is_in(State.ERROR_STATE):
                log_error("Estado de erro. Aguardando 2s antes de resetar.")
                time.sleep(2)
                sm.force_idle()

            time.sleep(0.05)  # Evita busy-loop

    except KeyboardInterrupt:
        log("Interrompido pelo usuário.")
    finally:
        camera.stop()
        arduino.disconnect()
        log("=== Master Trash 4.0 encerrado ===")


if __name__ == "__main__":
    run()
