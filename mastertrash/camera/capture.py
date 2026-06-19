"""
camera/capture.py

Captura imagens usando Picamera2 (Arducam 16MP no Raspberry Pi 5).
Retorna arrays NumPy prontos para inferência.
"""

import numpy as np
import os
import time
from pathlib import Path
from picamera2 import Picamera2
from utils.logger import log, log_error


class CameraCapture:
    def __init__(self, image_size: tuple[int, int] = (224, 224)):
        self.image_size = image_size  # (largura, altura)
        self.camera: Picamera2 | None = None

    def initialize(self) -> bool:
        try:
            self.camera = Picamera2()
            width, height = self.image_size
            config = self.camera.create_still_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(1)  # Aguarda estabilização do sensor
            log("Câmera inicializada.")
            return True
        except Exception as e:
            log_error(f"Falha ao inicializar câmera: {e}")
            return False

    def capture_image(self) -> np.ndarray | None:
        if not self.camera:
            log_error("Câmera não inicializada.")
            return None
        try:
            frame = self.camera.capture_array()
            log("Imagem capturada.")
            return frame
        except Exception as e:
            log_error(f"Erro ao capturar imagem: {e}")
            return None

    def save_debug_image(self, image: np.ndarray, folder: str = "debug_images/") -> None:
        try:
            import cv2
            Path(folder).mkdir(parents=True, exist_ok=True)
            filename = os.path.join(folder, f"debug_{int(time.time())}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            log(f"Imagem de debug salva em: {filename}")
        except Exception as e:
            log_error(f"Erro ao salvar imagem de debug: {e}")

    def stop(self):
        if self.camera:
            self.camera.stop()
            log("Câmera encerrada.")
