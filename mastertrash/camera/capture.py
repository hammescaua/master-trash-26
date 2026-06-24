"""
camera/capture.py

Captura imagens usando Picamera2 (Arducam IMX519 16MP no Raspberry Pi 5).
Usa preview_configuration + capture_array() — mais estável no Pi 5 que
still_configuration para capturas em loop contínuo.

Retorna arrays NumPy RGB prontos para inferência TFLite.
"""

import numpy as np
import time
from pathlib import Path
from picamera2 import Picamera2
from utils.logger import log, log_error


class CameraCapture:
    def __init__(self, image_size: tuple = (224, 224)):
        self.image_size = image_size  # (largura, altura) — entrada do modelo
        self.camera = None

    def initialize(self) -> bool:
        try:
            self.camera = Picamera2()
            width, height = self.image_size
            config = self.camera.create_preview_configuration(
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

    def capture_image(self) -> np.ndarray:
        """Retorna array NumPy RGB (H, W, 3) ou None em caso de erro."""
        if not self.camera:
            log_error("Câmera não inicializada.")
            return None
        try:
            frame = self.camera.capture_array()  # RGB888 -> ndarray uint8
            log("Imagem capturada.")
            return frame
        except Exception as e:
            log_error(f"Erro ao capturar imagem: {e}")
            return None

    def save_debug_image(self, image: np.ndarray, folder: str = "debug_images/") -> None:
        """Salva imagem em disco para inspeção manual (RGB -> BGR para OpenCV)."""
        try:
            import cv2
            dest = Path(folder)
            dest.mkdir(parents=True, exist_ok=True)
            filename = dest / f"debug_{int(time.time())}.jpg"
            # Picamera2 entrega RGB; OpenCV salva em BGR
            cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            log(f"Debug salvo: {filename}")
        except Exception as e:
            log_error(f"Erro ao salvar debug: {e}")

    def stop(self):
        if self.camera:
            self.camera.stop()
            log("Câmera encerrada.")
