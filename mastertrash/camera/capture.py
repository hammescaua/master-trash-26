"""
camera/capture.py

Captura imagens usando Picamera2 (Arducam IMX519 16MP no Raspberry Pi 5).

Correções aplicadas para a IMX519:
    - AfMode Continuous  -> foco automático contínuo (resolve blur)
    - AwbMode Auto + 3s  -> branco correto (resolve cor roxa/azul)
    - ScalerCrop full    -> campo visual completo (resolve zoom excessivo)
"""

import numpy as np
import time
from pathlib import Path
from picamera2 import Picamera2
from libcamera import controls
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
                main={"size": (width, height), "format": "RGB888"},
            )
            self.camera.configure(config)
            self.camera.start()

            # Fix foco — autofocus contínuo da IMX519
            self.camera.set_controls({
                "AfMode":  controls.AfModeEnum.Continuous,
                "AfSpeed": controls.AfSpeedEnum.Fast,
            })

            # Fix cor — AWB explícito
            self.camera.set_controls({
                "AwbMode": controls.AwbModeEnum.Auto,
            })

            # Fix zoom — usa área total do sensor
            sensor_w, sensor_h = self.camera.camera_properties["PixelArraySize"]
            self.camera.set_controls({
                "ScalerCrop": (0, 0, sensor_w, sensor_h),
            })

            # Aguarda AWB e AF estabilizarem antes do primeiro uso
            time.sleep(3)
            log("Câmera inicializada (AF + AWB estabilizados).")
            return True

        except Exception as e:
            log_error(f"Falha ao inicializar câmera: {e}")
            return False

    def capture_image(self) -> np.ndarray:
        """Retorna array NumPy RGB (H, W, 3) pronto para inferência TFLite."""
        if not self.camera:
            log_error("Câmera não inicializada.")
            return None
        try:
            frame = self.camera.capture_array()  # RGB888
            log("Imagem capturada.")
            return frame
        except Exception as e:
            log_error(f"Erro ao capturar imagem: {e}")
            return None

    def save_debug_image(self, image: np.ndarray, folder: str = "debug_images/") -> None:
        """Salva imagem RGB em disco no formato BGR (OpenCV)."""
        try:
            import cv2
            dest = Path(folder)
            dest.mkdir(parents=True, exist_ok=True)
            filename = dest / f"debug_{int(time.time())}.jpg"
            cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            log(f"Debug salvo: {filename}")
        except Exception as e:
            log_error(f"Erro ao salvar debug: {e}")

    def stop(self):
        if self.camera:
            self.camera.stop()
            log("Câmera encerrada.")
