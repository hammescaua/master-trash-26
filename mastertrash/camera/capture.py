"""
camera/capture.py

Captura imagens usando Picamera2 (Arducam IMX519 16MP no Raspberry Pi 5).

Usa BGR888 nativo para evitar conversão de cor incorreta.
ScalerCrop aplicado em chamada separada após start() para cobrir sensor completo.
Retorna arrays NumPy BGR para uso interno e RGB para o modelo TFLite.
"""

import numpy as np
import cv2
import time
from pathlib import Path
from picamera2 import Picamera2
from utils.logger import log, log_error


class CameraCapture:
    def __init__(self, image_size: tuple = (224, 224)):
        self.image_size = image_size
        self.camera = None

    def initialize(self) -> bool:
        try:
            self.camera = Picamera2()

            sensor_w, sensor_h = self.camera.camera_properties["PixelArraySize"]
            log(f"Sensor detectado: {sensor_w}x{sensor_h}")

            width, height = self.image_size
            config = self.camera.create_preview_configuration(
                main={"size": (width, height), "format": "BGR888"},
                controls={"FrameRate": 30},
            )
            self.camera.configure(config)
            self.camera.start()

            # Pausa 1: sensor ligar
            time.sleep(1)

            # ScalerCrop isolado → evita race condition com outros controles
            self.camera.set_controls({"ScalerCrop": [0, 0, sensor_w, sensor_h]})

            # Pausa 2: AWB estabilizar
            log("Aguardando AWB estabilizar...")
            time.sleep(2)
            log("Câmera inicializada.")
            return True

        except Exception as e:
            log_error(f"Falha ao inicializar câmera: {e}")
            return False

    def capture_image(self) -> np.ndarray:
        """
        Retorna imagem RGB (H, W, 3) pronta para inferência TFLite.
        A câmera entrega BGR888; convertemos para RGB aqui.
        """
        if not self.camera:
            log_error("Câmera não inicializada.")
            return None
        try:
            bgr   = self.camera.capture_array()           # BGR888
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # RGB para o modelo
            log("Imagem capturada.")
            return frame
        except Exception as e:
            log_error(f"Erro ao capturar imagem: {e}")
            return None

    def save_debug_image(self, image_rgb: np.ndarray, folder: str = "debug_images/") -> None:
        """Salva imagem RGB em disco (converte para BGR para o OpenCV)."""
        try:
            dest = Path(folder)
            dest.mkdir(parents=True, exist_ok=True)
            filename = dest / f"debug_{int(time.time())}.jpg"
            cv2.imwrite(str(filename), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            log(f"Debug salvo: {filename}")
        except Exception as e:
            log_error(f"Erro ao salvar debug: {e}")

    def stop(self):
        if self.camera:
            self.camera.stop()
            log("Câmera encerrada.")
