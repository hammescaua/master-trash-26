"""
camera/capture.py

Captura imagens usando Picamera2 (Arducam IMX519 16MP no Raspberry Pi 5).

Formato XBGR8888 no modo binned 2328x1748 — mesmo modo do rpicam-hello.
Garante campo visual completo (sem zoom) e cores corretas (sem cvtColor).
"""

import numpy as np
import cv2
import time
from pathlib import Path
from picamera2 import Picamera2
from utils.logger import log, log_error

# Resolução binned do sensor IMX519 (metade do sensor físico, sem crop)
SENSOR_SIZE = (2328, 1748)


class CameraCapture:
    def __init__(self, model_input_size: tuple = (224, 224)):
        self.model_input_size = model_input_size  # tamanho de entrada do TFLite
        self.camera = None

    def initialize(self) -> bool:
        try:
            self.camera = Picamera2()

            config = self.camera.create_preview_configuration(
                main={"size": SENSOR_SIZE, "format": "XBGR8888"},
                controls={"FrameRate": 15},
            )
            self.camera.configure(config)
            self.camera.start()

            log("Aguardando AWB estabilizar...")
            time.sleep(3)
            log("Câmera inicializada.")
            return True

        except Exception as e:
            log_error(f"Falha ao inicializar câmera: {e}")
            return False

    def capture_image(self) -> np.ndarray:
        """
        Retorna imagem RGB (H, W, 3) no tamanho do modelo, pronta para TFLite.

        Pipeline:
          XBGR8888 (4ch) → [:,:,:3] → BGR (3ch) → resize → RGB (para modelo)
        """
        if not self.camera:
            log_error("Câmera não inicializada.")
            return None
        try:
            raw  = self.camera.capture_array()           # (1748, 2328, 4) XBGR
            bgr  = raw[:, :, :3]                         # descarta canal X → BGR
            w, h = self.model_input_size
            resized = cv2.resize(bgr, (w, h))            # reduz para entrada do modelo
            rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # BGR → RGB para TFLite
            log("Imagem capturada.")
            return rgb
        except Exception as e:
            log_error(f"Erro ao capturar imagem: {e}")
            return None

    def save_debug_image(self, image_rgb: np.ndarray, folder: str = "debug_images/") -> None:
        """Salva imagem RGB em disco no formato BGR (padrão OpenCV/JPEG)."""
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
