"""
ai/classifier.py

Carrega o modelo TFLite e executa inferência sobre imagens capturadas.
Retorna a classe prevista e o nível de confiança.
"""

import numpy as np
from pathlib import Path
from tflite_runtime.interpreter import Interpreter
from utils.logger import log, log_error


class WasteClassifier:
    def __init__(self, model_path: str, labels_path: str):
        self.model_path = model_path
        self.labels_path = labels_path
        self.interpreter: Interpreter | None = None
        self.labels: list[str] = []
        self.input_details = None
        self.output_details = None

    def load_model(self) -> bool:
        if not Path(self.model_path).exists():
            log_error(f"Modelo não encontrado: {self.model_path}")
            return False
        if not Path(self.labels_path).exists():
            log_error(f"Labels não encontrado: {self.labels_path}")
            return False

        try:
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details  = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            with open(self.labels_path, "r") as f:
                self.labels = [line.strip() for line in f.readlines()]

            log(f"Modelo carregado. Classes: {self.labels}")
            return True
        except Exception as e:
            log_error(f"Erro ao carregar modelo: {e}")
            return False

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # Normaliza para [0, 1] e adiciona dimensão de batch
        image = image.astype(np.float32) / 255.0
        return np.expand_dims(image, axis=0)

    def predict(self, image: np.ndarray) -> np.ndarray:
        input_data = self.preprocess(image)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        return output[0]  # Remove dimensão de batch

    def get_top_prediction(self, image: np.ndarray) -> tuple[str, float]:
        scores = self.predict(image)
        top_index = int(np.argmax(scores))
        confidence = float(scores[top_index])
        label = self.labels[top_index]
        log(f"Predição: {label} ({confidence * 100:.1f}%)")
        return label, confidence
