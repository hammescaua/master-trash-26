"""
config.py

Todas as configurações do projeto em um único lugar.
Altere aqui sem precisar tocar no resto do código.
"""

# --- Serial ---
SERIAL_PORT = "/dev/ttyACM0"   # Porta do Arduino no Raspberry Pi
BAUD_RATE = 9600

# --- Sensor ---
DISTANCE_THRESHOLD = 15        # Distância em cm para detectar objeto

# --- Câmera ---
IMAGE_SIZE = (224, 224)        # Tamanho de entrada do modelo

# --- Modelo ---
MODEL_PATH = "ai/model.tflite"
LABELS_PATH = "ai/labels.txt"
CONFIDENCE_THRESHOLD = 0.70    # Confiança mínima para aceitar predição

# --- Servo ---
SERVO_WAIT_TIME = 3.0          # Segundos aguardando o servo completar o ciclo

# --- Captura ---
CAPTURE_DELAY = 0.5            # Segundos entre detecção e captura da imagem

# --- Debug ---
SAVE_DEBUG_IMAGES = False      # Salvar imagens capturadas para debug
DEBUG_IMAGE_PATH = "debug_images/"
