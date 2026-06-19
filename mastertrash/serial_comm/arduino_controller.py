"""
serial_comm/arduino_controller.py

Gerencia toda a comunicação serial com o Arduino.
Lê mensagens recebidas e envia comandos de controle do servo.
"""

import serial
import time
from hardware.commands import OBJECT_DETECTED
from utils.logger import log, log_error


class ArduinoController:
    def __init__(self, port: str, baud_rate: int):
        self.port = port
        self.baud_rate = baud_rate
        self.connection: serial.Serial | None = None

    def connect(self) -> bool:
        try:
            self.connection = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Aguarda o Arduino resetar após conexão serial
            log(f"Arduino conectado em {self.port} a {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            log_error(f"Falha ao conectar ao Arduino: {e}")
            return False

    def read_message(self) -> str | None:
        if not self.connection or not self.connection.in_waiting:
            return None
        try:
            line = self.connection.readline().decode("utf-8").strip()
            return line if line else None
        except Exception as e:
            log_error(f"Erro ao ler serial: {e}")
            return None

    def send_command(self, command: str) -> bool:
        if not self.connection:
            log_error("Serial não conectada. Comando não enviado.")
            return False
        try:
            self.connection.write(f"{command}\n".encode("utf-8"))
            log(f"Comando enviado: {command}")
            return True
        except Exception as e:
            log_error(f"Erro ao enviar comando: {e}")
            return False

    def is_object_detected(self) -> bool:
        message = self.read_message()
        if message:
            log(f"Serial recebido: {message}")
        return message == OBJECT_DETECTED

    def disconnect(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            log("Conexão serial encerrada.")
