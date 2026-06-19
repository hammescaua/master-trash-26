"""
utils/logger.py

Logger simples com timestamp no formato [HH:MM:SS] mensagem.
Sem dependências externas além da biblioteca padrão.
"""

import time


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def log_error(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] ERROR: {message}")


def log_warning(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] WARNING: {message}")
