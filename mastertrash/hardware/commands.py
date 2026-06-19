"""
hardware/commands.py

Constantes dos comandos trocados entre Raspberry Pi e Arduino via serial.
Centralizar aqui evita strings mágicas espalhadas pelo código.
"""

# Mensagem recebida do Arduino
OBJECT_DETECTED = "OBJECT_DETECTED"

# Comandos enviados ao Arduino
MOVE_PAPER   = "MOVE_PAPER"
MOVE_PLASTIC = "MOVE_PLASTIC"
MOVE_METAL   = "MOVE_METAL"
MOVE_ORGANIC = "MOVE_ORGANIC"

# Mapeamento: classe do modelo -> comando serial
CLASS_TO_COMMAND = {
    "paper":   MOVE_PAPER,
    "plastic": MOVE_PLASTIC,
    "metal":   MOVE_METAL,
    "organic": MOVE_ORGANIC,
}
