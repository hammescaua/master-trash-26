"""
state_machine/states.py

Define os estados do sistema com Enum e a classe StateMachine
que controla as transições de forma explícita e segura.
"""

from enum import Enum, auto
from utils.logger import log, log_error


class State(Enum):
    IDLE             = auto()
    OBJECT_DETECTED  = auto()
    CAPTURING_IMAGE  = auto()
    CLASSIFYING      = auto()
    SENDING_COMMAND  = auto()
    WAITING_SERVO    = auto()
    RETURN_TO_IDLE   = auto()
    ERROR_STATE      = auto()


# Transições válidas: estado atual -> estados permitidos
VALID_TRANSITIONS = {
    State.IDLE:            {State.OBJECT_DETECTED, State.ERROR_STATE},
    State.OBJECT_DETECTED: {State.CAPTURING_IMAGE, State.ERROR_STATE},
    State.CAPTURING_IMAGE: {State.CLASSIFYING,     State.ERROR_STATE},
    State.CLASSIFYING:     {State.SENDING_COMMAND,  State.ERROR_STATE},
    State.SENDING_COMMAND: {State.WAITING_SERVO,    State.ERROR_STATE},
    State.WAITING_SERVO:   {State.RETURN_TO_IDLE,   State.ERROR_STATE},
    State.RETURN_TO_IDLE:  {State.IDLE},
    State.ERROR_STATE:     {State.IDLE},
}


class StateMachine:
    def __init__(self):
        self.current = State.IDLE

    def transition(self, next_state: State) -> bool:
        allowed = VALID_TRANSITIONS.get(self.current, set())
        if next_state not in allowed:
            log_error(f"Transição inválida: {self.current.name} -> {next_state.name}")
            return False
        log(f"Estado: {self.current.name} -> {next_state.name}")
        self.current = next_state
        return True

    def is_in(self, state: State) -> bool:
        return self.current == state

    def force_idle(self):
        """Reseta para IDLE em qualquer situação (uso em erros críticos)."""
        self.current = State.IDLE
