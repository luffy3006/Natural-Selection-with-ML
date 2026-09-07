"""Backward-compatible imports for the shared game UI helpers."""

from .hud import draw_training_hud
from .human_input import HumanInput
from .stop_button import StopButton
from .training_state import TrainingState

__all__ = ["HumanInput", "StopButton", "TrainingState", "draw_training_hud"]
