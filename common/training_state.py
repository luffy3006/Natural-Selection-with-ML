"""Shared metrics collected while a NEAT generation is evaluated."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingState:
    """Metrics displayed by the training HUD."""

    generation: int = 0
    best_fitness: float | None = None
    average_fitness: float = 0.0
    improvement: float = 0.0
    score: int = 0
    stop_requested: bool = False

    def request_stop(self) -> None:
        self.stop_requested = True

    def record_generation(self, best: float, average: float) -> None:
        previous = self.best_fitness
        self.improvement = (
            0.0
            if previous is None or previous == 0
            else (best - previous) / abs(previous) * 100.0
        )
        self.best_fitness = best
        self.average_fitness = average
