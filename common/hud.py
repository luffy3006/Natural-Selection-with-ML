"""Shared training HUD drawing helpers."""

from __future__ import annotations

import pygame

from .stop_button import StopButton
from .training_state import TrainingState


def draw_training_hud(
    surface: pygame.Surface,
    fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font],
    state: TrainingState,
    alive_count: int,
    population_size: int,
    title: str,
    subtitle: str,
    stop_button: StopButton,
    *,
    panel_color: tuple[int, int, int] = (247, 251, 246),
    edge_color: tuple[int, int, int] = (205, 224, 211),
    ink: tuple[int, int, int] = (30, 43, 48),
    accent: tuple[int, int, int] = (27, 112, 91),
) -> None:
    """Draw the common generation, score, and fitness panels."""

    title_font, body_font, small_font = fonts
    width, _ = surface.get_size()
    panel = pygame.Surface((width - 24, 124), pygame.SRCALPHA)
    panel.fill((*panel_color, 238))
    pygame.draw.rect(panel, edge_color, panel.get_rect(), 2, border_radius=16)
    surface.blit(panel, (12, 12))
    surface.blit(title_font.render(title, True, accent), (28, 24))
    surface.blit(small_font.render(subtitle, True, (96, 121, 111)), (30, 54))
    metrics = (
        ("GENERATION", str(state.generation), 28),
        ("ALIVE", f"{alive_count}/{population_size}", 132),
        ("SCORE", str(state.score), 252),
    )
    for label, value, x in metrics:
        surface.blit(small_font.render(label, True, (96, 121, 111)), (x, 78))
        surface.blit(body_font.render(value, True, ink), (x, 94))
    stop_button.draw(surface, body_font)

    summary = pygame.Surface((width - 24, 74), pygame.SRCALPHA)
    summary.fill((*panel_color, 220))
    pygame.draw.rect(summary, edge_color, summary.get_rect(), 2, border_radius=14)
    surface.blit(summary, (12, 146))
    values = (
        ("LAST GEN BEST", "--" if state.best_fitness is None else f"{state.best_fitness:.1f}", 28),
        ("LAST GEN AVG", "--" if state.best_fitness is None else f"{state.average_fitness:.1f}", 170),
        ("IMPROVEMENT", "--" if state.best_fitness is None else f"{state.improvement:+.1f}%", 312),
    )
    for label, value, x in values:
        surface.blit(small_font.render(label, True, (96, 121, 111)), (x, 158))
        surface.blit(body_font.render(value, True, accent), (x, 174))
