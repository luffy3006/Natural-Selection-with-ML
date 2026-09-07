"""Reusable stop control for visual training windows."""

from __future__ import annotations

import pygame


class StopButton:
    """A small, reusable mouse button that requests a training stop."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str = "STOP",
        color: tuple[int, int, int] = (190, 67, 61),
        hover_color: tuple[int, int, int] = (215, 78, 69),
    ) -> None:
        self.rect = rect
        self.label = label
        self.color = color
        self.hover_color = hover_color

    def clicked(self, event: pygame.event.Event) -> bool:
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        color = self.hover_color if self.rect.collidepoint(pygame.mouse.get_pos()) else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        text = font.render(self.label, True, (255, 255, 255))
        surface.blit(text, text.get_rect(center=self.rect.center))
