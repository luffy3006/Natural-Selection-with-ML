"""Keyboard input shared by the interactive games."""

from __future__ import annotations

from typing import Iterable

import pygame


class HumanInput:
    """Translate keyboard events into game-independent action flags."""

    def __init__(self) -> None:
        self.paused = False
        self.quit_requested = False
        self.restart_requested = False
        self.jump_requested = False
        self.duck_requested = False
        self.duck_held = False
        # Flappy Bird uses the older, descriptive name.
        self.flap_requested = False

    def begin_frame(self) -> None:
        self.restart_requested = False
        self.jump_requested = False
        self.duck_requested = self.duck_held
        self.flap_requested = False

    def handle(self, events: Iterable[pygame.event.Event]) -> None:
        self.begin_frame()
        for event in events:
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_requested = True
                elif event.key in (pygame.K_SPACE, pygame.K_UP):
                    self.jump_requested = True
                    self.flap_requested = True
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.duck_held = True
                    self.duck_requested = True
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.restart_requested = True
            elif event.type == pygame.KEYUP and event.key in (pygame.K_DOWN, pygame.K_s):
                self.duck_held = False
                self.duck_requested = False
