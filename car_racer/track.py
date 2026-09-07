"""Track data for the car racer game."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame


@dataclass
class Track:
    center: tuple[float, float]
    radius_x: float
    radius_y: float
    width: float
    samples: int = 300

    def points(self) -> list[tuple[float, float]]:
        cx, cy = self.center
        points: list[tuple[float, float]] = []
        for i in range(self.samples):
            angle = (i / self.samples) * 2.0 * math.pi
            x = cx + self.radius_x * math.cos(angle)
            y = cy + self.radius_y * math.sin(angle)
            points.append((x, y))
        return points

    def draw(self, surface: pygame.Surface, *, color: tuple[int, int, int] = (72, 120, 160), line_width: int = 18) -> None:
        pygame.draw.ellipse(surface, color, (self.center[0] - self.radius_x, self.center[1] - self.radius_y, self.radius_x * 2, self.radius_y * 2), line_width)

    def nearest_point(self, x: float, y: float) -> tuple[float, float]:
        points = self.points()
        best = points[0]
        best_dist = float('inf')
        for point in points:
            dist = (point[0] - x) ** 2 + (point[1] - y) ** 2
            if dist < best_dist:
                best_dist = dist
                best = point
        return best

    def distance_to_centerline(self, x: float, y: float) -> float:
        px, py = self.nearest_point(x, y)
        return math.hypot(px - x, py - y)

    def is_on_track(self, x: float, y: float) -> bool:
        cx, cy = self.center
        dx = x - cx
        dy = y - cy
        start_pad = math.hypot(dx, dy) <= 18.0
        if start_pad:
            return True
        normalized = (dx / self.radius_x) ** 2 + (dy / self.radius_y) ** 2
        road_margin = 1.0 + (self.width / (2.0 * max(self.radius_x, self.radius_y)))
        return normalized <= road_margin ** 2


def make_track() -> Track:
    return Track(center=(350.0, 250.0), radius_x=190.0, radius_y=150.0, width=60.0)
