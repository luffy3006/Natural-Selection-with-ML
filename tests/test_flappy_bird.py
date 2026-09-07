import random

import pygame

from flappy_bird import main


def setup_module() -> None:
    pygame.init()


def teardown_module() -> None:
    pygame.quit()


def test_percentage_improvement_handles_first_generation() -> None:
    assert main.percentage_improvement(10.0, None) == 0.0
    assert main.percentage_improvement(10.0, 0.0) == 0.0
    assert main.percentage_improvement(15.0, 10.0) == 50.0


def test_pipe_seed_is_reproducible() -> None:
    first = main.Pipe(rng=random.Random(42))
    second = main.Pipe(rng=random.Random(42))
    assert first.gap_y == second.gap_y


def test_network_inputs_are_bounded() -> None:
    bird = main.Bird(y=-500)
    bird.velocity = 100
    pipe = main.Pipe(rng=random.Random(1))
    values = main.network_inputs(bird, pipe)
    assert all(-1.0 <= value <= 1.0 for value in values)


def test_bird_gravity_is_clamped() -> None:
    bird = main.Bird()
    for _ in range(100):
        bird.move()
    assert bird.velocity == main.MAX_FALL_SPEED


def test_pipe_collision_detects_overlap() -> None:
    pipe = main.Pipe(rng=random.Random(3))
    bird = main.Bird(y=pipe.gap_y - main.PIPE_GAP // 2 - 10)
    bird.x = pipe.x + 10
    bird._sync_rect()
    assert pipe.collides(bird)