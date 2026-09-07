import random

from pong import main


def test_pong_inputs_are_bounded() -> None:
    ball = main.Ball(vx=200, vy=200)
    paddle = main.Paddle(30, 20)
    values = main.network_inputs(paddle, ball)
    assert all(-1.0 <= value <= 1.0 for value in values)


def test_pong_action_dead_zone() -> None:
    assert main.paddle_action(0.05) == 0.0
    assert main.paddle_action(0.5) == 0.5
