import random

from dino_runner import main


def test_obstacle_seed_is_reproducible() -> None:
    first = main.Obstacle(rng=random.Random(42))
    second = main.Obstacle(rng=random.Random(42))
    assert first.height == second.height


def test_network_inputs_are_bounded() -> None:
    dino = main.Dino(y=-100)
    dino.velocity = 100
    values = main.network_inputs(dino, main.Obstacle(rng=random.Random(1)))
    assert all(-1.0 <= value <= 1.0 for value in values)


def test_dino_jump_lands_on_ground() -> None:
    dino = main.Dino()
    dino.jump()
    assert not dino.on_ground
    for _ in range(100):
        dino.move()
    assert dino.on_ground


def test_obstacle_types_and_two_action_tie_break() -> None:
    cactus = main.Obstacle(obstacle_type="cactus")
    bird = main.Obstacle(obstacle_type="bird")
    assert cactus.type_value == -1.0
    assert bird.type_value == 1.0
    assert main.action_from_outputs((0.5, 0.5)) == "jump"
    assert main.action_from_outputs((-0.5, 0.5)) == "duck"


def test_dino_network_input_order_has_five_bounded_values() -> None:
    dino = main.Dino()
    values = main.network_inputs(
        dino,
        main.Obstacle(obstacle_type="bird"),
        main.Obstacle(x=main.WIDTH + 200, obstacle_type="cactus"),
        speed=main.MAX_OBSTACLE_SPEED,
    )
    assert len(values) == len(main.INPUT_ORDER) == 5
    assert all(-1.0 <= value <= 1.0 for value in values)
