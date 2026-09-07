import math

from car_racer import main


def test_car_sensor_values_are_bounded() -> None:
    car = main.Car(350.0, 250.0, 0.0)
    values = main.network_inputs(car)
    assert len(values) == 6
    assert all(-1.0 <= value <= 1.0 for value in values)


def test_track_is_on_track_initial_position() -> None:
    assert main.TRACK.is_on_track(350.0, 250.0)


def test_action_from_outputs_is_continuous() -> None:
    steering, throttle = main.action_from_outputs((0.3, -0.8))
    assert math.isclose(steering, 0.3)
    assert math.isclose(throttle, -0.8)
