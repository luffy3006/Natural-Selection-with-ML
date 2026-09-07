# NEAT Game Lab

Two small procedural Pygame games trained with `neat-python`:

- `flappy_bird/` — the original shared-pipe Flappy Bird simulation.
- `dino_runner/` — an endless runner with cactus-jump and low-bird-duck avoidance.
- `pong/` — a two-paddle self-play winner-takes-all game trained with NEAT.
- `car_racer/` — a continuous-control track-following racer with ray sensors and throttle/steering outputs.
- `common/` — shared `hud.py`, `training_state.py`, `stop_button.py`, and `human_input.py` helpers (with `common.ui` kept as a compatibility import).
- `tests/` — pure-logic tests for the games.

## Setup

From the repository root:

```bash
python3 -m venv ../../.venv-flappy-bird
../../.venv-flappy-bird/bin/python -m pip install pygame-ce neat-python pytest
```

## Commands

```bash
../../.venv-flappy-bird/bin/python -m flappy_bird.main --generations 100
../../.venv-flappy-bird/bin/python -m flappy_bird.main --headless --generations 2 --seed 42
../../.venv-flappy-bird/bin/python -m flappy_bird.main --test
../../.venv-flappy-bird/bin/python -m flappy_bird.main --play-best
../../.venv-flappy-bird/bin/python -m flappy_bird.main --play

../../.venv-flappy-bird/bin/python -m dino_runner.main --generations 100
../../.venv-flappy-bird/bin/python -m dino_runner.main --headless --generations 2 --seed 42
../../.venv-flappy-bird/bin/python -m dino_runner.main --test
../../.venv-flappy-bird/bin/python -m dino_runner.main --play-best
../../.venv-flappy-bird/bin/python -m dino_runner.main --play

../../.venv-flappy-bird/bin/python -m pong.main --generations 100
../../.venv-flappy-bird/bin/python -m pong.main --headless --generations 2 --seed 42
../../.venv-flappy-bird/bin/python -m pong.main --test
../../.venv-flappy-bird/bin/python -m pong.main --play-best
../../.venv-flappy-bird/bin/python -m pong.main --play

../../.venv-flappy-bird/bin/python -m car_racer.main --generations 100
../../.venv-flappy-bird/bin/python -m car_racer.main --headless --generations 2 --seed 42
../../.venv-flappy-bird/bin/python -m car_racer.main --test
../../.venv-flappy-bird/bin/python -m car_racer.main --play-best
../../.venv-flappy-bird/bin/python -m car_racer.main --play
```

Use `--population`, `--time-limit`, `--checkpoint-interval`, and `--resume` to tune or continue a training run. `--checkpoint-interval 0` disables checkpoint output. Headless runs do not open a display and are suitable for CI.

## Pong notes

Pong uses one NEAT population with within-generation self-play pairings. Each match rewards a paddle with `+1` for a point scored and `-1` for a point conceded, plus a small `+0.01` per-frame engagement bonus so long rallies are rewarded. The network takes 5 inputs: this paddle's y, ball x/y, and ball velocity x/y. The single output is continuous in `[-1, 1]`, mapped to up/down movement with a small dead-zone (`|output| < 0.1` means stay still).

## Car Racer notes

Car Racer uses a continuous-control output pair `(steering, throttle)` rather than a discrete jump/duck action. The ray sensor array is fixed at `[-60, -30, 0, 30, 60]` degrees, with a current speed input, for a total of 6 inputs. The network outputs are direct tanh values in `[-1, 1]`: steering controls turn rate and throttle controls forward acceleration. Fitness is driven primarily by track progress, with an off-track death penalty and a smaller speed bonus.

## Validation

```bash
../../.venv-flappy-bird/bin/python -m compileall -q common flappy_bird dino_runner pong car_racer tests
../../.venv-flappy-bird/bin/python -m pytest -q
```

Generated genomes, checkpoints, Python caches, and virtual environments are ignored by Git.
