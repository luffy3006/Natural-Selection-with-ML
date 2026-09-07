# NEAT Game Lab

Two small procedural Pygame games trained with `neat-python`:

- `flappy_bird/` — the original shared-pipe Flappy Bird simulation.
- `dino_runner/` — an endless runner with cactus-jump and low-bird-duck avoidance.
- `common/` — shared `hud.py`, `training_state.py`, `stop_button.py`, and
  `human_input.py` helpers (with `common.ui` kept as a compatibility import).
- `tests/` — pure-logic tests for both games.

## Setup

From the repository root:

```bash
python3 -m venv ../../.venv-flappy-bird
../../.venv-flappy-bird/bin/python -m pip install pygame-ce neat-python pytest
```

## Commands

Both games support visual training, deterministic headless training, validation,
checkpoints, and replay of the saved best genome:

```bash
../../.venv-flappy-bird/bin/python -m flappy_bird.main --generations 100
../../.venv-flappy-bird/bin/python -m flappy_bird.main --generations 2 --headless --seed 42
../../.venv-flappy-bird/bin/python -m flappy_bird.main --test
../../.venv-flappy-bird/bin/python -m flappy_bird.main --play-best
../../.venv-flappy-bird/bin/python -m flappy_bird.main --human

../../.venv-flappy-bird/bin/python -m dino_runner.main --generations 100
../../.venv-flappy-bird/bin/python -m dino_runner.main --generations 2 --headless --seed 42
../../.venv-flappy-bird/bin/python -m dino_runner.main --test
../../.venv-flappy-bird/bin/python -m dino_runner.main --play-best
../../.venv-flappy-bird/bin/python -m dino_runner.main --play  # play beside the frozen best AI
```

Use `--population`, `--time-limit`, `--checkpoint-interval`, and `--resume`
to tune or continue a training run. `--checkpoint-interval 0` disables
checkpoint output. Headless runs do not open a display and are suitable for CI.

Dino Runner's five inputs are fixed and documented in `dino_runner/main.py`:
state (`-1` ducking, `0` running, `+1` jumping), normalized next-obstacle
distance, obstacle type (`-1` cactus, `+1` bird), normalized current speed,
and normalized second-obstacle distance. Its two outputs are `[jump, duck]`;
when both exceed the action threshold, jump wins. Cactus obstacles require
jumping and low floating birds require ducking. Fitness is `+0.05` per frame,
`+5` per obstacle passed, and `-5` on death. `--play` uses the exact same
seeded obstacle stream for the human and frozen AI; orange is the human,
blue is the AI, and the result banner reports both scores.

## Validation

```bash
../../.venv-flappy-bird/bin/python -m compileall -q common flappy_bird dino_runner tests
../../.venv-flappy-bird/bin/python -m pytest -q
```

Generated genomes, NEAT checkpoints, Python caches, and virtual environments
are ignored by Git.
