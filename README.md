# NEAT Flappy Bird

A procedurally drawn Flappy Bird simulation trained with `neat-python`. The game runs 50 birds concurrently and displays the current generation, living population, and score.

## Install

From this project directory:

```bash
python3 -m venv ../../.venv-flappy-bird
../../.venv-flappy-bird/bin/python -m pip install --upgrade pip
../../.venv-flappy-bird/bin/python -m pip install pygame-ce neat-python
```

`pygame-ce` provides the `pygame` import and is used here because it supports Python 3.14.

## Run

```bash
../../.venv-flappy-bird/bin/python main.py
```

Train visually for a chosen number of generations:

```bash
../../.venv-flappy-bird/bin/python main.py --generations 100
```

Fast, reproducible headless training:

```bash
../../.venv-flappy-bird/bin/python main.py --generations 100 --headless --seed 42
```

Checkpoints are saved every 10 generations by default. Resume from one with:

```bash
../../.venv-flappy-bird/bin/python main.py --resume neat-checkpoint-10 --generations 100 --headless
```

Disable checkpoint files with `--checkpoint-interval 0`.

Validate the files and installed dependencies:

```bash
../../.venv-flappy-bird/bin/python main.py --test
```

The best genome is saved as `best_bird.pkl` whenever a generation produces a new best fitness.

Replay the saved genome:

```bash
../../.venv-flappy-bird/bin/python main.py --play-best
```

The saved genome includes its fitness, generation, pipe score, seed, and save timestamp. Use `pytest` to run the pure-logic tests:

```bash
../../.venv-flappy-bird/bin/python -m pytest -q
```

## GPU usage

NEAT evaluates the neural networks on the CPU. Pygame's 2D drawing is also primarily CPU-side, so this project does not require CUDA or a dedicated GPU. A GPU may still be used by the desktop compositor for presenting the window, but it does not accelerate training.