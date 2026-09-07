# NEAT Flappy Bird

Run commands from the repository root with `python -m flappy_bird.main`.

A procedurally drawn Flappy Bird simulation trained with `neat-python`. The game runs 50 birds concurrently and displays the current generation, living population, and score.

## Install

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install pygame-ce neat-python
```

`pygame-ce` provides the `pygame` import and is used here because it supports Python 3.14.

## Run

```bash
.venv/bin/python -m flappy_bird.main
```

Train visually for a chosen number of generations:

```bash
.venv/bin/python -m flappy_bird.main --generations 100
```

Fast, reproducible headless training:

```bash
.venv/bin/python -m flappy_bird.main --generations 100 --headless --seed 42
```

Experiment with population size and generation length:

```bash
.venv/bin/python -m flappy_bird.main --generations 100 --headless --population 100 --time-limit 900
```

Checkpoints are saved every 10 generations by default. Resume from one with:

```bash
.venv/bin/python -m flappy_bird.main --resume flappy_bird/neat-checkpoint-10 --generations 100 --headless
```

Disable checkpoint files with `--checkpoint-interval 0`.

Validate the files and installed dependencies:

```bash
.venv/bin/python -m flappy_bird.main --test
```

The best genome is saved as `best_bird.pkl` after the score exceeds 50.

Replay the saved genome:

```bash
.venv/bin/python -m flappy_bird.main --play-best
```

Play against the frozen saved genome on the same pipe sequence:

```bash
.venv/bin/python -m flappy_bird.main --play
```

`--play` uses `Space` to flap the orange human bird and shows separate You/AI
scores and the result. It requires `best_bird.pkl`; train first if it is missing.

Play manually with `Space` to flap, `P` to pause, and `R` to restart:

```bash
.venv/bin/python -m flappy_bird.main --human
```

Human mode does not train the AI or make training faster. It is a gameplay and
comparison mode. While playing, it shows the saved AI's best pipe score and the
generation in which that saved record was reached. At the end, the terminal
prints your best score for the session and the AI record.

The saved genome includes its fitness, generation, pipe score, seed, and save timestamp. Use `pytest` to run the pure-logic tests:

```bash
.venv/bin/python -m pytest -q tests/test_flappy_bird.py
```

The current generation evaluates all genomes against one shared pipe timeline, so `neat.ParallelEvaluator` is intentionally not used. Headless mode removes the rendering cost while preserving that shared-environment comparison. Multiprocessing can be added after moving each genome to an independent environment, but doing so now would change the fitness comparison semantics.

## GPU usage

NEAT evaluates the neural networks on the CPU. Pygame's 2D drawing is also primarily CPU-side, so this project does not require CUDA or a dedicated GPU. A GPU may still be used by the desktop compositor for presenting the window, but it does not accelerate training.