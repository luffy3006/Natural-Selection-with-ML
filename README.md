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

Train for a chosen number of generations:

```bash
../../.venv-flappy-bird/bin/python main.py --generations 100
```

Validate the files and installed dependencies:

```bash
../../.venv-flappy-bird/bin/python main.py --test
```

The best genome is saved as `best_bird.pkl` after the score exceeds 50.

## GPU usage

NEAT evaluates the neural networks on the CPU. Pygame's 2D drawing is also primarily CPU-side, so this project does not require CUDA or a dedicated GPU. A GPU may still be used by the desktop compositor for presenting the window, but it does not accelerate training.