# NEAT Flappy Bird: Project Logic Summary

Copy this document into Claude when asking for architecture, AI-training, gameplay, or optimization advice.

## Project goal

This is a Python Flappy Bird simulation that uses Pygame for the game loop and drawing, and `neat-python` for neuroevolution. Fifty birds are evaluated concurrently in each generation. Each bird is controlled by its own evolved neural-network genome. There are no external image or sound assets; the game is drawn procedurally with Pygame primitives.

## Files

- `main.py`: gameplay, physics, collision detection, rendering, HUD, NEAT evaluation callback, training loop, and genome saving.
- `config-feedforward.txt`: NEAT population, genome, mutation, species, and reproduction settings.
- `README.md`: installation and run instructions.
- `.gitignore`: ignores Python caches, virtual environments, `.env`, and `best_bird.pkl`.
- `best_bird.pkl`: generated whenever a generation produces a new best fitness; it is ignored by Git.

## Runtime setup

- Python: tested with Python 3.14.
- Pygame implementation: `pygame-ce 2.5.8`, imported in code as `pygame`.
- AI library: `neat-python 2.0.0`.
- The virtual environment is outside the project folder at `../../.venv-flappy-bird` because the project directory name contains a colon, which prevents creating a venv inside it.
- Normal run:

```bash
../../.venv-flappy-bird/bin/python main.py --generations 100
```

- Validation mode currently prints a validation message and does not open a window:

```bash
../../.venv-flappy-bird/bin/python main.py --test
```

## Global game constants

- Window: `500 x 800` pixels.
- Frame rate: `30 FPS`.
- Ground height: `90` pixels; ground begins at `y = 710`.
- Bird x-position: fixed at `x = 100`.
- Bird radius: `16` pixels; collision uses a `32 x 32` Pygame rectangle centered on the bird.
- Pipe width: `70` pixels.
- Pipe gap: `175` pixels.
- Pipe speed: `5` pixels per frame to the left.
- Bird gravity: `+0.8` velocity per frame.
- Jump impulse: velocity becomes `-8.5`.
- Maximum downward velocity: `12`.
- Generation time limit: `900` frames, or about `30` seconds at 30 FPS.
- Population size: `50`, configured in NEAT.

## Main classes

### `Bird`

State:

- `x`, `y`: floating-point position; x is normally fixed.
- `velocity`: vertical speed.
- `alive`: whether this genome is still active in the current generation.
- `score`: survival reward accumulated for this bird.
- `pipes_passed`: count used for fitness.
- `rect`: Pygame rectangle synchronized to the floating-point position.

Methods:

- `_sync_rect()`: rounds position and updates the collision rectangle center.
- `jump()`: sets velocity to `-8.5` if alive.
- `move()`: applies gravity, clamps downward speed to `12`, moves the bird, and updates the rectangle.
- `draw()`: draws a circular yellow bird with wing, eye, and beak using Pygame shapes.

### `Pipe`

State:

- `x`: horizontal position shared by the top and bottom pipe.
- `gap_y`: randomized gap center between `150` and `GROUND_Y - 150`.
- `passed`: whether this pipe pair has crossed the bird and awarded score.

Properties and methods:

- `top_rect`: top obstacle from the top of the screen to the upper gap edge.
- `bottom_rect`: bottom obstacle from the lower gap edge to the ground.
- `horizontal_distance`: pipe right edge minus the fixed bird x-position.
- `move()`: subtracts `5` from x.
- `collides(bird)`: checks rectangle intersection with either pipe.
- `draw()`: draws the two pipes and darker/cap sections.

## Frame-by-frame simulation logic

For each generation:

1. Pygame initializes a `500 x 800` display and clock.
2. Every genome receives `fitness = 0`.
3. One `Bird` and one `FeedForwardNetwork` are created for every genome.
4. The pipe list begins with one pipe at `x = WIDTH + 80`.
5. Each frame is capped to 30 FPS and increments the frame counter.
6. Window close, Escape, or the in-game Stop button sets the global stop flag.
7. A new pipe is added when the last pipe moves left of `WIDTH - 230`.
8. Every living bird selects the next pipe and activates its network.
9. If the network's only output is greater than `0.5`, the bird jumps.
10. Bird physics runs, then the bird receives `+0.1` fitness for surviving the frame.
11. A bird dies if its rectangle touches the ceiling, ground, top pipe, or bottom pipe. Death applies `-1.0` fitness.
12. All pipes move left.
13. When a pipe pair's right edge passes the fixed bird x-position, it is marked passed, the displayed score increases by `1`, and each currently living bird receives `+5.0` fitness through `pipes_passed`/fitness updates.
14. Pipes that have moved sufficiently off-screen are removed.
15. The background, pipes, living birds, and HUD are drawn.
16. The frame is presented with `pygame.display.flip()`.
17. The generation ends when every bird dies, the time limit is reached, the window closes, Escape is pressed, or Stop is clicked.
18. Generation best fitness, average fitness, and improvement are calculated for the HUD.

## Neural-network inputs

`network_inputs()` returns five values in this exact order:

1. `bird.y / HEIGHT`: normalized bird vertical position.
2. `(bird.y - pipe.top_rect.bottom) / HEIGHT`: normalized signed distance from bird y to the bottom of the top pipe. Positive means the bird is below the top pipe edge.
3. `(bird.y - pipe.bottom_rect.top) / HEIGHT`: normalized signed distance from bird y to the top of the bottom pipe. Negative means the bird is above the bottom pipe edge.
4. `pipe.horizontal_distance / WIDTH`: normalized horizontal distance from the bird to the selected pipe's right edge.
5. `bird.velocity / MAX_FALL_SPEED`: normalized vertical velocity. It can be negative while rising and is capped at `1` while falling.

The NEAT config declares `num_inputs = 5` and `num_outputs = 1`.

## Neural-network action

- Network type: `neat.nn.FeedForwardNetwork`.
- Activation: tanh by default.
- One output is produced.
- Output greater than `0.5` means jump; otherwise no action.
- There is no explicit cooldown, so a bird can jump on consecutive frames if its network output remains above the threshold.
- The network is feed-forward and initially fully connected according to the config.

## Fitness function

For each genome during one generation:

- `+0.1` for every frame alive.
- `+5.0` for each pipe pair passed while the bird is alive.
- `-1.0` on collision or boundary death.
- The intended effective formula is approximately:

```text
fitness = survival_frames * 0.1 + passed_pipes * 5.0 - death_penalty
```

NEAT maximizes fitness because the config uses `fitness_criterion = max`.

## Pipe and collision rules

- Pipe gaps are randomized per pipe pair.
- The gap is fixed at `175` pixels.
- Collision is axis-aligned rectangle collision through `pygame.Rect.colliderect`.
- The bird is represented by a square collision box even though it is drawn as a circle.
- Ceiling collision is `bird.rect.top <= 0`.
- Ground collision is `bird.rect.bottom >= GROUND_Y`.
- Pipe collision checks every active pipe pair.
- There is no pixel-perfect mask collision.

## Generation and improvement metrics

Global state tracks:

- `CURRENT_GENERATION`: incremented before each NEAT evaluation ca
- `CURRENT_GENERATION`: incremented before each NEAT evaluation callback.
- `BEST_SCORE`: highest displayed pipe score encountered during the process.
- `LAST_GENERATION_BEST`: highest genome fitness from the most recently completed generation.
- `LAST_GENERATION_AVERAGE`: average genome fitness from the most recently completed generation.
- `LAST_GENERATION_IMPROVEMENT`: percentage change from the previous generation best fitness.
- `STOP_REQUESTED`: shared cancellation flag.

Improvement is calculated as:

```text
(current_generation_best - previous_generation_best)
/ abs(previous_generation_best) * 100
```

The first generation displays `--` before a previous result exists, and the first calculated comparison is `0.0%`.

The HUD contains:

- Current generation.
- Living birds out of population size.
- Current pipe score.
- Last generation best fitness.
- Last generation average fitness.
- Improvement percentage.
- A red Stop button with a hover color.

## Stop behavior

The player can stop by:

- Clicking the Stop button.
- Pressing Escape.
- Closing the window.

Stopping raises `TrainingStopped` after the current evaluation cleans up Pygame. `run_training()` catches it and keeps the best genome saved during completed generations.

## Genome persistence

- Output file: `best_bird.pkl` next to `main.py`.
- The winning genome is serialized with Python `pickle`.
- Saving occurs only when `BEST_SCORE > 50`.
- The file is ignored by Git.
- `--play-best` loads and replays the saved genome as a single bird.

## NEAT configuration

### Population

- Population size: `50`.
- Fitness criterion: maximum.
- Fitness threshold: `100000`.
- Extinction reset: disabled.

### Genome

- Activation: tanh only.
- Aggregation: sum only.
- Five inputs and one output.
- No hidden nodes initially.
- Feed-forward network.
- Fully connected initial topology.
- Connection add probability: `0.5`.
- Connection delete probability: `0.5`.
- Node add probability: `0.2`.
- Node delete probability: `0.2`.
- Bias mutation rate: `0.7`.
- Weight mutation rate: `0.8`.
- Weight mutation power: `0.5`.
- Single structural mutation is disabled.

### Species and reproduction

- Compatibility threshold: `3.0`.
- Maximum species stagnation: `20` generations.
- Species elitism: `2`.
- Reproduction elitism: `2`.
- Survival threshold: `0.2`.
- Minimum species size: `2`.

## Rendering and user interface

- Procedural background: sky, sun, clouds, ground, grass, and ground lines.
- Procedural bird and pipe graphics.
- Semi-transparent metric panels with rounded borders.
- Pygame font rendering for labels and values.
- The window title is `NEAT Flappy Bird`.
- The simulation is visual and runs all 50 birds on one screen.

## Hardware behavior

- NEAT network evaluation runs on the CPU.
- Pygame 2D drawing and collision checks are primarily CPU-side.
- No CUDA, PyTorch, TensorFlow, or GPU neural-network backend is used.
- A desktop compositor may use the GPU to present the window, but that does not accelerate training.
- GPU acceleration is probably unnecessary for a population of 50 tiny feed-forward networks; larger populations or parallel/headless simulations could benefit from multiprocessing or a specialized numerical backend.

## Implemented roadmap improvements

- `--headless` skips Pygame display, event, font, and frame-rate work for fast training.
- `--seed N` makes pipe generation and training randomness reproducible.
- `--checkpoint-interval N` writes NEAT checkpoints; `--resume FILE` continues from one.
- Inputs are clipped to `[-1, 1]` before entering the tanh network.
- The tanh action boundary is explicit and symmetric at `0.0`.
- Fitness weights are named constants: survival `0.05`, pipe pass `8.0`, death `-5.0`.
- Best-genome files contain the genome plus generation, fitness, pipe score, seed, and timestamp.
- `--play-best` replays the saved genome in a single-bird visual window.
- `test_main.py` covers deterministic pipes, input bounds, gravity, collision, and improvement math.

## Current limitations and possible improvement areas

These are areas to ask Claude to review rather than claims that they are already fixed:

1. The game only uses one fixed bird x-position, so horizontal bird movement is not learned.
2. The input distance is to the pipe right edge, not the pipe center or left edge; clarify which distance gives the best learning signal.
3. Input normalization is only approximate. Distances can be outside `[-1, 1]`, and tanh networks may saturate.
4. All birds share the same randomly generated pipe sequence within a generation, which is useful for comparison but can reduce environmental diversity.
5. The displayed score is global to a generation, while fitness is per bird. Decide whether score should be best-bird score, average score, or pipe count shared by the environment.
6. Best-genome metadata and replay are now implemented; a future improvement could add multi-run evaluation statistics.
7. There is no automated benchmark comparing multiple saved genomes across fixed seeds.
8. There are no automated unit tests for physics, pipe placement, collision, input normalization, fitness, or stop behavior.
9. Headless mode is implemented; a future render interval could support occasional visual monitoring during fast training.
10. Pygame is initialized and quit for every generation. Keeping one window alive across generations may reduce overhead and preserve smoother UI state.
11. The button is drawn and checked manually; a small UI abstraction would help if more controls are added.
12. The tanh output threshold is now `0.0`; compare it against other action designs through benchmark runs.
13. The generation time limit is fixed at 30 seconds. Consider ending based on progress, maximum fitness, or a configurable frame limit.
14. A `--seed` option is implemented; reproducibility still depends on keeping the same code, config, and dependency versions.
15. The current summary compares best fitness values, not pipe scores. Decide which metric is more meaningful for users.
16. The project does not use GPU acceleration. For this small problem, CPU is appropriate, but multiprocessing could be tested before adding GPU complexity.
17. Checkpoint/resume support is implemented; a future improvement could checkpoint UI state and benchmark metadata too.
18. The CLI now supports generations, headless mode, seed, checkpoint interval, resume, play-best, and test; population size and physics overrides remain config-level changes.
19. The code uses global state for HUD/training metrics. A training-state object would make the code easier to test and extend.
20. The game has no human-play mode, pause control, restart control, or separate trained-agent demonstration mode.

## Questions for Claude

Please review this project as a senior Python/game-AI engineer and answer:

1. What are the most important correctness issues in the physics, pipe generation, collision order, fitness accounting, and generation lifecycle?
2. Is the five-input state representation sufficient and well normalized for NEAT? What inputs should be added or changed?
3. Is the fitness function balanced, especially the `+0.1` survival reward, `+5` pipe reward, and `-1` death penalty?
4. Should the game use tanh with a `0.5` action threshold, sigmoid with `0.5`, or another action design?
5. Which NEAT configuration values are likely to improve learning speed and stability for this environment?
6. Should all birds share pipe layouts, or should each bird/environment receive independent layouts?
7. How should the project save, reload, replay, and evaluate the best genome correctly?
8. What is the cleanest way to add deterministic seeds, checkpointing, unit tests, and a headless benchmark mode?
9. Which optimizations matter most, and is multiprocessing more appropriate than GPU acceleration here?
10. How should the UI distinguish current run score, generation best score, historical best score, fitness, and improvement percentage?
11. Please propose a prioritized roadmap with low-risk fixes first and larger architectural changes later.
12. Include concrete code-level examples where useful, but preserve the current two-file requirement if possible.
