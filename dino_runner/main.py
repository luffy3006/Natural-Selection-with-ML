"""NEAT endless runner with visual, headless, and human-play modes."""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import neat
import pygame

try:
    from common.hud import draw_training_hud
    from common.human_input import HumanInput
    from common.stop_button import StopButton
    from common.training_state import TrainingState
except ModuleNotFoundError:  # Support ``python dino_runner/main.py`` from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.hud import draw_training_hud
    from common.human_input import HumanInput
    from common.stop_button import StopButton
    from common.training_state import TrainingState


WIDTH = 700
HEIGHT = 400
FPS = 30
GROUND_Y = 330
DINO_X = 100
DINO_WIDTH = 34
DINO_HEIGHT = 48
DUCK_HEIGHT = 28
OBSTACLE_WIDTH = 24
BASE_OBSTACLE_SPEED = 7.0
OBSTACLE_SPEED = BASE_OBSTACLE_SPEED  # Compatibility alias for callers using the old constant.
MAX_OBSTACLE_SPEED = 13.0
SPEED_RAMP_PER_FRAME = 0.006
SPEED_RAMP_MAX = MAX_OBSTACLE_SPEED - BASE_OBSTACLE_SPEED
BIRD_OBSTACLE_WIDTH = 34
BIRD_OBSTACLE_HEIGHT = 24
BIRD_OBSTACLE_Y = GROUND_Y - 52
GRAVITY = 1.0
JUMP_SPEED = -15.0
MAX_FALL_SPEED = 16.0
GENERATION_TIME_LIMIT = 30 * FPS
POPULATION_SIZE = 50
ACTION_THRESHOLD = 0.0
SURVIVAL_REWARD = 0.05
OBSTACLE_REWARD = 5.0
DEATH_PENALTY = 5.0
CACTUS = -1.0
BIRD = 1.0
RUNNING = "running"
JUMPING = "jumping"
DUCKING = "ducking"
INPUT_ORDER = (
    "dino_state (-1 ducking, 0 running, +1 jumping)",
    "normalized distance to next obstacle",
    "obstacle type (-1 cactus, +1 bird)",
    "normalized current speed",
    "normalized distance to second obstacle (or 1.0)",
)
BEST_GENOME_PATH = Path(__file__).with_name("best_dino.pkl")
CONFIG_PATH = Path(__file__).with_name("config-feedforward.txt")
CHECKPOINT_PREFIX = str(Path(__file__).with_name("neat-checkpoint-"))

SKY = (186, 225, 246)
GROUND = (218, 177, 92)
GRASS = (95, 184, 76)
INK = (30, 43, 48)
OBSTACLE = (54, 129, 74)
PANEL = (247, 251, 246)
PANEL_EDGE = (205, 224, 211)
ACCENT = (27, 112, 91)
WHITE = (255, 255, 255)


class TrainingStopped(Exception):
    """Raised after a visual training session is cancelled."""


def clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def percentage_improvement(current: float, previous: float | None) -> float:
    if previous is None or previous == 0:
        return 0.0
    return (current - previous) / abs(previous) * 100.0


class Dino:
    RUNNING = RUNNING
    JUMPING = JUMPING
    DUCKING = DUCKING

    def __init__(self, x: float = DINO_X, y: float = GROUND_Y - DINO_HEIGHT) -> None:
        self.x = x
        self.y = y
        self.velocity = 0.0
        self.on_ground = True
        self.state = RUNNING
        self.alive = True
        self.score = 0.0
        self.obstacles_passed = 0
        self.rect = pygame.Rect(0, 0, DINO_WIDTH, DINO_HEIGHT)
        self._sync_rect()

    def _sync_rect(self) -> None:
        height = DUCK_HEIGHT if self.state == DUCKING else DINO_HEIGHT
        top = GROUND_Y - height if self.state == DUCKING and self.on_ground else self.y
        self.rect = pygame.Rect(round(self.x), round(top), DINO_WIDTH, height)
        if self.state == DUCKING and self.on_ground:
            self.y = float(self.rect.y)

    def jump(self) -> None:
        if self.alive and self.on_ground:
            self.state = JUMPING
            self.velocity = JUMP_SPEED
            self.on_ground = False
            self._sync_rect()

    def duck(self) -> None:
        if self.alive and self.on_ground:
            self.state = DUCKING
            self.velocity = 0.0
            self._sync_rect()

    def stand(self) -> None:
        if self.alive and self.on_ground:
            self.state = RUNNING
            self._sync_rect()

    def move(self) -> None:
        if self.state == DUCKING and self.on_ground:
            self._sync_rect()
            return
        self.velocity = min(self.velocity + GRAVITY, MAX_FALL_SPEED)
        self.y += self.velocity
        if self.y >= GROUND_Y - DINO_HEIGHT:
            self.y = GROUND_Y - DINO_HEIGHT
            self.velocity = 0.0
            self.on_ground = True
            if self.state == JUMPING:
                self.state = RUNNING
        self._sync_rect()

    def draw(self, surface: pygame.Surface, *, color: tuple[int, int, int] = INK) -> None:
        pygame.draw.rect(surface, color, self.rect, border_radius=7)
        pygame.draw.rect(surface, (104, 174, 86), (self.rect.x + 7, self.rect.y + 5, 20, max(8, self.rect.height - 16)), border_radius=4)
        pygame.draw.circle(surface, WHITE, (self.rect.right - 8, self.rect.top + 12), 4)
        pygame.draw.circle(surface, color, (self.rect.right - 8, self.rect.top + 12), 2)
        if self.state != DUCKING:
            pygame.draw.line(surface, color, (self.rect.left + 9, self.rect.bottom), (self.rect.left + 7, self.rect.bottom + 7), 4)
            pygame.draw.line(surface, color, (self.rect.right - 9, self.rect.bottom), (self.rect.right - 7, self.rect.bottom + 7), 4)


class Obstacle:
    """A cactus on the ground or a low bird that can only be ducked under."""

    def __init__(
        self,
        x: float = WIDTH + 40,
        rng: random.Random | None = None,
        obstacle_type: str | float | None = None,
        kind: str | float | None = None,
    ) -> None:
        self.x = x
        generator = rng or random
        if obstacle_type is None:
            obstacle_type = kind
        if obstacle_type is None:
            obstacle_type = generator.choice(("cactus", "bird"))
        if obstacle_type in ("bird", BIRD, 1, 1.0):
            self.obstacle_type = "bird"
            self.type_value = BIRD
            self.height = BIRD_OBSTACLE_HEIGHT
        else:
            self.obstacle_type = "cactus"
            self.type_value = CACTUS
            self.height = generator.randint(30, 68)
        self.kind = self.obstacle_type
        self.passed = False

    @property
    def is_bird(self) -> bool:
        return self.obstacle_type == "bird"

    @property
    def type(self) -> float:
        """Numeric input encoding retained for simple callers and tests."""
        return self.type_value

    @property
    def rect(self) -> pygame.Rect:
        if self.obstacle_type == "bird":
            return pygame.Rect(round(self.x), BIRD_OBSTACLE_Y, BIRD_OBSTACLE_WIDTH, BIRD_OBSTACLE_HEIGHT)
        return pygame.Rect(round(self.x), GROUND_Y - self.height, OBSTACLE_WIDTH, self.height)

    @property
    def horizontal_distance(self) -> float:
        return self.x + self.rect.width - DINO_X

    def move(self, speed: float = OBSTACLE_SPEED) -> None:
        self.x -= speed

    def collides(self, dino: Dino) -> bool:
        return dino.rect.colliderect(self.rect)

    def draw(self, surface: pygame.Surface) -> None:
        rect = self.rect
        if self.obstacle_type == "bird":
            pygame.draw.ellipse(surface, (165, 71, 85), rect)
            pygame.draw.line(surface, (111, 47, 67), (rect.left + 4, rect.centery), (rect.right - 4, rect.centery), 4)
            pygame.draw.circle(surface, WHITE, (rect.right - 7, rect.top + 8), 3)
            return
        pygame.draw.rect(surface, OBSTACLE, rect, border_radius=4)
        pygame.draw.rect(surface, (36, 96, 52), (rect.x - 4, rect.y + 8, rect.width + 8, 8), border_radius=3)


def current_speed(frame: int) -> float:
    # Gradually ramp the scrolling speed so agents learn timing, not one fixed pace.
    return min(MAX_OBSTACLE_SPEED, BASE_OBSTACLE_SPEED + frame * SPEED_RAMP_PER_FRAME)


def state_value(dino: Dino) -> float:
    return {DUCKING: -1.0, RUNNING: 0.0, JUMPING: 1.0}[dino.state]


def network_inputs(
    dino: Dino,
    obstacle: Obstacle,
    second_obstacle: Obstacle | None = None,
    speed: float = BASE_OBSTACLE_SPEED,
) -> tuple[float, ...]:
    """Return inputs in the documented ``INPUT_ORDER`` (always normalized)."""
    second_distance = (
        clamp(second_obstacle.horizontal_distance / WIDTH)
        if second_obstacle is not None
        else 1.0
    )
    return (
        state_value(dino),
        clamp(obstacle.horizontal_distance / WIDTH),
        obstacle.type_value,
        clamp(speed / MAX_OBSTACLE_SPEED),
        second_distance,
    )


def action_from_outputs(outputs: tuple[float, ...] | list[float]) -> str | None:
    """Map [jump, duck] outputs to an action; jump wins ties."""
    jump_output = outputs[0] if outputs else -1.0
    duck_output = outputs[1] if len(outputs) > 1 else -1.0
    if jump_output > ACTION_THRESHOLD:
        return "jump"
    if duck_output > ACTION_THRESHOLD:
        return "duck"
    return None


def nearby_obstacles(obstacles: list[Obstacle], dino: Dino) -> tuple[Obstacle, Obstacle | None]:
    ordered = sorted(
        (obstacle for obstacle in obstacles if obstacle.x + obstacle.rect.width >= dino.x),
        key=lambda obstacle: obstacle.x,
    )
    if not ordered:
        ordered = sorted(obstacles, key=lambda obstacle: obstacle.x)
    return ordered[0], ordered[1] if len(ordered) > 1 else None


def apply_action(dino: Dino, action: str | None) -> None:
    if action == "jump":
        dino.jump()
    elif action == "duck":
        dino.duck()
    elif dino.on_ground:
        dino.stand()


def next_obstacle(obstacles: list[Obstacle], dino: Dino) -> Obstacle:
    upcoming = [obstacle for obstacle in obstacles if obstacle.x + OBSTACLE_WIDTH >= dino.x]
    return min(upcoming or obstacles, key=lambda obstacle: obstacle.x)


def remove_dead(obstacles: list[Obstacle]) -> list[Obstacle]:
    return [obstacle for obstacle in obstacles if obstacle.x + OBSTACLE_WIDTH > -20]


def draw_background(surface: pygame.Surface) -> None:
    surface.fill(SKY)
    pygame.draw.circle(surface, (255, 239, 169), (590, 72), 34)
    pygame.draw.ellipse(surface, (235, 248, 247), (90, 95, 130, 34))
    pygame.draw.ellipse(surface, (235, 248, 247), (385, 150, 120, 30))
    pygame.draw.rect(surface, GROUND, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
    pygame.draw.rect(surface, GRASS, (0, GROUND_Y, WIDTH, 10))
    for x in range(-20, WIDTH, 35):
        pygame.draw.line(surface, (189, 144, 71), (x, GROUND_Y + 25), (x + 16, HEIGHT), 2)


TRAINING_STATE = TrainingState()
CURRENT_GENERATION = 0
STOP_REQUESTED = False
BEST_SCORE = 0
BEST_GENOME_FITNESS: float | None = None


def evaluate_genomes(
    genomes: list[tuple[int, neat.DefaultGenome]],
    config: neat.Config,
    *,
    render: bool = True,
    rng: random.Random | None = None,
    time_limit: int = GENERATION_TIME_LIMIT,
) -> None:
    global BEST_SCORE, STOP_REQUESTED
    rng = rng or random
    screen = None
    clock = None
    fonts = None
    if render:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NEAT Dino Runner")
        clock = pygame.time.Clock()
        fonts = (pygame.font.Font(None, 25), pygame.font.Font(None, 23), pygame.font.Font(None, 16))

    dinos: list[Dino] = []
    networks: list[Any] = []
    for _, genome in genomes:
        genome.fitness = 0.0
        dinos.append(Dino())
        networks.append(neat.nn.FeedForwardNetwork.create(genome, config))
    obstacles = [Obstacle(WIDTH + 40, rng)]
    frame = 0
    score = 0
    running = True
    stop_button = StopButton(pygame.Rect(WIDTH - 136, 28, 112, 38))

    while running and not STOP_REQUESTED and dinos and frame < time_limit:
        if clock is not None:
            clock.tick(FPS)
        frame += 1
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                    STOP_REQUESTED = True
                    TRAINING_STATE.request_stop()
                elif stop_button.clicked(event):
                    running = False
                    STOP_REQUESTED = True
                    TRAINING_STATE.request_stop()
        if not running:
            break
        if obstacles[-1].x < WIDTH - 260:
            obstacles.append(Obstacle(WIDTH + rng.randint(20, 100), rng))
        speed = current_speed(frame)

        for index, dino in enumerate(dinos):
            if not dino.alive:
                continue
            first, second = nearby_obstacles(obstacles, dino)
            action = action_from_outputs(
                networks[index].activate(network_inputs(dino, first, second, speed))
            )
            apply_action(dino, action)
            dino.move()
            dino.score += SURVIVAL_REWARD
            genomes[index][1].fitness = dino.score
            if any(obstacle.collides(dino) for obstacle in obstacles):
                dino.alive = False
                genomes[index][1].fitness -= DEATH_PENALTY

        for obstacle in obstacles:
            obstacle.move(speed)
            if not obstacle.passed and obstacle.x + obstacle.rect.width < DINO_X:
                obstacle.passed = True
                score += 1
                BEST_SCORE = max(BEST_SCORE, score)
                for index, dino in enumerate(dinos):
                    if dino.alive:
                        dino.obstacles_passed += 1
                        dino.score += OBSTACLE_REWARD
                        genomes[index][1].fitness += OBSTACLE_REWARD
        obstacles = remove_dead(obstacles)

        if render and screen is not None and fonts is not None:
            draw_background(screen)
            for obstacle in obstacles:
                obstacle.draw(screen)
            for dino in dinos:
                if dino.alive:
                    dino.draw(screen)
            TRAINING_STATE.generation = CURRENT_GENERATION
            TRAINING_STATE.score = score
            draw_training_hud(
                screen, fonts, TRAINING_STATE, sum(dino.alive for dino in dinos), len(dinos),
                "NEAT DINO RUNNER", "LIVE EVOLUTION LAB", stop_button,
            )
            pygame.display.flip()
        if not any(dino.alive for dino in dinos):
            break

    fitnesses = [genome.fitness for _, genome in genomes]
    best = max(fitnesses, default=0.0)
    average = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
    TRAINING_STATE.generation = CURRENT_GENERATION
    TRAINING_STATE.score = score
    TRAINING_STATE.record_generation(best, average)
    if render:
        pygame.quit()
    if STOP_REQUESTED:
        raise TrainingStopped


def save_best_genome(winner: neat.DefaultGenome, generation: int, seed: int | None) -> None:
    global BEST_GENOME_FITNESS
    payload = {
        "genome": copy.deepcopy(winner),
        "generation": generation,
        "fitness": winner.fitness,
        "obstacle_score": BEST_SCORE,
        "pipe_score": BEST_SCORE,
        "seed": seed,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with BEST_GENOME_PATH.open("wb") as file:
        pickle.dump(payload, file)
    BEST_GENOME_FITNESS = winner.fitness
    print(f"Saved best dino genome to {BEST_GENOME_PATH} (fitness: {winner.fitness:.1f})")


def load_best_genome() -> tuple[neat.DefaultGenome, dict[str, Any]]:
    with BEST_GENOME_PATH.open("rb") as file:
        payload = pickle.load(file)
    if isinstance(payload, dict) and "genome" in payload:
        return payload["genome"], payload
    return payload, {}


def make_config(path: Path) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
        neat.DefaultStagnation, str(path),
    )


def play_best(config_path: Path) -> None:
    genome, metadata = load_best_genome()
    config = make_config(config_path)
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    play_network(config, network, title="NEAT Dino Runner - Best Genome", metadata=metadata)


def play_network(
    config: neat.Config,
    network: neat.nn.FeedForwardNetwork,
    *,
    title: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    del config  # The network has already been constructed from this config.
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    dino = Dino()
    metadata = metadata or {}
    rng = random.Random(metadata.get("seed"))
    obstacles = [Obstacle(WIDTH + 40, rng)]
    score = 0
    running = True
    frame = 0
    while running and dino.alive:
        clock.tick(FPS)
        frame += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        if obstacles[-1].x < WIDTH - 260:
            obstacles.append(Obstacle(WIDTH + rng.randint(20, 100), rng))
        speed = current_speed(frame)
        first, second = nearby_obstacles(obstacles, dino)
        apply_action(dino, action_from_outputs(network.activate(
            network_inputs(dino, first, second, speed)
        )))
        dino.move()
        for obstacle in obstacles:
            if obstacle.collides(dino):
                dino.alive = False
            obstacle.move(speed)
            if not obstacle.passed and obstacle.x + obstacle.rect.width < DINO_X:
                obstacle.passed = True
                score += 1
        obstacles = remove_dead(obstacles)
        draw_background(screen)
        for obstacle in obstacles:
            obstacle.draw(screen)
        if dino.alive:
            dino.draw(screen)
        screen.blit(font.render(f"Best genome | score: {score} | fitness: {metadata.get('fitness', '--')}", True, INK), (16, HEIGHT - 32))
        pygame.display.flip()
    pygame.quit()


def play_against_ai(config_path: Path) -> None:
    """Play beside a frozen best genome on one deterministic obstacle timeline."""
    genome, metadata = load_best_genome()
    config = make_config(config_path)
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    seed = metadata.get("seed")
    rng = random.Random(seed)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT Dino Runner - You vs Frozen AI")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    result_font = pygame.font.Font(None, 30)
    input_state = HumanInput()
    human, ai = Dino(), Dino()
    obstacles = [Obstacle(WIDTH + 40, rng)]
    human_score = ai_score = frame = 0
    running = True
    human_color, ai_color = (224, 112, 48), (57, 93, 181)

    while running:
        clock.tick(FPS)
        events = pygame.event.get()
        input_state.handle(events)
        if input_state.quit_requested:
            running = False
        if input_state.restart_requested:
            rng = random.Random(seed)
            human, ai = Dino(), Dino()
            obstacles = [Obstacle(WIDTH + 40, rng)]
            human_score = ai_score = frame = 0
            input_state.paused = False
        if not input_state.paused and (human.alive or ai.alive):
            frame += 1
            if obstacles[-1].x < WIDTH - 260:
                obstacles.append(Obstacle(WIDTH + rng.randint(20, 100), rng))
            speed = current_speed(frame)
            if human.alive:
                human_action = "jump" if input_state.jump_requested else (
                    "duck" if input_state.duck_requested else None
                )
                apply_action(human, human_action)
                human.move()
            if ai.alive:
                first, second = nearby_obstacles(obstacles, ai)
                apply_action(ai, action_from_outputs(network.activate(
                    network_inputs(ai, first, second, speed)
                )))
                ai.move()
            for obstacle in obstacles:
                if human.alive and obstacle.collides(human):
                    human.alive = False
                if ai.alive and obstacle.collides(ai):
                    ai.alive = False
                obstacle.move(speed)
                if not obstacle.passed and obstacle.x + obstacle.rect.width < DINO_X:
                    obstacle.passed = True
                    human_score += int(human.alive)
                    ai_score += int(ai.alive)
            obstacles = remove_dead(obstacles)

        draw_background(screen)
        for obstacle in obstacles:
            obstacle.draw(screen)
        if human.alive:
            human.draw(screen, color=human_color)
        if ai.alive:
            ai.draw(screen, color=ai_color)
        screen.blit(font.render(f"YOU (orange): {human_score}", True, human_color), (18, 18))
        screen.blit(font.render(f"FROZEN AI (blue): {ai_score}", True, ai_color), (18, 42))
        screen.blit(font.render("SPACE/UP jump | DOWN duck | P pause | R restart", True, INK), (18, HEIGHT - 25))
        if not human.alive or not ai.alive:
            winner = "YOU WIN" if human_score > ai_score else "AI WINS" if ai_score > human_score else "DRAW"
            banner = result_font.render(
                f"{winner}   YOU {human_score} - AI {ai_score}   (R restart)",
                True,
                INK,
            )
            screen.blit(banner, banner.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
        pygame.display.flip()
    print(f"Play result: you={human_score}, frozen_ai={ai_score}.")
    pygame.quit()


def play_human() -> None:
    """Compatibility alias for the side-by-side ``--play`` mode."""
    play_against_ai(CONFIG_PATH)


def run_training(
    config_path: Path,
    generations: int,
    *,
    render: bool,
    seed: int | None,
    population_size: int,
    time_limit: int,
    resume: Path | None,
    checkpoint_interval: int,
) -> None:
    global CURRENT_GENERATION, STOP_REQUESTED, POPULATION_SIZE, BEST_GENOME_FITNESS
    POPULATION_SIZE = population_size
    CURRENT_GENERATION = 0
    STOP_REQUESTED = False
    TRAINING_STATE.stop_requested = False
    if seed is not None:
        random.seed(seed)
    rng = random.Random(seed)
    config = make_config(config_path)
    config.pop_size = population_size
    population = neat.Checkpointer.restore_checkpoint(str(resume)) if resume else neat.Population(config)
    if resume:
        CURRENT_GENERATION = population.generation
    population.add_reporter(neat.StdOutReporter(render))
    population.add_reporter(neat.StatisticsReporter())
    if checkpoint_interval:
        population.add_reporter(neat.Checkpointer(checkpoint_interval, filename_prefix=CHECKPOINT_PREFIX))

    def run_generation(genomes: list[tuple[int, neat.DefaultGenome]], neat_config: neat.Config) -> None:
        global CURRENT_GENERATION
        CURRENT_GENERATION += 1
        evaluate_genomes(genomes, neat_config, render=render, rng=rng, time_limit=time_limit)
        winner = max((genome for _, genome in genomes), key=lambda genome: genome.fitness)
        if BEST_GENOME_FITNESS is None or winner.fitness > BEST_GENOME_FITNESS:
            save_best_genome(winner, CURRENT_GENERATION, seed)

    try:
        winner = population.run(run_generation, generations)
    except TrainingStopped:
        print("Training stopped by user.")
        return
    print(f"Training complete. Winner fitness: {winner.fitness:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NEAT agents to play Dino Runner.")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument("--time-limit", type=int, default=GENERATION_TIME_LIMIT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--play-best", action="store_true", help="Replay the saved best genome")
    parser.add_argument("--play", action="store_true", help="Play beside the frozen best AI on shared obstacles")
    parser.add_argument("--human", action="store_true", help="Alias for --play")
    parser.add_argument("--test", action="store_true", help="Validate imports and configuration")
    args = parser.parse_args()
    if args.test:
        make_config(CONFIG_PATH)
        print("Dino Runner project files and Python syntax are valid.")
        return
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing NEAT configuration: {CONFIG_PATH}")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if args.play_best:
        if not BEST_GENOME_PATH.exists():
            raise FileNotFoundError(f"No saved genome found at {BEST_GENOME_PATH}")
        play_best(CONFIG_PATH)
        return
    if args.play or args.human:
        if not BEST_GENOME_PATH.exists():
            raise FileNotFoundError(f"No saved genome found at {BEST_GENOME_PATH}; train first")
        play_against_ai(CONFIG_PATH)
        return
    if args.population < 2:
        raise ValueError("--population must be at least 2")
    run_training(
        CONFIG_PATH, max(1, args.generations), render=not args.headless, seed=args.seed,
        population_size=args.population, time_limit=max(1, args.time_limit),
        resume=args.resume, checkpoint_interval=max(0, args.checkpoint_interval),
    )


if __name__ == "__main__":
    main()
