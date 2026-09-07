"""Neuroevolutionary Flappy Bird using pygame and neat-python.

Run normally for the visual simulation from the repository root:
    python -m flappy_bird.main

Run a project validation check:
    python -m flappy_bird.main --test
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import neat
import pygame

try:
    from common.hud import draw_training_hud
    from common.human_input import HumanInput
    from common.stop_button import StopButton
    from common.training_state import TrainingState
except ModuleNotFoundError:  # Support ``python flappy_bird/main.py`` from the repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.hud import draw_training_hud
    from common.human_input import HumanInput
    from common.stop_button import StopButton
    from common.training_state import TrainingState


WIDTH = 500
HEIGHT = 800
FPS = 30
GROUND_HEIGHT = 90
GROUND_Y = HEIGHT - GROUND_HEIGHT
BIRD_X = 100
BIRD_RADIUS = 16
PIPE_WIDTH = 70
PIPE_GAP = 175
PIPE_SPEED = 5
MAX_FALL_SPEED = 12
GENERATION_TIME_LIMIT = 30 * FPS
POPULATION_SIZE = 50
BEST_GENOME_PATH = Path(__file__).with_name("best_bird.pkl")
CONFIG_PATH = Path(__file__).with_name("config-feedforward.txt")
CHECKPOINT_PREFIX = str(Path(__file__).with_name("neat-checkpoint-"))
ACTION_THRESHOLD = 0.0
SURVIVAL_REWARD = 0.05
PIPE_REWARD = 8.0
DEATH_PENALTY = 5.0

SKY = (117, 205, 238)
GROUND = (218, 177, 92)
GRASS = (95, 184, 76)
PIPE = (54, 177, 74)
PIPE_DARK = (36, 125, 52)
BIRD = (248, 198, 54)
BIRD_WING = (232, 145, 35)
INK = (30, 43, 48)
WHITE = (255, 255, 255)
PANEL = (247, 251, 246)
PANEL_EDGE = (205, 224, 211)
ACCENT = (27, 112, 91)
STOP = (190, 67, 61)
STOP_HOVER = (215, 78, 69)


class TrainingStopped(Exception):
    """Raised when the player clicks the in-game stop control."""


def clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def percentage_improvement(current: float, previous: float | None) -> float:
    if previous is None or previous == 0:
        return 0.0
    return (current - previous) / abs(previous) * 100.0


class Bird:
    """A player controlled by one NEAT genome."""

    def __init__(self, x: float = BIRD_X, y: float = HEIGHT / 2) -> None:
        self.x = x
        self.y = y
        self.velocity = 0.0
        self.alive = True
        self.score = 0.0
        self.pipes_passed = 0
        self.rect = pygame.Rect(0, 0, BIRD_RADIUS * 2, BIRD_RADIUS * 2)
        self._sync_rect()

    def _sync_rect(self) -> None:
        self.rect.center = (round(self.x), round(self.y))

    def jump(self) -> None:
        if self.alive:
            self.velocity = -8.5

    def move(self) -> None:
        self.velocity = min(self.velocity + 0.8, MAX_FALL_SPEED)
        self.y += self.velocity
        self._sync_rect()

    def draw(
        self,
        surface: pygame.Surface,
        *,
        body_color: tuple[int, int, int] = BIRD,
        wing_color: tuple[int, int, int] = BIRD_WING,
    ) -> None:
        pygame.draw.circle(surface, body_color, self.rect.center, BIRD_RADIUS)
        wing = pygame.Rect(self.rect.left + 2, self.rect.centery + 2, 15, 8)
        pygame.draw.ellipse(surface, wing_color, wing)
        pygame.draw.circle(surface, WHITE, (self.rect.right - 5, self.rect.top + 7), 5)
        pygame.draw.circle(surface, INK, (self.rect.right - 4, self.rect.top + 7), 2)
        pygame.draw.polygon(
            surface,
            (238, 116, 42),
            [(self.rect.right - 1, self.rect.centery - 2),
             (self.rect.right + 8, self.rect.centery + 2),
             (self.rect.right - 1, self.rect.centery + 5)],
        )


class Pipe:
    """A pair of rectangular obstacles with a fixed gap."""

    def __init__(self, x: float = WIDTH + 20, rng: random.Random | None = None) -> None:
        self.x = x
        self.gap_y = (rng or random).randint(150, GROUND_Y - 150)
        self.passed = False

    @property
    def top_rect(self) -> pygame.Rect:
        return pygame.Rect(round(self.x), 0, PIPE_WIDTH, self.gap_y - PIPE_GAP // 2)

    @property
    def bottom_rect(self) -> pygame.Rect:
        bottom_y = self.gap_y + PIPE_GAP // 2
        return pygame.Rect(round(self.x), bottom_y, PIPE_WIDTH, GROUND_Y - bottom_y)

    @property
    def horizontal_distance(self) -> float:
        return self.x + PIPE_WIDTH - BIRD_X

    def move(self) -> None:
        self.x -= PIPE_SPEED

    def collides(self, bird: Bird) -> bool:
        return bird.rect.colliderect(self.top_rect) or bird.rect.colliderect(self.bottom_rect)

    def draw(self, surface: pygame.Surface) -> None:
        top = self.top_rect
        bottom = self.bottom_rect
        pygame.draw.rect(surface, PIPE, top)
        pygame.draw.rect(surface, PIPE, bottom)
        pygame.draw.rect(surface, PIPE_DARK, (top.x, top.bottom - 8, top.width, 8))
        pygame.draw.rect(surface, PIPE_DARK, (bottom.x, bottom.y, bottom.width, 8))
        pygame.draw.rect(surface, PIPE, (top.x - 5, top.bottom - 18, PIPE_WIDTH + 10, 18))
        pygame.draw.rect(surface, PIPE, (bottom.x - 5, bottom.y, PIPE_WIDTH + 10, 18))


def draw_background(surface: pygame.Surface) -> None:
    surface.fill(SKY)
    pygame.draw.circle(surface, (255, 239, 169), (420, 105), 42)
    pygame.draw.ellipse(surface, (235, 248, 247), (55, 100, 105, 34))
    pygame.draw.ellipse(surface, (235, 248, 247), (300, 225, 125, 38))
    pygame.draw.rect(surface, GROUND, (0, GROUND_Y, WIDTH, GROUND_HEIGHT))
    pygame.draw.rect(surface, GRASS, (0, GROUND_Y, WIDTH, 12))
    for x in range(-20, WIDTH, 35):
        pygame.draw.line(surface, (189, 144, 71), (x, GROUND_Y + 28), (x + 16, HEIGHT), 2)


def network_inputs(bird: Bird, pipe: Pipe) -> tuple[float, ...]:
    """Return bounded normalized values in the same order as the NEAT config."""
    return (
        clamp((bird.y / HEIGHT) * 2.0 - 1.0),
        clamp((bird.y - pipe.top_rect.bottom) / (HEIGHT / 2.0)),
        clamp((bird.y - pipe.bottom_rect.top) / (HEIGHT / 2.0)),
        clamp(pipe.horizontal_distance / WIDTH),
        clamp(bird.velocity / MAX_FALL_SPEED),
    )


def next_pipe(pipes: list[Pipe], bird: Bird) -> Pipe:
    upcoming = [pipe for pipe in pipes if pipe.x + PIPE_WIDTH >= bird.x]
    return min(upcoming or pipes, key=lambda pipe: pipe.x)


def remove_dead(pipes: list[Pipe]) -> list[Pipe]:
    return [pipe for pipe in pipes if pipe.x + PIPE_WIDTH > -20]


def draw_hud(
    surface: pygame.Surface,
    fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font],
    alive_count: int,
    population_size: int,
    score: int,
    stop_hovered: bool,
) -> pygame.Rect:
    TRAINING_STATE.generation = CURRENT_GENERATION
    TRAINING_STATE.score = score
    button = StopButton(pygame.Rect(WIDTH - 136, 28, 112, 38))
    draw_training_hud(
        surface,
        fonts,
        TRAINING_STATE,
        alive_count,
        population_size,
        "NEAT FLAPPY BIRD",
        "LIVE EVOLUTION LAB",
        button,
    )
    return button.rect


def draw_simple_status(surface: pygame.Surface, font: pygame.font.Font, text: str) -> None:
    label = font.render(text, True, INK)
    surface.blit(label, (16, HEIGHT - 42))


def evaluate_genomes(
    genomes: list[tuple[int, neat.DefaultGenome]],
    config: neat.Config,
    render: bool = True,
    rng: random.Random | None = None,
) -> None:
    """Run one generation, with every genome represented on one screen."""
    global BEST_SCORE, LAST_GENERATION_BEST, LAST_GENERATION_AVERAGE
    global LAST_GENERATION_IMPROVEMENT, STOP_REQUESTED
    rng = rng or random
    screen = None
    clock = None
    fonts = None
    if render:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("NEAT Flappy Bird")
        clock = pygame.time.Clock()
        fonts = (
            pygame.font.Font(None, 25),
            pygame.font.Font(None, 23),
            pygame.font.Font(None, 16),
        )

    birds: list[Bird] = []
    networks: list[Any] = []
    for _, genome in genomes:
        genome.fitness = 0.0
        birds.append(Bird())
        networks.append(neat.nn.FeedForwardNetwork.create(genome, config))

    pipes = [Pipe(WIDTH + 80, rng)]
    frame = 0
    score = 0
    running = True
    stop_button = StopButton(pygame.Rect(WIDTH - 136, 28, 112, 38))

    while running and not STOP_REQUESTED and birds and frame < GENERATION_TIME_LIMIT:
        if clock is not None:
            clock.tick(FPS)
        frame += 1
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    STOP_REQUESTED = True
                    TRAINING_STATE.request_stop()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    STOP_REQUESTED = True
                    TRAINING_STATE.request_stop()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if stop_button.clicked(event):
                        running = False
                        STOP_REQUESTED = True
                        TRAINING_STATE.request_stop()

        if not running:
            break

        if pipes[-1].x < WIDTH - 230:
            pipes.append(Pipe(WIDTH + 30, rng))

        for index, bird in enumerate(birds):
            if not bird.alive:
                continue
            current_pipe = next_pipe(pipes, bird)
            if networks[index].activate(network_inputs(bird, current_pipe))[0] > ACTION_THRESHOLD:
                bird.jump()
            bird.move()
            bird.score += SURVIVAL_REWARD
            genomes[index][1].fitness = bird.score + bird.pipes_passed * PIPE_REWARD

            hit_boundary = bird.rect.top <= 0 or bird.rect.bottom >= GROUND_Y
            if hit_boundary or any(pipe.collides(bird) for pipe in pipes):
                bird.alive = False
                genomes[index][1].fitness -= DEATH_PENALTY

        for pipe in pipes:
            pipe.move()
            if not pipe.passed and pipe.x + PIPE_WIDTH < BIRD_X:
                pipe.passed = True
                score += 1
                BEST_SCORE = max(BEST_SCORE, score)
                for index, bird in enumerate(birds):
                    if bird.alive:
                        bird.pipes_passed += 1
                        genomes[index][1].fitness += PIPE_REWARD
        pipes = remove_dead(pipes)

        if render and screen is not None and fonts is not None:
            draw_background(screen)
            for pipe in pipes:
                pipe.draw(screen)
            for bird in birds:
                if bird.alive:
                    bird.draw(screen)

            alive_count = sum(bird.alive for bird in birds)
            draw_hud(
                screen,
                fonts,
                alive_count,
                len(birds),
                score,
                stop_button.rect.collidepoint(pygame.mouse.get_pos()),
            )
            pygame.display.flip()

        if not any(bird.alive for bird in birds):
            break

    generation_fitness = [genome.fitness for _, genome in genomes]
    generation_best = max(generation_fitness, default=0.0)
    generation_average = sum(generation_fitness) / len(generation_fitness) if generation_fitness else 0.0
    LAST_GENERATION_IMPROVEMENT = percentage_improvement(generation_best, LAST_GENERATION_BEST)
    LAST_GENERATION_BEST = generation_best
    LAST_GENERATION_AVERAGE = generation_average
    TRAINING_STATE.generation = CURRENT_GENERATION
    TRAINING_STATE.score = score
    TRAINING_STATE.record_generation(generation_best, generation_average)
    if render:
        pygame.quit()
    if STOP_REQUESTED:
        raise TrainingStopped


def save_best_genome(
    winner: neat.DefaultGenome,
    score: int,
    generation: int,
    fitness: float,
    seed: int | None,
) -> None:
    global BEST_GENOME_FITNESS
    payload = {
        "genome": copy.deepcopy(winner),
        "generation": generation,
        "fitness": fitness,
        "pipe_score": score,
        "seed": seed,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with BEST_GENOME_PATH.open("wb") as file:
        pickle.dump(payload, file)
    BEST_GENOME_FITNESS = fitness
    print(
        f"Saved best genome to {BEST_GENOME_PATH} "
        f"(fitness: {fitness:.1f}, score: {score})"
    )


def load_best_genome() -> tuple[neat.DefaultGenome, dict[str, Any]]:
    with BEST_GENOME_PATH.open("rb") as file:
        payload = pickle.load(file)
    if isinstance(payload, dict) and "genome" in payload:
        return payload["genome"], payload
    return payload, {"fitness": None, "pipe_score": None}


def play_best(config_path: Path) -> None:
    genome, metadata = load_best_genome()
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT Flappy Bird - Best Genome")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    bird = Bird()
    pipes = [Pipe(WIDTH + 80)]
    score = 0
    running = True

    while running and bird.alive:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
        if not running:
            break
        if pipes[-1].x < WIDTH - 230:
            pipes.append(Pipe(WIDTH + 30))
        current_pipe = next_pipe(pipes, bird)
        if network.activate(network_inputs(bird, current_pipe))[0] > ACTION_THRESHOLD:
            bird.jump()
        bird.move()
        if bird.rect.top <= 0 or bird.rect.bottom >= GROUND_Y:
            bird.alive = False
        for pipe in pipes:
            if pipe.collides(bird):
                bird.alive = False
            pipe.move()
            if not pipe.passed and pipe.x + PIPE_WIDTH < BIRD_X:
                pipe.passed = True
                score += 1
        pipes = remove_dead(pipes)
        draw_background(screen)
        for pipe in pipes:
            pipe.draw(screen)
        if bird.alive:
            bird.draw(screen)
        label = font.render(
            f"Best genome | score: {score} | fitness: {metadata.get('fitness', '--')}",
            True,
            INK,
        )
        screen.blit(label, (16, HEIGHT - 42))
        pygame.display.flip()
    pygame.quit()


def play_against_ai(config_path: Path) -> None:
    """Play as a human beside the frozen best genome on shared pipes."""
    genome, metadata = load_best_genome()
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    seed = metadata.get("seed")
    if seed is None:
        # Older genome files may not contain a seed; keep replay deterministic.
        seed = 0
    rng = random.Random(seed)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT Flappy Bird - You vs Frozen AI")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    result_font = pygame.font.Font(None, 32)
    input_state = HumanInput()
    human_color = (238, 116, 42)
    ai_color = (55, 100, 190)
    human = Bird()
    ai = Bird()
    pipes = [Pipe(WIDTH + 80, rng)]
    human_score = 0
    ai_score = 0
    running = True

    while running:
        clock.tick(FPS)
        events = pygame.event.get()
        input_state.handle(events)
        if input_state.quit_requested:
            running = False
        if input_state.restart_requested:
            rng = random.Random(seed)
            human = Bird()
            ai = Bird()
            pipes = [Pipe(WIDTH + 80, rng)]
            human_score = 0
            ai_score = 0
            input_state.paused = False
        if not running:
            break

        if not input_state.paused and (human.alive or ai.alive):
            if pipes[-1].x < WIDTH - 230:
                pipes.append(Pipe(WIDTH + 30, rng))

            if human.alive and input_state.flap_requested:
                human.jump()
            if human.alive:
                human.move()

            if ai.alive:
                current_pipe = next_pipe(pipes, ai)
                if network.activate(network_inputs(ai, current_pipe))[0] > ACTION_THRESHOLD:
                    ai.jump()
                ai.move()

            for bird in (human, ai):
                if bird.alive and (
                    bird.rect.top <= 0
                    or bird.rect.bottom >= GROUND_Y
                    or any(pipe.collides(bird) for pipe in pipes)
                ):
                    bird.alive = False

            for pipe in pipes:
                pipe.move()
                if not pipe.passed and pipe.x + PIPE_WIDTH < BIRD_X:
                    pipe.passed = True
                    human_score += int(human.alive)
                    ai_score += int(ai.alive)
            pipes = remove_dead(pipes)

        draw_background(screen)
        for pipe in pipes:
            pipe.draw(screen)
        if human.alive:
            human.draw(screen, body_color=human_color, wing_color=(205, 77, 25))
        if ai.alive:
            ai.draw(screen, body_color=ai_color, wing_color=(35, 62, 135))

        screen.blit(font.render(f"YOU (orange): {human_score}", True, human_color), (16, 24))
        screen.blit(font.render(f"AI (blue): {ai_score}", True, ai_color), (16, 50))
        screen.blit(
            font.render("SPACE flap | P pause | R restart | Esc quit", True, INK),
            (16, HEIGHT - 34),
        )

        if not human.alive and not ai.alive:
            if human_score > ai_score:
                result = "YOU WIN"
            elif ai_score > human_score:
                result = "AI WINS"
            else:
                result = "DRAW"
            banner = result_font.render(
                f"{result}   YOU {human_score} - AI {ai_score}   (R restart)",
                True,
                INK,
            )
            banner_rect = banner.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            pygame.draw.rect(screen, PANEL, banner_rect.inflate(24, 20), border_radius=10)
            pygame.draw.rect(
                screen,
                PANEL_EDGE,
                banner_rect.inflate(24, 20),
                2,
                border_radius=10,
            )
            screen.blit(banner, banner_rect)
        pygame.display.flip()

    print(f"Play result: you={human_score}, ai={ai_score}.")
    pygame.quit()


def play_human() -> None:
    """Run a keyboard-controlled game for physics and collision comparison."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT Flappy Bird - Human Mode")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    button_font = pygame.font.Font(None, 22)
    comparison_font = pygame.font.Font(None, 20)
    restart_button = pygame.Rect(WIDTH - 238, 18, 106, 36)
    exit_button = pygame.Rect(WIDTH - 122, 18, 106, 36)
    human_input = HumanInput()
    try:
        _, best_metadata = load_best_genome()
    except (OSError, pickle.PickleError, EOFError, AttributeError, KeyError):
        best_metadata = {}
    ai_best_score = int(best_metadata.get("pipe_score") or 0)
    ai_best_generation = best_metadata.get("generation", "--")
    bird = Bird()
    pipes = [Pipe(WIDTH + 80)]
    score = 0
    session_best_score = 0
    paused = False
    running = True

    while running:
        clock.tick(FPS)
        events = pygame.event.get()
        human_input.handle(events)
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if restart_button.collidepoint(event.pos):
                    bird = Bird()
                    pipes = [Pipe(WIDTH + 80)]
                    score = 0
                    paused = False
                    human_input.paused = False
                elif exit_button.collidepoint(event.pos):
                    running = False
        if human_input.quit_requested:
            running = False
        paused = human_input.paused
        if human_input.restart_requested:
            bird = Bird()
            pipes = [Pipe(WIDTH + 80)]
            score = 0
            paused = False
            human_input.paused = False
        if human_input.flap_requested and bird.alive and not paused:
            bird.jump()
        if not running:
            break
        if not paused and bird.alive:
            if pipes[-1].x < WIDTH - 230:
                pipes.append(Pipe(WIDTH + 30))
            bird.move()
            if bird.rect.top <= 0 or bird.rect.bottom >= GROUND_Y:
                bird.alive = False
            for pipe in pipes:
                if pipe.collides(bird):
                    bird.alive = False
                pipe.move()
                if not pipe.passed and pipe.x + PIPE_WIDTH < BIRD_X:
                    pipe.passed = True
                    score += 1
                    session_best_score = max(session_best_score, score)
            pipes = remove_dead(pipes)

        draw_background(screen)
        for pipe in pipes:
            pipe.draw(screen)
        if bird.alive:
            bird.draw(screen)
        score_label = font.render(
            f"SCORE: {score}    SESSION BEST: {session_best_score}",
            True,
            INK,
        )
        screen.blit(score_label, (16, 28))
        status = "PAUSED - P resume | R restart" if paused else "SPACE jump | P pause | R restart"
        if not bird.alive:
            status = f"Game over: {score} pipes | R restart | Esc quit"
        info_panel = pygame.Surface((WIDTH - 24, 76), pygame.SRCALPHA)
        info_panel.fill((*PANEL, 225))
        pygame.draw.rect(info_panel, PANEL_EDGE, info_panel.get_rect(), 2, border_radius=12)
        screen.blit(info_panel, (12, HEIGHT - 88))
        draw_simple_status(screen, font, status)
        if ai_best_score:
            if score <= ai_best_score:
                comparison = (
                    f"AI reached this position in Gen {ai_best_generation} "
                    f"(AI best: {ai_best_score})"
                )
            else:
                comparison = (
                    f"You passed AI best! AI record: {ai_best_score} "
                    f"(Gen {ai_best_generation})"
                )
        else:
            comparison = "No saved AI record yet - train the AI first"
        comparison_label = comparison_font.render(comparison, True, INK)
        screen.blit(comparison_label, (16, HEIGHT - 68))
        mouse_position = pygame.mouse.get_pos()
        for button, label in (
            (restart_button, "RESTART"),
            (exit_button, "EXIT"),
        ):
            hovered = button.collidepoint(mouse_position)
            color = ACCENT if hovered else (45, 91, 80)
            pygame.draw.rect(screen, color, button, border_radius=8)
            button_label = button_font.render(label, True, WHITE)
            screen.blit(button_label, button_label.get_rect(center=button.center))
        pygame.display.flip()
    print(
        f"Human session best: {session_best_score} pipes. "
        f"AI record: {ai_best_score} pipes, reached in generation {ai_best_generation}."
    )
    pygame.quit()


CURRENT_GENERATION = 0
BEST_SCORE = 0
BEST_GENOME_FITNESS: float | None = None
ALL_TIME_BEST_FITNESS: float | None = None
ALL_TIME_BEST_SCORE = 0
LAST_GENERATION_BEST: float | None = None
LAST_GENERATION_AVERAGE = 0.0
LAST_GENERATION_IMPROVEMENT = 0.0
RECENT_IMPROVEMENTS: deque[float] = deque(maxlen=5)
STOP_REQUESTED = False
TRAINING_STATE = TrainingState()


def run_training(
    config_path: Path,
    generations: int,
    render: bool = True,
    seed: int | None = None,
    resume: Path | None = None,
    checkpoint_interval: int = 10,
) -> None:
    global CURRENT_GENERATION, STOP_REQUESTED, BEST_GENOME_FITNESS
    global ALL_TIME_BEST_FITNESS, ALL_TIME_BEST_SCORE
    STOP_REQUESTED = False
    TRAINING_STATE.stop_requested = False
    if seed is not None:
        random.seed(seed)
    rng = random.Random(seed)
    if BEST_GENOME_PATH.exists():
        try:
            _, metadata = load_best_genome()
            BEST_GENOME_FITNESS = metadata.get("fitness")
            ALL_TIME_BEST_FITNESS = BEST_GENOME_FITNESS
            ALL_TIME_BEST_SCORE = metadata.get("pipe_score") or 0
        except (OSError, pickle.PickleError, EOFError, AttributeError, KeyError):
            BEST_GENOME_FITNESS = None
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )
    config.pop_size = POPULATION_SIZE
    if resume is not None:
        population = neat.Checkpointer.restore_checkpoint(str(resume))
        CURRENT_GENERATION = population.generation
    else:
        population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    if checkpoint_interval > 0:
        population.add_reporter(
            neat.Checkpointer(
                checkpoint_interval,
                filename_prefix=CHECKPOINT_PREFIX,
            )
        )

    def run_generation(genomes: list[tuple[int, neat.DefaultGenome]], neat_config: neat.Config) -> None:
        global CURRENT_GENERATION, ALL_TIME_BEST_FITNESS, ALL_TIME_BEST_SCORE
        CURRENT_GENERATION += 1
        evaluate_genomes(genomes, neat_config, render=render, rng=rng)
        generation_winner = max(genomes, key=lambda item: item[1].fitness)[1]
        if BEST_GENOME_FITNESS is None or generation_winner.fitness > BEST_GENOME_FITNESS:
            save_best_genome(
                generation_winner,
                BEST_SCORE,
                CURRENT_GENERATION,
                generation_winner.fitness,
                seed,
            )
        ALL_TIME_BEST_FITNESS = max(
            ALL_TIME_BEST_FITNESS or generation_winner.fitness,
            generation_winner.fitness,
        )
        ALL_TIME_BEST_SCORE = max(ALL_TIME_BEST_SCORE, BEST_SCORE)

    try:
        winner = population.run(run_generation, generations)
    except TrainingStopped:
        print("Training stopped by user.")
        return
    print(f"Training complete. Winner fitness: {winner.fitness:.1f}")


def main() -> None:
    global POPULATION_SIZE, GENERATION_TIME_LIMIT
    parser = argparse.ArgumentParser(description="Train NEAT agents to play Flappy Bird.")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations to train")
    parser.add_argument("--headless", action="store_true", help="Train without opening a Pygame window")
    parser.add_argument("--play-best", action="store_true", help="Replay the saved best genome")
    parser.add_argument("--play", action="store_true", help="Play beside the frozen best AI on shared pipes")
    parser.add_argument("--human", action="store_true", help="Play manually with Space, P, and R")
    parser.add_argument("--seed", type=int, help="Seed random generation for reproducible runs")
    parser.add_argument("--population", type=int, default=50, help="Population size for new runs")
    parser.add_argument("--time-limit", type=int, default=GENERATION_TIME_LIMIT, help="Maximum frames per generation")
    parser.add_argument("--resume", type=Path, help="Resume from a neat-checkpoint file")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save a NEAT checkpoint every N generations; use 0 to disable",
    )
    parser.add_argument("--test", action="store_true", help="Validate imports and configuration without opening a game")
    args = parser.parse_args()

    if args.test:
        print("Flappy Bird project files and Python syntax are valid.")
        return

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing NEAT configuration: {CONFIG_PATH}")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if args.play_best:
        if not BEST_GENOME_PATH.exists():
            raise FileNotFoundError(f"No saved genome found at {BEST_GENOME_PATH}")
        play_best(CONFIG_PATH)
        return
    if args.play:
        if not BEST_GENOME_PATH.exists():
            raise FileNotFoundError(
                f"No saved genome found at {BEST_GENOME_PATH}; train the AI first"
            )
        play_against_ai(CONFIG_PATH)
        return
    if args.human:
        play_human()
        return
    if args.population < 2:
        raise ValueError("--population must be at least 2")
    POPULATION_SIZE = args.population
    GENERATION_TIME_LIMIT = max(1, args.time_limit)
    run_training(
        CONFIG_PATH,
        max(1, args.generations),
        render=not args.headless,
        seed=args.seed,
        resume=args.resume,
        checkpoint_interval=max(0, args.checkpoint_interval),
    )


if __name__ == "__main__":
    main()
