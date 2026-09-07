"""NEAT car racer using ray sensors and continuous steering/throttle output."""

from __future__ import annotations

import argparse
import copy
import math
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
    from car_racer.track import make_track
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.hud import draw_training_hud
    from common.human_input import HumanInput
    from common.stop_button import StopButton
    from common.training_state import TrainingState
    from car_racer.track import make_track

WIDTH = 700
HEIGHT = 500
FPS = 30
TRACK = make_track()
MAX_SPEED = 6.0
MAX_TURN_RATE = 0.08
ACCELERATION = 0.16
FRICTION = 0.04
GENERATION_TIME_LIMIT = 30 * FPS
POPULATION_SIZE = 50
BEST_GENOME_PATH = Path(__file__).with_name('best_car.pkl')
CONFIG_PATH = Path(__file__).with_name('config-feedforward.txt')
CHECKPOINT_PREFIX = str(Path(__file__).with_name('neat-checkpoint-'))
RAY_ANGLES = (-math.radians(60), -math.radians(30), 0.0, math.radians(30), math.radians(60))
RAY_LENGTH = 150.0
SURVIVAL_REWARD = 0.05
DEATH_PENALTY = 10.0
PROGRESS_REWARD = 0.8
PANEL = (247, 251, 246)
PANEL_EDGE = (205, 224, 211)
INK = (30, 43, 48)
ACCENT = (27, 112, 91)
WHITE = (255,255,255)
AI_COLOR = (70, 130, 240)
HUMAN_COLOR = (235, 127, 52)
TRAINING_STATE = TrainingState()
STOP_REQUESTED = False
CURRENT_GENERATION = 0


def clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def percentage_improvement(current: float, previous: float | None) -> float:
    if previous is None or previous == 0:
        return 0.0
    return (current - previous) / abs(previous) * 100.0


class Car:
    def __init__(self, x: float = 350.0, y: float = 250.0, angle: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0.0
        self.alive = True
        self.score = 0.0
        self.progress = 0.0
        self.radius = 12

    def update(self, steering: float, throttle: float) -> None:
        self.speed += throttle * ACCELERATION
        self.speed = max(0.0, min(MAX_SPEED, self.speed))
        self.speed *= 1.0 - FRICTION
        if self.speed < 0.02:
            self.speed = 0.0
        self.angle += steering * MAX_TURN_RATE * (0.4 + self.speed / MAX_SPEED)
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))

    def draw(self, surface: pygame.Surface, color: tuple[int, int, int]) -> None:
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.radius)
        nose_x = self.x + math.cos(self.angle) * 18
        nose_y = self.y + math.sin(self.angle) * 18
        pygame.draw.line(surface, color, (int(self.x), int(self.y)), (int(nose_x), int(nose_y)), 3)


def sensor_distance(car: Car, angle_offset: float) -> float:
    max_distance = RAY_LENGTH
    angle = car.angle + angle_offset
    x = car.x
    y = car.y
    for step in range(int(max_distance) + 1):
        x = car.x + math.cos(angle) * step
        y = car.y + math.sin(angle) * step
        if not TRACK.is_on_track(x, y):
            return step / max_distance
    return 1.0


def network_inputs(car: Car) -> tuple[float, ...]:
    sensors = [sensor_distance(car, angle) for angle in RAY_ANGLES]
    speed_norm = clamp(car.speed / MAX_SPEED)
    return tuple(clamp(value) for value in sensors + [speed_norm])


def action_from_outputs(outputs: tuple[float, ...] | list[float]) -> tuple[float, float]:
    steering = outputs[0] if len(outputs) > 0 else 0.0
    throttle = outputs[1] if len(outputs) > 1 else 0.0
    return float(steering), float(throttle)


def draw_background(surface: pygame.Surface) -> None:
    surface.fill((181, 219, 233))
    TRACK.draw(surface, color=(72, 120, 160), line_width=28)
    TRACK.draw(surface, color=(245, 245, 245), line_width=4)


def save_best_genome(winner: neat.DefaultGenome, generation: int, fitness: float, seed: int | None) -> None:
    payload = {'genome': copy.deepcopy(winner), 'generation': generation, 'fitness': fitness, 'seed': seed, 'saved_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'config_path': str(CONFIG_PATH)}
    with BEST_GENOME_PATH.open('wb') as file:
        pickle.dump(payload, file)
    print(f'Saved best racer genome to {BEST_GENOME_PATH} (fitness: {fitness:.2f})')


def load_best_genome() -> tuple[neat.DefaultGenome, dict[str, Any]]:
    with BEST_GENOME_PATH.open('rb') as file:
        payload = pickle.load(file)
    if isinstance(payload, dict) and 'genome' in payload:
        return payload['genome'], payload
    return payload, {'fitness': None}


def make_config(path: Path) -> neat.Config:
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))


def play_best(config_path: Path) -> None:
    genome, metadata = load_best_genome()
    config = make_config(config_path)
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    car = Car()
    rng = random.Random(metadata.get('seed'))
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Car Racer - Best Genome')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        if not running:
            break
        outputs = network.activate(network_inputs(car))
        steering, throttle = action_from_outputs(outputs)
        car.update(steering, throttle)
        if not TRACK.is_on_track(car.x, car.y):
            running = False
        draw_background(screen)
        car.draw(screen, AI_COLOR)
        screen.blit(font.render(f'Best genome | progress: {car.progress:.2f} | speed: {car.speed:.2f}', True, INK), (16, HEIGHT - 28))
        pygame.display.flip()
    pygame.quit()


def play_human_vs_ai(config_path: Path) -> None:
    genome, metadata = load_best_genome()
    config = make_config(config_path)
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    rng = random.Random(metadata.get('seed'))
    human = Car(350.0, 250.0, 0.0)
    ai = Car(350.0, 250.0, 0.0)
    ai.x = 350.0 + 30
    ai.y = 250.0
    human_score = ai_score = 0.0
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Car Racer - You vs Frozen AI')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    result_font = pygame.font.Font(None, 30)
    input_state = HumanInput()
    running = True
    while running:
        clock.tick(FPS)
        events = pygame.event.get()
        input_state.handle(events)
        if input_state.quit_requested:
            running = False
        if input_state.restart_requested:
            human = Car(350.0, 250.0, 0.0)
            ai = Car(350.0, 250.0, 0.0)
            ai.x += 30
            human_score = ai_score = 0.0
        if not input_state.paused:
            steer = 0.0
            throttle = 0.0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                steer -= 1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                steer += 1.0
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                throttle += 1.0
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                throttle -= 0.8
            human.update(steer, throttle)
            ai_outputs = network.activate(network_inputs(ai))
            ai_steer, ai_throttle = action_from_outputs(ai_outputs)
            ai.update(ai_steer, ai_throttle)
            if not TRACK.is_on_track(human.x, human.y):
                human.alive = False
            if not TRACK.is_on_track(ai.x, ai.y):
                ai.alive = False
            if not human.alive or not ai.alive:
                running = False
        draw_background(screen)
        human.draw(screen, HUMAN_COLOR)
        ai.draw(screen, AI_COLOR)
        screen.blit(font.render(f'YOU {human_score:.0f}  :  {ai_score:.0f} AI', True, INK), (220, 20))
        screen.blit(font.render('WASD / arrows | P pause | R restart | Esc quit', True, INK), (110, HEIGHT - 24))
        pygame.display.flip()
    pygame.quit()


class TrainingStopped(Exception):
    pass


def evaluate_genomes(genomes: list[tuple[int, neat.DefaultGenome]], config: neat.Config, *, render: bool = True, rng: random.Random | None = None, time_limit: int = GENERATION_TIME_LIMIT) -> None:
    global STOP_REQUESTED, TRAINING_STATE
    rng = rng or random
    screen = None
    clock = None
    fonts = None
    if render:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('NEAT Car Racer')
        clock = pygame.time.Clock()
        fonts = (pygame.font.Font(None, 25), pygame.font.Font(None, 23), pygame.font.Font(None, 16))
    cars = []
    networks = {}
    for idx, (_, genome) in enumerate(genomes):
        genome.fitness = 0.0
        cars.append(Car())
        networks[idx] = neat.nn.FeedForwardNetwork.create(genome, config)
    stop_button = StopButton(pygame.Rect(WIDTH - 136, 28, 112, 38))
    frame = 0
    while frame < time_limit and cars:
        if clock is not None:
            clock.tick(FPS)
        frame += 1
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    STOP_REQUESTED = True
                    TRAINING_STATE.request_stop()
                    raise TrainingStopped
                elif stop_button.clicked(event):
                    STOP_REQUESTED = True
                    TRAINING_STATE.request_stop()
                    raise TrainingStopped
        if STOP_REQUESTED:
            raise TrainingStopped
        for idx, car in enumerate(cars):
            if not car.alive:
                continue
            outputs = networks[idx].activate(network_inputs(car))
            steering, throttle = action_from_outputs(outputs)
            car.update(steering, throttle)
            if not TRACK.is_on_track(car.x, car.y):
                car.alive = False
                genomes[idx][1].fitness -= DEATH_PENALTY
                continue
            car.progress = max(car.progress, (car.x * 0.001 + car.y * 0.001))
            car.score += SURVIVAL_REWARD
            genomes[idx][1].fitness = car.score + car.progress * 100.0
        if render and screen is not None:
            draw_background(screen)
            for car in cars:
                if car.alive:
                    car.draw(screen, AI_COLOR)
            TRAINING_STATE.generation = CURRENT_GENERATION
            TRAINING_STATE.score = int(max(car.progress for car in cars))
            draw_training_hud(screen, fonts, TRAINING_STATE, sum(car.alive for car in cars), len(cars), 'NEAT CAR RACER', 'CONTINUOUS CONTROL', stop_button)
            pygame.display.flip()
    if render and screen is not None:
        pygame.quit()
    best = max((genome.fitness for _, genome in genomes), default=0.0)
    average = sum(genome.fitness for _, genome in genomes) / max(len(genomes), 1)
    TRAINING_STATE.record_generation(best, average)


def run_training(config_path: Path, generations: int, render: bool = True, seed: int | None = None, resume: Path | None = None, checkpoint_interval: int = 10, population_size: int = POPULATION_SIZE, time_limit: int = GENERATION_TIME_LIMIT) -> None:
    global CURRENT_GENERATION, STOP_REQUESTED
    STOP_REQUESTED = False
    if seed is not None:
        random.seed(seed)
    rng = random.Random(seed)
    config = make_config(config_path)
    config.pop_size = population_size
    population = neat.Checkpointer.restore_checkpoint(str(resume)) if resume else neat.Population(config)
    if resume:
        CURRENT_GENERATION = population.generation
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    if checkpoint_interval > 0:
        population.add_reporter(neat.Checkpointer(checkpoint_interval, filename_prefix=CHECKPOINT_PREFIX))

    def run_generation(genomes: list[tuple[int, neat.DefaultGenome]], neat_config: neat.Config) -> None:
        global CURRENT_GENERATION
        CURRENT_GENERATION += 1
        evaluate_genomes(genomes, neat_config, render=render, rng=rng, time_limit=time_limit)
        best_genome = max(genomes, key=lambda item: item[1].fitness)[1]
        if best_genome.fitness > 0.0:
            save_best_genome(best_genome, CURRENT_GENERATION, best_genome.fitness, seed)

    try:
        winner = population.run(run_generation, generations)
    except TrainingStopped:
        print('Training stopped by user.')
        return
    print(f'Training complete. Winner fitness: {winner.fitness:.2f}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Train NEAT agents to race a track.')
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--population', type=int, default=POPULATION_SIZE)
    parser.add_argument('--time-limit', type=int, default=GENERATION_TIME_LIMIT)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--play-best', action='store_true')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        make_config(CONFIG_PATH)
        print('Car Racer project files and Python syntax are valid.')
        return
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f'Missing NEAT configuration: {CONFIG_PATH}')
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    if args.play_best:
        if not BEST_GENOME_PATH.exists():
            raise FileNotFoundError(f'No saved genome found at {BEST_GENOME_PATH}')
        play_best(CONFIG_PATH)
        return
    if args.play:
        if not BEST_GENOME_PATH.exists():
            raise FileNotFoundError(f'No saved genome found at {BEST_GENOME_PATH}; train the AI first')
        play_human_vs_ai(CONFIG_PATH)
        return
    run_training(CONFIG_PATH, max(1, args.generations), render=not args.headless, seed=args.seed, population_size=max(2, args.population), time_limit=max(1, args.time_limit), resume=args.resume, checkpoint_interval=max(0, args.checkpoint_interval))


if __name__ == '__main__':
    main()
