"""NEAT Pong self-play using a simple within-generation pairing strategy."""

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
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.hud import draw_training_hud
    from common.human_input import HumanInput
    from common.stop_button import StopButton
    from common.training_state import TrainingState

WIDTH = 700
HEIGHT = 500
FPS = 30
PADDLE_WIDTH = 16
PADDLE_HEIGHT = 90
PADDLE_SPEED = 8
BALL_RADIUS = 10
MAX_BALL_SPEED = 13
BALL_SPEED_START = 5.5
WIN_SCORE = 5
GENERATION_TIME_LIMIT = 30 * FPS
POPULATION_SIZE = 50
ACTION_THRESHOLD = 0.1
SURVIVAL_REWARD = 0.01
POINT_REWARD = 1.0
BEST_GENOME_PATH = Path(__file__).with_name('best_pong.pkl')
CONFIG_PATH = Path(__file__).with_name('config-feedforward.txt')
CHECKPOINT_PREFIX = str(Path(__file__).with_name('neat-checkpoint-'))

PANEL = (247, 251, 246)
PANEL_EDGE = (205, 224, 211)
INK = (30, 43, 48)
ACCENT = (27, 112, 91)
WHITE = (255,255,255)
LEFT_COLOR = (80, 160, 255)
RIGHT_COLOR = (255, 117, 91)

TRAINING_STATE = TrainingState()
STOP_REQUESTED = False
CURRENT_GENERATION = 0


def clamp(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def percentage_improvement(current: float, previous: float | None) -> float:
    if previous is None or previous == 0:
        return 0.0
    return (current - previous) / abs(previous) * 100.0


class Paddle:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
        self.rect = pygame.Rect(int(x), int(y), self.width, self.height)

    def move(self, direction: float) -> None:
        self.y += direction * PADDLE_SPEED
        self.y = max(0, min(HEIGHT - self.height, self.y))
        self.rect.y = int(self.y)

    def draw(self, surface: pygame.Surface, color: tuple[int, int, int]) -> None:
        pygame.draw.rect(surface, color, self.rect, border_radius=8)


class Ball:
    def __init__(self, x: float = WIDTH / 2, y: float = HEIGHT / 2, vx: float = BALL_SPEED_START, vy: float = 2.5) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = BALL_RADIUS

    def reset(self, direction: float = 1.0) -> None:
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.vx = direction * BALL_SPEED_START
        self.vy = random.uniform(-3.0, 3.0)

    def move(self) -> None:
        self.x += self.vx
        self.y += self.vy
        if self.y - self.radius <= 0:
            self.y = self.radius
            self.vy = abs(self.vy)
        if self.y + self.radius >= HEIGHT:
            self.y = HEIGHT - self.radius
            self.vy = -abs(self.vy)
        self.vx = max(-MAX_BALL_SPEED, min(MAX_BALL_SPEED, self.vx))
        self.vy = max(-MAX_BALL_SPEED, min(MAX_BALL_SPEED, self.vy))

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), self.radius)


def network_inputs(paddle: Paddle, ball: Ball) -> tuple[float, ...]:
    return (
        clamp((paddle.y + paddle.height / 2) / HEIGHT * 2.0 - 1.0),
        clamp((ball.x / WIDTH) * 2.0 - 1.0),
        clamp((ball.y / HEIGHT) * 2.0 - 1.0),
        clamp(ball.vx / MAX_BALL_SPEED),
        clamp(ball.vy / MAX_BALL_SPEED),
    )


def paddle_action(output: float) -> float:
    if abs(output) < ACTION_THRESHOLD:
        return 0.0
    return output


def make_config(path: Path) -> neat.Config:
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))


def draw_background(surface: pygame.Surface) -> None:
    surface.fill((18, 22, 33))
    center_line = pygame.Rect(WIDTH // 2 - 2, 0, 4, HEIGHT)
    pygame.draw.rect(surface, (80, 90, 110), center_line)
    for i in range(0, HEIGHT, 24):
        pygame.draw.rect(surface, (100, 110, 130), (WIDTH // 2 - 2, i, 4, 14))


class TrainingStopped(Exception):
    pass


def save_best_genome(winner: neat.DefaultGenome, generation: int, fitness: float, seed: int | None) -> None:
    payload = {
        'genome': copy.deepcopy(winner),
        'generation': generation,
        'fitness': fitness,
        'seed': seed,
        'saved_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'config_path': str(CONFIG_PATH),
    }
    with BEST_GENOME_PATH.open('wb') as file:
        pickle.dump(payload, file)
    print(f'Saved best pong genome to {BEST_GENOME_PATH} (fitness: {fitness:.2f})')


def load_best_genome() -> tuple[neat.DefaultGenome, dict[str, Any]]:
    with BEST_GENOME_PATH.open('rb') as file:
        payload = pickle.load(file)
    if isinstance(payload, dict) and 'genome' in payload:
        return payload['genome'], payload
    return payload, {'fitness': None}


def play_best(config_path: Path) -> None:
    genome, metadata = load_best_genome()
    config = make_config(config_path)
    network_left = neat.nn.FeedForwardNetwork.create(genome, config)
    network_right = neat.nn.FeedForwardNetwork.create(genome, config)
    rng = random.Random(metadata.get('seed'))
    left_paddle = Paddle(30, HEIGHT // 2 - PADDLE_HEIGHT // 2)
    right_paddle = Paddle(WIDTH - 30 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2)
    ball = Ball(vx=-BALL_SPEED_START, vy=rng.uniform(-2.5, 2.5))
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Pong - Best Genome')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    left_score = right_score = 0
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        if not running:
            break
        left_out = network_left.activate(network_inputs(left_paddle, ball))[0]
        right_out = network_right.activate(network_inputs(right_paddle, ball))[0]
        left_paddle.move(paddle_action(left_out))
        right_paddle.move(-paddle_action(right_out))
        ball.move()
        if ball.y - ball.radius <= 0 or ball.y + ball.radius >= HEIGHT:
            ball.vy *= -1
        if ball.x - ball.radius <= left_paddle.x + left_paddle.width and left_paddle.y <= ball.y <= left_paddle.y + left_paddle.height:
            ball.x = left_paddle.x + left_paddle.width + ball.radius
            ball.vx = abs(ball.vx) * 1.08
            ball.vy += (ball.y - (left_paddle.y + left_paddle.height / 2)) * 0.12
        if ball.x + ball.radius >= right_paddle.x and right_paddle.y <= ball.y <= right_paddle.y + right_paddle.height:
            ball.x = right_paddle.x - ball.radius
            ball.vx = -abs(ball.vx) * 1.08
            ball.vy += (ball.y - (right_paddle.y + right_paddle.height / 2)) * 0.12
        if ball.x < -30:
            right_score += 1
            ball.reset(1.0)
        elif ball.x > WIDTH + 30:
            left_score += 1
            ball.reset(-1.0)
        draw_background(screen)
        left_paddle.draw(screen, LEFT_COLOR)
        right_paddle.draw(screen, RIGHT_COLOR)
        ball.draw(screen)
        screen.blit(font.render(f'LEFT {left_score}  :  {right_score} RIGHT', True, WHITE), (230, 18))
        screen.blit(font.render(f'Best fitness: {metadata.get("fitness", "--")}', True, WHITE), (16, HEIGHT - 32))
        pygame.display.flip()
    pygame.quit()


def play_human_vs_ai(config_path: Path) -> None:
    genome, metadata = load_best_genome()
    config = make_config(config_path)
    ai_net = neat.nn.FeedForwardNetwork.create(genome, config)
    rng = random.Random(metadata.get('seed'))
    left_paddle = Paddle(30, HEIGHT // 2 - PADDLE_HEIGHT // 2)
    right_paddle = Paddle(WIDTH - 30 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2)
    ball = Ball(vx=BALL_SPEED_START, vy=rng.uniform(-2.5, 2.5))
    human_score = 0
    ai_score = 0
    input_state = HumanInput()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Pong - You vs Frozen AI')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    result_font = pygame.font.Font(None, 36)
    running = True
    while running:
        clock.tick(FPS)
        events = pygame.event.get()
        input_state.handle(events)
        if input_state.quit_requested:
            running = False
        if input_state.restart_requested:
            ball.reset(1.0 if random.random() > 0.5 else -1.0)
            left_paddle = Paddle(30, HEIGHT // 2 - PADDLE_HEIGHT // 2)
            right_paddle = Paddle(WIDTH - 30 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2)
            human_score = ai_score = 0
        if input_state.paused:
            pass
        else:
            dir_input = 0.0
            if input_state.jump_requested:
                dir_input = -1.0
            if input_state.duck_requested:
                dir_input = 1.0
            left_paddle.move(dir_input)
            ai_out = ai_net.activate(network_inputs(right_paddle, ball))[0]
            right_paddle.move(-paddle_action(ai_out))
            ball.move()
            if ball.y - ball.radius <= 0 or ball.y + ball.radius >= HEIGHT:
                ball.vy *= -1
            if ball.x - ball.radius <= left_paddle.x + left_paddle.width and left_paddle.y <= ball.y <= left_paddle.y + left_paddle.height:
                ball.x = left_paddle.x + left_paddle.width + ball.radius
                ball.vx = abs(ball.vx) * 1.08
                ball.vy += (ball.y - (left_paddle.y + left_paddle.height / 2)) * 0.12
            if ball.x + ball.radius >= right_paddle.x and right_paddle.y <= ball.y <= right_paddle.y + right_paddle.height:
                ball.x = right_paddle.x - ball.radius
                ball.vx = -abs(ball.vx) * 1.08
                ball.vy += (ball.y - (right_paddle.y + right_paddle.height / 2)) * 0.12
            if ball.x < -30:
                ai_score += 1
                ball.reset(1.0)
            elif ball.x > WIDTH + 30:
                human_score += 1
                ball.reset(-1.0)
        draw_background(screen)
        left_paddle.draw(screen, LEFT_COLOR)
        right_paddle.draw(screen, RIGHT_COLOR)
        ball.draw(screen)
        screen.blit(font.render(f'YOU {human_score}  :  {ai_score} AI', True, WHITE), (240, 18))
        screen.blit(font.render('UP/DOWN move | P pause | R restart | Esc quit', True, WHITE), (150, HEIGHT - 24))
        if human_score >= WIN_SCORE or ai_score >= WIN_SCORE:
            result = 'YOU WIN' if human_score > ai_score else 'AI WINS'
            screen.blit(result_font.render(result, True, WHITE), (260, HEIGHT // 2 - 20))
            if human_score >= WIN_SCORE or ai_score >= WIN_SCORE:
                pygame.display.flip(); pygame.time.delay(1200); running = False
        pygame.display.flip()
    pygame.quit()


def play_heuristic_vs_ai(config_path: Path) -> None:
    # Backwards-compatible alias used by the CLI when a generic `--play` flag is requested.
    play_human_vs_ai(config_path)


def evaluate_genomes(genomes: list[tuple[int, neat.DefaultGenome]], config: neat.Config, *, render: bool = True, rng: random.Random | None = None, time_limit: int = GENERATION_TIME_LIMIT) -> None:
    global CURRENT_GENERATION, STOP_REQUESTED
    rng = rng or random
    screen = None
    clock = None
    fonts = None
    if render:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('NEAT Pong')
        clock = pygame.time.Clock()
        fonts = (pygame.font.Font(None, 25), pygame.font.Font(None, 23), pygame.font.Font(None, 16))
    matches: list[tuple[int, int, Paddle, Paddle, Ball]] = []
    networks: dict[int, Any] = {}
    for idx, (_, genome) in enumerate(genomes):
        genome.fitness = 0.0
        networks[idx] = neat.nn.FeedForwardNetwork.create(genome, config)
    for i in range(0, len(genomes) - 1, 2):
        left_id, left_genome = genomes[i]
        right_id, right_genome = genomes[i + 1]
        scores = {'left': 0, 'right': 0}
        left_paddle = Paddle(30, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        right_paddle = Paddle(WIDTH - 30 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        ball = Ball(vx=rng.choice((-1, 1)) * BALL_SPEED_START, vy=rng.uniform(-3.0, 3.0))
        matches.append((left_id, right_id, left_paddle, right_paddle, ball))
    for left_id, right_id, left_paddle, right_paddle, ball in matches:
        left_score = right_score = 0
        frame = 0
        while frame < time_limit and left_score < WIN_SCORE and right_score < WIN_SCORE:
            if clock is not None:
                clock.tick(FPS)
            frame += 1
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        STOP_REQUESTED = True
                        TRAINING_STATE.request_stop()
                        raise TrainingStopped
            if STOP_REQUESTED:
                raise TrainingStopped
            left_net = networks[left_id]
            right_net = networks[right_id]
            left_move = paddle_action(left_net.activate(network_inputs(left_paddle, ball))[0])
            right_move = paddle_action(right_net.activate(network_inputs(right_paddle, ball))[0])
            left_paddle.move(left_move)
            right_paddle.move(-right_move)
            ball.move()
            if ball.y - ball.radius <= 0 or ball.y + ball.radius >= HEIGHT:
                ball.vy *= -1
            if ball.x - ball.radius <= left_paddle.x + left_paddle.width and left_paddle.y <= ball.y <= left_paddle.y + left_paddle.height:
                ball.x = left_paddle.x + left_paddle.width + ball.radius
                ball.vx = abs(ball.vx) * 1.08
                ball.vy += (ball.y - (left_paddle.y + left_paddle.height / 2)) * 0.12
            if ball.x + ball.radius >= right_paddle.x and right_paddle.y <= ball.y <= right_paddle.y + right_paddle.height:
                ball.x = right_paddle.x - ball.radius
                ball.vx = -abs(ball.vx) * 1.08
                ball.vy += (ball.y - (right_paddle.y + right_paddle.height / 2)) * 0.12
            if ball.x < -30:
                right_score += 1
                ball.reset(1.0)
            elif ball.x > WIDTH + 30:
                left_score += 1
                ball.reset(-1.0)
            for _, genome in genomes:
                genome.fitness += 0.01
            if render and screen is not None:
                draw_background(screen)
                left_paddle.draw(screen, LEFT_COLOR)
                right_paddle.draw(screen, RIGHT_COLOR)
                ball.draw(screen)
                TRAINING_STATE.generation = CURRENT_GENERATION
                TRAINING_STATE.score = max(left_score, right_score)
                draw_training_hud(screen, fonts, TRAINING_STATE, 2, 2, 'NEAT PONG', 'SELF-PLAY', StopButton(pygame.Rect(WIDTH - 136, 28, 112, 38)))
                pygame.display.flip()
        for _, genome in genomes:
            genome.fitness += 0.0
        left_fit = left_score - right_score + 0.01 * frame
        right_fit = right_score - left_score + 0.01 * frame
        for idx, genome in genomes:
            if idx == left_id:
                genome.fitness += left_fit
            elif idx == right_id:
                genome.fitness += right_fit
    if render and screen is not None:
        pygame.quit()


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
        winner = max(genomes, key=lambda item: item[1].fitness)[1]
        if winner.fitness > 0:
            save_best_genome(winner, CURRENT_GENERATION, winner.fitness, seed)

    try:
        winner = population.run(run_generation, generations)
    except TrainingStopped:
        print('Training stopped by user.')
        return
    print(f'Training complete. Winner fitness: {winner.fitness:.2f}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Train NEAT agents to play Pong.')
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
        print('Pong project files and Python syntax are valid.')
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
