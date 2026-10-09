import random
from collections import deque

import torch

from game import Game
from model import MyModel
from q_trainer import QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 64
MIN_MEMORY = 1_000
TARGET_SYNC_EVERY = 500

LR = 0.001
GAMMA = 0.99

EPS_START = 1.0
EPS_MIN = 0.05
EPS_DECAY_STEPS = 30_000

STATE_SIZE = 6
N_ACTIONS = 3

class Agent:
    def __init__(self, game: Game):
        self._game = game
        self._gamma = GAMMA
        self._nb_games = 0
        self._steps = 0
        self._train_steps = 0
        self._memory = deque(maxlen=MAX_MEMORY)

        self._model = MyModel(STATE_SIZE, 128, N_ACTIONS)
        self._target_model = MyModel(STATE_SIZE, 128, N_ACTIONS)
        self._target_model.load_state_dict(self._model.state_dict())

        self._trainer = QTrainer(
            self._model, self._target_model, lr=LR, gamma=self._gamma
        )

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self._steps / EPS_DECAY_STEPS)
        return EPS_START + progress * (EPS_MIN - EPS_START)

    def get_state(self):
        ball = self._game.get_ball()
        player = self._game.get_player()

        return (
            (player.position.x - 300) / 250,
            (ball.position.x - 300) / 250,
            (ball.position.y - 300) / 250,
            ball.velocity.x / 400,
            ball.velocity.y / 400,
            (ball.position.x - player.position.x) / 250,  # ecart relatif
        )

    def get_action(self, state) -> int:
        self._steps += 1

        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)

        with torch.no_grad():
            q = self._model(torch.tensor(state, dtype=torch.float32))
        return int(torch.argmax(q).item())

    def remember(self, state, action, reward, next_state, done):
        self._memory.append((state, action, reward, next_state, done))

    def train(self) -> float | None:
        if len(self._memory) < MIN_MEMORY:
            return None

        batch = random.sample(self._memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        loss = self._trainer.train_step(states, actions, rewards, next_states, dones)

        self._train_steps += 1
        if self._train_steps % TARGET_SYNC_EVERY == 0:
            self._trainer.sync_target()

        return loss

    def save_model(self):
        self._model.save()

    def new_game(self):
        self._nb_games += 1
