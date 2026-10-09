import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from model import MyModel


class QTrainer:
    def __init__(self, model: MyModel, target_model: MyModel, lr: float, gamma: float):
        self._model = model
        self._target_model = target_model
        self._gamma = gamma
        self._optimizer = optim.Adam(model.parameters(), lr=lr)

    def sync_target(self):
        self._target_model.load_state_dict(self._model.state_dict())

    def train_step(self, states, actions, rewards, next_states, dones) -> float:
        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        print(dones.shape)

        q_values = self._model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q = self._target_model(next_states).max(dim=1).values
            target = rewards + self._gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)

        self._optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()

        return loss.item()
