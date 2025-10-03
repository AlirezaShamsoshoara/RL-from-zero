from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from gym import spaces

from deepQN.dqn.networks import QNetwork


@dataclass
class UpdateStats:
    loss: float
    td_error: float


class DQNAgent:
    def __init__(
        self,
        obs_space: spaces.Box,
        act_space: spaces.Discrete,
        hidden_sizes: list[int],
        activation: str,
        lr: float,
        gamma: float,
        target_update_interval: int,
        max_grad_norm: float,
        double_dqn: bool,
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_space = act_space
        self.action_dim = int(act_space.n)
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.double_dqn = double_dqn

        self.q_network = QNetwork(self.obs_dim, self.action_dim, hidden_sizes, activation).to(self.device)
        self.target_q_network = QNetwork(self.obs_dim, self.action_dim, hidden_sizes, activation).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.step_count = 0
        self.train_steps = 0

    def act(self, obs: np.ndarray, epsilon: float, deterministic: bool = False) -> int:
        self.step_count += 1
        if not deterministic and np.random.rand() < epsilon:
            return int(self.act_space.sample())
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def update(self, batch: Dict[str, torch.Tensor]) -> UpdateStats:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].long().to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)

        q_values = self.q_network(obs)
        action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = torch.argmax(self.q_network(next_obs), dim=1, keepdim=True)
                next_target_values = self.target_q_network(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_target_values = self.target_q_network(next_obs).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_target_values

        loss = F.mse_loss(action_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.update_target_network()

        td_error = torch.mean(torch.abs(action_values.detach() - targets)).item()
        return UpdateStats(loss=float(loss.item()), td_error=td_error)

    def update_target_network(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def state_dict(self) -> Dict[str, Any]:
        return {
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "train_steps": self.train_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.q_network.load_state_dict(state["q_network"])
        target_state = state.get("target_q_network", state["q_network"])
        self.target_q_network.load_state_dict(target_state)
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
        self.step_count = int(state.get("step_count", 0))
        self.train_steps = int(state.get("train_steps", 0))
        self.update_target_network()

    def to(self, device: str) -> "DQNAgent":
        dev = torch.device(device)
        self.q_network.to(dev)
        self.target_q_network.to(dev)
        self.device = dev
        return self
