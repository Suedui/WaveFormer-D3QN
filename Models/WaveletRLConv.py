"""Dueling Double DQN components used to adaptively select wavelet kernels."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Transition(NamedTuple):
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


@dataclass(slots=True)
class AgentConfig:
    """Hyper-parameters controlling the D3QN agent."""

    state_dim: int
    action_dim: int
    gamma: float = 0.99
    batch_size: int = 32
    memory_size: int = 5_000
    lr: float = 1e-3
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 10_000


class DuelingQNetwork(nn.Module):
    """A simple dueling Q-network for discrete action spaces."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class WaveletSelectionAgent:
    """D3QN agent responsible for selecting the optimal wavelet kernel."""

    def __init__(self, config: AgentConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device("cpu")

        self.policy_net = DuelingQNetwork(config.state_dim, config.action_dim).to(self.device)
        self.target_net = DuelingQNetwork(config.state_dim, config.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)

        self.memory: Deque[Transition] = deque(maxlen=config.memory_size)
        self.step_count = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.step_count / max(1, self.config.epsilon_decay))
        return self.config.epsilon_start + frac * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_action(self, state: torch.Tensor) -> int:
        """Select an action using an epsilon-greedy exploration strategy."""

        self.step_count += 1
        eps = self.epsilon()
        if np.random.rand() < eps:
            return np.random.randint(self.config.action_dim)

        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.policy_net(state.unsqueeze(0))
            return int(torch.argmax(q_values, dim=-1).item())

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.memory.append(Transition(state.detach(), action, reward, next_state.detach(), done))

    def _sample_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(len(self.memory), self.config.batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]

        states = torch.stack([transition.state for transition in batch]).to(self.device)
        actions = torch.tensor([transition.action for transition in batch], dtype=torch.long)
        rewards = torch.tensor([transition.reward for transition in batch], dtype=torch.float32)
        next_states = torch.stack([transition.next_state for transition in batch]).to(self.device)
        dones = torch.tensor([transition.done for transition in batch], dtype=torch.float32)
        return states, actions.to(self.device), rewards.to(self.device), next_states, dones.to(
            self.device
        )

    def update(self) -> float | None:
        """Perform a single optimisation step if enough samples are available."""

        if len(self.memory) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self._sample_batch()

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_action = torch.argmax(self.policy_net(next_states), dim=-1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_action).squeeze(-1)
            target = rewards + self.config.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.mul_(1.0 - self.config.tau)
                target_param.data.add_(self.config.tau * policy_param.data)

        return float(loss.item())

    def available_actions(self) -> Iterable[int]:
        """Return the indices of the available actions."""

        return range(self.config.action_dim)
