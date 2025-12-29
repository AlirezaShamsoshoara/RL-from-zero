"""DQN agent tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/deepqn/test_agent.py
    >>> # Run with unittest
    >>> # python -m unittest tests.deepqn.test_agent
"""

import unittest
from unittest import mock

import numpy as np
import torch
from gymnasium import spaces

from deepQN.dqn.agent import DQNAgent


def _make_agent(
    obs_dim: int = 4,
    act_dim: int = 3,
    hidden_sizes: list[int] | None = None,
    double_dqn: bool = False,
) -> DQNAgent:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Discrete(act_dim)
    return DQNAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=hidden_sizes or [],
        activation="relu",
        lr=1e-3,
        gamma=0.99,
        target_update_interval=1,
        max_grad_norm=1.0,
        double_dqn=double_dqn,
        device="cpu",
    )


class TestDQNAgent(unittest.TestCase):
    def test_act_deterministic_uses_argmax(self) -> None:
        agent = _make_agent(obs_dim=2, act_dim=3, hidden_sizes=[])
        with torch.no_grad():
            layer = agent.q_network.model[0]
            layer.weight.zero_()
            layer.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

        obs = np.array([0.5, -0.2], dtype=np.float32)
        action = agent.act(obs, epsilon=0.0, deterministic=True)

        self.assertEqual(action, 2)
        self.assertEqual(agent.step_count, 1)

    def test_act_epsilon_sampling_uses_action_space(self) -> None:
        agent = _make_agent(obs_dim=4, act_dim=3, hidden_sizes=[])
        obs = np.zeros(4, dtype=np.float32)

        with mock.patch.object(agent.act_space, "sample", return_value=1) as sampler:
            action = agent.act(obs, epsilon=1.0, deterministic=False)

        self.assertEqual(action, 1)
        sampler.assert_called_once()
        self.assertEqual(agent.step_count, 1)

    def test_update_changes_parameters_and_syncs_target(self) -> None:
        torch.manual_seed(0)
        agent = _make_agent(obs_dim=4, act_dim=2, hidden_sizes=[8], double_dqn=True)
        batch_size = 6
        batch = {
            "obs": torch.randn((batch_size, 4), dtype=torch.float32),
            "actions": torch.randint(0, 2, (batch_size,), dtype=torch.int64),
            "rewards": torch.randn((batch_size,), dtype=torch.float32),
            "next_obs": torch.randn((batch_size, 4), dtype=torch.float32),
            "dones": torch.zeros((batch_size,), dtype=torch.float32),
        }

        before = [p.detach().clone() for p in agent.q_network.parameters()]
        stats = agent.update(batch)

        changed = any(
            not torch.allclose(b, a) for b, a in zip(before, agent.q_network.parameters())
        )
        self.assertTrue(changed)
        self.assertEqual(agent.train_steps, 1)
        self.assertTrue(np.isfinite(stats.loss))
        self.assertTrue(np.isfinite(stats.td_error))
        for q_param, target_param in zip(
            agent.q_network.parameters(), agent.target_q_network.parameters()
        ):
            self.assertTrue(torch.allclose(q_param, target_param))

    def test_state_dict_round_trip(self) -> None:
        torch.manual_seed(1)
        agent = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=[8])
        agent.act(np.zeros(3, dtype=np.float32), epsilon=0.0, deterministic=True)
        batch = {
            "obs": torch.randn((4, 3), dtype=torch.float32),
            "actions": torch.randint(0, 2, (4,), dtype=torch.int64),
            "rewards": torch.randn((4,), dtype=torch.float32),
            "next_obs": torch.randn((4, 3), dtype=torch.float32),
            "dones": torch.zeros((4,), dtype=torch.float32),
        }
        agent.update(batch)

        state = agent.state_dict()
        restored = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=[8])
        restored.load_state_dict(state)

        self.assertEqual(restored.step_count, agent.step_count)
        self.assertEqual(restored.train_steps, agent.train_steps)
        for param, restored_param in zip(
            agent.q_network.parameters(), restored.q_network.parameters()
        ):
            self.assertTrue(torch.allclose(param, restored_param))
        for q_param, target_param in zip(
            restored.q_network.parameters(), restored.target_q_network.parameters()
        ):
            self.assertTrue(torch.allclose(q_param, target_param))


if __name__ == "__main__":
    unittest.main()
