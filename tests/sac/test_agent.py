"""SAC agent tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/sac/test_agent.py
    >>> # Run with unittest
    >>> # python -m unittest tests.sac.test_agent
"""

import unittest

import numpy as np
import torch
from gymnasium import spaces

from SAC.sac.agent import SACAgent


def _make_agent(
    obs_dim: int = 4,
    act_dim: int = 2,
    hidden_sizes: tuple[int, ...] = (8,),
    activation: str = "relu",
    tau: float = 0.5,
) -> SACAgent:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-2.0, high=2.0, shape=(act_dim,), dtype=np.float32)
    return SACAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=hidden_sizes,
        activation=activation,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        gamma=0.99,
        tau=tau,
        target_entropy_scale=1.0,
        device="cpu",
    )


class TestSACAgent(unittest.TestCase):
    def test_act_returns_action_in_bounds(self) -> None:
        agent = _make_agent(obs_dim=3, act_dim=2)
        obs = np.zeros(3, dtype=np.float32)

        deterministic = agent.act(obs, deterministic=True)
        stochastic = agent.act(obs, deterministic=False)

        self.assertEqual(deterministic.shape, (2,))
        self.assertEqual(stochastic.shape, (2,))
        self.assertTrue(np.all(deterministic <= 2.0 + 1e-5))
        self.assertTrue(np.all(deterministic >= -2.0 - 1e-5))
        self.assertTrue(np.all(stochastic <= 2.0 + 1e-5))
        self.assertTrue(np.all(stochastic >= -2.0 - 1e-5))

    def test_update_changes_parameters_and_targets(self) -> None:
        torch.manual_seed(0)
        agent = _make_agent(obs_dim=4, act_dim=2, hidden_sizes=(16,), tau=0.5)
        batch_size = 6
        obs = torch.randn((batch_size, 4), dtype=torch.float32)
        actions = torch.randn((batch_size, 2), dtype=torch.float32)
        rewards = torch.randn((batch_size, 1), dtype=torch.float32)
        next_obs = torch.randn((batch_size, 4), dtype=torch.float32)
        dones = torch.zeros((batch_size, 1), dtype=torch.float32)

        actor_before = [p.detach().clone() for p in agent.actor.parameters()]
        target_before = [p.detach().clone() for p in agent.q1_target.parameters()]

        stats = agent.update((obs, actions, rewards, next_obs, dones))

        self.assertEqual(agent.global_step, 1)
        self.assertGreater(stats.alpha_value, 0.0)
        self.assertTrue(np.isfinite(stats.actor_loss))
        self.assertTrue(np.isfinite(stats.critic_loss))
        self.assertTrue(np.isfinite(stats.alpha_loss))
        self.assertTrue(np.isfinite(stats.log_prob))
        self.assertTrue(
            any(
                not torch.allclose(b, a)
                for b, a in zip(actor_before, agent.actor.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.allclose(b, a)
                for b, a in zip(target_before, agent.q1_target.parameters())
            )
        )

    def test_state_dict_round_trip(self) -> None:
        torch.manual_seed(1)
        agent = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=(8,))
        batch_size = 4
        batch = (
            torch.randn((batch_size, 3), dtype=torch.float32),
            torch.randn((batch_size, 2), dtype=torch.float32),
            torch.randn((batch_size, 1), dtype=torch.float32),
            torch.randn((batch_size, 3), dtype=torch.float32),
            torch.zeros((batch_size, 1), dtype=torch.float32),
        )
        agent.update(batch)

        state = agent.state_dict()
        restored = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=(8,))
        restored.load_state_dict(state)

        self.assertEqual(restored.global_step, agent.global_step)
        self.assertTrue(
            torch.allclose(agent.log_alpha.detach(), restored.log_alpha.detach())
        )
        for param, restored_param in zip(agent.actor.parameters(), restored.actor.parameters()):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(agent.q1.parameters(), restored.q1.parameters()):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(agent.q2.parameters(), restored.q2.parameters()):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(
            agent.q1_target.parameters(), restored.q1_target.parameters()
        ):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(
            agent.q2_target.parameters(), restored.q2_target.parameters()
        ):
            self.assertTrue(torch.allclose(param, restored_param))


if __name__ == "__main__":
    unittest.main()
