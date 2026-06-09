"""DDPG agent tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/ddpg/test_agent.py
    >>> # Run with unittest
    >>> # python -m unittest tests.ddpg.test_agent
"""

import unittest

import numpy as np
import torch
from gymnasium import spaces

from DDPG.ddpg.agent import DDPGAgent


def _make_agent(
    obs_dim: int = 4,
    act_dim: int = 2,
    hidden_sizes: tuple[int, ...] = (8,),
    activation: str = "relu",
    tau: float = 0.5,
    target_policy_noise: float = 0.0,
    target_noise_clip: float = 0.0,
) -> DDPGAgent:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-2.0, high=2.0, shape=(act_dim,), dtype=np.float32)
    return DDPGAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=hidden_sizes,
        activation=activation,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=tau,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        device="cpu",
    )


def _random_batch(batch_size: int, obs_dim: int, act_dim: int):
    obs = torch.randn((batch_size, obs_dim), dtype=torch.float32)
    actions = torch.randn((batch_size, act_dim), dtype=torch.float32)
    rewards = torch.randn((batch_size, 1), dtype=torch.float32)
    next_obs = torch.randn((batch_size, obs_dim), dtype=torch.float32)
    dones = torch.zeros((batch_size, 1), dtype=torch.float32)
    return obs, actions, rewards, next_obs, dones


class TestDDPGAgent(unittest.TestCase):
    def test_act_returns_action_in_bounds(self) -> None:
        agent = _make_agent(obs_dim=3, act_dim=2)
        obs = np.zeros(3, dtype=np.float32)

        deterministic = agent.act(obs, deterministic=True)
        stochastic = agent.act(obs, noise=0.5, deterministic=False)

        self.assertEqual(deterministic.shape, (2,))
        self.assertEqual(stochastic.shape, (2,))
        self.assertTrue(np.all(deterministic <= 2.0 + 1e-5))
        self.assertTrue(np.all(deterministic >= -2.0 - 1e-5))
        self.assertTrue(np.all(stochastic <= 2.0 + 1e-5))
        self.assertTrue(np.all(stochastic >= -2.0 - 1e-5))

    def test_act_zero_noise_is_deterministic(self) -> None:
        agent = _make_agent(obs_dim=3, act_dim=2)
        obs = np.full(3, 0.3, dtype=np.float32)

        first = agent.act(obs, noise=0.0, deterministic=False)
        second = agent.act(obs, noise=0.0, deterministic=False)

        np.testing.assert_allclose(first, second)

    def test_update_changes_actor_and_critic_and_targets(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        agent = _make_agent(obs_dim=4, act_dim=2, hidden_sizes=(16,), tau=0.5)
        batch = _random_batch(batch_size=6, obs_dim=4, act_dim=2)

        actor_before = [p.detach().clone() for p in agent.actor.parameters()]
        critic_before = [p.detach().clone() for p in agent.critic.parameters()]
        actor_target_before = [
            p.detach().clone() for p in agent.actor_target.parameters()
        ]
        critic_target_before = [
            p.detach().clone() for p in agent.critic_target.parameters()
        ]

        stats = agent.update(batch)

        # DDPG updates the actor on every step (no policy delay).
        self.assertEqual(agent.global_step, 1)
        self.assertTrue(np.isfinite(stats.critic_loss))
        self.assertTrue(np.isfinite(stats.actor_loss))
        self.assertTrue(np.isfinite(stats.q_value))
        self.assertTrue(
            any(
                not torch.allclose(before, after)
                for before, after in zip(actor_before, agent.actor.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.allclose(before, after)
                for before, after in zip(critic_before, agent.critic.parameters())
            )
        )
        # Soft update should have nudged the target networks.
        self.assertTrue(
            any(
                not torch.allclose(before, after)
                for before, after in zip(
                    actor_target_before, agent.actor_target.parameters()
                )
            )
        )
        self.assertTrue(
            any(
                not torch.allclose(before, after)
                for before, after in zip(
                    critic_target_before, agent.critic_target.parameters()
                )
            )
        )

    def test_target_policy_smoothing_path_runs(self) -> None:
        # Exercises the optional TD3-style target smoothing branch in update().
        torch.manual_seed(1)
        np.random.seed(1)
        agent = _make_agent(
            obs_dim=3,
            act_dim=2,
            hidden_sizes=(8,),
            target_policy_noise=0.2,
            target_noise_clip=0.5,
        )
        batch = _random_batch(batch_size=4, obs_dim=3, act_dim=2)

        stats = agent.update(batch)

        self.assertTrue(np.isfinite(stats.critic_loss))
        self.assertTrue(np.isfinite(stats.actor_loss))

    def test_state_dict_round_trip(self) -> None:
        torch.manual_seed(1)
        agent = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=(8,))
        batch = _random_batch(batch_size=4, obs_dim=3, act_dim=2)
        agent.update(batch)

        state = agent.state_dict()
        restored = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=(8,))
        restored.load_state_dict(state)

        self.assertEqual(restored.global_step, agent.global_step)
        for param, restored_param in zip(
            agent.actor.parameters(), restored.actor.parameters()
        ):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(
            agent.critic.parameters(), restored.critic.parameters()
        ):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(
            agent.actor_target.parameters(), restored.actor_target.parameters()
        ):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(
            agent.critic_target.parameters(), restored.critic_target.parameters()
        ):
            self.assertTrue(torch.allclose(param, restored_param))


if __name__ == "__main__":
    unittest.main()
