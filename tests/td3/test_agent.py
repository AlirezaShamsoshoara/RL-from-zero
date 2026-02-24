"""TD3 agent tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/td3/test_agent.py
    >>> # Run with unittest
    >>> # python -m unittest tests.td3.test_agent
"""

import unittest

import numpy as np
import torch
from gymnasium import spaces

from TD3.td3.agent import TD3Agent


def _make_agent(
    obs_dim: int = 4,
    act_dim: int = 2,
    hidden_sizes: tuple[int, ...] = (8,),
    activation: str = "relu",
    tau: float = 0.5,
    policy_delay: int = 2,
) -> TD3Agent:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-2.0, high=2.0, shape=(act_dim,), dtype=np.float32)
    return TD3Agent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=hidden_sizes,
        activation=activation,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=tau,
        policy_delay=policy_delay,
        target_noise=0.2,
        noise_clip=0.5,
        device="cpu",
    )


def _random_batch(batch_size: int, obs_dim: int, act_dim: int):
    obs = torch.randn((batch_size, obs_dim), dtype=torch.float32)
    actions = torch.randn((batch_size, act_dim), dtype=torch.float32)
    rewards = torch.randn((batch_size, 1), dtype=torch.float32)
    next_obs = torch.randn((batch_size, obs_dim), dtype=torch.float32)
    dones = torch.zeros((batch_size, 1), dtype=torch.float32)
    return obs, actions, rewards, next_obs, dones


class TestTD3Agent(unittest.TestCase):
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

    def test_update_delays_actor_and_updates_targets(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        agent = _make_agent(
            obs_dim=4,
            act_dim=2,
            hidden_sizes=(16,),
            tau=0.5,
            policy_delay=2,
        )
        batch = _random_batch(batch_size=6, obs_dim=4, act_dim=2)

        actor_before_first = [p.detach().clone() for p in agent.actor.parameters()]
        stats_first = agent.update(batch)

        self.assertEqual(agent.global_step, 1)
        self.assertIsNone(stats_first.actor_loss)
        self.assertTrue(np.isfinite(stats_first.critic_loss))
        self.assertTrue(np.isfinite(stats_first.q1_value))
        self.assertTrue(np.isfinite(stats_first.q2_value))
        self.assertTrue(
            all(
                torch.allclose(before, after)
                for before, after in zip(actor_before_first, agent.actor.parameters())
            )
        )

        target_before_second = [
            p.detach().clone() for p in agent.actor_target.parameters()
        ]
        stats_second = agent.update(batch)

        self.assertEqual(agent.global_step, 2)
        self.assertIsNotNone(stats_second.actor_loss)
        self.assertTrue(np.isfinite(float(stats_second.actor_loss)))
        self.assertTrue(
            any(
                not torch.allclose(before, after)
                for before, after in zip(actor_before_first, agent.actor.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.allclose(before, after)
                for before, after in zip(
                    target_before_second, agent.actor_target.parameters()
                )
            )
        )

    def test_state_dict_round_trip(self) -> None:
        torch.manual_seed(1)
        agent = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=(8,), policy_delay=1)
        batch = _random_batch(batch_size=4, obs_dim=3, act_dim=2)
        agent.update(batch)

        state = agent.state_dict()
        restored = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=(8,), policy_delay=1)
        restored.load_state_dict(state)

        self.assertEqual(restored.global_step, agent.global_step)
        for param, restored_param in zip(agent.actor.parameters(), restored.actor.parameters()):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(agent.q1.parameters(), restored.q1.parameters()):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(agent.q2.parameters(), restored.q2.parameters()):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(
            agent.actor_target.parameters(), restored.actor_target.parameters()
        ):
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
