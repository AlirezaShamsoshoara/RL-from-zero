"""MADDPG agent tests.

Examples:
    >>> # python -m pytest tests/maddpg/test_agent.py
"""

import copy
import unittest

import numpy as np
import torch

from MADDPG.maddpg.agent import MADDPGAgent, MADDPGStats
from MADDPG.maddpg.replay_buffer import MultiAgentReplayBuffer


def _make_agent(
    n_agents: int = 2,
    obs_dims=(4, 4),
    act_dims=(2, 2),
    target_policy_noise: float = 0.0,
    target_noise_clip: float = 0.0,
) -> MADDPGAgent:
    action_lows = [np.zeros(d, dtype=np.float32) for d in act_dims]
    action_highs = [np.ones(d, dtype=np.float32) for d in act_dims]
    return MADDPGAgent(
        n_agents=n_agents,
        obs_dims=list(obs_dims),
        act_dims=list(act_dims),
        action_lows=action_lows,
        action_highs=action_highs,
        hidden_sizes=[8, 8],
        activation="relu",
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.9,
        tau=0.05,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        device="cpu",
    )


def _fill_buffer(agent: MADDPGAgent, capacity: int = 32, n_add: int = 20) -> MultiAgentReplayBuffer:
    buf = MultiAgentReplayBuffer(
        n_agents=agent.n_agents, obs_dims=agent.obs_dims, act_dims=agent.act_dims,
        capacity=capacity, device=torch.device("cpu"),
    )
    rng = np.random.default_rng(0)
    for _ in range(n_add):
        obs = [rng.standard_normal(d).astype(np.float32) for d in agent.obs_dims]
        acts = [rng.uniform(0.0, 1.0, d).astype(np.float32) for d in agent.act_dims]
        rews = [float(rng.standard_normal()) for _ in range(agent.n_agents)]
        next_obs = [rng.standard_normal(d).astype(np.float32) for d in agent.obs_dims]
        dones = [False] * agent.n_agents
        buf.add(obs, acts, rews, next_obs, dones)
    return buf


class TestMADDPGAgent(unittest.TestCase):
    def test_act_shapes_deterministic_and_bounds(self) -> None:
        agent = _make_agent()
        obs_list = [np.zeros(d, dtype=np.float32) for d in agent.obs_dims]
        actions_a = agent.act(obs_list, noise=0.0, deterministic=True)
        actions_b = agent.act(obs_list, noise=0.0, deterministic=True)
        self.assertEqual(len(actions_a), agent.n_agents)
        for i, a in enumerate(actions_a):
            self.assertEqual(a.shape, (agent.act_dims[i],))
            self.assertEqual(a.dtype, np.float32)  # matches PettingZoo Box dtype
            self.assertTrue((a >= agent.action_lows_np[i] - 1e-6).all())
            self.assertTrue((a <= agent.action_highs_np[i] + 1e-6).all())
        # Deterministic across calls.
        for a, b in zip(actions_a, actions_b):
            self.assertTrue(np.allclose(a, b))

    def test_act_noise_perturbs_and_stays_in_bounds(self) -> None:
        agent = _make_agent()
        obs_list = [np.zeros(d, dtype=np.float32) for d in agent.obs_dims]
        np.random.seed(0)
        noisy = agent.act(obs_list, noise=0.5, deterministic=False)
        clean = agent.act(obs_list, noise=0.0, deterministic=True)
        self.assertFalse(np.allclose(noisy[0], clean[0]))
        for i, a in enumerate(noisy):
            self.assertTrue((a >= agent.action_lows_np[i]).all())
            self.assertTrue((a <= agent.action_highs_np[i]).all())

    def test_update_returns_finite_stats_and_moves_params(self) -> None:
        agent = _make_agent()
        buf = _fill_buffer(agent, n_add=32)
        # Snapshot actor 0 params.
        pre = [p.detach().clone() for p in agent.actors[0].parameters()]
        stats = agent.update(buf.sample(16))
        self.assertIsInstance(stats, MADDPGStats)
        for v in (stats.critic_loss, stats.actor_loss, stats.q_value):
            self.assertTrue(np.isfinite(v))
        # At least one parameter should have moved after one update.
        post = list(agent.actors[0].parameters())
        moved = any(not torch.allclose(a, b) for a, b in zip(pre, post))
        self.assertTrue(moved)

    def test_soft_target_update_moves_targets_toward_source(self) -> None:
        agent = _make_agent()
        # Randomly perturb the online actor so that source != target.
        with torch.no_grad():
            for p in agent.actors[0].parameters():
                p.add_(torch.randn_like(p) * 0.5)
        target_before = [p.detach().clone() for p in agent.actors_target[0].parameters()]
        source = [p.detach().clone() for p in agent.actors[0].parameters()]
        agent._soft_update(agent.actors[0], agent.actors_target[0])
        for tgt_before, src, tgt_after in zip(
            target_before, source, agent.actors_target[0].parameters()
        ):
            expected = agent.tau * src + (1.0 - agent.tau) * tgt_before
            self.assertTrue(torch.allclose(tgt_after.data, expected, atol=1e-6))

    def test_state_dict_round_trip(self) -> None:
        agent = _make_agent()
        buf = _fill_buffer(agent, n_add=32)
        agent.update(buf.sample(16))  # produce some non-init weights
        state = agent.state_dict()
        # Fresh agent with the same shapes; different init.
        clone = _make_agent()
        self.assertFalse(
            torch.allclose(
                next(agent.actors[0].parameters()),
                next(clone.actors[0].parameters()),
            )
        )
        clone.load_state_dict(state)
        # After load every actor/critic weight should match.
        for i in range(agent.n_agents):
            for a, b in zip(agent.actors[i].parameters(), clone.actors[i].parameters()):
                self.assertTrue(torch.allclose(a, b))
            for a, b in zip(agent.critics[i].parameters(), clone.critics[i].parameters()):
                self.assertTrue(torch.allclose(a, b))
        self.assertEqual(clone.global_step, agent.global_step)

    def test_target_policy_smoothing_clip(self) -> None:
        # With smoothing enabled the update path must not crash and target actions
        # must remain in bounds.
        agent = _make_agent(target_policy_noise=0.2, target_noise_clip=0.5)
        buf = _fill_buffer(agent, n_add=32)
        stats = agent.update(buf.sample(16))
        for v in (stats.critic_loss, stats.actor_loss, stats.q_value):
            self.assertTrue(np.isfinite(v))


if __name__ == "__main__":
    unittest.main()
