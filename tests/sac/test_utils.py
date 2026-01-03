"""SAC utility tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/sac/test_utils.py
    >>> # Run with unittest
    >>> # python -m unittest tests.sac.test_utils
"""

import os
import tempfile
import unittest

import numpy as np
import torch
from gymnasium import spaces

from SAC.sac.agent import SACAgent
from SAC.sac.utils import load_checkpoint, save_checkpoint


def _make_agent(obs_dim: int = 3, act_dim: int = 2) -> SACAgent:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-2.0, high=2.0, shape=(act_dim,), dtype=np.float32)
    return SACAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=(8,),
        activation="relu",
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        gamma=0.99,
        tau=0.5,
        target_entropy_scale=1.0,
        device="cpu",
    )


class TestUtils(unittest.TestCase):
    def test_save_and_load_checkpoint_round_trip(self) -> None:
        torch.manual_seed(0)
        agent = _make_agent()
        agent.global_step = 7

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            save_checkpoint(path, agent, step=10, best_return=12.5)

            restored = _make_agent()
            data = load_checkpoint(path, restored)

        self.assertEqual(data["step"], 10)
        self.assertAlmostEqual(float(data["best_return"]), 12.5)
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


if __name__ == "__main__":
    unittest.main()
