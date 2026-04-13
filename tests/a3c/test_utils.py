"""A3C utility tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/a3c/test_utils.py
    >>> # Run with unittest
    >>> # python -m unittest tests.a3c.test_utils
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from A3C.a3c.utils import (
    compute_returns,
    make_env,
    make_vec_env,
    save_checkpoint,
    load_checkpoint,
    set_seed,
)


class TestComputeReturns(unittest.TestCase):
    def test_single_env_returns(self) -> None:
        """1D inputs (single env): standard n-step return computation."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        dones = torch.tensor([0.0, 0.0, 0.0])
        last_value = torch.tensor(1.0)
        gamma = 0.99

        returns = compute_returns(rewards, dones, last_value, gamma)

        # R[2] = 3 + 0.99 * 1.0 = 3.99
        # R[1] = 2 + 0.99 * 3.99 = 5.9501
        # R[0] = 1 + 0.99 * 5.9501 = 6.890599
        self.assertEqual(returns.shape, (3,))
        self.assertAlmostEqual(returns[2].item(), 3.99, places=4)
        self.assertAlmostEqual(returns[1].item(), 5.9501, places=4)
        self.assertAlmostEqual(returns[0].item(), 6.890599, places=4)

    def test_done_cuts_bootstrap(self) -> None:
        """done=1 at a step should zero out the bootstrap from future steps."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        dones = torch.tensor([0.0, 1.0, 0.0])
        last_value = torch.tensor(1.0)
        gamma = 0.99

        returns = compute_returns(rewards, dones, last_value, gamma)

        # R[2] = 3 + 0.99 * 1.0 = 3.99
        # R[1] = 2 + 0.99 * R[2] * (1 - 1) = 2.0  (done cuts bootstrap)
        # R[0] = 1 + 0.99 * 2.0 = 2.98
        self.assertAlmostEqual(returns[2].item(), 3.99, places=4)
        self.assertAlmostEqual(returns[1].item(), 2.0, places=4)
        self.assertAlmostEqual(returns[0].item(), 2.98, places=4)

    def test_multi_env_returns_2d(self) -> None:
        """2D inputs (steps, num_envs): returns computed independently per env."""
        num_envs = 3
        rewards = torch.tensor(
            [[1.0, 10.0, 100.0],
             [2.0, 20.0, 200.0]],
            dtype=torch.float32,
        )  # (2 steps, 3 envs)
        dones = torch.zeros(2, num_envs)
        last_value = torch.tensor([0.0, 0.0, 0.0])
        gamma = 0.5

        returns = compute_returns(rewards, dones, last_value, gamma)

        self.assertEqual(returns.shape, (2, num_envs))
        # Env 0: R[1]=2+0.5*0=2,   R[0]=1+0.5*2=2
        # Env 1: R[1]=20+0.5*0=20, R[0]=10+0.5*20=20
        # Env 2: R[1]=200,          R[0]=100+0.5*200=200
        self.assertAlmostEqual(returns[1, 0].item(), 2.0, places=4)
        self.assertAlmostEqual(returns[0, 0].item(), 2.0, places=4)
        self.assertAlmostEqual(returns[1, 1].item(), 20.0, places=4)
        self.assertAlmostEqual(returns[0, 1].item(), 20.0, places=4)
        self.assertAlmostEqual(returns[1, 2].item(), 200.0, places=4)
        self.assertAlmostEqual(returns[0, 2].item(), 200.0, places=4)

    def test_multi_env_done_isolates_envs(self) -> None:
        """A done in one env should not affect other envs' return computation."""
        rewards = torch.tensor(
            [[1.0, 1.0],
             [1.0, 1.0]],
            dtype=torch.float32,
        )  # (2 steps, 2 envs)
        dones = torch.tensor(
            [[0.0, 1.0],   # env 1 done at step 0
             [0.0, 0.0]],
            dtype=torch.float32,
        )
        last_value = torch.tensor([1.0, 1.0])
        gamma = 0.99

        returns = compute_returns(rewards, dones, last_value, gamma)

        # Env 0 (no dones): R[1]=1+0.99*1=1.99, R[0]=1+0.99*1.99=2.9701
        # Env 1 (done at step 0): R[1]=1+0.99*1=1.99, R[0]=1+0.99*R[1]*(1-1)=1.0
        self.assertAlmostEqual(returns[0, 0].item(), 2.9701, places=4)
        self.assertAlmostEqual(returns[0, 1].item(), 1.0, places=4)


class TestMakeVecEnv(unittest.TestCase):
    def test_returns_vector_env_with_correct_num_envs(self) -> None:
        env = make_vec_env("CartPole-v1", num_envs=3, seed=42)

        self.assertEqual(env.num_envs, 3)
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 3)
        env.close()

    def test_step_returns_batched_outputs(self) -> None:
        env = make_vec_env("CartPole-v1", num_envs=2, seed=42)
        obs, _ = env.reset()
        actions = np.array([0, 1])

        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        self.assertEqual(next_obs.shape[0], 2)
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(terminated.shape, (2,))
        self.assertEqual(truncated.shape, (2,))
        env.close()


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_save_and_load_preserves_weights(self) -> None:
        from A3C.a3c.networks import ActorCritic
        from A3C.a3c.utils import SharedAdam

        torch.manual_seed(0)
        model = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,))
        optimizer = SharedAdam(model.parameters(), lr=1e-3)

        # Run a step so optimizer has state
        obs = torch.randn((2, 6))
        logits, values = model.forward(obs)
        loss = logits.sum() + values.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        weights_before = [p.detach().clone() for p in model.parameters()]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            save_checkpoint(path, model, optimizer, step=100, best_return=42.0)

            model2 = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,))
            data = load_checkpoint(path, model2)

            self.assertEqual(data["step"], 100)
            self.assertAlmostEqual(data["best_return"], 42.0)
            for before, after in zip(weights_before, model2.parameters()):
                self.assertTrue(torch.allclose(before, after))


class TestSetSeed(unittest.TestCase):
    def test_reproducibility(self) -> None:
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
