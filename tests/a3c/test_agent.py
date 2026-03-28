"""A3C agent tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/a3c/test_agent.py
    >>> # Run with unittest
    >>> # python -m unittest tests.a3c.test_agent
"""

import unittest

import numpy as np
import torch
from gymnasium import spaces

from A3C.a3c.agent import A3CAgent


def _make_agent(
    obs_dim: int = 6,
    act_dim: int = 3,
    hidden_sizes: tuple[int, ...] = (8,),
    activation: str = "relu",
) -> A3CAgent:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Discrete(act_dim)
    return A3CAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=hidden_sizes,
        activation=activation,
        learning_rate=1e-3,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=40.0,
        device="cpu",
    )


class TestA3CAgent(unittest.TestCase):
    def test_model_shared_memory(self) -> None:
        agent = _make_agent()
        # model parameters should be in shared memory after construction
        for param in agent.model.parameters():
            self.assertTrue(param.is_shared())

    def test_new_local_model_copies_weights(self) -> None:
        torch.manual_seed(0)
        agent = _make_agent()
        local = agent.new_local_model()

        for gp, lp in zip(agent.model.parameters(), local.parameters()):
            self.assertTrue(torch.allclose(gp, lp))
        # local model should NOT be in shared memory
        for param in local.parameters():
            self.assertFalse(param.is_shared())

    def test_compute_loss_returns_valid_stats(self) -> None:
        torch.manual_seed(0)
        agent = _make_agent()
        batch_size = 4
        advantages = torch.randn(batch_size)
        log_probs = torch.randn(batch_size, requires_grad=True)
        values = torch.randn(batch_size, requires_grad=True)
        returns = torch.randn(batch_size)
        entropies = torch.rand(batch_size)

        total_loss, stats = agent.compute_loss(
            advantages=advantages,
            log_probs=log_probs,
            values=values,
            returns=returns,
            entropies=entropies,
        )

        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(np.isfinite(stats.policy_loss))
        self.assertTrue(np.isfinite(stats.value_loss))
        self.assertTrue(np.isfinite(stats.entropy))
        self.assertTrue(np.isfinite(stats.total_loss))

    def test_apply_gradients_updates_global_model(self) -> None:
        torch.manual_seed(0)
        agent = _make_agent(obs_dim=4, act_dim=2, hidden_sizes=(8,))
        local = agent.new_local_model()

        # Run a forward/backward pass on the local model
        obs = torch.randn((3, 4), dtype=torch.float32)
        logits, values = local.forward(obs)
        loss = logits.sum() + values.sum()
        loss.backward()

        params_before = [p.detach().clone() for p in agent.model.parameters()]
        agent.apply_gradients(local)

        changed = any(
            not torch.allclose(before, after)
            for before, after in zip(params_before, agent.model.parameters())
        )
        self.assertTrue(changed)

    def test_sync_local_reloads_weights(self) -> None:
        torch.manual_seed(0)
        agent = _make_agent()
        local = agent.new_local_model()

        # Mutate global model weights
        with torch.no_grad():
            for p in agent.model.parameters():
                p.add_(1.0)

        # Before sync, weights differ
        differs = any(
            not torch.allclose(gp, lp)
            for gp, lp in zip(agent.model.parameters(), local.parameters())
        )
        self.assertTrue(differs)

        agent.sync_local(local)

        # After sync, weights match
        for gp, lp in zip(agent.model.parameters(), local.parameters()):
            self.assertTrue(torch.allclose(gp, lp))


if __name__ == "__main__":
    unittest.main()
