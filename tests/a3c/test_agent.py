"""Focused A3C loss, shared optimizer, and synchronization regressions."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium import spaces

from A3C.a3c.agent import A3CAgent, A3CStats


class RecordingLock:
    def __init__(self):
        self.depth = 0
        self.entries = 0

    def __enter__(self):
        self.depth += 1
        self.entries += 1
        return self

    def __exit__(self, *_):
        self.depth -= 1


def _make_agent(obs_dim=6, act_dim=3, hidden_sizes=(8,), activation="relu", **kwargs):
    return A3CAgent(
        obs_space=spaces.Box(-1.0, 1.0, shape=(obs_dim,), dtype=np.float32),
        act_space=spaces.Discrete(act_dim),
        hidden_sizes=hidden_sizes,
        activation=activation,
        learning_rate=1e-3,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=40.0,
        device="cpu",
        **kwargs,
    )


def _loss_inputs():
    return {
        "advantages": torch.tensor([1.0, 3.0]),
        "log_probs": torch.tensor([-0.25, -1.0], requires_grad=True),
        "values": torch.tensor([2.0, 4.0], requires_grad=True),
        "returns": torch.tensor([3.0, 7.0]),
        "entropies": torch.tensor([0.2, 0.4]),
    }


def _prime_adam(agent):
    local = agent.new_local_model()
    for parameter in local.parameters():
        parameter.grad = torch.ones_like(parameter)
    assert agent.apply_gradients(local)
    return local


def _snapshot(agent):
    return copy.deepcopy(
        (
            agent.model.state_dict(),
            agent.optimizer.state_dict(),
            agent.last_learning_rate,
        )
    )


class TestA3CAgent(unittest.TestCase):
    def assert_nested_equal(self, before, after):
        if isinstance(before, torch.Tensor):
            self.assertTrue(torch.equal(before, after))
        elif isinstance(before, dict):
            self.assertEqual(before.keys(), after.keys())
            for key in before:
                self.assert_nested_equal(before[key], after[key])
        elif isinstance(before, (list, tuple)):
            self.assertEqual(len(before), len(after))
            for left, right in zip(before, after):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(before, after)

    def test_model_shared_memory(self):
        agent = _make_agent()
        for parameter in agent.model.parameters():
            self.assertTrue(parameter.is_shared())
        self.assertTrue(agent.last_learning_rate.is_shared())

    def test_new_local_model_copies_weights(self):
        agent = _make_agent()
        local = agent.new_local_model()
        for shared, private in zip(agent.model.parameters(), local.parameters()):
            self.assertTrue(torch.equal(shared, private))
            self.assertFalse(private.is_shared())

    def test_compute_loss_returns_valid_stats(self):
        agent = _make_agent()
        inputs = _loss_inputs()
        loss, stats = agent.compute_loss(**inputs)
        expected_policy = -(inputs["advantages"] * inputs["log_probs"]).mean()
        expected_value = (inputs["values"] - inputs["returns"]).square().mean()
        expected = expected_policy + 0.5 * expected_value - 0.01 * 0.3
        torch.testing.assert_close(loss, expected)
        self.assertTrue(all(np.isfinite(value) for value in vars(stats).values()))
        self.assertAlmostEqual(stats.value_mean, 3.0)
        self.assertAlmostEqual(stats.return_mean, 5.0)
        self.assertAlmostEqual(stats.advantage_mean, 2.0)

    def test_apply_gradients_updates_global_model(self):
        agent = _make_agent(obs_dim=4, act_dim=2)
        local = agent.new_local_model()
        logits, values = local(torch.ones(3, 4))
        (logits.sum() + values.sum()).backward()
        before = [p.detach().clone() for p in agent.model.parameters()]
        self.assertIs(agent.apply_gradients(local), True)
        self.assertTrue(
            any(not torch.equal(p, q) for p, q in zip(before, agent.model.parameters()))
        )
        self.assertTrue(all(p.grad is None for p in local.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.model.parameters()))
        for shared, private in zip(agent.model.parameters(), local.parameters()):
            self.assertTrue(torch.equal(shared, private))

    def test_sync_local_reloads_weights(self):
        agent = _make_agent()
        local = agent.new_local_model()
        with torch.no_grad():
            for parameter in agent.model.parameters():
                parameter.add_(1.0)
        agent.sync_local(local)
        for shared, private in zip(agent.model.parameters(), local.parameters()):
            self.assertTrue(torch.equal(shared, private))

    def test_population_std_normalization(self):
        inputs = _loss_inputs()
        _, stats = _make_agent().compute_loss(**inputs, normalize_advantages=True)
        normalized = (inputs["advantages"] - 2.0) / (
            inputs["advantages"].std(unbiased=False) + 1e-8
        )
        expected = -(normalized * inputs["log_probs"]).mean().item()
        self.assertAlmostEqual(stats.policy_loss, expected, places=6)

    def test_singleton_and_constant_advantages_normalize_to_zero(self):
        for size in (1, 3):
            with self.subTest(size=size):
                inputs = {
                    key: value[:1].repeat(size) for key, value in _loss_inputs().items()
                }
                loss, stats = _make_agent().compute_loss(
                    **inputs, normalize_advantages=True
                )
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(stats.policy_loss, 0.0)

    def test_loss_rejects_empty_mismatched_and_nonvector_inputs(self):
        for replacement in (torch.empty(0), torch.ones(3), torch.ones(2, 1)):
            with self.subTest(shape=tuple(replacement.shape)):
                inputs = _loss_inputs()
                inputs["values"] = replacement
                with self.assertRaises(ValueError):
                    _make_agent().compute_loss(**inputs)

    def test_loss_rejects_nonfinite_inputs(self):
        for key in _loss_inputs():
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(key=key, invalid=invalid):
                    inputs = _loss_inputs()
                    inputs[key] = torch.tensor([invalid, 1.0])
                    with self.assertRaises(ValueError):
                        _make_agent().compute_loss(**inputs)

    def test_stats_accepts_original_four_positional_fields(self):
        stats = A3CStats(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(
            (stats.value_mean, stats.return_mean, stats.advantage_mean), (0.0, 0.0, 0.0)
        )

    def test_no_gradient_clears_stale_gradients_without_stepping_primed_adam(self):
        agent = _make_agent()
        local = _prime_adam(agent)
        before = _snapshot(agent)
        for parameter in agent.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        with patch.object(agent.optimizer, "step", wraps=agent.optimizer.step) as step:
            self.assertIs(agent.apply_gradients(local), False)
        step.assert_not_called()
        self.assert_nested_equal(before, _snapshot(agent))
        self.assertTrue(all(p.grad is None for p in agent.model.parameters()))
        self.assertTrue(all(p.grad is None for p in local.parameters()))

    def test_nonfinite_gradient_does_not_mutate_primed_adam(self):
        for invalid in (float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                agent = _make_agent()
                local = _prime_adam(agent)
                before = _snapshot(agent)
                for parameter in local.parameters():
                    parameter.grad = torch.ones_like(parameter)
                next(local.parameters()).grad.flatten()[0] = invalid
                with patch.object(
                    agent.optimizer, "step", wraps=agent.optimizer.step
                ) as step:
                    self.assertIs(agent.apply_gradients(local), False)
                step.assert_not_called()
                self.assert_nested_equal(before, _snapshot(agent))
                self.assertTrue(all(p.grad is None for p in local.parameters()))
                self.assertTrue(all(p.grad is None for p in agent.model.parameters()))

    def test_finite_gradient_with_overflowing_norm_is_rejected(self):
        agent = _make_agent()
        local = _prime_adam(agent)
        before = _snapshot(agent)
        for parameter in local.parameters():
            parameter.grad = torch.full_like(parameter, 1e30)
        with patch.object(agent.optimizer, "step", wraps=agent.optimizer.step) as step:
            self.assertIs(agent.apply_gradients(local), False)
        step.assert_not_called()
        self.assert_nested_equal(before, _snapshot(agent))
        self.assertTrue(all(p.grad is None for p in local.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.model.parameters()))

    def test_update_lr_and_optimizer_step_are_under_lock_with_optional_sync(self):
        lock = RecordingLock()
        agent = _make_agent(update_lock=lock)
        local = agent.new_local_model()
        before_local = [p.detach().clone() for p in local.parameters()]
        for parameter in local.parameters():
            parameter.grad = torch.ones_like(parameter)
        real_step = agent.optimizer.step

        def checked_step(*args, **kwargs):
            self.assertGreater(lock.depth, 0)
            self.assertEqual(agent.optimizer.param_groups[0]["lr"], 2e-4)
            return real_step(*args, **kwargs)

        with patch.object(agent.optimizer, "step", side_effect=checked_step):
            self.assertTrue(
                agent.apply_gradients(local, learning_rate=2e-4, sync=False)
            )
        for before, after in zip(before_local, local.parameters()):
            self.assertTrue(torch.equal(before, after))
        self.assertAlmostEqual(agent.last_learning_rate.item(), 2e-4, places=8)
        self.assertEqual(lock.depth, 0)

    def test_new_model_and_sync_read_coherent_snapshot_under_lock(self):
        lock = RecordingLock()
        agent = _make_agent(update_lock=lock)
        real_state_dict = agent.model.state_dict

        def checked_snapshot(*args, **kwargs):
            self.assertGreater(lock.depth, 0)
            return real_state_dict(*args, **kwargs)

        with patch.object(agent.model, "state_dict", side_effect=checked_snapshot):
            local = agent.new_local_model()
            agent.sync_local(local)
        self.assertGreaterEqual(lock.entries, 2)
        self.assertEqual(lock.depth, 0)
