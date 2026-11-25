"""SAC agent tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/sac/test_agent.py
    >>> # Run with unittest
    >>> # python -m unittest tests.sac.test_agent
"""

import copy
import math
import unittest
from unittest.mock import patch

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
        actions = torch.tanh(torch.randn((batch_size, 2), dtype=torch.float32))
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
            torch.tanh(torch.randn((batch_size, 2), dtype=torch.float32)),
            torch.randn((batch_size, 1), dtype=torch.float32),
            torch.randn((batch_size, 3), dtype=torch.float32),
            torch.zeros((batch_size, 1), dtype=torch.float32),
        )
        agent.update(batch)
        with torch.no_grad():
            for target in (agent.q1_target, agent.q2_target):
                for param in target.parameters():
                    param.add_(0.125)

        state = agent.state_dict()
        restored = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=(8,))
        restored.load_state_dict(state)

        self.assertEqual(restored.global_step, agent.global_step)
        self.assertTrue(
            torch.allclose(agent.log_alpha.detach(), restored.log_alpha.detach())
        )
        for param, restored_param in zip(
            agent.actor.parameters(), restored.actor.parameters()
        ):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(
            agent.q1.parameters(), restored.q1.parameters()
        ):
            self.assertTrue(torch.allclose(param, restored_param))
        for param, restored_param in zip(
            agent.q2.parameters(), restored.q2.parameters()
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

    def test_deterministic_single_action_consumes_no_rng_or_update_count(self) -> None:
        agent = _make_agent(obs_dim=3)
        obs = np.zeros(3, dtype=np.float32)
        torch_before = torch.random.get_rng_state().clone()
        numpy_before = np.random.get_state()
        with patch.object(
            agent.actor, "sample", side_effect=AssertionError("stochastic path")
        ):
            first = agent.act(obs, deterministic=True)
            second = agent.act(obs, deterministic=True)
        np.testing.assert_array_equal(first, second)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_before)
        after = np.random.get_state()
        self.assertEqual(numpy_before[0], after[0])
        np.testing.assert_array_equal(numpy_before[1], after[1])
        self.assertEqual(numpy_before[2:], after[2:])
        self.assertEqual(agent.global_step, 0)

    def test_batched_action_preserves_every_observation_and_singleton_axis(
        self,
    ) -> None:
        agent = _make_agent(obs_dim=3)
        obs = np.array([[0.0, 0.5, -0.5], [0.8, -0.2, 0.1]], dtype=np.float32)
        before = torch.random.get_rng_state().clone()
        batched = agent.act(obs, deterministic=True)
        self.assertEqual(batched.shape, (2, 2))
        for row, action in zip(obs, batched):
            np.testing.assert_allclose(
                action, agent.act(row, deterministic=True), atol=1e-6
            )
        self.assertEqual(agent.act(obs[:1], deterministic=True).shape, (1, 2))
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        self.assertEqual(agent.global_step, 0)

    def test_act_rejects_malformed_or_nonfinite_observations(self) -> None:
        agent = _make_agent(obs_dim=3)
        for obs in [
            np.zeros(2),
            np.zeros((0, 3)),
            np.zeros((2, 2)),
            np.zeros((1, 1, 3)),
            np.array([0.0, np.nan, 0.0]),
        ]:
            with self.subTest(shape=obs.shape), self.assertRaises(ValueError):
                agent.act(obs, deterministic=True)
        self.assertEqual(agent.global_step, 0)

    def test_target_critics_are_distinct_frozen_and_never_accumulate_gradients(
        self,
    ) -> None:
        agent = _make_agent(obs_dim=3)
        for online, target in [
            (agent.q1, agent.q1_target),
            (agent.q2, agent.q2_target),
        ]:
            for source, dest in zip(online.parameters(), target.parameters()):
                torch.testing.assert_close(source, dest)
                self.assertNotEqual(source.data_ptr(), dest.data_ptr())
                self.assertFalse(dest.requires_grad)
                self.assertIsNone(dest.grad)
        agent.update(self._batch())
        for target in (agent.q1_target, agent.q2_target):
            self.assertTrue(
                all(not p.requires_grad and p.grad is None for p in target.parameters())
            )

    def test_target_entropy_scales_with_action_dimension(self) -> None:
        agent = _make_agent(act_dim=3)
        self.assertEqual(agent.target_entropy, -3.0)
        kwargs = dict(
            obs_space=spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
            act_space=spaces.Box(-2.0, 2.0, (2,), dtype=np.float32),
            hidden_sizes=(),
            activation="relu",
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
            gamma=0.99,
            tau=0.5,
            target_entropy_scale=0.25,
            device="cpu",
        )
        self.assertEqual(SACAgent(**kwargs).target_entropy, -0.5)

    def test_sac_target_uses_minimum_q_entropy_and_terminated_only_mask(self) -> None:
        agent = _make_agent(obs_dim=3)
        agent.gamma = 0.5
        with torch.no_grad():
            agent.log_alpha.fill_(math.log(2.0))
            for module in (agent.q1, agent.q2):
                for param in module.parameters():
                    param.zero_()
        obs, acts, _, next_obs, _ = self._batch(n=2)
        rewards = torch.tensor([1.0, 2.0])
        dones = torch.tensor([0.0, 1.0])

        def fixed_sample(value):
            actions = agent.actor.deterministic(value)
            log_prob = actions.sum(-1, keepdim=True) * 0.0 - 0.5
            return actions, log_prob, actions

        with (
            patch.object(agent.actor, "sample", side_effect=fixed_sample),
            patch.object(
                agent.q1_target, "forward", return_value=torch.full((2, 1), 4.0)
            ),
            patch.object(
                agent.q2_target, "forward", return_value=torch.full((2, 1), 6.0)
            ),
            patch.object(agent.critic_opt, "step"),
            patch.object(agent.actor_opt, "step"),
        ):
            stats = agent.update((obs, acts, rewards, next_obs, dones))
        expected_target = torch.tensor([3.5, 2.0])
        self.assertAlmostEqual(
            stats.critic_loss, float(2.0 * expected_target.square().mean()), places=5
        )
        self.assertAlmostEqual(stats.target_q_mean, 2.75, places=5)
        self.assertAlmostEqual(stats.actor_loss, -1.0, places=5)
        self.assertAlmostEqual(stats.alpha_loss, math.log(2.0) * 2.5, places=5)
        self.assertEqual(agent.global_step, 1)

    def test_actor_backprop_freezes_online_critics_without_extra_critic_step(
        self,
    ) -> None:
        torch.manual_seed(8)
        agent = _make_agent(obs_dim=3)
        flags = {"q1": [], "q2": []}
        before = [p.detach().clone() for p in agent.actor.parameters()]
        handles = [
            module.register_forward_pre_hook(
                lambda module, args, name=name: flags[name].append(
                    tuple(p.requires_grad for p in module.parameters())
                )
            )
            for name, module in [("q1", agent.q1), ("q2", agent.q2)]
        ]
        try:
            with patch.object(
                agent.critic_opt, "step", wraps=agent.critic_opt.step
            ) as step:
                agent.update(self._batch())
            self.assertEqual(step.call_count, 1)
        finally:
            for handle in handles:
                handle.remove()
        for stages in flags.values():
            self.assertEqual(len(stages), 2)
            self.assertTrue(all(stages[0]))
            self.assertFalse(any(stages[1]))
        self.assertTrue(
            all(
                p.requires_grad
                for module in (agent.q1, agent.q2)
                for p in module.parameters()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, agent.actor.parameters())
            )
        )

    def test_soft_update_exact_interpolation_uses_post_update_online_weights(
        self,
    ) -> None:
        agent = _make_agent(obs_dim=3, tau=0.25)
        before = [p.detach().clone() for p in agent.q1_target.parameters()]
        agent.update(self._batch())
        for old, online, target in zip(
            before, agent.q1.parameters(), agent.q1_target.parameters()
        ):
            torch.testing.assert_close(target, 0.75 * old + 0.25 * online.detach())

    def test_state_snapshot_is_owned_cpu_data_including_adam_moments(self) -> None:
        agent = _make_agent(obs_dim=3)
        agent.update(self._batch())
        state = agent.state_dict()
        baseline = copy.deepcopy(state)
        for tensor in self._tensors(state):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertFalse(tensor.requires_grad)
        self.assertTrue(state["alpha_optimizer_state_dict"]["state"])
        agent.update(self._batch())
        self._assert_tree_equal(state, baseline)
        state["actor_state_dict"][next(iter(state["actor_state_dict"]))].add_(10.0)
        self.assertFalse(
            torch.equal(
                state["actor_state_dict"][next(iter(state["actor_state_dict"]))],
                agent.actor.state_dict()[next(iter(state["actor_state_dict"]))],
            )
        )

    def test_restore_does_not_alias_caller_optimizer_moments(self) -> None:
        source = _make_agent(obs_dim=3)
        source.update(self._batch())
        state = source.state_dict()
        restored = _make_agent(obs_dim=3)
        restored.load_state_dict(state)
        baseline = restored.state_dict()
        for tensor in self._tensors(state):
            tensor.zero_()
        self._assert_tree_equal(restored.state_dict(), baseline)

    def test_alpha_parameter_identity_and_optimizer_restore_survive_update(
        self,
    ) -> None:
        source = _make_agent(obs_dim=3)
        source.update(self._batch())
        restored = _make_agent(obs_dim=3)
        parameter = restored.log_alpha
        restored.load_state_dict(source.state_dict())
        self.assertIs(restored.log_alpha, parameter)
        self.assertIs(restored.alpha_opt.param_groups[0]["params"][0], parameter)
        torch.testing.assert_close(
            restored.alpha_opt.state[parameter]["exp_avg"],
            source.alpha_opt.state[source.log_alpha]["exp_avg"],
        )
        before = parameter.detach().clone()
        stats = restored.update(self._batch())
        self.assertFalse(torch.equal(before, parameter.detach()))
        self.assertTrue(np.isfinite(stats.alpha_loss))
        self.assertTrue(np.isfinite(stats.alpha_value))
        self.assertGreater(stats.alpha_value, 0.0)
        self.assertEqual(restored.global_step, 2)

    def test_malformed_checkpoint_is_validated_before_any_mutation(self) -> None:
        agent = _make_agent(obs_dim=3)
        agent.update(self._batch())
        baseline = agent.state_dict()
        alpha_parameter = agent.log_alpha
        optimizers = (agent.actor_opt, agent.critic_opt, agent.alpha_opt)
        corruptions = []
        missing = copy.deepcopy(baseline)
        del missing["q1_state_dict"]
        corruptions.append(missing)
        wrong_shape = copy.deepcopy(baseline)
        key = next(iter(wrong_shape["q2_state_dict"]))
        wrong_shape["q2_state_dict"][key] = torch.zeros(99)
        corruptions.append(wrong_shape)
        nonfinite = copy.deepcopy(baseline)
        nonfinite["log_alpha"] = torch.tensor([float("nan")])
        corruptions.append(nonfinite)
        unknown = copy.deepcopy(baseline)
        unknown["q2_state_dict"]["unexpected_parameter"] = torch.zeros(1)
        corruptions.append(unknown)
        invalid_moment = copy.deepcopy(baseline)
        slots = next(
            iter(invalid_moment["alpha_optimizer_state_dict"]["state"].values())
        )
        slots["exp_avg_sq"] = torch.full_like(slots["exp_avg_sq"], -1.0)
        corruptions.append(invalid_moment)
        incomplete_amsgrad = copy.deepcopy(baseline)
        incomplete_amsgrad["alpha_optimizer_state_dict"]["param_groups"][0][
            "amsgrad"
        ] = True
        corruptions.append(incomplete_amsgrad)
        for state in corruptions:
            # A valid early model change exposes partially applied later failures.
            weight = next(
                key for key in state["actor_state_dict"] if key.endswith("weight")
            )
            state["actor_state_dict"][weight].add_(0.25)
            with self.subTest(keys=list(state)), self.assertRaises(ValueError):
                agent.load_state_dict(state)
            self._assert_tree_equal(agent.state_dict(), baseline)
            self.assertIs(agent.log_alpha, alpha_parameter)
            for actual, original in zip(
                (agent.actor_opt, agent.critic_opt, agent.alpha_opt), optimizers
            ):
                self.assertIs(actual, original)

    def test_incompatible_action_bounds_checkpoint_is_rejected_before_mutation(
        self,
    ) -> None:
        source = _make_agent(obs_dim=3)
        restored = SACAgent(
            spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
            spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
            (8,),
            "relu",
            1e-3,
            1e-3,
            1e-3,
            0.99,
            0.5,
            1.0,
            "cpu",
        )
        baseline = restored.state_dict()
        with self.assertRaises(ValueError):
            restored.load_state_dict(source.state_dict())
        self._assert_tree_equal(restored.state_dict(), baseline)

    def test_legacy_checkpoint_omitted_targets_fall_back_to_online_critics(
        self,
    ) -> None:
        source = _make_agent(obs_dim=3)
        state = source.state_dict()
        for key in [
            "q1_target_state_dict",
            "q2_target_state_dict",
            "actor_optimizer_state_dict",
            "critic_optimizer_state_dict",
            "alpha_optimizer_state_dict",
        ]:
            del state[key]
        restored = _make_agent(obs_dim=3)
        restored.load_state_dict(state)
        for online, target in [
            (restored.q1, restored.q1_target),
            (restored.q2, restored.q2_target),
        ]:
            self._assert_tree_equal(online.state_dict(), target.state_dict())
            self.assertTrue(all(not p.requires_grad for p in target.parameters()))

    def test_invalid_update_inputs_leave_parameters_moments_and_counter_unchanged(
        self,
    ) -> None:
        agent = _make_agent(obs_dim=3)
        agent.update(self._batch())
        before = agent.state_dict()
        bad_batches = []
        for index, value in [
            (0, torch.zeros(4, 2)),
            (1, torch.full((4, 2), 3.0)),
            (2, torch.full((4, 1), float("inf"))),
            (4, torch.full((4, 1), 1.5)),
        ]:
            batch = list(self._batch())
            batch[index] = value
            bad_batches.append(tuple(batch))
        for batch in bad_batches:
            with (
                self.subTest(shapes=[tuple(t.shape) for t in batch]),
                self.assertRaises(ValueError),
            ):
                agent.update(batch)
            self._assert_tree_equal(agent.state_dict(), before)

    @staticmethod
    def _batch(n=4):
        return (
            torch.linspace(-0.5, 0.5, n * 3).reshape(n, 3),
            torch.zeros(n, 2),
            torch.ones(n, 1),
            torch.linspace(0.5, -0.5, n * 3).reshape(n, 3),
            torch.zeros(n, 1),
        )

    @staticmethod
    def _tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from TestSACAgent._tensors(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from TestSACAgent._tensors(child)

    def _assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self._assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self._assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
