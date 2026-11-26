"""TD3 agent tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/td3/test_agent.py
    >>> # Run with unittest
    >>> # python -m unittest tests.td3.test_agent
"""

import copy
import unittest
from unittest.mock import patch

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
    actions = torch.randn((batch_size, act_dim), dtype=torch.float32).clamp(-2.0, 2.0)
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


def _fixed_batch(n=4, obs_dim=3, act_dim=2):
    obs = torch.arange(n * obs_dim, dtype=torch.float32).reshape(n, obs_dim) / 20
    actions = torch.zeros(n, act_dim)
    rewards = torch.arange(n, dtype=torch.float32).reshape(n, 1)
    return obs, actions, rewards, obs + 0.1, torch.zeros(n, 1)


def _assert_nested_equal(case, left, right):
    if isinstance(left, torch.Tensor):
        case.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        case.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_nested_equal(case, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            _assert_nested_equal(case, a, b)
    else:
        case.assertEqual(left, right)


class TestAgentContracts(unittest.TestCase):
    def _agent(self, **kwargs):
        agent = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=(), **kwargs)
        agent.target_noise = 0.0
        return agent

    def test_deterministic_batched_act_does_not_consume_rng_or_counter(self):
        agent = self._agent()
        obs = np.zeros((4, 3), dtype=np.float32)
        np_before = copy.deepcopy(np.random.get_state())
        torch_before = torch.get_rng_state().clone()
        with patch("numpy.random.normal", side_effect=AssertionError("RNG used")):
            first = agent.act(obs, noise=99.0, deterministic=True)
            second = agent.act(obs, noise=99.0, deterministic=True)
            zero_noise = agent.act(obs, noise=0.0, deterministic=False)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, zero_noise)
        self.assertEqual(first.shape, (4, 2))
        self.assertEqual(agent.global_step, 0)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        now = np.random.get_state()
        self.assertEqual(np_before[0], now[0])
        np.testing.assert_array_equal(np_before[1], now[1])
        self.assertEqual(np_before[2:], now[2:])

    def test_act_rejects_invalid_shapes_nonfinite_and_noise(self):
        agent = self._agent()
        for obs in [
            np.zeros(2),
            np.zeros((2, 4)),
            np.zeros((2, 1, 3)),
            np.full(3, np.nan),
            np.full(3, np.inf),
        ]:
            with (
                self.subTest(obs=obs.shape),
                self.assertRaises((ValueError, TypeError)),
            ):
                agent.act(obs)
        for noise in [-0.1, float("nan"), float("inf"), True]:
            with self.subTest(noise=noise), self.assertRaises((ValueError, TypeError)):
                agent.act(np.zeros(3), noise=noise)

    def test_target_uses_twin_min_and_terminal_mask(self):
        agent = self._agent()
        with torch.no_grad():
            for critic, value in [(agent.q1_target, 7.0), (agent.q2_target, 2.0)]:
                critic.head.weight.zero_()
                critic.head.bias.fill_(value)
        batch = list(_fixed_batch())
        batch[-1] = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        expected = batch[2] + (1 - batch[-1]) * agent.gamma * 2.0
        import TD3.td3.agent as module

        real_mse = module.F.mse_loss
        targets = []

        def record(current, target, *args, **kwargs):
            targets.append(target.detach().clone())
            self.assertFalse(target.requires_grad)
            return real_mse(current, target, *args, **kwargs)

        with patch.object(module.F, "mse_loss", side_effect=record):
            agent.update(tuple(batch))
        self.assertEqual(len(targets), 2)
        for target in targets:
            torch.testing.assert_close(target, expected)

    def test_actor_uses_q1_not_twin_min_and_preserves_action_gradient(self):
        agent = self._agent(policy_delay=1)
        with torch.no_grad():
            agent.actor.head.weight.zero_()
            agent.actor.head.bias.zero_()
            agent.q1.head.weight.zero_()
            agent.q1.head.weight[0, -2:] = 1.0
            agent.q1.head.bias.zero_()
            agent.q2.head.weight.zero_()
            agent.q2.head.weight[0, -2:] = -100.0
            agent.q2.head.bias.fill_(-100.0)
        with patch.object(agent.critic_opt, "step"):
            stats = agent.update(_fixed_batch())
        self.assertAlmostEqual(stats.actor_loss, 0.0, places=6)
        self.assertTrue((agent.actor.head.bias > 0).all().item())

    def test_all_target_networks_wait_for_policy_delay_and_soft_update(self):
        agent = self._agent(policy_delay=2, tau=0.5)
        names = ["actor", "q1", "q2"]
        before = {
            name: [
                p.detach().clone()
                for p in getattr(agent, name + "_target").parameters()
            ]
            for name in names
        }
        agent.update(_fixed_batch())
        for name in names:
            for old, target in zip(
                before[name], getattr(agent, name + "_target").parameters()
            ):
                torch.testing.assert_close(old, target, rtol=0, atol=0)
        agent.update(_fixed_batch())
        for name in names:
            for old, source, target in zip(
                before[name],
                getattr(agent, name).parameters(),
                getattr(agent, name + "_target").parameters(),
            ):
                torch.testing.assert_close(target, 0.5 * source + 0.5 * old)

    def test_targets_are_frozen_and_never_receive_gradients(self):
        agent = self._agent(policy_delay=1)
        agent.update(_fixed_batch())
        for name in ["actor_target", "q1_target", "q2_target"]:
            for param in getattr(agent, name).parameters():
                self.assertFalse(param.requires_grad)
                self.assertIsNone(param.grad)

    def test_online_critic_freeze_is_temporary_during_actor_backward(self):
        agent = self._agent(policy_delay=1)
        actor_phase_flags = []

        def capture(module, args, output):
            if args[1].requires_grad:
                actor_phase_flags.append([p.requires_grad for p in module.parameters()])
                self.assertTrue(output.requires_grad)

        handle = agent.q1.register_forward_hook(capture)
        try:
            agent.update(_fixed_batch())
        finally:
            handle.remove()
        self.assertEqual(len(actor_phase_flags), 1)
        self.assertFalse(any(actor_phase_flags[0]))
        self.assertTrue(all(p.requires_grad for p in agent.q1.parameters()))
        self.assertTrue(all(p.requires_grad for p in agent.q2.parameters()))

    def test_vector_and_column_rewards_dones_are_identical_no_nxn_target(self):
        initial = self._agent().state_dict()
        results = []
        for columns in [True, False]:
            agent = self._agent()
            agent.load_state_dict(initial)
            batch = list(_fixed_batch())
            if not columns:
                batch[2] = batch[2].flatten()
                batch[4] = batch[4].flatten()
            import TD3.td3.agent as module

            real_mse = module.F.mse_loss

            def record(current, target, *args, **kwargs):
                self.assertEqual(tuple(target.shape), (4, 1))
                return real_mse(current, target, *args, **kwargs)

            with patch.object(module.F, "mse_loss", side_effect=record):
                stats = agent.update(tuple(batch))
            results.append((stats, agent.state_dict()))
        self.assertEqual(results[0][0], results[1][0])
        _assert_nested_equal(self, results[0][1], results[1][1])

    def test_invalid_batch_prevalidation_leaves_live_state_uncorrupted(self):
        agent = self._agent(policy_delay=1)
        agent.update(_fixed_batch())
        before = agent.state_dict()
        for field in range(5):
            bad = list(_fixed_batch())
            bad[field] = bad[field].clone()
            bad[field].flatten()[0] = float("nan")
            with self.subTest(field=field), self.assertRaises((ValueError, TypeError)):
                agent.update(tuple(bad))
            _assert_nested_equal(self, before, agent.state_dict())
        for index, value in [
            (1, torch.full((4, 2), 3.0)),
            (4, torch.full((4, 1), 0.5)),
            (2, torch.zeros(4, 4)),
            (0, torch.zeros(3, 3)),
        ]:
            bad = list(_fixed_batch())
            bad[index] = value
            with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                agent.update(tuple(bad))
            _assert_nested_equal(self, before, agent.state_dict())

    def test_snapshot_is_deep_cpu_owned_including_adam(self):
        agent = self._agent(policy_delay=1)
        agent.update(_fixed_batch())
        snapshot = agent.state_dict()
        frozen = copy.deepcopy(snapshot)

        def check(value):
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device.type, "cpu")
                self.assertFalse(value.requires_grad)
            elif isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    check(item)

        check(snapshot)
        agent.update(_fixed_batch())
        _assert_nested_equal(self, snapshot, frozen)
        snapshot["actor_state_dict"]["head.bias"].add_(100)
        self.assertFalse(
            torch.equal(
                snapshot["actor_state_dict"]["head.bias"], agent.actor.head.bias
            )
        )

    def test_restore_retains_distinct_targets_adam_and_policy_counter(self):
        agent = self._agent(policy_delay=2)
        for _ in range(3):
            agent.update(_fixed_batch())
        state = agent.state_dict()
        self.assertTrue(state["actor_optimizer_state_dict"]["state"])
        self.assertTrue(state["critic_optimizer_state_dict"]["state"])
        for name in ["actor", "q1", "q2"]:
            self.assertTrue(
                any(
                    not torch.equal(a, b)
                    for a, b in zip(
                        state[name + "_state_dict"].values(),
                        state[name + "_target_state_dict"].values(),
                    )
                )
            )
        restored = self._agent(policy_delay=2)
        restored.load_state_dict(state)
        _assert_nested_equal(self, restored.state_dict(), state)
        expected = agent.update(_fixed_batch())
        actual = restored.update(_fixed_batch())
        self.assertIsNotNone(actual.actor_loss)
        self.assertEqual(expected, actual)
        _assert_nested_equal(self, agent.state_dict(), restored.state_dict())

    def test_legacy_online_only_state_initializes_targets(self):
        agent = self._agent()
        state = agent.state_dict()
        for name in ["actor", "q1", "q2"]:
            state.pop(name + "_target_state_dict")
        restored = self._agent()
        restored.load_state_dict(state)
        for name in ["actor", "q1", "q2"]:
            _assert_nested_equal(
                self,
                getattr(restored, name).state_dict(),
                getattr(restored, name + "_target").state_dict(),
            )

    def test_invalid_model_and_adam_restore_is_atomic(self):
        agent = self._agent(policy_delay=1)
        agent.update(_fixed_batch())
        before = agent.state_dict()
        invalid = copy.deepcopy(before)
        invalid["q2_state_dict"]["head.bias"].fill_(float("nan"))
        missing = copy.deepcopy(before)
        missing.pop("q1_state_dict")
        bad_adam = copy.deepcopy(before)
        slots = bad_adam["critic_optimizer_state_dict"]["state"]
        next(iter(slots.values()))["exp_avg"].fill_(float("nan"))
        for payload in [invalid, missing, bad_adam]:
            with self.assertRaises((ValueError, TypeError, KeyError, RuntimeError)):
                agent.load_state_dict(payload)
            _assert_nested_equal(self, before, agent.state_dict())

    def test_hyperparameters_are_typed_and_strictly_bounded(self):
        obs = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        act = spaces.Box(-2.0, 2.0, (2,), dtype=np.float32)
        base = dict(
            obs_space=obs,
            act_space=act,
            hidden_sizes=(),
            activation="relu",
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            policy_delay=2,
            target_noise=0.2,
            noise_clip=0.5,
            max_grad_norm=0.0,
            device="cpu",
        )
        invalid = {
            "actor_lr": [0, True, float("nan")],
            "critic_lr": [-1, float("inf")],
            "gamma": [-0.1, 1.1, True],
            "tau": [-0.1, 1.1, float("nan")],
            "policy_delay": [0, 1.5, True],
            "target_noise": [-0.1, float("inf")],
            "noise_clip": [-1, float("nan")],
            "max_grad_norm": [-1, float("inf")],
        }
        for name, values in invalid.items():
            for value in values:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    TD3Agent(**{**base, name: value})

    def test_smoothing_noise_clipped_then_exact_asymmetric_action_bounds(self):
        low = np.array([-0.001, 2.0], dtype=np.float32)
        high = np.array([0.002, 3.0], dtype=np.float32)
        agent = TD3Agent(
            spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
            spaces.Box(low, high, dtype=np.float32),
            (),
            "relu",
            1e-3,
            1e-3,
            0.99,
            0.5,
            2,
            0.2,
            0.1,
            "cpu",
        )
        with torch.no_grad():
            agent.actor_target.head.weight.zero_()
            agent.actor_target.head.bias.zero_()
        seen = []
        handle = agent.q1_target.register_forward_pre_hook(
            lambda module, args: seen.append(args[1].detach().clone())
        )
        try:
            with patch(
                "torch.randn_like", side_effect=lambda x: torch.full_like(x, 1000)
            ):
                batch = list(_fixed_batch())
                batch[1][:, 1] = 2.5
                agent.update(tuple(batch))
        finally:
            handle.remove()
        expected = torch.tensor([[0.002, 2.6]]).expand(4, 2)
        torch.testing.assert_close(seen[0], expected)
        with patch("numpy.random.normal", return_value=np.full((4, 2), 1000.0)):
            actions = agent.act(np.zeros((4, 3), dtype=np.float32), noise=1.0)
        self.assertTrue(np.all(actions <= high))
        self.assertTrue(np.all(actions >= low))


if __name__ == "__main__":
    unittest.main()
