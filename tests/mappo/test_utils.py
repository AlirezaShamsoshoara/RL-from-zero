"""MAPPO checkpoint integrity and lazy, mocked PettingZoo factory tests."""

import copy
import os
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from gymnasium import spaces

from MAPPO.mappo import utils


class SpaceEnv:
    def __init__(self):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents.copy()
        self.obs = {
            name: spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
            for name in self.possible_agents
        }
        self.actions = {
            name: spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
            for name in self.possible_agents
        }
        self.reset = MagicMock(return_value=({}, {}))
        self.close = MagicMock()

    def observation_space(self, name):
        return self.obs[name]

    def action_space(self, name):
        return self.actions[name]


def fake_pettingzoo(env):
    package = types.ModuleType("pettingzoo")
    package.__path__ = []
    sisl = types.ModuleType("pettingzoo.sisl")
    sisl.__path__ = []
    multiwalker = types.ModuleType("pettingzoo.sisl.multiwalker_v9")
    multiwalker.parallel_env = MagicMock(return_value=env)
    sisl.multiwalker_v9 = multiwalker
    package.sisl = sisl
    return {
        "pettingzoo": package,
        "pettingzoo.sisl": sisl,
        "pettingzoo.sisl.multiwalker_v9": multiwalker,
    }, multiwalker.parallel_env


class TestUtils(unittest.TestCase):
    def assert_nested_equal(self, one, two):
        if isinstance(one, torch.Tensor):
            torch.testing.assert_close(one, two, rtol=0, atol=0)
        elif isinstance(one, dict):
            self.assertEqual(set(one), set(two))
            for key in one:
                self.assert_nested_equal(one[key], two[key])
        elif isinstance(one, (list, tuple)):
            self.assertEqual(len(one), len(two))
            for left, right in zip(one, two):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(one, two)

    def optimize(self, models, optimizers):
        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad()
            (model(torch.ones((2, 3))) - 1.0).square().mean().backward()
            optimizer.step()

    def test_set_seed_reproducible(self):
        utils.set_seed(123)
        first = (random.random(), np.random.rand(), float(torch.rand(1).item()))
        utils.set_seed(123)
        second = (random.random(), np.random.rand(), float(torch.rand(1).item()))
        self.assertEqual(first, second)

    def test_get_space_dims_uses_real_box_or_discrete_spaces(self):
        env = SpaceEnv()
        self.assertEqual(utils.get_space_dims(env)[:4], (3, 2, 6, 2))
        env.actions = {name: spaces.Discrete(4) for name in env.possible_agents}
        self.assertEqual(utils.get_space_dims(env)[:4], (3, 4, 6, 2))

    def test_shared_and_separate_checkpoint_restore_adam_and_exact_next_step(self):
        for shared in (True, False):
            with self.subTest(shared=shared):
                torch.manual_seed(5)
                if shared:
                    model = torch.nn.Linear(3, 2)
                    models = [model, model]
                    optimizers = [torch.optim.Adam(model.parameters(), lr=0.001)]
                else:
                    models = [torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)]
                    optimizers = [
                        torch.optim.Adam(model.parameters(), lr=0.001)
                        for model in models
                    ]
                self.optimize(models, optimizers)
                expected_models = [
                    copy.deepcopy(model.state_dict()) for model in models
                ]
                expected_optimizers = [
                    copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers
                ]
                restored_one = torch.nn.Linear(3, 2)
                restored = (
                    [restored_one, restored_one]
                    if shared
                    else [restored_one, torch.nn.Linear(3, 2)]
                )
                restored_optimizers = [
                    torch.optim.Adam(model.parameters(), lr=0.5)
                    for model in (restored[:1] if shared else restored)
                ]
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "checkpoint.pt")
                    utils.save_checkpoint(
                        path, models, optimizers, 7, 2.5, shared, config={"n_agents": 2}
                    )
                    data = utils.load_checkpoint(path, restored, restored_optimizers)
                self.assertEqual(
                    (data["step"], data["best_return"], data["share_policy"]),
                    (7, 2.5, shared),
                )
                self.assertEqual(data["config"], {"n_agents": 2})
                for expected, actual in zip(expected_models, restored):
                    self.assert_nested_equal(expected, actual.state_dict())
                for expected, actual in zip(expected_optimizers, restored_optimizers):
                    self.assertTrue(expected["state"])
                    self.assert_nested_equal(expected, actual.state_dict())
                self.optimize(models, optimizers)
                self.optimize(restored, restored_optimizers)
                for expected, actual in zip(models, restored):
                    self.assert_nested_equal(expected.state_dict(), actual.state_dict())

    def test_shared_inference_can_restore_multiple_unique_models(self):
        shared = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(shared.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "shared.pt")
            utils.save_checkpoint(path, [shared, shared], [optimizer], 1, 1.0, True)
            restored = [torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)]
            utils.load_checkpoint(path, restored, None)
        for model in restored:
            self.assert_nested_equal(shared.state_dict(), model.state_dict())

    def test_wrong_model_optimizer_counts_and_step_metadata_are_rejected(self):
        models = [torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)]
        optimizers = [torch.optim.Adam(model.parameters()) for model in models]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "separate.pt")
            utils.save_checkpoint(path, models, optimizers, 1, 0.0, False)
            with self.assertRaises(ValueError):
                utils.load_checkpoint(path, models[:1], optimizers[:1])
            with self.assertRaises(ValueError):
                utils.load_checkpoint(path, models, optimizers[:1])
            for step in (1.5, True, -1):
                with self.subTest(step=step), self.assertRaises(ValueError):
                    utils.save_checkpoint(path, models, optimizers, step, 0.0, False)

    def test_wrong_checkpoint_architecture_does_not_silently_load(self):
        model = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "shared.pt")
            utils.save_checkpoint(
                path, [model], [torch.optim.Adam(model.parameters())], 1, 0.0, True
            )
            with self.assertRaises((ValueError, RuntimeError)):
                utils.load_checkpoint(path, [torch.nn.Linear(4, 2)], None)

    def test_factory_lazily_imports_stub_and_preserves_overridable_defaults(self):
        env = SpaceEnv()
        modules, factory = fake_pettingzoo(env)
        with (
            patch.dict(sys.modules, modules),
            patch.object(env.actions["agent_0"], "seed") as first,
            patch.object(env.actions["agent_1"], "seed") as second,
        ):
            actual = utils.make_multiwalker_env(
                n_walkers=2,
                seed=17,
                discretize_actions=False,
                env_kwargs={"max_cycles": 13, "terminate_on_fall": False},
            )
        self.assertIs(getattr(actual, "env", actual), env)
        self.assertEqual(factory.call_args.kwargs["n_walkers"], 2)
        self.assertEqual(factory.call_args.kwargs["max_cycles"], 13)
        self.assertIs(factory.call_args.kwargs["terminate_on_fall"], False)
        self.assertEqual(factory.call_args.kwargs["forward_reward"], 1.0)
        env.reset.assert_called_once_with(seed=17)
        first.assert_called_once()
        second.assert_called_once()

    def test_native_time_limit_adapter_preserves_rewards_and_distinguishes_same_step_failure(
        self,
    ):
        for frames, true_dones, expected_terms, expected_truncs in (
            (
                2,
                {0: False, 1: False},
                {"agent_0": False, "agent_1": False},
                {"agent_0": True, "agent_1": True},
            ),
            (
                2,
                {0: True, 1: False},
                {"agent_0": True, "agent_1": False},
                {"agent_0": False, "agent_1": True},
            ),
            (
                1,
                {0: False, 1: False},
                {"agent_0": True, "agent_1": True},
                {"agent_0": False, "agent_1": False},
            ),
        ):
            with self.subTest(frames=frames, true_dones=true_dones):
                engine = types.SimpleNamespace(
                    frames=frames, max_cycles=2, get_last_dones=lambda: true_dones
                )
                env = SpaceEnv()
                if true_dones.get(0):
                    env.agents = [
                        "agent_1"
                    ]  # Engine indices remain original-agent indices.
                env.aec_env = types.SimpleNamespace(
                    unwrapped=types.SimpleNamespace(env=engine)
                )
                observations = {
                    name: np.ones(3, dtype=np.float32) for name in env.possible_agents
                }
                rewards = {"agent_0": -100.0, "agent_1": 1.0}
                raw_terms = dict.fromkeys(env.possible_agents, True)
                raw_truncs = dict.fromkeys(env.possible_agents, False)
                env.step = MagicMock(
                    return_value=(observations, rewards, raw_terms, raw_truncs, {})
                )
                adapter = utils.MultiwalkerTimeLimitAdapter(env)
                actual_obs, actual_rewards, terms, truncs, _ = adapter.step(
                    {"agent_0": 0, "agent_1": 0}
                )
                self.assertIs(actual_obs, observations)
                self.assertIs(actual_rewards, rewards)
                self.assertEqual(terms, expected_terms)
                self.assertEqual(truncs, expected_truncs)
                self.assertEqual(raw_terms, dict.fromkeys(env.possible_agents, True))

    def test_continuous_agent_checkpoints_restore_adam_and_reproduce_next_joint_update(
        self,
    ):
        from MAPPO.mappo.agent import MAPPOAgent, Batch

        for shared in (True, False):
            with (
                self.subTest(shared=shared),
                tempfile.TemporaryDirectory() as directory,
            ):

                def learner():
                    return MAPPOAgent(
                        2,
                        2,
                        2,
                        4,
                        [4],
                        [4],
                        "tanh",
                        0.001,
                        0.2,
                        0.01,
                        0.5,
                        0.5,
                        "cpu",
                        share_policy=shared,
                        action_type="continuous",
                    )

                original, restored = learner(), learner()
                batches = []
                for model in original.models:
                    obs, state = torch.ones((2, 2)), torch.ones((2, 4))
                    with torch.no_grad():
                        action, logprob, _ = model.act(obs, deterministic=True)
                        value = model.get_value(state)
                    advantage = torch.tensor([-1.0, 1.0])
                    batches.append(
                        Batch(
                            obs,
                            state,
                            action.detach(),
                            logprob.detach(),
                            value.detach() + advantage,
                            advantage,
                            value.detach(),
                            torch.ones(2),
                        )
                    )
                original.update(batches)
                path = os.path.join(directory, "continuous.pt")
                utils.save_checkpoint(
                    path, original.models, original.optimizers, 2, 1.0, shared
                )
                utils.load_checkpoint(path, restored.models, restored.optimizers)
                for one, two in zip(original.models, restored.models):
                    self.assert_nested_equal(one.state_dict(), two.state_dict())
                for one, two in zip(original.optimizers, restored.optimizers):
                    self.assertTrue(one.state_dict()["state"])
                    self.assert_nested_equal(one.state_dict(), two.state_dict())
                left, left_steps = original.update(batches)
                right, right_steps = restored.update(batches)
                self.assertEqual(left_steps, right_steps)
                self.assertEqual(left, right)
                for one, two in zip(original.models, restored.models):
                    self.assert_nested_equal(one.state_dict(), two.state_dict())

    def test_continuous_checkpoint_action_bounds_mismatch_is_rejected_shared_and_separate(
        self,
    ):
        from MAPPO.mappo.agent import MAPPOAgent

        def learner(shared, low, high):
            return MAPPOAgent(
                2,
                2,
                2,
                4,
                [4],
                [4],
                "tanh",
                0.001,
                0.2,
                0.0,
                0.5,
                0.5,
                "cpu",
                share_policy=shared,
                action_type="continuous",
                action_low=low,
                action_high=high,
            )

        for shared in (True, False):
            with (
                self.subTest(shared=shared),
                tempfile.TemporaryDirectory() as directory,
            ):
                original = learner(shared, [-1.0, -1.0], [1.0, 1.0])
                destination = learner(shared, [-2.0, -1.0], [2.0, 1.0])
                path = os.path.join(directory, "continuous.pt")
                utils.save_checkpoint(
                    path, original.models, original.optimizers, 1, 0.0, shared
                )
                with self.assertRaisesRegex(ValueError, "bounds"):
                    utils.load_checkpoint(
                        path, destination.models, destination.optimizers
                    )

    def test_factory_reset_or_action_seed_failure_closes_environment(self):
        for failure in ("reset", "seed"):
            with self.subTest(failure=failure):
                env = SpaceEnv()
                modules, _ = fake_pettingzoo(env)
                with (
                    patch.dict(sys.modules, modules),
                    patch.object(env.actions["agent_0"], "seed") as seed,
                ):
                    if failure == "reset":
                        env.reset.side_effect = RuntimeError("factory failure")
                    else:
                        seed.side_effect = RuntimeError("factory failure")
                    with self.assertRaisesRegex(RuntimeError, "factory failure"):
                        utils.make_multiwalker_env(
                            n_walkers=2, seed=17, discretize_actions=False
                        )
                env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
