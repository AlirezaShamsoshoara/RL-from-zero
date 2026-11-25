"""A3C returns, mocked environment factories, and trusted checkpoint regression tests."""

import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.vector import AutoresetMode

from A3C.a3c.networks import ActorCritic
from A3C.a3c.utils import (
    SharedAdam,
    compute_returns,
    load_checkpoint,
    make_env,
    make_vec_env,
    save_checkpoint,
    set_seed,
)

from ._support import StubEnv, StubVecEnv


class TestComputeReturns(unittest.TestCase):
    def test_single_env_returns(self):
        result = compute_returns(
            torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3), torch.tensor(1.0), 0.99
        )
        self.assertEqual(result.shape, (3,))
        torch.testing.assert_close(result, torch.tensor([6.890599, 5.9501, 3.99]))

    def test_done_cuts_bootstrap(self):
        result = compute_returns(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor(1.0),
            0.99,
        )
        torch.testing.assert_close(result, torch.tensor([2.98, 2.0, 3.99]))

    def test_multi_env_returns_2d(self):
        rewards = torch.tensor([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
        result = compute_returns(
            rewards, torch.zeros_like(rewards), torch.zeros(3), 0.5
        )
        self.assertEqual(result.shape, (2, 3))
        torch.testing.assert_close(
            result, torch.tensor([[2.0, 20.0, 200.0], [2.0, 20.0, 200.0]])
        )

    def test_multi_env_done_isolates_envs(self):
        result = compute_returns(
            torch.ones(2, 2),
            torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            torch.ones(2),
            0.99,
        )
        torch.testing.assert_close(result, torch.tensor([[2.9701, 1.0], [1.99, 1.99]]))

    def test_timeout_bootstraps_actual_endpoint_without_crossing_reset_episode(self):
        result = compute_returns(
            torch.tensor([1.0, 2.0, 100.0]),
            torch.tensor([False, True, False]),
            torch.tensor(0.0),
            0.5,
            terminations=torch.zeros(3, dtype=torch.bool),
            next_values_sequence=torch.tensor([0.0, 10.0, 0.0]),
        )
        torch.testing.assert_close(result, torch.tensor([4.5, 7.0, 100.0]))

    def test_vector_timeout_and_termination_masks_are_independent(self):
        rewards = torch.ones(2, 2)
        dones = torch.tensor([[False, False], [True, True]])
        terminals = torch.tensor([[False, False], [False, True]])
        endpoints = torch.tensor([[0.0, 0.0], [10.0, 999.0]])
        result = compute_returns(
            rewards,
            dones,
            torch.tensor([88.0, 88.0]),
            0.5,
            terminations=terminals,
            next_values_sequence=endpoints,
        )
        torch.testing.assert_close(result, torch.tensor([[4.0, 1.5], [6.0, 1.0]]))

    def test_returns_validate_shapes_boundaries_gamma_and_timeout_endpoints(self):
        rewards = torch.ones(2)
        for kwargs in (
            {"dones": torch.zeros(3)},
            {"gamma": float("nan")},
            {"gamma": 1.1},
            {"terminations": torch.ones(3)},
            {"terminations": torch.tensor([True, False])},
            {
                "dones": torch.tensor([True, False]),
                "terminations": torch.zeros(2, dtype=torch.bool),
            },
            {"next_values_sequence": torch.zeros(3)},
            {
                "dones": torch.tensor([True, False]),
                "terminations": torch.zeros(2, dtype=torch.bool),
                "next_values_sequence": torch.tensor([float("nan"), 0.0]),
            },
        ):
            with self.subTest(keys=tuple(kwargs)):
                arguments = dict(
                    rewards=rewards,
                    dones=torch.zeros(2),
                    last_value=torch.tensor(0.0),
                    gamma=0.99,
                )
                arguments.update(kwargs)
                with self.assertRaises(ValueError):
                    compute_returns(**arguments)


class FactoryVecEnv(StubVecEnv):
    def __init__(self, thunks, *, autoreset_mode):
        self.envs = [thunk() for thunk in thunks]
        self.autoreset_mode = autoreset_mode
        super().__init__(num_envs=len(self.envs))

    def close(self):
        for env in self.envs:
            env.close()
        super().close()


class TestEnvironmentFactories(unittest.TestCase):
    def test_returns_vector_env_with_correct_num_envs(self):
        raw_envs = [StubEnv() for _ in range(3)]
        with (
            patch("A3C.a3c.utils.gym.make", side_effect=raw_envs) as make,
            patch("A3C.a3c.utils.gym.vector.SyncVectorEnv", side_effect=FactoryVecEnv),
        ):
            env = make_vec_env("UnitStub-v0", num_envs=3, seed=42)
            self.assertEqual(env.num_envs, 3)
            self.assertIs(env.autoreset_mode, AutoresetMode.SAME_STEP)
            self.assertEqual(make.call_count, 3)
            self.assertTrue(all(raw.reset_seeds == [] for raw in raw_envs))
            obs, _ = env.reset(seed=[42, 43, 44])
            self.assertEqual(obs.shape, (3, 6))
            env.close()
        self.assertTrue(all(raw.close_called for raw in raw_envs))

    def test_step_returns_batched_outputs(self):
        with (
            patch("A3C.a3c.utils.gym.make", side_effect=lambda *a, **k: StubEnv()),
            patch("A3C.a3c.utils.gym.vector.SyncVectorEnv", side_effect=FactoryVecEnv),
        ):
            env = make_vec_env("UnitStub-v0", num_envs=2, seed=42)
            env.reset(seed=[42, 43])
            obs, rewards, terminated, truncated, _ = env.step(np.array([0, 1]))
            self.assertEqual(obs.shape, (2, 6))
            for vector in (rewards, terminated, truncated):
                self.assertEqual(vector.shape, (2,))
            env.close()

    def test_scalar_native_statistics_survive_learning_reward_scaling(self):
        raw = StubEnv(reward=2.0)
        with patch("A3C.a3c.utils.gym.make", return_value=raw) as make:
            env = make_env(
                "UnitStub-v0", seed=4, reward_scale=0.01, env_kwargs={"custom": 9}
            )
            self.assertEqual(raw.reset_seeds, [])
            make.assert_called_once_with("UnitStub-v0", render_mode=None, custom=9)
            env.reset(seed=4)
            _, reward, _, _, info = env.step(0)
            self.assertAlmostEqual(reward, 0.02)
            self.assertEqual(info["native_reward"], 2.0)
            _, _, terminated, _, info = env.step(0)
            self.assertTrue(terminated)
            self.assertEqual(info["native_episode_return"], 4.0)
            env.reset()
            env.step(0)
            _, _, _, _, info = env.step(0)
            self.assertEqual(info["native_episode_return"], 4.0)
            env.close()
        self.assertTrue(raw.close_called)

    def test_native_statistics_are_innermost_before_shaping_and_scaling(self):
        raw = StubEnv(reward=2.0)
        with patch("A3C.a3c.utils.gym.make", return_value=raw):
            env = make_env(
                "Acrobot-UnitStub-v0", seed=4, reward_shaping=True, reward_scale=0.1
            )
            env.reset()
            _, reward, _, _, info = env.step(0)
            self.assertNotEqual(reward, 2.0)
            self.assertEqual(info["native_reward"], 2.0)
            _, _, _, _, info = env.step(0)
            self.assertEqual(info["native_episode_return"], 4.0)
            env.close()

    def test_failed_vector_constructor_closes_every_created_stub(self):
        raw_envs = [StubEnv(), StubEnv()]

        def broken_constructor(thunks, **kwargs):
            for thunk in thunks:
                thunk()
            raise RuntimeError("stub vector constructor failed")

        with (
            patch("A3C.a3c.utils.gym.make", side_effect=raw_envs),
            patch(
                "A3C.a3c.utils.gym.vector.SyncVectorEnv", side_effect=broken_constructor
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "constructor failed"):
                make_vec_env("UnitStub-v0", num_envs=2, seed=42)
        self.assertTrue(all(env.close_called for env in raw_envs))


class TestCheckpointRoundTrip(unittest.TestCase):
    def _model_optimizer(self):
        model = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,))
        optimizer = SharedAdam(model.parameters(), lr=1e-3)
        logits, values = model(torch.ones(2, 6))
        (logits.sum() + values.sum()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return model, optimizer

    def test_save_and_load_preserves_weights(self):
        model, optimizer = self._model_optimizer()
        before = [p.detach().clone() for p in model.parameters()]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ckpt.pt")
            save_checkpoint(path, model, optimizer, 100, 42.0)
            restored = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,))
            data = load_checkpoint(path, restored)
        self.assertEqual(data["step"], 100)
        self.assertEqual(data["best_return"], 42.0)
        for left, right in zip(before, restored.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_restored_shared_adam_moments_and_steps_remain_shared(self):
        model, optimizer = self._model_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ckpt.pt")
            save_checkpoint(path, model, optimizer, 1, 1.0)
            restored_model = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,))
            restored_optimizer = SharedAdam(restored_model.parameters(), lr=0.01)
            load_checkpoint(path, restored_model, restored_optimizer)
        for original, restored in zip(
            optimizer.state.values(), restored_optimizer.state.values()
        ):
            for name in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(restored[name].is_shared())
                self.assertTrue(torch.equal(original[name], restored[name]))
            self.assertEqual(restored["step"].ndim, 0)

    def test_basename_checkpoint_and_nested_metadata_redaction(self):
        model, optimizer = self._model_optimizer()
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                save_checkpoint(
                    "basename.pt",
                    model,
                    optimizer,
                    0,
                    -float("inf"),
                    config={"env_id": "UnitStub-v0", "wandb_key": "do-not-save"},
                    metadata={"nested": {"api_key": "secret", "note": "keep"}},
                )
                data = load_checkpoint("basename.pt", model)
            finally:
                os.chdir(previous)
        self.assertEqual(data["config"], {"env_id": "UnitStub-v0"})
        self.assertEqual(data["metadata"], {"nested": {"note": "keep"}})
        self.assertNotIn("do-not-save", repr(data))

    def test_invalid_save_step_score_and_metadata_fail_before_file_write(self):
        model, optimizer = self._model_optimizer()
        for kwargs in (
            {"step": -1},
            {"step": True},
            {"step": 1.5},
            {"best_return": float("nan")},
            {"best_return": float("inf")},
            {"metadata": [1]},
            {"config": [1]},
        ):
            with self.subTest(kwargs=kwargs), patch("A3C.a3c.utils.torch.save") as save:
                arguments = dict(
                    path="never-written.pt",
                    model=model,
                    optimizer=optimizer,
                    step=1,
                    best_return=1.0,
                )
                arguments.update(kwargs)
                with self.assertRaises((ValueError, TypeError)):
                    save_checkpoint(**arguments)
                save.assert_not_called()

    def test_invalid_loaded_checkpoint_does_not_mutate_model_or_adam(self):
        model, optimizer = self._model_optimizer()
        baseline_weights = copy.deepcopy(model.state_dict())
        baseline_optimizer = copy.deepcopy(optimizer.state_dict())
        payload = dict(
            model_state_dict=copy.deepcopy(baseline_weights),
            optimizer_state_dict=copy.deepcopy(baseline_optimizer),
            step=1,
            best_return=1.0,
        )
        corruptions = []
        for invalid in (-1, True, 1.5):
            bad = copy.deepcopy(payload)
            bad["step"] = invalid
            corruptions.append(bad)
        bad = copy.deepcopy(payload)
        next(iter(bad["model_state_dict"].values())).flatten()[0] = float("nan")
        corruptions.append(bad)
        bad = copy.deepcopy(payload)
        next(iter(bad["optimizer_state_dict"]["state"].values()))["exp_avg"] = (
            torch.zeros(1)
        )
        corruptions.append(bad)
        bad = copy.deepcopy(payload)
        bad["metadata"] = ["not-a-mapping"]
        corruptions.append(bad)
        for bad in corruptions:
            with (
                self.subTest(step=bad["step"]),
                patch("A3C.a3c.utils.torch.load", return_value=bad) as load,
            ):
                with self.assertRaises((ValueError, TypeError)):
                    load_checkpoint("not-read.pt", model, optimizer)
                self.assertTrue(load.call_args.kwargs["weights_only"])
                for name, tensor in model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, baseline_weights[name]))
                for key, state in optimizer.state_dict()["state"].items():
                    for name, tensor in state.items():
                        self.assertTrue(
                            torch.equal(tensor, baseline_optimizer["state"][key][name])
                        )


class TestSetSeed(unittest.TestCase):
    def test_reproducibility(self):
        set_seed(123)
        first = torch.randn(5)
        set_seed(123)
        self.assertTrue(torch.equal(first, torch.randn(5)))
