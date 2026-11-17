"""Checkpoint integrity and mocked vector-environment construction tests."""

import copy
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from PPO.ppo.networks import ActorCritic
import PPO.ppo.utils as ppo_utils


class TestCheckpoints(unittest.TestCase):
    def _assert_nested_equal(self, expected, actual) -> None:
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self._assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self._assert_nested_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def test_checkpoint_restores_weights_adam_moments_and_metadata(self) -> None:
        torch.manual_seed(7)
        model = ActorCritic(4, 3, hidden_sizes=(8,))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        obs = torch.randn((5, 4))
        logits, values = model(obs)
        loss = logits.square().mean() + (values - 1.0).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        self.assertTrue(expected_optimizer["state"])
        self.assertTrue(
            any(
                torch.count_nonzero(state["exp_avg"]).item() > 0
                for state in expected_optimizer["state"].values()
            )
        )

        restored = ActorCritic(4, 3, hidden_sizes=(8,))
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "best.pt")
            ppo_utils.save_checkpoint(
                path, model, optimizer, step=37, best_return=123.5
            )
            data = ppo_utils.load_checkpoint(path, restored, restored_optimizer)
        self.assertEqual(data["step"], 37)
        self.assertEqual(data["best_return"], 123.5)
        self._assert_nested_equal(expected_model, restored.state_dict())
        self._assert_nested_equal(expected_optimizer, restored_optimizer.state_dict())
        with torch.no_grad():
            for expected, actual in zip(model(obs), restored(obs)):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

        # Matching optimizer state must also produce the same next optimization step.
        for current, current_optimizer in (
            (model, optimizer),
            (restored, restored_optimizer),
        ):
            current_optimizer.zero_grad()
            current_logits, current_values = current(obs)
            (current_logits.square().mean() + current_values.square().mean()).backward()
            current_optimizer.step()
        self._assert_nested_equal(model.state_dict(), restored.state_dict())

    def test_checkpoint_can_load_model_without_optimizer(self) -> None:
        model = ActorCritic(4, 2, hidden_sizes=(8,))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        restored = ActorCritic(4, 2, hidden_sizes=(8,))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            ppo_utils.save_checkpoint(path, model, optimizer, step=2, best_return=-3.0)
            data = ppo_utils.load_checkpoint(path, restored)
        self.assertEqual(data["step"], 2)
        self._assert_nested_equal(model.state_dict(), restored.state_dict())


class TestVectorFactories(unittest.TestCase):
    def _assert_factory(self, num_envs: int, constructor_name: str) -> None:
        environments = [MagicMock() for _ in range(num_envs)]
        vector = object()
        with (
            patch.object(ppo_utils.gym, "make", side_effect=environments) as make,
            patch.object(
                ppo_utils.gym.vector, constructor_name, return_value=vector
            ) as constructor,
        ):
            actual = ppo_utils.make_vec_env(
                "Stub-v0", num_envs, 17, render_mode="rgb_array"
            )
            self.assertIs(actual, vector)
            factories = constructor.call_args.args[0]
            self.assertEqual(len(factories), num_envs)
            for rank, factory in enumerate(factories):
                self.assertIs(factory(), environments[rank])
                environments[rank].reset.assert_called_once_with(seed=17 + rank)
                environments[rank].action_space.seed.assert_called_once_with(17 + rank)
            self.assertEqual(make.call_count, num_envs)
            for call in make.call_args_list:
                self.assertEqual(call.args, ("Stub-v0",))
                self.assertEqual(
                    call.kwargs["render_mode"], "rgb_array" if num_envs == 1 else None
                )
            mode = getattr(ppo_utils.gym.vector, "AutoresetMode", None)
            if mode is not None:
                self.assertEqual(
                    constructor.call_args.kwargs["autoreset_mode"], mode.SAME_STEP
                )

    def test_gymnasium_10_without_autoreset_enum_is_rejected(self) -> None:
        # A namespace replacement simulates an absent attribute without changing
        # the installed gym module or starting a vector worker.
        from types import SimpleNamespace

        vector = SimpleNamespace(SyncVectorEnv=MagicMock(), AsyncVectorEnv=MagicMock())
        gym = SimpleNamespace(__version__="1.0.0", vector=vector, make=MagicMock())
        with patch.object(ppo_utils, "gym", gym):
            with self.assertRaisesRegex(RuntimeError, "Gymnasium 1.0 is not supported"):
                ppo_utils.make_vec_env("Stub-v0", 1, 17)
        vector.SyncVectorEnv.assert_not_called()
        vector.AsyncVectorEnv.assert_not_called()
        gym.make.assert_not_called()

    def test_legacy_029_without_autoreset_enum_keeps_default_same_step_api(
        self,
    ) -> None:
        from types import SimpleNamespace

        env = MagicMock()
        vector = SimpleNamespace(SyncVectorEnv=MagicMock(), AsyncVectorEnv=MagicMock())
        gym = SimpleNamespace(
            __version__="0.29.1", vector=vector, make=MagicMock(return_value=env)
        )
        with patch.object(ppo_utils, "gym", gym):
            result = ppo_utils.make_vec_env("Stub-v0", 1, 17)
            self.assertIs(result, vector.SyncVectorEnv.return_value)
            self.assertEqual(vector.SyncVectorEnv.call_args.kwargs, {})
            factory = vector.SyncVectorEnv.call_args.args[0][0]
            self.assertIs(factory(), env)
        env.reset.assert_called_once_with(seed=17)
        env.action_space.seed.assert_called_once_with(17)
        vector.AsyncVectorEnv.assert_not_called()

    def test_single_environment_factory_seeds_actions_and_requests_same_step(
        self,
    ) -> None:
        self._assert_factory(1, "SyncVectorEnv")

    def test_multiple_environment_factory_uses_rank_seeds_without_processes(
        self,
    ) -> None:
        self._assert_factory(3, "AsyncVectorEnv")


if __name__ == "__main__":
    unittest.main()
