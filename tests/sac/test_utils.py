"""SAC utility tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/sac/test_utils.py
    >>> # Run with unittest
    >>> # python -m unittest tests.sac.test_utils
"""

import os
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from gymnasium import spaces

from SAC.sac.agent import SACAgent
from SAC.sac.utils import load_checkpoint, make_env, save_checkpoint, set_seed


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

    def test_factory_seeds_action_space_but_leaves_reset_to_caller_and_closes_on_failure(
        self,
    ) -> None:
        env = Mock()
        env.observation_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        env.action_space = spaces.Box(-2.0, 2.0, (2,), dtype=np.float32)
        env.action_space.seed = Mock()
        kwargs = {"g": 9.81}
        with (
            patch("SAC.sac.utils.gym.make", return_value=env) as make,
            patch(
                "SAC.sac.utils.gym.wrappers.RecordEpisodeStatistics", return_value=env
            ),
        ):
            result = make_env("StubEnv-v0", 17, render_mode=None, env_kwargs=kwargs)
        self.assertIs(result, env)
        make.assert_called_once_with("StubEnv-v0", render_mode=None, g=9.81)
        env.action_space.seed.assert_called_once_with(17)
        env.reset.assert_not_called()
        self.assertEqual(kwargs, {"g": 9.81})
        with (
            patch("SAC.sac.utils.gym.make", return_value=env),
            patch(
                "SAC.sac.utils.gym.wrappers.RecordEpisodeStatistics",
                side_effect=RuntimeError("wrapper"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "wrapper"):
                make_env("StubEnv-v0", 17)
        env.close.assert_called_once()

    def test_set_seed_reproduces_python_numpy_and_cpu_torch_rng(self) -> None:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            with patch("torch.cuda.manual_seed_all"):
                set_seed(17)
                first = (random.random(), np.random.random(3), torch.rand(3))
                set_seed(17)
                second = (random.random(), np.random.random(3), torch.rand(3))
            self.assertEqual(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            torch.testing.assert_close(first[2], second[2], atol=0, rtol=0)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

    def test_checkpoint_accepts_a_basename_and_unknown_best_score(self) -> None:
        agent = _make_agent()
        previous_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                save_checkpoint("checkpoint.pt", agent, step=0, best_return=None)
                payload = torch.load(
                    "checkpoint.pt", map_location="cpu", weights_only=True
                )
                self.assertIsNone(payload["best_return"])
                self.assertEqual(payload["step"], 0)
                self.assertEqual(os.listdir(tmpdir), ["checkpoint.pt"])
            finally:
                os.chdir(previous_directory)

    def test_failed_atomic_checkpoint_preserves_previous_file_and_removes_temporary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            with open(path, "wb") as handle:
                handle.write(b"previous checkpoint")
            with patch(
                "SAC.sac.utils.torch.save", side_effect=RuntimeError("write failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    save_checkpoint(path, _make_agent(), 1, None)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"previous checkpoint")
            self.assertEqual(os.listdir(tmpdir), ["checkpoint.pt"])

    def test_load_uses_cpu_weights_only_and_rejects_invalid_payload_before_loading_agent(
        self,
    ) -> None:
        agent = _make_agent()
        state = agent.state_dict()
        state.update(step=3, best_return=None)
        with (
            patch("SAC.sac.utils.torch.load", return_value=state) as load,
            patch.object(agent, "load_state_dict") as restore,
        ):
            result = load_checkpoint("unused.pt", agent)
            load.assert_called_once_with(
                "unused.pt", map_location="cpu", weights_only=True
            )
            restore.assert_called_once_with(state)
            self.assertIs(result, state)
        for payload in [
            [],
            {"step": -1},
            {"step": True},
            {"best_return": float("nan")},
            {"best_return": float("inf")},
        ]:
            with (
                self.subTest(payload=payload),
                patch("SAC.sac.utils.torch.load", return_value=payload),
                patch.object(agent, "load_state_dict") as restore,
            ):
                with self.assertRaises(ValueError):
                    load_checkpoint("unused.pt", agent)
                restore.assert_not_called()
        with self.assertRaises(TypeError):
            load_checkpoint("unused.pt", object())

    def test_checkpoint_metadata_is_redacted_owned_and_does_not_claim_exact_resume(
        self,
    ) -> None:
        agent = _make_agent()
        config = {
            "wandb_key": "hidden",
            "env_kwargs": {"api_key": "hidden-too", "g": 9.81},
            "nested": [{"password": "private", "value": 1}],
        }
        metadata = {
            "environment_steps": 10,
            "optimizer_updates": 7,
            "training_completed": True,
            "access_token": "private",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            save_checkpoint(path, agent, 10, None, config=config, metadata=metadata)
            config["env_kwargs"]["g"] = 0.0
            metadata["optimizer_updates"] = 99
            data = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(
            data["config"], {"env_kwargs": {"g": 9.81}, "nested": [{"value": 1}]}
        )
        self.assertEqual(
            data["metadata"],
            {
                "environment_steps": 10,
                "optimizer_updates": 7,
                "training_completed": True,
            },
        )
        self.assertNotIn("replay", data)
        self.assertNotIn("rng_state", data)
        self.assertNotIn("environment_state", data)
        self.assertIn("actor_optimizer_state_dict", data)
        self.assertIn("alpha_optimizer_state_dict", data)


if __name__ == "__main__":
    unittest.main()
