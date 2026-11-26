"""Factory and checkpoint contracts with mocked Gym constructors."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from gymnasium.vector import AutoresetMode

from TD3.td3.agent import TD3Agent
from TD3.td3.config import Config
import TD3.td3.utils as utils


class TestUtils(unittest.TestCase):
    def _agent(self):
        agent = Mock(spec=TD3Agent)
        agent.state_dict.return_value = {
            "actor_state_dict": {"weight": torch.tensor([3.0])}
        }
        return agent

    def test_single_factory_seeds_actions_never_resets_and_preserves_optional_kwargs(
        self,
    ):
        for kwargs in [None, {"max_episode_steps": 999}]:
            env = Mock()
            original = None if kwargs is None else dict(kwargs)
            with (
                patch.object(utils.gym, "make", return_value=env) as make,
                patch.object(
                    utils.gym.wrappers,
                    "RecordEpisodeStatistics",
                    side_effect=lambda value: value,
                ),
            ):
                result = utils.make_env(
                    "stub-v0", 7, render_mode=None, env_kwargs=kwargs
                )
            self.assertIs(result, env)
            env.reset.assert_not_called()
            env.action_space.seed.assert_called_once_with(7)
            make.assert_called_once_with("stub-v0", render_mode=None, **(kwargs or {}))
            self.assertEqual(kwargs, original)

    def test_vector_factory_is_sync_same_step_with_seed_plus_rank_and_no_factory_reset(
        self,
    ):
        singles = [Mock(), Mock()]
        vector = Mock()

        def build(thunks, *, autoreset_mode):
            self.assertEqual(autoreset_mode, AutoresetMode.SAME_STEP)
            self.assertEqual([thunk() for thunk in thunks], singles)
            return vector

        with (
            patch.object(utils.gym, "make", side_effect=singles) as make,
            patch.object(
                utils.gym.wrappers,
                "RecordEpisodeStatistics",
                side_effect=lambda value: value,
            ),
            patch.object(utils.gym.vector, "SyncVectorEnv", side_effect=build),
        ):
            result = utils.make_vec_env(
                "stub-v0", 2, 10, env_kwargs={"max_episode_steps": 999}
            )
        self.assertIs(result, vector)
        self.assertEqual(make.call_count, 2)
        for rank, env in enumerate(singles):
            env.reset.assert_not_called()
            env.action_space.seed.assert_called_once_with(10 + rank)
        vector.reset.assert_not_called()
        vector.action_space.seed.assert_called_once_with(10)
        self.assertTrue(
            all(call.kwargs["render_mode"] is None for call in make.call_args_list)
        )
        # Acquisition must close every prior sub-env even if the first close fails.
        partial = [Mock(), Mock()]
        partial[0].close.side_effect = RuntimeError("close failed")

        def failing_build(thunks, **kwargs):
            return [thunk() for thunk in thunks]

        with (
            patch.object(
                utils,
                "make_env",
                side_effect=[*partial, TypeError("acquisition failed")],
            ),
            patch.object(utils.gym.vector, "SyncVectorEnv", side_effect=failing_build),
        ):
            with self.assertRaisesRegex(TypeError, "acquisition failed"):
                utils.make_vec_env("stub-v0", 3, 10)
        for env in partial:
            env.close.assert_called_once_with()

    def test_checkpoint_bare_filename_optional_config_metadata_and_deep_safe_payload(
        self,
    ):
        agent = self._agent()
        cfg = Config(device="cpu", wandb_key="never-store-this")
        metadata = {
            "environment_steps": 8,
            "vector_steps": 4,
            "optimizer_updates": 8,
            "actor_updates": 4,
            "training_completed": True,
            "api_key": "redact",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            original_abspath = utils.os.path.abspath
            with patch.object(
                utils.os.path,
                "abspath",
                side_effect=lambda value: (
                    str(path) if value == "weights.pt" else original_abspath(value)
                ),
            ) as absolute:
                utils.save_checkpoint(
                    "weights.pt", agent, 8, None, config=cfg, metadata=metadata
                )
            absolute.assert_any_call("weights.pt")
            data = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(data["step"], 8)
        self.assertIsNone(data["best_return"])
        self.assertEqual(data["metadata"]["vector_steps"], 4)
        self.assertNotIn("api_key", data["metadata"])
        self.assertNotIn("wandb_key", data["config"])
        self.assertEqual(metadata["api_key"], "redact")
        from TD3.utils._common import checkpoint_counters

        counters = checkpoint_counters({**data, "global_step": 999})
        self.assertEqual(counters["environment_steps"], 8)
        self.assertEqual(counters["vector_steps"], 4)
        self.assertEqual(counters["optimizer_updates"], 8)
        self.assertEqual(counters["critic_updates"], 8)
        self.assertEqual(counters["actor_updates"], 4)
        self.assertTrue(counters["checkpoint_metadata"]["training_completed"])
        self.assertEqual(
            counters["counter_sources"]["optimizer_updates"],
            "metadata.optimizer_updates",
        )
        agent.state_dict.return_value["actor_state_dict"]["weight"].fill_(99)
        torch.testing.assert_close(
            data["actor_state_dict"]["weight"], torch.tensor([3.0])
        )

    def test_checkpoint_failure_is_atomic_and_removes_temporary_files(self):
        agent = self._agent()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            path.write_bytes(b"previous-checkpoint")
            with patch.object(
                utils.torch, "save", side_effect=RuntimeError("write failed")
            ):
                with self.assertRaises(RuntimeError):
                    utils.save_checkpoint(str(path), agent, 1, None)
            self.assertEqual(path.read_bytes(), b"previous-checkpoint")
            self.assertEqual(
                sorted(p.name for p in Path(directory).iterdir()), ["weights.pt"]
            )
            for step, best in [
                (True, None),
                (-1, None),
                (1.5, None),
                (1, float("nan")),
                (1, float("inf")),
            ]:
                with (
                    self.subTest(step=step, best=best),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    utils.save_checkpoint(str(path), agent, step, best)
                self.assertEqual(path.read_bytes(), b"previous-checkpoint")

    def test_checkpoint_load_explicit_weights_only_cpu_and_rejects_bad_payload_before_agent(
        self,
    ):
        agent = self._agent()
        payload = {
            "actor_state_dict": {"weight": torch.tensor([1.0])},
            "step": 2,
            "best_return": None,
        }
        with patch.object(utils.torch, "load", return_value=payload) as load:
            result = utils.load_checkpoint("weights.pt", agent)
        load.assert_called_once_with(
            "weights.pt", map_location="cpu", weights_only=True
        )
        self.assertIs(result, payload)
        agent.load_state_dict.assert_called_once_with(payload)
        agent.load_state_dict.reset_mock()
        for invalid in [
            [],
            {"step": -1},
            {"step": True},
            {"best_return": float("nan")},
        ]:
            with patch.object(utils.torch, "load", return_value=invalid):
                with (
                    self.subTest(invalid=invalid),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    utils.load_checkpoint("weights.pt", agent)
            agent.load_state_dict.assert_not_called()
