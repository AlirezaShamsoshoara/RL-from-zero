"""Offline native-reward DQN helper tests with stub rollouts and asset writers."""

from argparse import Namespace
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from gymnasium import spaces

from deepQN.dqn.agent import DQNAgent
from deepQN.dqn.config import Config
from deepQN.utils import _common as common
from deepQN.utils import evaluate as evaluation
from deepQN.utils import make_charts as charts
from deepQN.utils import make_gif as gif


class ToolEnv:
    def __init__(self, lengths=(3, 1, 2), end="terminated", fail=False):
        self.observation_space = spaces.Box(-10.0, 10.0, (2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.lengths, self.end, self.fail = lengths, end, fail
        self.episode = -1
        self.steps = 0
        self.actions, self.seeds = [], []
        self.close_calls = 0
        self.frame = np.zeros((1, 1, 3), dtype=np.uint8)
        self.spec = SimpleNamespace(max_episode_steps=200)

    def reset(self, seed=None):
        self.episode += 1
        self.steps = 0
        self.seeds.append(seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise AssertionError("invalid action")
        self.actions.append(action)
        self.steps += 1
        if self.fail:
            raise RuntimeError("stub rollout failure")
        ended = self.steps >= self.lengths[self.episode]
        terminated, truncated = (
            ended and self.end == "terminated",
            ended and self.end == "truncated",
        )
        obs = np.array(
            [0.6 if terminated else self.steps / 10.0, 0.01], dtype=np.float32
        )
        return obs, -1.0, terminated, truncated, {}

    def render(self):
        self.frame[:] = self.episode * 10 + self.steps
        return self.frame

    def close(self):
        self.close_calls += 1


def agent_and_config():
    cfg = Config()
    cfg.device = "cpu"
    cfg.hidden_sizes = []
    cfg.gamma = 0.5
    cfg.max_steps_per_episode = 4
    cfg.wandb_key = "never-log-this-test-key"
    env = ToolEnv()
    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        [],
        "relu",
        0.001,
        0.5,
        3,
        1.0,
        True,
        "cpu",
    )
    with torch.no_grad():
        agent.q_network.model[0].weight.zero_()
        agent.q_network.model[0].bias.copy_(torch.tensor([0.0, 3.0]))
    agent.step_count = 17
    return agent, cfg


class OfflineWandb:
    def __init__(self):
        self.init_kwargs = None
        self.logged = []
        self.run = SimpleNamespace(
            id="stub-run", dir="stub-dir", summary={}, finish=MagicMock()
        )

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def log(self, data, step=None):
        self.logged.append((dict(data), step))


@contextmanager
def evaluate_stubs(cfg, env, agent):
    wandb = OfflineWandb()
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"WANDB_MODE": "online"}))
        stack.enter_context(
            patch.object(evaluation.Config, "from_yaml", return_value=cfg)
        )
        make = stack.enter_context(
            patch.object(evaluation, "make_env", return_value=env)
        )
        load = stack.enter_context(
            patch.object(evaluation, "load_agent", return_value=(agent, {}))
        )
        stack.enter_context(
            patch.object(evaluation, "checkpoint_metadata", return_value={"step": 9})
        )
        stack.enter_context(patch.object(evaluation, "wandb", wandb))
        stack.enter_context(patch.object(evaluation, "set_seed"))
        yield wandb, make, load


class TestRolloutTools(unittest.TestCase):
    def test_stop_flags_final_frames_native_reward_and_readonly_counters(self):
        for length, cap, end, flags in (
            (2, 4, "terminated", (True, False, False)),
            (2, 4, "truncated", (False, True, False)),
            (999, 2, "terminated", (False, False, True)),
        ):
            with self.subTest(end=end, length=length):
                agent, _ = agent_and_config()
                env = ToolEnv(lengths=(length,), end=end)
                with (
                    patch.object(
                        agent, "update", side_effect=AssertionError("learning used")
                    ),
                    patch.object(
                        agent.act_space,
                        "sample",
                        side_effect=AssertionError("random action used"),
                    ),
                ):
                    record = common.rollout_episode(
                        env, agent, 91, cap, 0.5, "MountainCar-v0", capture_frames=True
                    )
                self.assertEqual(record["return"], -2.0)
                self.assertEqual(record["discounted_return"], -1.5)
                self.assertEqual(record["initial_q"], 3.0)
                self.assertEqual(
                    tuple(record[key] for key in ("terminated", "truncated", "capped")),
                    flags,
                )
                self.assertEqual(record["success"], flags[0])
                self.assertEqual(
                    [int(frame[0, 0, 0]) for frame in record["frames"]], [0, 1, 2]
                )
                self.assertEqual(agent.step_count, 17)
                self.assertEqual(agent.train_steps, 0)

    def test_generic_negative_terminal_reward_does_not_define_success(self):
        agent, _ = agent_and_config()
        record = common.rollout_episode(
            ToolEnv(lengths=(1,)), agent, 91, 4, 0.5, "Generic-v0"
        )
        self.assertEqual(record["return"], -1.0)
        self.assertTrue(record["terminated"])
        self.assertIsNone(record["success"])
        self.assertIsNone(record["max_position"])

    def test_counter_metadata_rejects_nonintegral_bool_and_negative_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stub.pt"
            path.write_text("stub", encoding="utf-8")
            for field in ("step", "step_count", "train_steps"):
                for value in (1.5, True, -0.5, -1):
                    with self.subTest(field=field, value=value):
                        payload = {
                            "step": 2,
                            "step_count": 2,
                            "train_steps": 1,
                            "best_return": -1.0,
                        }
                        payload[field] = value
                        with self.assertRaises((ValueError, TypeError)):
                            common.checkpoint_metadata(path, payload)

    def test_loader_preserves_raw_counter_validation_and_uses_cpu_agent(self):
        _, cfg = agent_and_config()
        env = ToolEnv()
        with patch.object(common, "load_checkpoint") as load:
            common.load_agent(cfg, env, Path("unused.pt"))
        self.assertEqual(load.call_args.args[0], "unused.pt")
        self.assertEqual(load.call_args.args[1].device.type, "cpu")


class TestEvaluateTool(unittest.TestCase):
    def args(self, directory, random=False):
        return Namespace(
            config=Path("unused.yaml"),
            checkpoint=None,
            episodes=3,
            seed=100,
            out=Path(directory) / "nested" / "evaluation.json",
            random=random,
        )

    def test_offline_unfiltered_json_native_returns_and_negative_goal_success(self):
        agent, cfg = agent_and_config()
        env = ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            with (
                evaluate_stubs(cfg, env, agent) as (wandb, make, load),
                patch.object(
                    env.action_space,
                    "sample",
                    side_effect=AssertionError("random action used"),
                ),
            ):
                result = evaluation.evaluate(args)
                self.assertEqual(os.environ["WANDB_MODE"], "offline")
            persisted = json.loads(args.out.read_text(encoding="utf-8"))
        load.assert_called_once_with(cfg, env, Path(cfg.inference_model_path))
        make.assert_called_once_with(
            cfg.env_id, args.seed, render_mode=None, env_kwargs=cfg.env_kwargs
        )
        self.assertEqual(result["returns"], [-3.0, -1.0, -2.0])
        self.assertEqual(result["discounted_returns"], [-1.75, -1.0, -1.5])
        self.assertEqual(result["episode_lengths"], [3, 1, 2])
        self.assertEqual(result["seeds"], [100, 101, 102])
        self.assertEqual(result["initial_q_values"], [3.0, 3.0, 3.0])
        self.assertEqual(result["summary"]["count"], 3)
        self.assertEqual(result["summary"]["mean"], -2.0)
        self.assertAlmostEqual(result["summary"]["std"], np.std([-3.0, -1.0, -2.0]))
        self.assertEqual(result["summary"]["success_rate"], 1.0)
        self.assertEqual(result["summary"]["termination_rate"], 1.0)
        self.assertEqual(
            (result["summary"]["min"], result["summary"]["max"]), (-3.0, -1.0)
        )
        self.assertEqual(persisted, result)
        self.assertEqual(wandb.init_kwargs["mode"], "offline")
        self.assertNotIn(cfg.wandb_key, repr((result, wandb.init_kwargs, wandb.logged)))
        self.assertEqual(agent.step_count, 17)
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_random_baseline_skips_checkpoint_and_generic_success_is_unknown(self):
        agent, cfg = agent_and_config()
        cfg.env_id = "Generic-v0"
        env = ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            with (
                evaluate_stubs(cfg, env, agent) as (wandb, _, load),
                patch.object(env.action_space, "sample", return_value=0) as sample,
            ):
                result = evaluation.evaluate(self.args(directory, random=True))
        load.assert_not_called()
        self.assertEqual(sample.call_count, 6)
        self.assertEqual(result["returns"], [-3.0, -1.0, -2.0])
        self.assertEqual(result["initial_q_values"], [None, None, None])
        self.assertIsNone(result["checkpoint"])
        self.assertIsNone(result["checkpoint_training_config"])
        self.assertEqual(result["config_mismatches"], {})
        self.assertEqual(result["config"], result["evaluation_config"])
        self.assertIsNone(result["summary"]["success_rate"])
        self.assertEqual(wandb.init_kwargs["mode"], "offline")
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_saved_training_config_is_distinct_redacted_and_mismatches_are_flagged(
        self,
    ):
        agent, cfg = agent_and_config()
        cfg.env_kwargs = {
            "max_episode_steps": 200,
            "nested": {"api_key": "eval-nested-secret", "keep": 2},
        }
        saved = {
            "env_id": cfg.env_id,
            "gamma": 0.9,
            "hidden_sizes": [],
            "env_kwargs": {
                "max_episode_steps": 123,
                "nested": {"token": "saved-token", "keep": 1},
            },
            "WANDB_API_KEY": "saved-top-secret",
            "details": [{"password": "saved-password", "ordinary": "retained"}],
        }
        env = ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            with evaluate_stubs(cfg, env, agent) as (wandb, _, load):
                load.return_value = (agent, {"config": saved})
                result = evaluation.evaluate(args)
            persisted = json.loads(args.out.read_text(encoding="utf-8"))
        self.assertEqual(result["config"], result["evaluation_config"])
        self.assertEqual(result["evaluation_config"]["gamma"], 0.5)
        self.assertEqual(result["checkpoint_training_config"]["gamma"], 0.9)
        self.assertEqual(
            result["checkpoint_training_config"]["env_kwargs"]["max_episode_steps"], 123
        )
        self.assertEqual(
            result["evaluation_config"]["env_kwargs"]["max_episode_steps"], 200
        )
        self.assertIn("gamma", result["config_mismatches"])
        self.assertIn("env_kwargs", result["config_mismatches"])
        self.assertTrue(result["config_mismatches"]["gamma"])
        self.assertEqual(
            result["checkpoint_training_config"]["details"], [{"ordinary": "retained"}]
        )
        wb_config = wandb.init_kwargs["config"]
        self.assertNotIn("training_config", wb_config)
        self.assertEqual(wb_config["evaluation_config"], result["evaluation_config"])
        self.assertEqual(
            wb_config["checkpoint_training_config"],
            result["checkpoint_training_config"],
        )
        self.assertEqual(persisted, result)
        public = repr((result, persisted, wandb.init_kwargs, wandb.logged))
        for secret in (
            "eval-nested-secret",
            "saved-token",
            "saved-top-secret",
            "saved-password",
        ):
            self.assertNotIn(secret, public)
        self.assertEqual(saved["env_kwargs"]["nested"]["token"], "saved-token")
        self.assertEqual(cfg.env_kwargs["nested"]["api_key"], "eval-nested-secret")
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_legacy_checkpoint_has_unknown_training_provenance_not_selected_config(
        self,
    ):
        for payload in ({}, {"config": None}):
            with self.subTest(payload=payload):
                agent, cfg = agent_and_config()
                env = ToolEnv()
                with tempfile.TemporaryDirectory() as directory:
                    with evaluate_stubs(cfg, env, agent) as (wandb, _, load):
                        load.return_value = (agent, payload)
                        result = evaluation.evaluate(self.args(directory))
                self.assertEqual(result["config"], result["evaluation_config"])
                self.assertIsNone(result["checkpoint_training_config"])
                self.assertEqual(result["config_mismatches"], {})
                self.assertIsNone(
                    wandb.init_kwargs["config"]["checkpoint_training_config"]
                )
                self.assertNotIn("training_config", wandb.init_kwargs["config"])

    def test_recursive_redaction_handles_nested_mappings_lists_and_case_insensitive_keys(
        self,
    ):
        source = {
            "gamma": 0.9,
            "TOKEN": "top-token",
            "nested": {"api_key": "nested-key", "keep": 3},
            "items": [
                {"access_token": "list-token", "keep": True},
                ({"password": "tuple-secret", "keep": "value"},),
            ],
        }
        result = common.redact_config(source)
        self.assertEqual(
            result,
            {
                "gamma": 0.9,
                "nested": {"keep": 3},
                "items": [{"keep": True}, [{"keep": "value"}]],
            },
        )
        self.assertEqual(source["TOKEN"], "top-token")
        self.assertEqual(source["nested"]["api_key"], "nested-key")

    def test_nonmapping_saved_training_config_is_rejected_with_cleanup(self):
        agent, cfg = agent_and_config()
        env = ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            with evaluate_stubs(cfg, env, agent) as (_, _, load):
                load.return_value = (agent, {"config": "not-a-mapping"})
                with self.assertRaises(ValueError):
                    evaluation.evaluate(self.args(directory))
        self.assertEqual(env.close_calls, 1)

    def test_rollout_failure_cleans_up_without_output_file(self):
        agent, cfg = agent_and_config()
        env = ToolEnv(fail=True)
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            with evaluate_stubs(cfg, env, agent) as (wandb, _, _):
                with self.assertRaisesRegex(RuntimeError, "stub rollout failure"):
                    evaluation.evaluate(args)
            self.assertFalse(args.out.exists())
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()


class TestGifTool(unittest.TestCase):
    def test_selection_retains_all_returns_final_frames_and_fps(self):
        agent, cfg = agent_and_config()
        env = ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                config=Path("unused.yaml"),
                checkpoint=None,
                out=Path(directory) / "demo.gif",
                episodes=3,
                keep_top=2,
                fps=5,
                stride=2,
                seed=100,
            )

            def encode(path, frames, **kwargs):
                Path(path).write_text("stub", encoding="utf-8")

            with (
                patch.dict(os.environ),
                patch.object(gif.Config, "from_yaml", return_value=cfg),
                patch.object(gif, "make_env", return_value=env),
                patch.object(gif, "load_agent", return_value=(agent, {})) as load,
                patch.object(gif, "set_seed"),
                patch.object(gif.imageio, "mimsave", side_effect=encode) as save,
                patch("builtins.print"),
            ):
                result = gif.make_gif(args)
            load.assert_called_once_with(cfg, env, Path(cfg.inference_model_path))
            self.assertEqual(
                [int(frame[0, 0, 0]) for frame in save.call_args.args[1]],
                [10, 11, 11, 11, 20, 22, 22, 22],
            )
            self.assertEqual(
                save.call_args.kwargs,
                {"format": "pillow", "duration": 200.0, "loop": 0},
            )
        self.assertEqual(result["returns"], [-3.0, -1.0, -2.0])
        self.assertEqual(result["selected_returns"], [-1.0, -2.0])
        self.assertEqual(result["selected_seeds"], [101, 102])
        self.assertEqual(result["selected_success"], [True, True])
        self.assertEqual((result["frames"], result["fps"]), (8, 5))
        self.assertEqual(agent.step_count, 17)
        self.assertEqual(env.close_calls, 1)

    def test_gif_hyphenated_cli_and_invalid_top_count(self):
        args = gif.parse_args(["--episodes", "4", "--keep-top", "2", "--stride", "3"])
        self.assertEqual((args.episodes, args.keep_top, args.stride), (4, 2, 3))
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            gif.parse_args(["--episodes", "1", "--keep-top", "2"])


class TestChartTool(unittest.TestCase):
    def record(self, items):
        record = charts.pb.Record()
        for key, value, nested in items:
            item = record.history.item.add()
            if nested:
                item.nested_key.extend(key.split("/"))
                item.key = "wrong-scalar-key"
            else:
                item.key = key
            item.value_json = json.dumps(value)
        return record.SerializeToString()

    def test_nested_scalar_history_keys_sorting_and_finite_step_filter(self):
        ds = MagicMock()
        ds.scan_data.side_effect = [
            self.record([("progress/step", 20, True), ("stats/q_value", 2.0, True)]),
            charts.pb.Record().SerializeToString(),
            self.record([("progress/step", 10, False), ("stats/q_value", 1.0, False)]),
            self.record([("progress/step", True, True)]),
            self.record([("progress/step", float("nan"), True)]),
            None,
        ]
        with patch.object(charts.datastore, "DataStore", return_value=ds):
            rows = charts.load_history("stub.wandb")
        self.assertEqual(
            rows,
            [
                {"progress/step": 10, "stats/q_value": 1.0},
                {"progress/step": 20, "stats/q_value": 2.0},
            ],
        )
        ds.close.assert_called_once()

    def row(self, step=10):
        return {
            "progress/step": step,
            "charts/avg_return": -2.0,
            "charts/avg_length": 2.0,
            "charts/epsilon": 0.1,
            "charts/termination_rate": 1.0,
            "charts/success_rate": 1.0,
            "stats/q_value": -1.0,
            "stats/target_value": -1.5,
            "stats/td_error": 0.5,
            "loss/q_loss": 0.25,
        }

    def assert_chart_error(self, rows, message):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(charts, "load_history", return_value=rows),
            patch.object(charts.plt, "subplots") as plots,
        ):
            args = Namespace(
                run=Path("stub.wandb"),
                out_dir=Path(directory) / "charts",
                title="Stub DQN",
                env_id="MountainCar-v0",
                smooth=3,
                summary=None,
            )
            with self.assertRaisesRegex(ValueError, message):
                charts.make_charts(args)
        plots.assert_not_called()

    def test_empty_missing_nonfinite_and_unmatched_metrics_are_rejected(self):
        self.assert_chart_error([], "No DQN training history")
        for value in (None, True, np.nan, "bad"):
            with self.subTest(value=value):
                row = self.row()
                row["loss/q_loss"] = value
                self.assert_chart_error(
                    [row], "Missing finite training metrics: loss/q_loss"
                )
        first, second = self.row(10), self.row(20)
        del first["stats/q_value"]
        del second["stats/target_value"]
        self.assert_chart_error([first, second], "No matched replay rows")

    def test_fraction_and_loss_constraints_are_checked(self):
        for key, value, message in (
            ("charts/epsilon", 1.1, "fraction"),
            ("charts/success_rate", -0.1, "fraction"),
            ("stats/td_error", -0.1, "nonnegative"),
            ("loss/q_loss", -0.1, "nonnegative"),
        ):
            with self.subTest(key=key):
                row = self.row()
                row[key] = value
                self.assert_chart_error([row], message)

    def test_raw_summary_preserves_native_reward_series_without_smoothing(self):
        first, second = self.row(10), self.row(20)
        first["charts/avg_return"], second["charts/avg_return"] = -3.0, -1.0
        summary = charts.summarize(
            [first, second], Path("stub.wandb"), "MountainCar-v0"
        )
        metric = summary["metrics"]["charts/avg_return"]
        self.assertEqual(metric["values"], [-3.0, -1.0])
        self.assertEqual(metric["steps"], [10.0, 20.0])
        self.assertEqual(metric["mean"], -2.0)
        self.assertIn("native", summary["units"]["charts/avg_return"])
        self.assertIn("stats/start_q", summary["missing_metrics"])
        np.testing.assert_allclose(
            charts.smooth(np.array([-3.0, -1.0]), 3), [-2.0, -2.0]
        )


if __name__ == "__main__":
    unittest.main()
