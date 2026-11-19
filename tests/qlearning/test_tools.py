"""Tabular evaluation, GIF, and local-history tools with no real environments/assets."""

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
from gymnasium import spaces

from Qlearning.ql.agent import QLearningAgent
from Qlearning.ql.config import Config
from Qlearning.utils import _common as common
from Qlearning.utils import evaluate as evaluation
from Qlearning.utils import make_charts as charts
from Qlearning.utils import make_gif as gif


class ToolEnv:
    def __init__(self, rewards=(-1.0, 3.0, 1.0), length=2, timeout=False, fail=False):
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self.rewards, self.length = rewards, length
        self.timeout, self.fail = timeout, fail
        self.episode = -1
        self.steps = 0
        self.actions, self.seeds = [], []
        self.close_calls = 0
        self.frame = np.zeros((1, 1, 3), dtype=np.uint8)

    def reset(self, seed=None):
        self.episode += 1
        self.steps = 0
        self.seeds.append(seed)
        return 0, {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise AssertionError("invalid action")
        self.actions.append(action)
        self.steps += 1
        if self.fail:
            raise RuntimeError("stub rollout failure")
        done = self.steps >= self.length
        return (
            min(self.steps, 2),
            self.rewards[self.episode],
            done and not self.timeout,
            done and self.timeout,
            {},
        )

    def render(self):
        self.frame[:] = self.episode * 10 + self.steps
        return self.frame

    def close(self):
        self.close_calls += 1


def agent_and_config():
    cfg = Config()
    cfg.env_id = "FrozenLake-v1"
    cfg.gamma = 0.5
    cfg.max_steps_per_episode = 4
    cfg.wandb_key = "never-publish-test-key"
    agent = QLearningAgent(3, 2, 0.5, 0.5, 1.0, 1.0, 0.0)
    agent.Q[:] = [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]
    agent.global_step = 17
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


class TestRolloutHelper(unittest.TestCase):
    def test_terminal_timeout_and_cap_return_flags_without_learning_or_counter_changes(
        self,
    ):
        for length, max_steps, timeout, expected_flags in (
            (2, 4, False, (True, False, False)),
            (2, 4, True, (False, True, False)),
            (999, 2, False, (False, False, True)),
        ):
            with self.subTest(timeout=timeout, length=length):
                agent, _ = agent_and_config()
                table = agent.Q.copy()
                env = ToolEnv(rewards=(1.0,), length=length, timeout=timeout)
                with patch.object(
                    agent, "update", side_effect=AssertionError("learning used")
                ):
                    result = common.rollout_episode(env, agent, 91, max_steps, 0.5)
                self.assertEqual(result["return"], 2.0)
                self.assertEqual(result["discounted_return"], 1.5)
                self.assertEqual(result["length"], 2)
                self.assertEqual(result["start_value"], 2.0)
                self.assertEqual(
                    tuple(
                        result[name] for name in ("terminated", "truncated", "capped")
                    ),
                    expected_flags,
                )
                self.assertEqual(env.actions, [1, 1])
                self.assertEqual(agent.global_step, 17)
                np.testing.assert_array_equal(agent.Q, table)

    def test_helper_checkpoint_loader_rejects_malformed_exploration_counters(self):
        _, cfg = agent_and_config()
        env = ToolEnv()
        for step in (1.5, True, -0.5):
            with (
                self.subTest(step=step),
                patch.object(
                    common,
                    "load_checkpoint",
                    return_value={
                        "q_table": np.zeros((3, 2), dtype=np.float32),
                        "global_step": step,
                    },
                ),
            ):
                with self.assertRaises((ValueError, TypeError)):
                    common.load_agent(cfg, env, Path("unused.pt"))

    def test_helper_checkpoint_loader_copies_table_and_restores_counter(self):
        _, cfg = agent_and_config()
        table = np.arange(6, dtype=np.float32).reshape(3, 2)
        with patch.object(
            common,
            "load_checkpoint",
            return_value={"q_table": table, "global_step": 19},
        ):
            agent, _ = common.load_agent(cfg, ToolEnv(), Path("unused.pt"))
        table[:] = -999.0
        self.assertEqual(agent.Q[2, 1], 5.0)
        self.assertEqual(agent.global_step, 19)

    def test_zero_based_helper_spaces_are_required(self):
        for field in ("observation_space", "action_space"):
            with self.subTest(field=field):
                env = ToolEnv()
                setattr(env, field, spaces.Discrete(3, start=1))
                with self.assertRaisesRegex(ValueError, "zero-based"):
                    common.validate_env(env)

    def test_timeout_and_cap_keep_the_reached_final_frame(self):
        for length, max_steps, timeout in ((1, 4, True), (999, 1, False)):
            with self.subTest(timeout=timeout):
                agent, _ = agent_and_config()
                env = ToolEnv(rewards=(1.0,), length=length, timeout=timeout)
                result = common.rollout_episode(
                    env, agent, 91, max_steps, 0.5, capture_frames=True
                )
                self.assertEqual(
                    [int(frame[0, 0, 0]) for frame in result["frames"]], [0, 1]
                )
                self.assertEqual(result["truncated"], timeout)
                self.assertEqual(result["capped"], not timeout)
                self.assertEqual(agent.global_step, 17)

    def test_nonfinite_reward_is_rejected(self):
        agent, _ = agent_and_config()
        with self.assertRaisesRegex(ValueError, "non-finite reward"):
            common.rollout_episode(
                ToolEnv(rewards=(np.nan,), length=1), agent, 91, 2, 0.5
            )


class TestEvaluateTool(unittest.TestCase):
    def _args(self, directory, random=False):
        return Namespace(
            config=Path("unused.yaml"),
            checkpoint=None,
            episodes=3,
            seed=100,
            out=Path(directory) / "nested" / "evaluation.json",
            random=random,
        )

    def test_offline_deterministic_evaluation_keeps_unfiltered_returns_and_discounted_units(
        self,
    ):
        agent, cfg = agent_and_config()
        env = ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory)
            with (
                evaluate_stubs(cfg, env, agent) as (wandb, make, load),
                patch.object(agent, "act", wraps=agent.act) as act,
                patch.object(env.action_space, "sample") as sample,
            ):
                result = evaluation.evaluate(args)
                self.assertEqual(os.environ["WANDB_MODE"], "offline")
            persisted = json.loads(args.out.read_text(encoding="utf-8"))
        make.assert_called_once_with(
            cfg.env_id, args.seed, render_mode=None, env_kwargs=cfg.env_kwargs
        )
        load.assert_called_once_with(cfg, env, Path(cfg.inference_model_path))
        sample.assert_not_called()
        self.assertEqual(act.call_count, 6)
        self.assertTrue(
            all(call.kwargs["deterministic"] for call in act.call_args_list)
        )
        self.assertEqual(result["returns"], [-2.0, 6.0, 2.0])
        self.assertEqual(result["discounted_returns"], [-1.5, 4.5, 1.5])
        self.assertEqual(result["start_values"], [2.0, 2.0, 2.0])
        self.assertEqual(result["seeds"], [100, 101, 102])
        self.assertEqual(result["episode_lengths"], [2, 2, 2])
        self.assertEqual(result["summary"]["count"], 3)
        self.assertEqual(result["summary"]["min"], -2.0)
        self.assertEqual(result["summary"]["max"], 6.0)
        self.assertAlmostEqual(result["summary"]["mean"], 2.0)
        self.assertAlmostEqual(result["summary"]["std"], np.std([-2.0, 6.0, 2.0]))
        self.assertAlmostEqual(result["summary"]["success_rate"], 2.0 / 3.0)
        self.assertEqual(persisted, result)
        self.assertNotIn(cfg.wandb_key, repr((result, wandb.init_kwargs, wandb.logged)))
        self.assertEqual(wandb.init_kwargs["mode"], "offline")
        self.assertEqual(result["wandb"]["mode"], "offline")
        self.assertEqual(agent.global_step, 17)
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_random_baseline_does_not_load_checkpoint_and_has_no_start_value(self):
        agent, cfg = agent_and_config()
        cfg.env_id = "Stub-v0"  # FrozenLake success has no generic-environment meaning.
        env = ToolEnv(length=1)
        with tempfile.TemporaryDirectory() as directory:
            with (
                evaluate_stubs(cfg, env, agent) as (wandb, _, load),
                patch.object(env.action_space, "sample", return_value=0) as sample,
                patch.object(agent, "act") as act,
            ):
                result = evaluation.evaluate(self._args(directory, random=True))
        load.assert_not_called()
        act.assert_not_called()
        self.assertEqual(sample.call_count, 3)
        self.assertEqual(result["returns"], [-1.0, 3.0, 1.0])
        self.assertEqual(result["start_values"], [None, None, None])
        self.assertIsNone(result["checkpoint"])
        self.assertIsNone(result["summary"]["success_rate"])
        self.assertEqual(env.seeds, [100, 101, 102])
        self.assertEqual(wandb.init_kwargs["mode"], "offline")
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_rollout_failure_closes_env_finishes_run_and_does_not_publish_json(self):
        agent, cfg = agent_and_config()
        env = ToolEnv(fail=True)
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory)
            with evaluate_stubs(cfg, env, agent) as (wandb, _, _):
                with self.assertRaisesRegex(RuntimeError, "stub rollout failure"):
                    evaluation.evaluate(args)
            self.assertFalse(args.out.exists())
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()


class TestGifTool(unittest.TestCase):
    def test_top_returns_stride_and_final_goal_frames_with_stub_encoder(self):
        agent, cfg = agent_and_config()
        env = ToolEnv(rewards=(1.0, 3.0, 2.0), length=3)
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                config=Path("unused.yaml"),
                checkpoint=None,
                out=Path(directory) / "demo.gif",
                episodes=3,
                keep_top=2,
                fps=2,
                stride=2,
                seed=100,
            )

            def encoder(path, frames, **kwargs):
                Path(path).write_text("stub", encoding="utf-8")

            with (
                patch.dict(os.environ),
                patch.object(gif.Config, "from_yaml", return_value=cfg),
                patch.object(gif, "make_env", return_value=env),
                patch.object(gif, "load_agent", return_value=(agent, {})),
                patch.object(gif, "set_seed"),
                patch.object(gif.imageio, "mimsave", side_effect=encoder) as save,
                patch("builtins.print"),
            ):
                result = gif.make_gif(args)
                self.assertEqual(os.environ["SDL_VIDEODRIVER"], "dummy")
            frames = save.call_args.args[1]
            self.assertEqual(
                [int(frame[0, 0, 0]) for frame in frames],
                [10, 12, 13, 13, 13, 20, 22, 23, 23, 23],
            )
            self.assertEqual(
                save.call_args.kwargs,
                {"format": "pillow", "duration": 500.0, "loop": 0},
            )
        self.assertEqual(result["returns"], [3.0, 9.0, 6.0])
        self.assertEqual(result["selected_returns"], [9.0, 6.0])
        self.assertEqual(result["selected_seeds"], [101, 102])
        self.assertEqual(result["frames"], 10)
        self.assertEqual(agent.global_step, 17)
        self.assertEqual(env.close_calls, 1)

    def test_missing_frame_closes_environment_before_encoding(self):
        agent, cfg = agent_and_config()
        env = ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                config=Path("unused.yaml"),
                checkpoint=None,
                out=Path(directory) / "demo.gif",
                episodes=1,
                keep_top=1,
                fps=2,
                stride=1,
                seed=100,
            )
            with (
                patch.dict(os.environ),
                patch.object(gif.Config, "from_yaml", return_value=cfg),
                patch.object(gif, "make_env", return_value=env),
                patch.object(gif, "load_agent", return_value=(agent, {})),
                patch.object(gif, "set_seed"),
                patch.object(env, "render", return_value=None),
                patch.object(gif.imageio, "mimsave") as save,
            ):
                with self.assertRaisesRegex(RuntimeError, "returned no frame"):
                    gif.make_gif(args)
        save.assert_not_called()
        self.assertEqual(env.close_calls, 1)

    def test_hyphenated_gif_cli_and_invalid_top_count(self):
        args = gif.parse_args(["--episodes", "4", "--keep-top", "2", "--stride", "3"])
        self.assertEqual((args.episodes, args.keep_top, args.stride), (4, 2, 3))
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            gif.parse_args(["--episodes", "1", "--keep-top", "2"])


class TestEvaluationDefaults(unittest.TestCase):
    def test_default_evaluation_seed_is_outside_default_training_reset_range(self):
        cfg = Config()
        args = evaluation.parse_args([])
        self.assertGreater(args.seed, cfg.seed + cfg.total_episodes - 1)


class TestChartTool(unittest.TestCase):
    def _record(self, items):
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

    def test_history_nested_keys_scalar_fallback_sorting_and_invalid_episode_filter(
        self,
    ):
        ds = MagicMock()
        ds.scan_data.side_effect = [
            self._record(
                [("progress/episode", 20, True), ("stats/start_q", 2.0, True)]
            ),
            charts.pb.Record().SerializeToString(),
            self._record(
                [("progress/episode", 10, False), ("stats/start_q", 1.0, False)]
            ),
            self._record([("progress/episode", True, True)]),
            self._record([("progress/episode", float("nan"), True)]),
            self._record([("stats/start_q", 999.0, True)]),
            None,
        ]
        with patch.object(charts.datastore, "DataStore", return_value=ds):
            rows = charts.load_history("stub.wandb")
        self.assertEqual(
            rows,
            [
                {"progress/episode": 10, "stats/start_q": 1.0},
                {"progress/episode": 20, "stats/start_q": 2.0},
            ],
        )
        ds.open_for_scan.assert_called_once_with("stub.wandb")
        ds.close.assert_called_once()

    def test_invalid_history_json_closes_datastore(self):
        record = charts.pb.Record()
        item = record.history.item.add()
        item.key, item.value_json = "progress/episode", "invalid-json"
        ds = MagicMock()
        ds.scan_data.side_effect = [record.SerializeToString()]
        with patch.object(charts.datastore, "DataStore", return_value=ds):
            with self.assertRaisesRegex(ValueError, "Malformed W&B history JSON"):
                charts.load_history("stub.wandb")
        ds.close.assert_called_once()

    def _row(self, episode=10):
        return {
            "progress/episode": episode,
            "charts/avg_return": 1.0,
            "charts/avg_length": 2.0,
            "charts/success_rate": 1.0,
            "charts/epsilon": 0.1,
            "stats/start_q": 0.5,
            "stats/discounted_return": 0.75,
            "stats/td_error": 0.1,
            "loss/td_mse": 0.01,
        }

    def _args(self, directory):
        return Namespace(
            run=Path("stub.wandb"),
            out_dir=Path(directory) / "charts",
            title="Stub Qlearning",
            smooth=3,
            summary=None,
        )

    def _assert_chart_error(self, rows, message):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(charts, "load_history", return_value=rows),
            patch.object(charts.plt, "subplots") as plots,
        ):
            with self.assertRaisesRegex(ValueError, message):
                charts.make_charts(self._args(directory))
        plots.assert_not_called()

    def test_no_training_history_and_nonfinite_required_metrics_are_rejected(self):
        self._assert_chart_error([], "No training history")
        for value in (None, True, "bad", float("nan"), float("inf")):
            with self.subTest(value=value):
                row = self._row()
                row["loss/td_mse"] = value
                self._assert_chart_error(
                    [row], "Missing finite training metrics: loss/td_mse"
                )

    def test_q_and_discounted_behavior_return_need_matched_rows(self):
        first, second = self._row(10), self._row(20)
        del first["stats/start_q"]
        del second["stats/discounted_return"]
        self._assert_chart_error([first, second], "No matched rows")

    def test_fraction_and_nonnegative_td_diagnostics_are_validated(self):
        for key, value, message in (
            ("charts/epsilon", 1.1, "fractions"),
            ("charts/success_rate", -0.1, "fractions"),
            ("stats/td_error", -0.1, "nonnegative"),
            ("loss/td_mse", -0.1, "nonnegative"),
        ):
            with self.subTest(key=key):
                row = self._row()
                row[key] = value
                self._assert_chart_error([row], message)

    def test_raw_summary_preserves_series_and_tabular_units(self):
        first, second = self._row(10), self._row(20)
        first["charts/avg_return"], second["charts/avg_return"] = -1.0, 3.0
        summary = charts.summarize([first, second], Path("stub.wandb"))
        metric = summary["metrics"]["charts/avg_return"]
        self.assertEqual(metric["episodes"], [10.0, 20.0])
        self.assertEqual(metric["values"], [-1.0, 3.0])
        self.assertEqual(metric["mean"], 1.0)
        self.assertEqual((metric["min"], metric["max"]), (-1.0, 3.0))
        self.assertIn("progress/steps", summary["missing_metrics"])
        self.assertIn("not a neural-network loss", summary["units"]["loss/td_mse"])
        np.testing.assert_allclose(charts.smooth(np.array([-1.0, 3.0]), 3), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
