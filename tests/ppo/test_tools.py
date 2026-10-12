"""Offline PPO tool regressions without real rollouts, charts, or GIF encoding."""

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

from PPO.ppo.config import Config
from PPO.utils import evaluate as evaluate_tool
from PPO.utils import make_charts as charts_tool
from PPO.utils import make_gif as gif_tool


class ToolEnv:
    """Single-environment stub with distinguishable rewards and reused RGB frames."""

    def __init__(self, rewards=(1.0, 3.0, 2.0), length=3, fail_at=None, timeout=False):
        self.observation_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.rewards = rewards
        self.length = length
        self.fail_at = fail_at
        self.timeout = timeout
        self.episode = -1
        self.steps = 0
        self.actions = []
        self.seeds = []
        self.close_calls = 0
        self.frame = np.zeros((1, 1, 3), dtype=np.uint8)

    def reset(self, seed=None):
        self.episode += 1
        self.steps = 0
        self.seeds.append(seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise AssertionError("invalid action")
        self.actions.append(int(action))
        self.steps += 1
        if self.steps == self.fail_at:
            raise RuntimeError("stub rollout failure")
        done = self.steps >= self.length
        return (
            np.zeros(4, dtype=np.float32),
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


class OfflineWandb:
    def __init__(self):
        self.logged = []
        self.init_kwargs = None
        self.run = SimpleNamespace(
            id="stub-run", dir="stub-local-dir", summary={}, finish=MagicMock()
        )

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def log(self, data, step=None):
        self.logged.append((dict(data), step))


def tool_config():
    cfg = Config()
    cfg.env_id = "Stub-v0"
    cfg.wandb_key = "do-not-log-test-key"
    return cfg


def stub_agent():
    agent = MagicMock()
    agent.act.return_value = (np.array([1]), np.array([0.0]), np.array([0.0]))
    return agent


def evaluation_args(directory, random=False):
    return Namespace(
        config=Path("unused.yaml"),
        checkpoint=Path("unused.pt"),
        episodes=3,
        seed=100,
        out=Path(directory) / "nested" / "evaluation.json",
        random=random,
    )


@contextmanager
def evaluation_stubs(cfg, env):
    wandb, agent = OfflineWandb(), stub_agent()
    metadata = {"sha256": "stub-digest", "step": 10}
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"WANDB_MODE": "online"}))
        stack.enter_context(
            patch.object(evaluate_tool.Config, "from_yaml", return_value=cfg)
        )
        make = stack.enter_context(
            patch.object(evaluate_tool.gym, "make", return_value=env)
        )
        load = stack.enter_context(
            patch.object(evaluate_tool, "load_agent", return_value=(agent, {}))
        )
        meta = stack.enter_context(
            patch.object(evaluate_tool, "checkpoint_metadata", return_value=metadata)
        )
        stack.enter_context(patch.object(evaluate_tool, "wandb", wandb))
        stack.enter_context(patch.object(evaluate_tool, "set_seed"))
        yield wandb, agent, make, load, meta


class TestEvaluationTool(unittest.TestCase):
    def test_deterministic_evaluation_is_offline_and_keeps_every_return(self) -> None:
        cfg, env = (
            tool_config(),
            ToolEnv(rewards=(-1.0, 3.0, 1.0), length=2, timeout=True),
        )
        with tempfile.TemporaryDirectory() as directory:
            args = evaluation_args(directory)
            with (
                evaluation_stubs(cfg, env) as (wandb, agent, make, load, meta),
                patch.object(env.action_space, "sample") as sample,
                patch.object(
                    env.action_space, "seed", wraps=env.action_space.seed
                ) as seed,
            ):
                result = evaluate_tool.evaluate(args)
                self.assertEqual(os.environ["WANDB_MODE"], "offline")
            persisted = json.loads(args.out.read_text(encoding="utf-8"))
        make.assert_called_once_with(cfg.env_id, render_mode=None)
        load.assert_called_once_with(cfg, env, args.checkpoint)
        meta.assert_called_once_with(args.checkpoint, {})
        sample.assert_not_called()
        self.assertEqual(
            seed.call_args_list,
            [unittest.mock.call(100), unittest.mock.call(101), unittest.mock.call(102)],
        )
        self.assertEqual(agent.act.call_count, 6)
        for call in agent.act.call_args_list:
            self.assertIs(call.kwargs["deterministic"], True)
            self.assertEqual(call.args[0].shape, (1, 4))
        self.assertEqual(result["returns"], [-2.0, 6.0, 2.0])
        self.assertEqual(result["seeds"], [100, 101, 102])
        self.assertEqual(result["episode_lengths"], [2, 2, 2])
        self.assertEqual(result["policy"], "deterministic")
        self.assertEqual(result["summary"]["count"], 3)
        self.assertEqual(result["summary"]["min"], -2.0)
        self.assertEqual(result["summary"]["max"], 6.0)
        self.assertAlmostEqual(result["summary"]["mean"], 2.0)
        self.assertAlmostEqual(result["summary"]["std"], np.std([-2.0, 6.0, 2.0]))
        self.assertEqual(persisted, result)
        self.assertEqual(wandb.init_kwargs["mode"], "offline")
        self.assertEqual(result["wandb"]["mode"], "offline")
        self.assertNotIn(cfg.wandb_key, repr((result, wandb.init_kwargs, wandb.logged)))
        self.assertEqual(
            [data["eval/episode_return"] for data, _ in wandb.logged[:-1]],
            [-2.0, 6.0, 2.0],
        )
        self.assertEqual(wandb.run.summary["eval/count"], 3)
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_random_evaluation_seeds_and_samples_without_loading_checkpoint(
        self,
    ) -> None:
        cfg, env = tool_config(), ToolEnv(length=1)
        with tempfile.TemporaryDirectory() as directory:
            args = evaluation_args(directory, random=True)
            with (
                evaluation_stubs(cfg, env) as (wandb, agent, _, load, meta),
                patch.object(env.action_space, "sample", return_value=2) as sample,
            ):
                result = evaluate_tool.evaluate(args)
        load.assert_not_called()
        meta.assert_not_called()
        agent.act.assert_not_called()
        self.assertEqual(sample.call_count, 3)
        self.assertEqual(env.actions, [2, 2, 2])
        self.assertEqual(env.seeds, [100, 101, 102])
        self.assertEqual(result["returns"], [1.0, 3.0, 2.0])
        self.assertEqual(result["policy"], "random")
        self.assertIsNone(result["checkpoint"])
        self.assertEqual(wandb.init_kwargs["mode"], "offline")
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_rollout_failure_closes_environment_and_finishes_offline_run(self) -> None:
        cfg, env = tool_config(), ToolEnv(fail_at=1)
        with tempfile.TemporaryDirectory() as directory:
            args = evaluation_args(directory)
            with evaluation_stubs(cfg, env) as (wandb, _, _, _, _):
                with self.assertRaisesRegex(RuntimeError, "stub rollout failure"):
                    evaluate_tool.evaluate(args)
            self.assertFalse(args.out.exists())
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_checkpoint_failure_closes_environment_without_initializing_run(
        self,
    ) -> None:
        cfg, env = tool_config(), ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            with evaluation_stubs(cfg, env) as (wandb, _, _, load, _):
                load.side_effect = RuntimeError("stub checkpoint failure")
                with self.assertRaisesRegex(RuntimeError, "stub checkpoint failure"):
                    evaluate_tool.evaluate(evaluation_args(directory))
        self.assertIsNone(wandb.init_kwargs)
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_not_called()

    def test_nonfinite_episode_return_is_rejected_before_writing_summary(self) -> None:
        cfg, env = tool_config(), ToolEnv(rewards=(float("nan"),), length=1)
        with tempfile.TemporaryDirectory() as directory:
            args = evaluation_args(directory)
            args.episodes = 1
            with evaluation_stubs(cfg, env) as (wandb, _, _, _, _):
                with self.assertRaisesRegex(ValueError, "Non-finite return"):
                    evaluate_tool.evaluate(args)
            self.assertFalse(args.out.exists())
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()


class TestGifTool(unittest.TestCase):
    def _args(self, directory):
        return Namespace(
            config=Path("unused.yaml"),
            checkpoint=Path("unused.pt"),
            out=Path(directory) / "nested" / "demo.gif",
            episodes=3,
            keep_top=2,
            seed=100,
            fps=5,
            stride=2,
        )

    def test_top_selection_stride_and_end_hold_do_not_filter_reported_returns(
        self,
    ) -> None:
        cfg, env, agent = tool_config(), ToolEnv(), stub_agent()
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory)

            def encode_stub(path, frames, **kwargs):
                # Only the writer is mocked; no GIF bytes are encoded or rendered.
                Path(path).write_text("stub", encoding="utf-8")

            with (
                patch.dict(os.environ),
                patch.object(gif_tool.Config, "from_yaml", return_value=cfg),
                patch.object(gif_tool.gym, "make", return_value=env) as make,
                patch.object(gif_tool, "load_agent", return_value=(agent, {})),
                patch.object(gif_tool, "set_seed"),
                patch.object(
                    gif_tool.imageio, "mimsave", side_effect=encode_stub
                ) as save,
                patch("builtins.print"),
            ):
                result = gif_tool.make_gif(args)
                self.assertEqual(os.environ["SDL_VIDEODRIVER"], "dummy")
            make.assert_called_once_with(cfg.env_id, render_mode="rgb_array")
            frames = save.call_args.args[1]
            self.assertEqual(
                [int(frame[0, 0, 0]) for frame in frames],
                [10, 12, 12, 12, 20, 22, 22, 22],
            )
            self.assertEqual(save.call_args.kwargs, {"duration": 200.0, "loop": 0})
        self.assertEqual(result["returns"], [3.0, 9.0, 6.0])
        self.assertEqual(result["selected_returns"], [9.0, 6.0])
        self.assertEqual(result["selected_seeds"], [101, 102])
        self.assertEqual(result["frames"], 8)
        self.assertEqual(agent.act.call_count, 9)
        self.assertTrue(
            all(call.kwargs["deterministic"] for call in agent.act.call_args_list)
        )
        self.assertEqual(env.close_calls, 1)

    def test_missing_rgb_frame_closes_environment_without_encoding(self) -> None:
        cfg, env = tool_config(), ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict(os.environ),
                patch.object(gif_tool.Config, "from_yaml", return_value=cfg),
                patch.object(gif_tool.gym, "make", return_value=env),
                patch.object(gif_tool, "load_agent", return_value=(stub_agent(), {})),
                patch.object(gif_tool, "set_seed"),
                patch.object(env, "render", return_value=None),
                patch.object(gif_tool.imageio, "mimsave") as save,
            ):
                with self.assertRaisesRegex(RuntimeError, "returned no frame"):
                    gif_tool.make_gif(self._args(directory))
        save.assert_not_called()
        self.assertEqual(env.close_calls, 1)

    def test_gif_cli_uses_hyphenated_flags_and_rejects_invalid_top_count(self) -> None:
        args = gif_tool.parse_args(
            ["--episodes", "4", "--keep-top", "2", "--stride", "3"]
        )
        self.assertEqual((args.episodes, args.keep_top, args.stride), (4, 2, 3))
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            gif_tool.parse_args(["--episodes", "1", "--keep-top", "2"])


class TestChartTool(unittest.TestCase):
    def _record(self, items):
        record = charts_tool.pb.Record()
        for key, value, nested in items:
            item = record.history.item.add()
            if nested:
                item.nested_key.extend(key.split("/"))
                item.key = "wrong-scalar-key"
            else:
                item.key = key
            item.value_json = json.dumps(value)
        return record.SerializeToString()

    def test_load_history_reads_repeated_nested_keys_falls_back_to_scalar_and_sorts(
        self,
    ) -> None:
        records = [
            self._record(
                [("progress/global_step", 10, True), ("charts/avg_return", -2.0, True)]
            ),
            charts_tool.pb.Record().SerializeToString(),  # not a history record
            self._record(
                [("progress/global_step", 5, False), ("charts/avg_return", -5.0, False)]
            ),
            self._record([("charts/avg_return", 99.0, True)]),  # no environment step
            self._record([("progress/global_step", "not-numeric", True)]),
            self._record([("progress/global_step", float("nan"), True)]),
            None,
        ]
        ds = MagicMock()
        ds.scan_data.side_effect = records
        with patch.object(charts_tool.datastore, "DataStore", return_value=ds):
            rows = charts_tool.load_history(Path("stub.wandb"))
        ds.open_for_scan.assert_called_once_with("stub.wandb")
        ds.close.assert_called_once()
        self.assertEqual(
            rows,
            [
                {"progress/global_step": 5, "charts/avg_return": -5.0},
                {"progress/global_step": 10, "charts/avg_return": -2.0},
            ],
        )
        self.assertTrue(all("wrong-scalar-key" not in row for row in rows))

    def test_load_history_closes_datastore_on_invalid_json(self) -> None:
        record = charts_tool.pb.Record()
        item = record.history.item.add()
        item.key = "progress/global_step"
        item.value_json = "invalid-json"
        ds = MagicMock()
        ds.scan_data.side_effect = [record.SerializeToString()]
        with patch.object(charts_tool.datastore, "DataStore", return_value=ds):
            with self.assertRaises(json.JSONDecodeError):
                charts_tool.load_history("stub.wandb")
        ds.close.assert_called_once()

    def _args(self, directory):
        return Namespace(
            run=Path("stub.wandb"),
            out_dir=Path(directory) / "charts",
            title="Stub PPO",
            smooth=3,
            summary=None,
        )

    def _row(self, step=10):
        return {
            "progress/global_step": step,
            "charts/avg_return": 3.0,
            "charts/avg_length": 3.0,
            "stats/value_mean": 1.0,
            "stats/return_target_mean": 2.0,
            "loss/policy": 0.1,
            "loss/value": 0.2,
        }

    def test_empty_history_fails_without_creating_figures(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(charts_tool, "load_history", return_value=[]),
            patch.object(charts_tool.plt, "subplots") as subplots,
        ):
            with self.assertRaisesRegex(ValueError, "No training history"):
                charts_tool.make_charts(self._args(directory))
        subplots.assert_not_called()

    def test_required_metrics_must_have_finite_numeric_values(self) -> None:
        for bad_value in (None, "bad", float("nan"), float("inf")):
            with (
                self.subTest(bad_value=bad_value),
                tempfile.TemporaryDirectory() as directory,
            ):
                row = self._row()
                row["loss/value"] = bad_value
                with (
                    patch.object(charts_tool, "load_history", return_value=[row]),
                    patch.object(charts_tool.plt, "subplots") as subplots,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "Missing finite training metrics: loss/value"
                    ):
                        charts_tool.make_charts(self._args(directory))
                subplots.assert_not_called()

    def test_value_and_target_metrics_must_exist_on_matching_rows(self) -> None:
        first, second = self._row(10), self._row(20)
        del first["stats/return_target_mean"]
        del second["stats/value_mean"]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(charts_tool, "load_history", return_value=[first, second]),
            patch.object(charts_tool.plt, "subplots") as subplots,
        ):
            with self.assertRaisesRegex(ValueError, "No matched rows"):
                charts_tool.make_charts(self._args(directory))
        subplots.assert_not_called()

    def test_summary_uses_unsmoothed_metrics_and_marks_missing_metrics(self) -> None:
        first, second = self._row(10), self._row(20)
        first["charts/avg_return"], second["charts/avg_return"] = -3.0, 9.0
        summary = charts_tool.summarize([first, second], Path("stub.wandb"))
        metrics = summary["metrics"]["charts/avg_return"]
        self.assertEqual(metrics["count"], 2)
        self.assertEqual(
            (metrics["first"], metrics["last"], metrics["mean"]), (-3.0, 9.0, 3.0)
        )
        self.assertEqual((metrics["min"], metrics["max"]), (-3.0, 9.0))
        self.assertEqual((metrics["first_step"], metrics["last_step"]), (10.0, 20.0))
        self.assertIn("stats/entropy", summary["missing_metrics"])
        self.assertIn("discounted", summary["units"]["stats/value_mean"])
        np.testing.assert_allclose(
            charts_tool.smooth(np.array([-3.0, 9.0]), 3), [3.0, 3.0]
        )


if __name__ == "__main__":
    unittest.main()
