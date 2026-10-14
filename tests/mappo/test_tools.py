"""MAPPO offline evaluation/provenance and asset helpers with deterministic stubs."""

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

from MAPPO.mappo.config import Config
from MAPPO.utils import _common as common
from MAPPO.utils import evaluate as evaluation
from MAPPO.utils import make_gif as gif
from MAPPO.utils import make_charts as charts


class ToolEnv:
    def __init__(self, lengths=(3, 1, 2), mode="terminated", fail=False):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents.copy()
        self.observation = spaces.Box(-20.0, 20.0, (2,), dtype=np.float32)
        self.action = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.lengths, self.mode, self.fail = lengths, mode, fail
        self.episode = -1
        self.steps = self.close_calls = 0
        self.action_keys = []
        self.frame = np.zeros((1, 1, 3), dtype=np.uint8)

    def observation_space(self, name):
        return self.observation

    def action_space(self, name):
        return self.action

    def reset(self, seed=None):
        self.episode += 1
        self.steps = 0
        self.agents = self.possible_agents.copy()
        return {name: np.zeros(2, dtype=np.float32) for name in self.agents}, {}

    def step(self, actions):
        if not actions or set(actions) != set(self.agents):
            raise AssertionError("must step active agents only")
        for value in actions.values():
            if not self.action.contains(value):
                raise AssertionError("invalid action")
        self.action_keys.append(set(actions))
        self.steps += 1
        if self.fail:
            raise RuntimeError("stub rollout failure")
        ended = self.steps >= self.lengths[self.episode]
        terms = {name: ended and self.mode == "terminated" for name in self.agents}
        truncs = {name: ended and self.mode == "truncated" for name in self.agents}
        rewards = {"agent_0": -1.0, "agent_1": -3.0}
        obs = {name: np.full(2, self.steps, dtype=np.float32) for name in self.agents}
        if ended:
            self.agents = []
        return (
            obs,
            rewards,
            terms,
            truncs,
            {"agent_0": {"access_token": "private-info-secret"}},
        )

    def render(self):
        self.frame[:] = self.episode * 10 + self.steps
        return self.frame

    def close(self):
        self.close_calls += 1


def cfg_and_agent():
    cfg = Config()
    cfg.n_walkers = cfg.n_agents = 2
    cfg.gamma = 0.5
    cfg.env_kwargs = {"max_cycles": 4}
    cfg.max_steps_per_episode = 4
    cfg.wandb_key = "private-config-secret"
    agent = MagicMock()
    agent.values.return_value = np.array([1.0, 2.0])
    agent.act.return_value = (
        np.zeros((2, 2), dtype=np.float32),
        np.zeros(2),
        np.array([1.0, 2.0]),
    )
    return cfg, agent


class OfflineWandb:
    def __init__(self):
        self.init_kwargs = None
        self.logged = []
        self.run = MagicMock()
        self.run.id, self.run.dir, self.run.summary = "stub-run", "stub-dir", {}

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def log(self, data, step=None):
        self.logged.append((dict(data), step))


@contextmanager
def evaluation_stubs(cfg, env, agent, payload=None):
    wandb = OfflineWandb()
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"WANDB_MODE": "online"}))
        stack.enter_context(
            patch.object(evaluation.Config, "from_yaml", return_value=cfg)
        )
        stack.enter_context(patch.object(evaluation, "create_env", return_value=env))
        load = stack.enter_context(
            patch.object(evaluation, "load_agent", return_value=(agent, payload or {}))
        )
        stack.enter_context(
            patch.object(evaluation, "checkpoint_metadata", return_value={"step": 9})
        )
        stack.enter_context(patch.object(evaluation, "wandb", wandb))
        yield wandb, load


class TestRolloutTools(unittest.TestCase):
    def test_terminal_timeout_cap_flags_and_final_frames_are_not_task_success(self):
        for length, cap, mode, flags, survived in (
            (2, 4, "terminated", (True, False, False), False),
            (2, 4, "truncated", (False, True, False), True),
            (999, 2, "terminated", (False, False, True), False),
        ):
            with self.subTest(mode=mode, length=length):
                cfg, agent = cfg_and_agent()
                env = ToolEnv(lengths=(length,), mode=mode)
                record = common.rollout_episode(
                    env,
                    cfg,
                    agent,
                    100,
                    cap,
                    2 if length == 2 else 4,
                    capture_frames=True,
                )
                self.assertEqual(
                    tuple(record[key] for key in ("terminated", "truncated", "capped")),
                    flags,
                )
                self.assertEqual(record["survived_horizon"], survived)
                self.assertIsNone(record["success"])
                self.assertEqual(record["team_return"], -4.0)
                self.assertEqual(record["discounted_team_return"], -3.0)
                self.assertEqual(
                    record["per_agent_returns"], {"agent_0": -2.0, "agent_1": -6.0}
                )
                self.assertEqual(record["agent_transitions"], 4)
                self.assertEqual(
                    [int(frame[0, 0, 0]) for frame in record["frames"]], [0, 1, 2]
                )
                self.assertNotIn("private-info-secret", repr(record))

    def test_provenance_keeps_training_config_separate_and_legacy_unknown(self):
        cfg, _ = cfg_and_agent()
        saved = {
            "gamma": 0.9,
            "env_kwargs": {"max_cycles": 13, "token": "saved-secret"},
        }
        source = common.provenance(cfg, {"config": saved}, 100, 3, None)
        self.assertEqual(source["evaluation_config"]["gamma"], 0.5)
        self.assertEqual(source["checkpoint_config"]["gamma"], 0.9)
        self.assertEqual(source["checkpoint_config_status"], "recorded")
        self.assertIn(
            "gamma", [difference["path"] for difference in source["mismatches"]]
        )
        self.assertNotIn("saved-secret", repr(source))
        legacy = common.provenance(cfg, {}, 100, 3, None)
        self.assertEqual(legacy["checkpoint_config_status"], "legacy_unknown")
        self.assertIsNone(legacy["checkpoint_config"])
        self.assertIsNone(legacy["mismatches"])

    def test_malformed_provenance_and_raw_checkpoint_steps_are_rejected(self):
        cfg, _ = cfg_and_agent()
        for saved in ([], "not-a-mapping"):
            with self.subTest(saved=saved), self.assertRaises(ValueError):
                common.provenance(cfg, {"config": saved}, 100, 3, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stub.pt"
            path.write_text("stub", encoding="utf-8")
            for step in (-1, True, 1.5):
                with self.subTest(step=step), self.assertRaises(ValueError):
                    common.checkpoint_metadata(
                        path, {"step": step, "share_policy": True}
                    )

    def test_agent_removal_uses_fixed_denominator_and_never_steps_empty_actions(self):
        cfg, agent = cfg_and_agent()
        env = ToolEnv(lengths=(2,))
        original = env.step

        def removing(actions):
            result = original(actions)
            obs, rewards, terms, truncs, infos = result
            rewards = {"agent_1": -3.0}
            if env.steps == 1:
                terms["agent_0"] = True
                obs.pop("agent_0")
                env.agents = ["agent_1"]
            return obs, rewards, terms, truncs, infos

        with patch.object(env, "step", side_effect=removing):
            record = common.rollout_episode(env, cfg, agent, 100, 4, 4)
        self.assertEqual(record["team_return"], -3.0)
        self.assertEqual(record["per_agent_returns"], {"agent_0": 0.0, "agent_1": -6.0})
        self.assertEqual(record["agent_transitions"], 3)
        self.assertEqual(env.action_keys, [{"agent_0", "agent_1"}, {"agent_1"}])


class TestEvaluateTool(unittest.TestCase):
    def args(self, directory, random=False):
        return Namespace(
            config=Path("unused.yaml"),
            checkpoint=None,
            episodes=3,
            seed=100,
            out=Path(directory) / "evaluation.json",
            max_steps=None,
            random=random,
            stochastic=False,
        )

    def test_forced_offline_unfiltered_json_per_agent_returns_and_checkpoint_inference(
        self,
    ):
        cfg, agent = cfg_and_agent()
        env = ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            with evaluation_stubs(cfg, env, agent) as (wandb, load):
                result = evaluation.evaluate(args)
                self.assertEqual(os.environ["WANDB_MODE"], "offline")
            persisted = json.loads(args.out.read_text(encoding="utf-8"))
        load.assert_called_once_with(cfg, env, Path(cfg.inference_model_path))
        self.assertEqual(result["returns"], [-6.0, -2.0, -4.0])
        self.assertEqual(result["discounted_returns"], [-3.5, -2.0, -3.0])
        self.assertEqual(result["lengths"], [3, 1, 2])
        self.assertEqual(result["summary"]["count"], 3)
        self.assertEqual(result["summary"]["mean"], -4.0)
        self.assertAlmostEqual(result["summary"]["std"], np.std([-6.0, -2.0, -4.0]))
        self.assertEqual(result["summary"]["survival_rate"], 0.0)
        self.assertEqual(result["per_agent_summary"]["agent_1"]["mean"], -6.0)
        self.assertIsNone(result["success"])
        self.assertEqual(persisted, result)
        self.assertEqual(wandb.init_kwargs["mode"], "offline")
        self.assertNotIn(
            "private-config-secret", repr((result, wandb.init_kwargs, wandb.logged))
        )
        self.assertNotIn("private-info-secret", repr(result))
        self.assertTrue(
            all(call.kwargs["deterministic"] for call in agent.act.call_args_list)
        )
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_package_positions_track_native_displacement_and_available_summary(self):
        cfg, agent = cfg_and_agent()
        env = ToolEnv()
        env.engine = SimpleNamespace(
            package=SimpleNamespace(position=SimpleNamespace(x=0.0))
        )
        original_reset, original_step = env.reset, env.step

        def reset_with_position(seed=None):
            result = original_reset(seed=seed)
            env.engine.package.position.x = 10.0 + env.episode
            return result

        def advancing_package(actions):
            result = original_step(actions)
            env.engine.package.position.x += 0.5
            return result

        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            with (
                evaluation_stubs(cfg, env, agent) as (wandb, _),
                patch.object(env, "reset", side_effect=reset_with_position),
                patch.object(env, "step", side_effect=advancing_package),
            ):
                result = evaluation.evaluate(args)
            persisted = json.loads(args.out.read_text(encoding="utf-8"))
        self.assertEqual(
            [episode["initial_package_x"] for episode in result["episodes"]],
            [10.0, 11.0, 12.0],
        )
        self.assertEqual(
            [episode["final_package_x"] for episode in result["episodes"]],
            [11.5, 11.5, 13.0],
        )
        self.assertEqual(
            [episode["package_displacement"] for episode in result["episodes"]],
            [1.5, 0.5, 1.0],
        )
        self.assertEqual(result["summary"]["package_displacement_available_count"], 3)
        self.assertEqual(result["summary"]["package_displacement_mean"], 1.0)
        self.assertEqual(result["summary"]["package_displacement_min"], 0.5)
        self.assertEqual(result["summary"]["package_displacement_max"], 1.5)
        self.assertEqual(
            [data["eval/package_displacement"] for data, _ in wandb.logged[:-1]],
            [1.5, 0.5, 1.0],
        )
        self.assertEqual(
            wandb.logged[-1][0]["eval/package_displacement_available_count"], 3
        )
        self.assertEqual(persisted, result)
        self.assertIsNone(result["success"])
        # Diagnostics must not append package state to actors' local observations.
        for call, native_value in zip(
            agent.act.call_args_list, (0.0, 1.0, 2.0, 0.0, 0.0, 1.0)
        ):
            self.assertEqual(np.asarray(call.args[0]).shape, (2, 2))
            np.testing.assert_array_equal(call.args[0], np.full((2, 2), native_value))
            np.testing.assert_array_equal(
                call.kwargs["state"], np.full(4, native_value)
            )
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_unavailable_package_positions_remain_none_and_are_not_logged_as_zero(self):
        cfg, agent = cfg_and_agent()
        env = ToolEnv()
        env.engine = None
        with tempfile.TemporaryDirectory() as directory:
            with evaluation_stubs(cfg, env, agent) as (wandb, _):
                result = evaluation.evaluate(self.args(directory))
        for episode in result["episodes"]:
            for key in ("initial_package_x", "final_package_x", "package_displacement"):
                self.assertIsNone(episode[key])
        self.assertEqual(result["summary"]["package_displacement_available_count"], 0)
        for key in (
            "package_displacement_mean",
            "package_displacement_min",
            "package_displacement_max",
        ):
            self.assertIsNone(result["summary"][key])
            self.assertTrue(all(f"eval/{key}" not in data for data, _ in wandb.logged))
        for data, _ in wandb.logged[:-1]:
            self.assertNotIn("eval/initial_package_x", data)
            self.assertNotIn("eval/final_package_x", data)
            self.assertNotIn("eval/package_displacement", data)
        self.assertIsNone(result["success"])
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_nonfinite_native_package_position_fails_and_closes_env_and_run(self):
        cfg, agent = cfg_and_agent()
        env = ToolEnv(lengths=(1,))
        env.engine = SimpleNamespace(
            package=SimpleNamespace(position=SimpleNamespace(x=1.0))
        )
        original_step = env.step

        def nonfinite_final_position(actions):
            result = original_step(actions)
            env.engine.package.position.x = float("nan")
            return result

        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            args.episodes = 1
            with (
                evaluation_stubs(cfg, env, agent) as (wandb, _),
                patch.object(env, "step", side_effect=nonfinite_final_position),
            ):
                with self.assertRaisesRegex(
                    ValueError, "package x position must be finite"
                ):
                    evaluation.evaluate(args)
            self.assertFalse(args.out.exists())
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_random_policy_skips_checkpoint_and_critic_and_cleanup_on_failure(self):
        cfg, agent = cfg_and_agent()
        env = ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            with (
                evaluation_stubs(cfg, env, agent) as (wandb, load),
                patch.object(
                    env.action, "sample", return_value=np.zeros(2, dtype=np.float32)
                ) as sample,
            ):
                result = evaluation.evaluate(self.args(directory, random=True))
        load.assert_not_called()
        agent.act.assert_not_called()
        agent.values.assert_not_called()
        self.assertEqual(sample.call_count, 12)
        self.assertIsNone(result["checkpoint"])
        self.assertEqual(
            result["provenance"]["checkpoint_config_status"], "no_checkpoint"
        )
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()
        env = ToolEnv(fail=True)
        with tempfile.TemporaryDirectory() as directory:
            with evaluation_stubs(cfg, env, agent) as (wandb, _):
                with self.assertRaisesRegex(RuntimeError, "stub rollout failure"):
                    evaluation.evaluate(self.args(directory))
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()


class TestGifTool(unittest.TestCase):
    def test_selected_showcase_keeps_unfiltered_returns_and_final_frames(self):
        cfg, agent = cfg_and_agent()
        env = ToolEnv()
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                config=Path("unused.yaml"),
                checkpoint=None,
                episodes=3,
                seed=100,
                out=Path(directory) / "demo.gif",
                max_steps=None,
                keep_top=2,
                fps=5,
                stride=2,
            )
            with (
                patch.dict(os.environ),
                patch.object(gif.Config, "from_yaml", return_value=cfg),
                patch.object(gif, "create_env", return_value=env),
                patch.object(gif, "load_agent", return_value=(agent, {})) as load,
                patch.object(gif.imageio, "mimsave") as save,
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
        self.assertEqual(result["returns"], [-6.0, -2.0, -4.0])
        self.assertEqual(result["selected_returns"], [-2.0, -4.0])
        self.assertEqual(result["selected_seeds"], [101, 102])
        self.assertEqual(result["frames"], 8)
        self.assertEqual(env.close_calls, 1)


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

    def test_datastore_nested_scalar_keys_and_finite_parallel_step_filter(self):
        ds = MagicMock()
        ds.scan_data.side_effect = [
            self.record(
                [("progress/global_step", 20, True), ("stats/value_mean", 2.0, True)]
            ),
            charts.pb.Record().SerializeToString(),
            self.record(
                [("progress/global_step", 10, False), ("stats/value_mean", 1.0, False)]
            ),
            self.record([("progress/global_step", True, True)]),
            None,
        ]
        with patch.object(charts.datastore, "DataStore", return_value=ds):
            rows = charts.load_history("stub.wandb")
        self.assertEqual(
            rows,
            [
                {"progress/global_step": 10, "stats/value_mean": 1.0},
                {"progress/global_step": 20, "stats/value_mean": 2.0},
            ],
        )
        ds.close.assert_called_once()

    def row(self, step=10):
        return {
            "progress/global_step": step,
            "charts/avg_return": -4.0,
            "charts/avg_length": 2.0,
            "stats/value_mean": -1.0,
            "stats/return_target_mean": -2.0,
            "loss/policy": 0.1,
            "loss/value": 0.2,
            "stats/entropy": 1.0,
            "stats/approx_kl": 0.01,
            "charts/agent_0_avg_return": -2.0,
            "charts/agent_1_avg_return": -6.0,
        }

    def assert_chart_error(self, rows, pattern):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(charts, "load_history", return_value=rows),
            patch.object(charts.plt, "subplots") as subplots,
        ):
            args = Namespace(
                run=Path("stub.wandb"),
                out_dir=Path(directory),
                title="Stub",
                smooth=3,
                summary=None,
            )
            with self.assertRaisesRegex(ValueError, pattern):
                charts.make_charts(args)
        subplots.assert_not_called()

    def test_required_finite_and_matched_metrics_are_validated(self):
        self.assert_chart_error([], "No MAPPO training history")
        row = self.row()
        row["loss/value"] = np.nan
        self.assert_chart_error([row], "Missing finite training metrics: loss/value")
        first, second = self.row(10), self.row(20)
        del first["stats/value_mean"]
        del second["stats/return_target_mean"]
        self.assert_chart_error([first, second], "No matched rows")

    def test_raw_summary_preserves_agent_series_and_parallel_units(self):
        summary = charts.summarize([self.row(10), self.row(20)], Path("stub.wandb"))
        self.assertEqual(
            summary["metrics"]["charts/agent_1_avg_return"]["values"], [-6.0, -6.0]
        )
        self.assertEqual(summary["metrics"]["charts/avg_return"]["steps"], [10.0, 20.0])
        self.assertIn("parallel", summary["step_units"])
        self.assertIn("not goal success", summary["units"]["charts/survival_rate"])


if __name__ == "__main__":
    unittest.main()
