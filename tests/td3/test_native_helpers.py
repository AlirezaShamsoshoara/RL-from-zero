"""Native helper contracts use a tiny engine stub, never gym.make or rendering."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from gymnasium import spaces

from TD3.utils import _common as common


class ZeroActor(torch.nn.Module):
    def forward(self, obs):
        return torch.zeros((len(obs), 1), dtype=torch.float32)


class NativeStub:
    def __init__(self):
        self.observation_space = spaces.Box(
            np.array([-1.2, -0.07], dtype=np.float32),
            np.array([0.6, 0.07], dtype=np.float32),
        )
        self.action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.spec = SimpleNamespace(max_episode_steps=999)
        self.unwrapped = SimpleNamespace(**common.PHYSICS, state=np.array([0.0, 0.0]))
        self.render = Mock(side_effect=AssertionError("render forbidden"))
        self.resets = []
        self.steps = 0

    def reset(self, seed=None):
        self.resets.append(seed)
        self.steps = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.array([0.0, 0.0], dtype=np.float32),
            100.0,
            False,
            self.steps == 999,
            {},
        )


def _source():
    return {"goal_predicate_confirmed": True, "observed_physics": dict(common.PHYSICS)}


class TestNativeHelpers(unittest.TestCase):
    def test_goal_requires_engine_position_velocity_termination_and_no_cap(self):
        env = NativeStub()
        for position, velocity, terminated, capped, expected in [
            (0.45, 0.0, True, False, True),
            (0.44, 0.0, True, False, False),
            (0.6, -0.001, True, False, False),
            (0.6, 0.01, False, False, False),
            (0.6, 0.01, True, True, False),
        ]:
            env.unwrapped.state = np.array([position, velocity])
            with self.subTest(
                position=position,
                velocity=velocity,
                terminated=terminated,
                capped=capped,
            ):
                outcome = common.native_outcome(
                    env, terminated, False, capped, _source()
                )
                self.assertIs(outcome["native_goal"], expected)
                self.assertIs(outcome["failure"], not expected)
        for source in [{}, {"goal_predicate_confirmed": False}]:
            outcome = common.native_outcome(env, True, False, False, source)
            self.assertIsNone(outcome["native_goal"])
            self.assertIsNone(outcome["failure"])
        env.unwrapped.state = None
        self.assertIsNone(
            common.native_outcome(env, True, False, False, _source())["native_goal"]
        )

    def test_native_kwargs_remove_training_horizon_but_reject_changed_physics(self):
        config = {"env_id": common.ENV_ID, "env_kwargs": {"max_episode_steps": 25}}
        self.assertEqual(common.native_env_kwargs(config), {"goal_velocity": 0})
        self.assertEqual(config["env_kwargs"]["max_episode_steps"], 25)
        common.validate_env(NativeStub())
        shortened = NativeStub()
        shortened.spec.max_episode_steps = 25
        with self.assertRaises(ValueError):
            common.validate_env(shortened)
        for kwargs in [
            {"goal_velocity": 0.01},
            {"max_episode_steps": True},
            {"max_episode_steps": 0},
            {"power": 0.01},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                common.native_env_kwargs(
                    {"env_id": common.ENV_ID, "env_kwargs": kwargs}
                )

    def test_full_native_999_stub_rollout_has_raw_returns_but_no_return_based_success(
        self,
    ):
        env = NativeStub()
        model = common.InferenceModel(ZeroActor())
        rng_before = torch.get_rng_state().clone()
        action, values = common.deterministic_action(
            model, np.zeros(2, dtype=np.float32)
        )
        self.assertEqual(action.shape, (1,))
        self.assertEqual(values, {"q1": None, "q2": None})
        action[:] = 99
        again, _ = common.deterministic_action(model, np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(again, [0.0])
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        record, frames = common.rollout_episode(
            model,
            env,
            {
                "gamma": 1.0,
                "max_steps_per_episode": 2,
                "env_kwargs": {"max_episode_steps": 25},
            },
            seed=11,
            source=_source(),
        )
        self.assertEqual(env.resets, [11])
        self.assertEqual(record["episode_length"], 999)
        self.assertEqual(record["native_return"], 99900.0)
        self.assertEqual(record["native_discounted_return"], 99900.0)
        self.assertTrue(record["truncated"])
        self.assertFalse(record["terminated"])
        self.assertFalse(record["native_goal"])
        self.assertTrue(record["timeout"])
        self.assertFalse(record["capped"])
        self.assertFalse(record["calibration_eligible"])
        self.assertEqual(len(record["observations"]), 1000)
        self.assertEqual(len(record["native_return_targets"]), 999)
        self.assertEqual(record["native_return_targets"][-1], 100.0)
        self.assertEqual(frames, [])
        env.render.assert_not_called()
        from TD3.utils.evaluate import summarize

        summary = summarize([record])
        self.assertTrue(summary["mean_native_return_ge_90"])
        self.assertEqual(summary["native_goal_rate"], 0.0)
        self.assertEqual(summary["count"], 1)
        unknown = {**record, "native_goal": None}
        summary = summarize([record, unknown])
        self.assertIsNone(summary["native_goal_rate"])
        self.assertIsNone(summary["native_goal_count"])
        self.assertIsNone(summary["failure_count"])
        self.assertEqual(summary["native_goal_known_count"], 1)

    def test_tool_parsers_and_chart_series_preserve_absence_and_full_span_selection(
        self,
    ):
        from TD3.utils import evaluate, make_charts, make_gif

        self.assertEqual(evaluate.parse_args([]).episodes, 100)
        self.assertIsNone(evaluate.parse_args([]).checkpoint)
        for parser, argv in [
            (evaluate.parse_args, ["--episodes", "0"]),
            (make_gif.parse_args, ["--fps", "101"]),
            (make_charts.parse_args, ["--run", "stub.wandb", "--smooth", "0"]),
        ]:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parser(argv)
        rows = [
            {"progress/step": 1, "loss/critic_loss": 2.0},
            {"progress/global_step": 2, "loss/actor_loss": None},
            {"progress/step": 3, "loss/actor_loss": -5.0},
        ]
        steps, values = make_charts.series(rows, "loss/actor_loss")
        np.testing.assert_array_equal(steps, [3.0])
        np.testing.assert_array_equal(values, [-5.0])
        with self.assertRaises(ValueError):
            make_charts.step_value({"progress/step": 1, "progress/global_step": 2})
        incomplete = [
            {"progress/step": 1, "stats/q1_value": 1.0},
            {"progress/step": 1, "stats/q2_value": 2.0, "stats/target_q_value": 3.0},
        ]
        with self.assertRaises(ValueError):
            make_charts.matched_q_rows(incomplete)
        complete = {
            "progress/step": 2,
            "stats/q1_value": 1.0,
            "stats/q2_value": 2.0,
            "stats/target_q_value": 3.0,
        }
        self.assertEqual(
            make_charts.matched_q_rows(incomplete + [complete]), [complete]
        )
        self.assertEqual(make_gif.frame_indices(10, 4), [0, 4, 8, 9])
        self.assertEqual(
            make_gif.select_top_episodes(
                [
                    {"native_return": 2.0, "native_goal": False},
                    {"native_return": 1.0, "native_goal": True},
                    {"native_return": 2.0, "native_goal": False},
                ],
                2,
            ),
            [0, 2],
        )
