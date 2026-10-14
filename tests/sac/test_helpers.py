"""Native helper contracts with stub rollouts, tiny encoded GIFs, and pure charts."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from gymnasium import spaces
from PIL import Image
import torch

from SAC.utils import _common, evaluate, make_charts, make_gif


class NativeStubEnv:
    """A unit fixture, not a physics simulation or performance measurement."""

    def __init__(self):
        self.observation_space = spaces.Box(-8.0, 8.0, (3,), dtype=np.float32)
        self.action_space = spaces.Box(-2.0, 2.0, (1,), dtype=np.float32)
        self.spec = SimpleNamespace(max_episode_steps=200)
        self.unwrapped = self
        self.g, self.m, self.l, self.dt = 10.0, 1.0, 1.0, 0.05
        self.max_speed, self.max_torque = 8.0, 2.0
        self.state = np.zeros(2)
        self.steps = 0
        self.seed = None
        self.reset_seeds = []
        self.render = Mock(
            side_effect=AssertionError("no environment rendering in units")
        )

    def reset(self, seed=None):
        self.seed = seed
        self.reset_seeds.append(seed)
        self.steps = 0
        self.state[:] = 0.0
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        self.state[:] = [0.0, 0.0] if self.steps % 2 == 0 else [0.2, 0.6]
        theta, velocity = self.state
        obs = np.array([np.cos(theta), np.sin(theta), velocity], dtype=np.float32)
        return obs, -float(self.seed), False, self.steps == 200, {}


class TestHelpers(unittest.TestCase):
    def test_native_rollouts_retain_all_full_horizons_and_observer_diagnostics(
        self,
    ) -> None:
        env = NativeStubEnv()
        actor = Mock()
        actor.deterministic.return_value = torch.zeros(1, 1)
        actor.sample.side_effect = AssertionError("inference must not sample")
        model = _common.InferenceModel(actor)
        records = []
        for index, seed in enumerate([7, 8, 9]):
            record, frames = _common.rollout_episode(
                model,
                env,
                {"gamma": 0.99},
                seed,
                episode=index,
                source={"observed_physics": {"dt": 0.05}},
            )
            records.append(record)
            self.assertEqual(frames, [])
            self.assertEqual(record["episode_length"], 200)
            self.assertEqual(record["native_return"], -200.0 * seed)
            self.assertEqual(len(record["native_rewards"]), 200)
            self.assertEqual(len(record["theta_trajectory"]), 201)
            self.assertEqual(len(record["angular_velocity_trajectory"]), 201)
            self.assertEqual(len(record["soft_q_reward_points"]), 200)
            self.assertTrue(record["native_200_step_episode"])
            self.assertTrue(record["timeout"])
            self.assertFalse(record["capped"])
            self.assertFalse(record["calibration_eligible"])
            self.assertEqual(record["upright_diagnostic"]["upright_fraction"], 0.5)
            self.assertEqual(record["upright_diagnostic"]["upright_time_seconds"], 5.0)
        summary = evaluate.summarize(records, diagnostic_return_threshold=-1500.0)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["native_return"]["mean"], -1600.0)
        self.assertEqual(summary["native_200_step_episode_count"], 3)
        self.assertIsNone(summary["solved_threshold"])
        self.assertEqual(
            summary["optional_return_diagnostic"]["episode_count_ge_threshold"], 1
        )
        self.assertFalse(
            summary["optional_return_diagnostic"]["all_episode_mean_ge_threshold"]
        )
        self.assertEqual(env.reset_seeds, [7, 8, 9])
        actor.sample.assert_not_called()
        env.render.assert_not_called()

    def test_gif_selection_is_illustrative_full_span_and_metadata_reads_encoded_bytes(
        self,
    ) -> None:
        records = [{"native_return": score} for score in [-10.0, -2.0, -2.0, -20.0]]
        self.assertEqual(make_gif.select_top_episodes(records, 2), [1, 2])
        indices = make_gif.frame_indices(201, 3)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 200)
        self.assertEqual(len(records), 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny.gif"
            first = Image.new("RGB", (2, 2), "red")
            second = Image.new("RGB", (2, 2), "blue")
            first.save(
                path, save_all=True, append_images=[second], duration=[40, 70], loop=0
            )
            metadata = make_gif.encoded_gif_metadata(path)
        self.assertEqual(metadata["frames"], 2)
        self.assertEqual(metadata["encoded_frames"], 2)
        self.assertEqual(metadata["duration_ms"], 110.0)
        self.assertIn("encoded bytes", metadata["encoder_note"])
        for keep in [0, 5, True]:
            with self.subTest(keep=keep), self.assertRaises(ValueError):
                make_gif.select_top_episodes(records, keep)

    def test_soft_q_chart_data_retains_entropy_units_and_rejects_unmatched_steps(
        self,
    ) -> None:
        row = {
            "progress/step": 20,
            "progress/global_step": 20,
            "charts/avg_return": -200.0,
            "charts/avg_length": 200.0,
            "stats/q1_mean": -100.0,
            "stats/q2_mean": -101.0,
            "stats/target_q_mean": -102.0,
        }
        rows = [row, {"progress/step": 21, "stats/q1_mean": -99.0}]
        matched = make_charts.matched_q_rows(rows)
        self.assertEqual(matched, [row])
        with patch("SAC.utils.make_charts.sha256", return_value="static-fixture"):
            summary = make_charts.summarize(rows, "unused.wandb")
        self.assertIn("entropy-augmented", summary["units"]["stats/q1_mean"])
        self.assertIn("undiscounted", summary["units"]["charts/avg_return"])
        self.assertIn("including bootstrap", summary["interpretation"])
        self.assertIn("not a calibration error", summary["interpretation"])
        bad = dict(row, **{"progress/global_step": 22})
        with self.assertRaises(ValueError):
            make_charts.series([bad], "stats/q1_mean")
        with self.assertRaises(ValueError):
            make_charts.matched_q_rows([rows[1]])


if __name__ == "__main__":
    unittest.main()
