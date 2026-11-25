"""Native helper regressions with stub physics and tiny local media fixtures."""

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import torch

from A3C.a3c.networks import ActorCritic
from A3C.utils import _common, evaluate, make_charts, make_gif

from ._support import StubEnv


def _config(**overrides):
    config = dict(
        env_id="LunarLander-v3",
        gamma=0.5,
        reward_scale=0.01,
        reward_shaping=False,
        hidden_sizes=[8],
        activation="relu",
        env_kwargs={},
    )
    config.update(overrides)
    return config


class ConstantModel(torch.nn.Module):
    def forward(self, obs):
        return torch.tensor([[0.0, 1.0, 3.0, 2.0]]), torch.tensor([2.0])


def _record(seed, native_return):
    return dict(
        seed=seed,
        native_return=native_return,
        episode_length=1,
        native_discounted_return=native_return,
        native_rewards=[native_return],
        training_value_predictions=[2.0],
        native_return_targets=[native_return],
        benchmark_ge_200=native_return >= 200.0,
        terminated=True,
        truncated=False,
        timeout=False,
        capped=False,
        calibration_eligible=True,
        native_landing=None,
        landed_on_pad=None,
        crash=None,
        out_of_bounds=None,
        failure=None,
        initial_native_value=200.0,
    )


class TestHelpers(unittest.TestCase):
    def test_deterministic_action_does_not_sample_and_validates_finite_shapes(self):
        model = ConstantModel()
        rng = torch.random.get_rng_state().clone()
        with patch(
            "torch.distributions.Categorical.sample",
            side_effect=AssertionError("sampled"),
        ):
            action, value = _common.deterministic_action(model, np.zeros(8))
        self.assertEqual((action, value), (2, 2.0))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for obs in (np.zeros(7), np.full(8, np.nan)):
            with self.subTest(obs=obs.shape), self.assertRaises(ValueError):
                _common.deterministic_action(model, obs)
        for outputs in (
            (torch.ones(1, 3), torch.zeros(1)),
            (torch.full((1, 4), float("inf")), torch.zeros(1)),
            (torch.zeros(1, 4), torch.tensor([float("nan")])),
        ):
            with (
                self.subTest(outputs=tuple(t.shape for t in outputs)),
                self.assertRaises(ValueError),
            ):
                _common.deterministic_action(
                    MagicMock(return_value=outputs), np.zeros(8)
                )

    def test_build_evaluation_model_preserves_rng_and_constructs_no_shared_optimizer(
        self,
    ):
        source = ActorCritic(obs_dim=8, act_dim=4, hidden_sizes=[8])
        env = StubEnv(obs_dim=8, act_dim=4)
        before = torch.random.get_rng_state().clone()
        with patch(
            "A3C.a3c.utils.SharedAdam",
            side_effect=AssertionError("optimizer constructed"),
        ):
            model = _common.build_agent(_config(), env, source.state_dict())
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertFalse(model.training)
        for original, restored in zip(source.parameters(), model.parameters()):
            self.assertTrue(torch.equal(original, restored))
            self.assertFalse(restored.is_shared())

    def test_rollout_preserves_native_units_frames_and_censored_or_shaped_targets(self):
        for shaped, cap in ((False, None), (True, None), (False, 1)):
            with self.subTest(shaped=shaped, cap=cap):
                env = StubEnv(obs_dim=8, act_dim=4, reward=2.0)
                env.render = lambda: np.full((4, 4, 3), env.steps, dtype=np.uint8)
                record, frames = _common.rollout_episode(
                    ConstantModel(),
                    env,
                    _config(reward_shaping=shaped),
                    123,
                    max_steps=cap,
                    capture_frames=True,
                    source={"landing_predicate_confirmed": False},
                )
                self.assertEqual(env.reset_seeds, [123])
                self.assertEqual(record["native_return"], 2.0 if cap else 4.0)
                self.assertEqual(
                    record["native_return_targets"], [2.0] if cap else [3.0, 2.0]
                )
                self.assertEqual(record["actions"], [2] * record["episode_length"])
                self.assertEqual(len(frames), record["episode_length"] + 1)
                self.assertEqual(int(frames[-1][0, 0, 0]), record["episode_length"])
                self.assertIsNone(record["native_landing"])
                self.assertEqual(
                    record["calibration_eligible"], not shaped and cap is None
                )
                if shaped:
                    self.assertIsNone(record["native_value_predictions"])
                    self.assertEqual(record["value_return_pairs"], [])
                else:
                    self.assertEqual(
                        record["native_value_predictions"],
                        [200.0] * record["episode_length"],
                    )
                if cap:
                    self.assertTrue(record["capped"])
                    self.assertFalse(record["terminated"])
                    self.assertFalse(record["complete_episode"])

    def test_evaluate_keeps_all_episode_returns_and_closes_on_load_or_rollout_failure(
        self,
    ):
        args = SimpleNamespace(
            seed=100,
            episodes=3,
            max_steps=None,
            config=Path("mock.yaml"),
            checkpoint=Path("mock.pt"),
            out=Path("mock-eval.json"),
            source_root=None,
        )
        for failure in (None, "load", "rollout"):
            with self.subTest(failure=failure), ExitStack() as stack:
                env = StubEnv(obs_dim=8, act_dim=4)
                run = MagicMock(id="stub-run", dir="stub-dir")
                records = [
                    _record(100 + index, reward)
                    for index, reward in enumerate((-100.0, 300.0, 50.0))
                ]
                args.checkpoint = None
                stack.enter_context(patch.dict(os.environ))
                stack.enter_context(
                    patch(
                        "A3C.utils.evaluate.read_config",
                        return_value=_config(inference_model_path="tuned.pt"),
                    )
                )
                stack.enter_context(
                    patch("A3C.utils.evaluate.config_metadata", return_value={})
                )
                stack.enter_context(
                    patch("A3C.utils.evaluate.make_native_env", return_value=env)
                )
                load = stack.enter_context(
                    patch(
                        "A3C.utils.evaluate.load_agent",
                        return_value=(ConstantModel(), {}),
                        side_effect=RuntimeError("stub load failed")
                        if failure == "load"
                        else None,
                    )
                )
                stack.enter_context(
                    patch(
                        "A3C.utils.evaluate.native_source_metadata",
                        return_value={"landing_predicate_confirmed": False},
                    )
                )
                init = stack.enter_context(
                    patch("A3C.utils.evaluate.wandb.init", return_value=run)
                )
                rollout = stack.enter_context(
                    patch(
                        "A3C.utils.evaluate.rollout_episode",
                        side_effect=RuntimeError("stub rollout failed")
                        if failure == "rollout"
                        else [(record, []) for record in records],
                    )
                )
                write = stack.enter_context(patch("A3C.utils.evaluate.write_json"))
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "stub"):
                        evaluate.evaluate(args)
                else:
                    result = evaluate.evaluate(args)
                    self.assertEqual(result["returns"], [-100.0, 300.0, 50.0])
                    self.assertEqual(result["summary"]["count"], 3)
                    self.assertEqual(
                        result["summary"]["native_return"]["mean"], 250.0 / 3
                    )
                    self.assertIsNone(result["summary"]["native_landing_fraction"])
                    self.assertEqual(rollout.call_count, 3)
                    self.assertEqual(
                        [call.args[3] for call in rollout.call_args_list],
                        [100, 101, 102],
                    )
                    self.assertEqual(init.call_args.kwargs["mode"], "offline")
                    write.assert_called_once()
                load.assert_called_once()
                self.assertEqual(load.call_args.args[2], Path("tuned.pt"))
                self.assertTrue(env.close_called)
                if failure == "load":
                    init.assert_not_called()
                else:
                    run.finish.assert_called_once()
        args.checkpoint = None
        args.out = Path("tuned.pt")
        with (
            patch(
                "A3C.utils.evaluate.read_config",
                return_value=_config(inference_model_path="tuned.pt"),
            ),
            patch("A3C.utils.evaluate.make_native_env") as make,
        ):
            with self.assertRaisesRegex(ValueError, "overwrite"):
                evaluate.evaluate(args)
        make.assert_not_called()

    def test_gif_selects_after_all_rollouts_and_keeps_true_final_frame_and_cleanup(
        self,
    ):
        args = SimpleNamespace(
            seed=100,
            episodes=3,
            keep_top=1,
            fps=25,
            stride=2,
            hold_frames=2,
            config=Path("mock.yaml"),
            checkpoint=Path("mock.pt"),
            out=Path("mock.gif"),
            summary=None,
            source_root=None,
        )
        for writer_error in (None, RuntimeError("stub GIF writer failed")):
            with self.subTest(writer_error=writer_error), ExitStack() as stack:
                env = StubEnv(obs_dim=8, act_dim=4)
                run = MagicMock(id="stub-run", dir="stub-dir")
                records = [
                    _record(100 + index, reward)
                    for index, reward in enumerate((1.0, 9.0, 4.0))
                ]
                frame_sets = [
                    [
                        np.full((4, 4, 3), index * 10 + step, dtype=np.uint8)
                        for step in range(4)
                    ]
                    for index in range(3)
                ]
                args.checkpoint = None
                stack.enter_context(patch.dict(os.environ))
                stack.enter_context(
                    patch(
                        "A3C.utils.make_gif.read_config",
                        return_value=_config(inference_model_path="tuned-gif.pt"),
                    )
                )
                stack.enter_context(
                    patch("A3C.utils.make_gif.config_metadata", return_value={})
                )
                stack.enter_context(
                    patch("A3C.utils.make_gif.make_native_env", return_value=env)
                )
                load = stack.enter_context(
                    patch(
                        "A3C.utils.make_gif.load_agent",
                        return_value=(ConstantModel(), {}),
                    )
                )
                stack.enter_context(
                    patch("A3C.utils.make_gif.native_source_metadata", return_value={})
                )
                stack.enter_context(
                    patch("A3C.utils.make_gif.wandb.init", return_value=run)
                )
                rollout = stack.enter_context(
                    patch(
                        "A3C.utils.make_gif.rollout_episode",
                        side_effect=list(zip(records, frame_sets)),
                    )
                )
                stack.enter_context(
                    patch(
                        "A3C.utils.make_gif._label_frame",
                        side_effect=lambda frame, *args: frame,
                    )
                )
                stack.enter_context(patch("pathlib.Path.mkdir"))
                writer = stack.enter_context(
                    patch(
                        "A3C.utils.make_gif.imageio.mimsave", side_effect=writer_error
                    )
                )
                encoded = stack.enter_context(
                    patch(
                        "A3C.utils.make_gif.encoded_gif_metadata",
                        return_value={
                            "frames": 3,
                            "encoded_frames": 3,
                            "duration_ms": 200.0,
                            "encoder_note": "duplicate final holds merged",
                        },
                    )
                )
                stack.enter_context(patch("A3C.utils.make_gif.write_json"))
                if writer_error:
                    with self.assertRaisesRegex(RuntimeError, "writer failed"):
                        make_gif.make_gif(args)
                    encoded.assert_not_called()
                else:
                    result = make_gif.make_gif(args)
                    self.assertEqual(result["frames"], 3)
                    self.assertEqual(result["encoded_frames"], 3)
                    self.assertEqual(result["submitted_frames"], 5)
                    self.assertEqual(result["duration_ms"], 200.0)
                    encoded.assert_called_once_with(args.out)
                    self.assertEqual(result["returns"], [1.0, 9.0, 4.0])
                    self.assertEqual(result["selected_episode_indices"], [1])
                    self.assertEqual(result["summary_all_episodes"]["count"], 3)
                    self.assertIn("not unbiased", result["selection"])
                self.assertEqual(rollout.call_count, 3)
                self.assertEqual(load.call_args.args[2], Path("tuned-gif.pt"))
                frames = writer.call_args.args[1]
                self.assertEqual(
                    [int(frame[0, 0, 0]) for frame in frames], [10, 12, 13, 13, 13]
                )
                self.assertTrue(env.close_called)
                run.finish.assert_called_once()
        args.checkpoint = None
        args.out = Path("tuned-gif.pt")
        with (
            patch(
                "A3C.utils.make_gif.read_config",
                return_value=_config(inference_model_path="tuned-gif.pt"),
            ),
            patch("A3C.utils.make_gif.make_native_env") as make,
        ):
            with self.assertRaisesRegex(ValueError, "overwrite"):
                make_gif.make_gif(args)
        make.assert_not_called()

    def test_encoded_gif_metadata_counts_coalesced_frames_and_actual_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "tiny.gif"
            red = Image.new("RGB", (3, 3), (255, 0, 0))
            blue = Image.new("RGB", (3, 3), (0, 0, 255))
            submitted = [red, red.copy(), blue, blue.copy(), blue.copy()]
            submitted[0].save(
                path,
                format="GIF",
                save_all=True,
                append_images=submitted[1:],
                duration=40,
                loop=0,
            )
            metadata = make_gif.encoded_gif_metadata(path)
            self.assertEqual(metadata["frames"], 2)
            self.assertEqual(metadata["encoded_frames"], 2)
            self.assertLess(metadata["frames"], len(submitted))
            self.assertEqual(metadata["duration_ms"], 200.0)
            with Image.open(path) as encoded:
                durations = []
                for index in range(encoded.n_frames):
                    encoded.seek(index)
                    durations.append(encoded.info["duration"])
            self.assertEqual(durations, [80, 120])
            self.assertEqual(metadata["duration_ms"], sum(durations))
            png = root / "not-a-gif.png"
            red.save(png)
            with self.assertRaises(ValueError):
                make_gif.encoded_gif_metadata(png)
            with self.assertRaises(FileNotFoundError):
                make_gif.encoded_gif_metadata(root / "missing.gif")

    def test_charts_read_nested_keys_match_same_rows_and_validate_finite_smoothing(
        self,
    ):
        records = []
        for step in (2, 1):
            record = make_charts.pb.Record()
            item = record.history.item.add()
            item.nested_key.extend(["progress", "global_step"])
            item.value_json = str(step)
            for key, value in (
                ("stats/value_mean", 2.0),
                ("stats/return_mean", 3.0),
                ("wandb_key", "not-exported"),
            ):
                item = record.history.item.add()
                item.key = key
                item.value_json = (
                    repr(value) if isinstance(value, float) else '"' + value + '"'
                )
            records.append(record.SerializeToString())
        ds = MagicMock()
        ds.scan_data.side_effect = records + [None]
        with patch("A3C.utils.make_charts.datastore.DataStore", return_value=ds):
            rows = make_charts.load_history("mock.wandb")
        ds.close.assert_called_once()
        self.assertEqual([row["progress/global_step"] for row in rows], [1, 2])
        self.assertNotIn("wandb_key", repr(rows))
        unmatched = {"progress/global_step": 3, "stats/value_mean": 99.0}
        self.assertEqual(make_charts.matched_value_rows(rows + [unmatched]), rows)
        np.testing.assert_allclose(
            make_charts.smooth(np.array([1.0, 2.0, 3.0, 4.0]), 2), [1.5, 2.5, 3.5, 4.0]
        )
        for values, window in (
            (np.array([np.nan]), 1),
            (np.ones(2), 0),
            (np.ones(2), True),
        ):
            with self.subTest(window=window), self.assertRaises(ValueError):
                make_charts.smooth(values, window)
        with self.assertRaises(ValueError):
            make_charts.matched_value_rows([unmatched])
        with self.assertRaises(ValueError):
            make_charts.series(
                [{"progress/global_step": 1, "stats/value_mean": np.inf}],
                "stats/value_mean",
            )
