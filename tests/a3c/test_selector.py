"""Checkpoint selection uses complete native scores and unchanged trusted bytes.

These tests serialize tiny real tensor checkpoints, but mock all model loading,
environment creation, episode rollouts, and W&B. No physics or training is used.
"""

from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from A3C.utils import select_checkpoint as selector

from ._support import StubEnv
from .test_helpers import _config, _record


def _native_record(seed, reward):
    record = _record(seed, reward)
    record.update(
        complete_episode=True,
        native_landing=reward >= 0,
        landed_on_pad=reward >= 0,
        crash=reward < 0,
        failure=reward < 0,
    )
    return record


def _payload(tag):
    return {
        "model_state_dict": {
            "policy_head.weight": torch.tensor([[tag, 2.0], [3.0, 4.0]])
        },
        "optimizer_state_dict": {
            "state": {
                0: {
                    "step": torch.tensor(3.0),
                    "exp_avg": torch.tensor([0.2, 0.4]),
                    "exp_avg_sq": torch.tensor([0.04, 0.16]),
                }
            }
        },
        "step": 100,
        "best_return": 123.0,
        "metadata": {
            "training_completed": False,
            "progress_semantics": "reserved_transitions",
        },
    }


class TestCheckpointSelector(unittest.TestCase):
    def _fixture(self, root):
        checkpoints = root / "checkpoints"
        checkpoints.mkdir()
        rewards = {
            "best.pt": [-100.0, -100.0],
            "checkpoint_100.pt": [-200.0, 300.0],
            "checkpoint_200.pt": [-120.0, 240.0],
        }
        for index, name in enumerate(rewards):
            torch.save(_payload(float(index + 1)), checkpoints / name)
        nested = checkpoints / "nested"
        nested.mkdir()
        (nested / "checkpoint_999.pt").write_bytes(
            b"must not be discovered recursively"
        )
        out = checkpoints / "best_greedy.pt"
        out.write_bytes(b"previous inference alias")
        config_path = root / "config.json"
        config_path.write_text("{}", encoding="utf-8")
        args = SimpleNamespace(
            config=config_path,
            checkpoint_dir=None,
            out=None,
            summary=root / "selection.json",
            episodes=2,
            seed=90000,
            max_candidates=100,
        )
        config = _config(checkpoint_dir=str(checkpoints), run_name="unit-selector")
        return args, config, checkpoints, rewards, out

    def _mock_workflow(self, stack, config, rewards, *, failure=None):
        self.env = StubEnv(obs_dim=8, act_dim=4)
        self.run = MagicMock(id="unit-selection", dir="unit-run-dir")
        self.run.summary = {}
        stack.enter_context(patch.dict(os.environ))
        stack.enter_context(
            patch("A3C.utils.select_checkpoint.read_config", return_value=config)
        )
        self.make_env = stack.enter_context(
            patch("A3C.utils.select_checkpoint.make_native_env", return_value=self.env)
        )
        stack.enter_context(
            patch(
                "A3C.utils.select_checkpoint.native_source_metadata",
                return_value={"native_horizon": 999 if failure == "horizon" else 1000},
            )
        )
        self.init = stack.enter_context(
            patch("A3C.utils.select_checkpoint.wandb.init", return_value=self.run)
        )

        def load_model(config, env, path):
            if failure == "load":
                raise RuntimeError("unit model load failed")
            return SimpleNamespace(path=Path(path)), {
                "path": str(Path(path).resolve()),
                "sha256": selector.sha256(path),
            }

        def rollout(model, env, config, seed, *, episode, source):
            if failure == "rollout":
                raise RuntimeError("unit rollout failed")
            return _native_record(seed, rewards[model.path.name][episode]), []

        self.load = stack.enter_context(
            patch("A3C.utils.select_checkpoint.load_agent", side_effect=load_model)
        )
        self.rollout = stack.enter_context(
            patch("A3C.utils.select_checkpoint.rollout_episode", side_effect=rollout)
        )
        stack.enter_context(
            patch(
                "A3C.a3c.agent.A3CAgent",
                side_effect=AssertionError("training agent constructed"),
            )
        )
        stack.enter_context(
            patch(
                "A3C.a3c.utils.SharedAdam",
                side_effect=AssertionError("optimizer constructed"),
            )
        )
        if failure == "copy":
            stack.enter_context(
                patch(
                    "A3C.utils.select_checkpoint.atomic_copy_checkpoint",
                    side_effect=RuntimeError("unit alias copy failed"),
                )
            )

    def test_all_candidates_keep_negative_episodes_and_select_complete_native_mean(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            args, config, directory, rewards, out = self._fixture(Path(temporary))
            candidates = selector.enumerate_candidates(directory)
            self.assertEqual(
                [path.name for path in candidates],
                ["best.pt", "checkpoint_100.pt", "checkpoint_200.pt"],
            )
            before = {path: path.read_bytes() for path in candidates}
            previous_alias = out.read_bytes()
            real_copy = selector.atomic_copy_checkpoint

            def copy_after_complete_scores(source, destination, digest):
                scored = json.loads(args.summary.read_text(encoding="utf-8"))
                self.assertEqual(
                    scored["alias_status"], "scores complete; copy pending"
                )
                self.assertEqual(len(scored["candidates"]), 3)
                self.assertTrue(
                    all(
                        len(candidate["episodes"]) == 2
                        for candidate in scored["candidates"]
                    )
                )
                self.assertEqual(out.read_bytes(), previous_alias)
                return real_copy(source, destination, digest)

            with ExitStack() as stack:
                self._mock_workflow(stack, config, rewards)
                stack.enter_context(
                    patch(
                        "A3C.utils.select_checkpoint.atomic_copy_checkpoint",
                        side_effect=copy_after_complete_scores,
                    )
                )
                save = stack.enter_context(
                    patch(
                        "A3C.utils.select_checkpoint.torch.save",
                        side_effect=AssertionError("checkpoint reserialized"),
                    )
                )
                result = selector.select_checkpoint(args)
                save.assert_not_called()
            chosen = directory / "checkpoint_200.pt"
            self.assertEqual(result["chosen_source_path"], str(chosen.resolve()))
            self.assertEqual(result["chosen_summary"]["native_return"]["mean"], 60.0)
            self.assertEqual(result["development_seeds"], [90000, 90001])
            self.assertEqual(result["candidate_count"], 3)
            self.assertEqual(self.load.call_count, 3)
            self.assertEqual(self.rollout.call_count, 6)
            self.assertEqual(
                [call.args[3] for call in self.rollout.call_args_list],
                [90000, 90001] * 3,
            )
            self.assertTrue(
                all(
                    "max_steps" not in call.kwargs
                    for call in self.rollout.call_args_list
                )
            )
            for candidate in result["candidates"]:
                name = Path(candidate["source_path"]).name
                self.assertEqual(
                    [ep["native_return"] for ep in candidate["episodes"]], rewards[name]
                )
                self.assertEqual(candidate["summary"]["count"], 2)
            self.assertIs(
                result["chosen_snapshot_metadata"]["metadata"]["training_completed"],
                False,
            )
            self.assertEqual(result["alias_status"], "copied byte-identically")
            self.assertEqual(out.read_bytes(), before[chosen])
            self.assertEqual(result["out_sha256"], selector.sha256(out))
            self.assertEqual(result["chosen_source_sha256"], selector.sha256(chosen))
            self.assertTrue(
                all(path.read_bytes() == data for path, data in before.items())
            )
            exported = json.loads(args.summary.read_text(encoding="utf-8"))
            self.assertEqual(
                sum(len(item["episodes"]) for item in exported["candidates"]), 6
            )
            self.assertEqual(self.init.call_args.kwargs["mode"], "offline")
            self.assertEqual(self.init.call_count, 1)
            self.assertTrue(self.env.close_called)
            self.run.finish.assert_called_once()

        # Negative scores must still select a winner, and ties keep input order.
        negative = []
        for scores in ([-9.0, -1.0], [-4.0, -3.0], [-4.0, -3.0]):
            negative.append(
                {
                    "episodes": [
                        _native_record(1 + i, value) for i, value in enumerate(scores)
                    ],
                    "summary": {"count": 2, "native_return": {"mean": sum(scores) / 2}},
                }
            )
        self.assertIs(selector.choose_candidate(negative), negative[1])

    def test_atomic_alias_preserves_tensor_checkpoint_bytes_and_failed_copy_preserves_old_output(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, out = root / "source.pt", root / "alias.pt"
            torch.save(_payload(7.0), source)
            original = source.read_bytes()
            digest = selector.sha256(source)
            out.write_bytes(b"old alias")
            real_replace = os.replace
            with (
                patch(
                    "A3C.utils.select_checkpoint.os.replace", wraps=real_replace
                ) as replace,
                patch(
                    "A3C.utils.select_checkpoint.torch.save",
                    side_effect=AssertionError("reserialized"),
                ),
                patch(
                    "A3C.utils.select_checkpoint.torch.load",
                    side_effect=AssertionError("deserialized"),
                ),
            ):
                selector.atomic_copy_checkpoint(source, out, digest)
            replace.assert_called_once()
            self.assertEqual(Path(replace.call_args.args[1]), out)
            self.assertEqual(Path(replace.call_args.args[0]).parent, root)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(out.read_bytes(), original)
            self.assertEqual(selector.sha256(out), digest)
            restored = torch.load(out, map_location="cpu", weights_only=True)
            torch.testing.assert_close(
                restored["model_state_dict"]["policy_head.weight"],
                _payload(7.0)["model_state_dict"]["policy_head.weight"],
            )
            torch.testing.assert_close(
                restored["optimizer_state_dict"]["state"][0]["exp_avg"],
                torch.tensor([0.2, 0.4]),
            )
            self.assertIs(restored["metadata"]["training_completed"], False)

            for fault in ("hash", "replace"):
                with self.subTest(fault=fault):
                    out.write_bytes(b"old alias")
                    with ExitStack() as stack:
                        if fault == "replace":
                            stack.enter_context(
                                patch(
                                    "A3C.utils.select_checkpoint.os.replace",
                                    side_effect=OSError("unit replace failed"),
                                )
                            )
                        with self.assertRaises((ValueError, OSError)):
                            selector.atomic_copy_checkpoint(
                                source,
                                out,
                                "not-the-source-hash" if fault == "hash" else digest,
                            )
                    self.assertEqual(out.read_bytes(), b"old alias")
                    self.assertEqual(source.read_bytes(), original)
                    self.assertEqual(list(root.glob(".alias.pt.*.tmp")), [])

    def test_candidate_path_score_guards_and_workflow_failures_cleanup_without_overwriting_inputs(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            args, config, directory, rewards, out = self._fixture(Path(temporary))
            candidates = selector.enumerate_candidates(directory)
            empty = Path(temporary) / "empty"
            empty.mkdir()
            with self.assertRaises(ValueError):
                selector.enumerate_candidates(empty)
            for limit in (0, True, 2):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    selector.enumerate_candidates(directory, max_candidates=limit)
            symlink = directory / "checkpoint_symlink.pt"
            symlink.symlink_to(candidates[0])
            with self.assertRaises(ValueError):
                selector.enumerate_candidates(directory)
            symlink.unlink()

            for destination, summary in (
                (candidates[0], args.summary),
                (out, args.config),
                (out, candidates[0]),
            ):
                with (
                    self.subTest(destination=destination, summary=summary),
                    self.assertRaises(ValueError),
                ):
                    selector.validate_paths(
                        args.config, candidates, destination, summary
                    )
            with (
                patch("A3C.utils.select_checkpoint.read_config", return_value=config),
                patch("A3C.utils.select_checkpoint.make_native_env") as make,
                patch("A3C.utils.select_checkpoint.wandb.init") as init,
            ):
                unsafe = copy.copy(args)
                unsafe.out = candidates[0]
                with self.assertRaises(ValueError):
                    selector.select_checkpoint(unsafe)
            make.assert_not_called()
            init.assert_not_called()

            valid = {
                "episodes": [_native_record(1, 1.0), _native_record(2, 3.0)],
                "summary": {"count": 2, "native_return": {"mean": 2.0}},
            }
            for fault in ("count", "mean", "capped"):
                bad = copy.deepcopy(valid)
                if fault == "count":
                    bad["summary"]["count"] = 1
                elif fault == "mean":
                    bad["summary"]["native_return"]["mean"] = 3.0
                else:
                    bad["episodes"][0]["capped"] = True
                with self.subTest(score_fault=fault), self.assertRaises(ValueError):
                    selector.choose_candidate([bad])

            before = {path: path.read_bytes() for path in candidates}
            for failure in ("load", "rollout", "copy", "horizon"):
                with self.subTest(failure=failure), ExitStack() as stack:
                    out.write_bytes(b"old inference alias")
                    self._mock_workflow(stack, config, rewards, failure=failure)
                    with self.assertRaises((RuntimeError, ValueError)):
                        selector.select_checkpoint(args)
                    self.assertTrue(self.env.close_called)
                    if failure == "horizon":
                        self.init.assert_not_called()
                    else:
                        self.run.finish.assert_called_once()
                        report = json.loads(args.summary.read_text(encoding="utf-8"))
                        self.assertIn("error", report)
                        self.assertNotEqual(
                            report["alias_status"], "copied byte-identically"
                        )
                        if failure == "copy":
                            self.assertEqual(
                                sum(
                                    len(candidate["episodes"])
                                    for candidate in report["candidates"]
                                ),
                                6,
                            )
                    self.assertEqual(out.read_bytes(), b"old inference alias")
                    self.assertTrue(
                        all(path.read_bytes() == data for path, data in before.items())
                    )
