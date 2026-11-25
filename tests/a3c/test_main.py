"""Main lifecycle regressions using synchronous fake processes and queues.

Patching a parent module does not patch a real spawn child. These tests replace
get_context itself, so neither a subprocess nor a physics environment is used.
"""

from contextlib import ExitStack
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from gymnasium import spaces

from A3C.a3c.config import Config
import A3C.main as a3c_main

from ._support import DummyWandb, FakeContext, StubEnv


def _make_config(**overrides):
    values = dict(
        total_steps=20,
        num_workers=1,
        num_envs=1,
        t_max=5,
        hidden_sizes=[8],
        device="cpu",
        checkpoint_interval=0,
        save_best=False,
        log_interval=0,
        log_to_console=False,
        log_to_file=False,
        wandb_key="",
        worker_timeout=0.1,
        shutdown_timeout=0.1,
    )
    values.update(overrides)
    return Config(**values).validate()


def _complete_worker(worker_id, cfg, agent, global_step, result_queue, stop_event):
    transitions = cfg.total_steps if worker_id == 0 else 0
    global_step.value += transitions
    result_queue.put(
        {
            "kind": "worker_done",
            "worker_id": worker_id,
            "collected_transitions": transitions,
            "rollout_steps": transitions,
            "worker_rollout_steps": transitions // cfg.num_envs,
            "reserved_transitions": transitions,
            "episodes": 0,
            "optimizer_steps": 2 if transitions else 0,
            "stopped": False,
            "budget_exhausted": True,
            "global_step": global_step.value,
        }
    )


class TestMain(unittest.TestCase):
    def _run(
        self,
        cfg=None,
        *,
        env=None,
        ctx=None,
        worker=_complete_worker,
        wandb_key="",
        save_error=None,
        log_error=None,
    ):
        self.env = env or StubEnv()
        self.ctx = ctx or FakeContext()
        self.wandb = DummyWandb()
        self.pbar = MagicMock()
        if log_error is not None:
            self.wandb.log = MagicMock(side_effect=log_error)
        with ExitStack() as stack:
            stack.enter_context(
                patch("A3C.main.Config.from_yaml", return_value=cfg or _make_config())
            )
            stack.enter_context(patch("A3C.main.wandb", self.wandb))
            stack.enter_context(
                patch("A3C.main.setup_logger", return_value=MagicMock())
            )
            stack.enter_context(patch("A3C.main.gym.make", return_value=self.env))
            self.get_context = stack.enter_context(
                patch("A3C.main.mp.get_context", return_value=self.ctx)
            )
            stack.enter_context(patch("A3C.main.worker_process", side_effect=worker))
            stack.enter_context(patch("A3C.main.tqdm", return_value=self.pbar))
            self.save = stack.enter_context(
                patch("A3C.main.save_checkpoint", side_effect=save_error)
            )
            return a3c_main.train(config="unused.yaml", wandb_key=wandb_key)

    def assert_training_cleanup(self):
        self.assertTrue(self.env.close_called)
        self.assertTrue(self.wandb.run.finished)
        self.pbar.close.assert_called_once()
        self.assertTrue(self.ctx.result_queue.closed)
        self.assertTrue(self.ctx.result_queue.joined)
        for process in self.ctx.processes:
            if process.started:
                self.assertTrue(process.joined)

    def test_train_runs_with_stub_env_and_wandb(self):
        def child_applied_lr(worker_id, cfg, agent, counter, reports, stop):
            # Simulate the shared scalar changing in a child while the parent's
            # process-local optimizer param group retains its initial LR.
            agent.last_learning_rate.fill_(2e-4)
            _complete_worker(worker_id, cfg, agent, counter, reports, stop)

        result = self._run(worker=child_applied_lr)
        self.assertIsNotNone(self.wandb.init_kwargs)
        self.get_context.assert_called_once_with("spawn")
        self.assertEqual(result["global_step"], 20)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["episodes"], 0)
        self.assertIsNone(result["best_checkpoint"])
        self.assertIsNone(result["best_avg_return"])
        self.assertEqual(len(self.ctx.processes[0].args), 6)
        self.assertIs(self.ctx.processes[0].args[2].update_lock, self.ctx.update_lock)
        metadata = self.save.call_args.kwargs["metadata"]
        self.assertEqual(metadata["progress_semantics"], "completed_transitions")
        self.assertIs(metadata["training_completed"], True)
        self.assertAlmostEqual(metadata["last_applied_learning_rate"], 2e-4, places=8)
        self.assertAlmostEqual(
            self.save.call_args.args[2].param_groups[0]["lr"], 2e-4, places=8
        )
        self.assert_training_cleanup()

    def test_train_uses_env_wandb_key_when_cli_key_missing(self):
        with patch.dict(os.environ, {"WANDB_API_KEY": "env-key"}):
            self._run()
        self.assertEqual(self.wandb.login_keys, ["env-key"])
        self.assertNotIn("env-key", repr(self.wandb.init_kwargs["config"]))

    def test_cli_wandb_key_overrides_env_and_is_redacted(self):
        with patch.dict(os.environ, {"WANDB_API_KEY": "env-key"}):
            self._run(wandb_key="cli-key")
        self.assertEqual(self.wandb.login_keys, ["cli-key"])
        self.assertNotIn("wandb_key", self.wandb.init_kwargs["config"])
        self.assertNotIn("cli-key", repr(self.wandb.init_kwargs["config"]))

    def test_train_multi_env_config_accepted(self):
        result = self._run(_make_config(num_envs=2, num_workers=2, total_steps=6))
        self.assertEqual(self.wandb.init_kwargs["config"]["num_envs"], 2)
        self.assertEqual(result["global_step"], 6)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(len(self.ctx.processes), 2)
        self.assert_training_cleanup()

    def test_native_best_window_requires_full_twenty_episodes(self):
        for episode_count in (19, 20):
            with self.subTest(episode_count=episode_count):

                def episodes_then_done(worker_id, cfg, agent, counter, reports, stop):
                    for index in range(episode_count):
                        reports.put(
                            {
                                "kind": "episode",
                                "worker_id": worker_id,
                                "global_step": index + 1,
                                "episode_length": 1,
                                "episode_return": 100.0,
                                "learning_episode_return": 1.0,
                                "native_episode_return": 100.0,
                            }
                        )
                    _complete_worker(worker_id, cfg, agent, counter, reports, stop)

                result = self._run(
                    _make_config(save_best=True, reward_scale=0.01),
                    worker=episodes_then_done,
                )
                self.assertEqual(result["episodes"], episode_count)
                best_calls = [
                    call
                    for call in self.save.call_args_list
                    if str(call.args[0]).endswith("best.pt")
                ]
                if episode_count < 20:
                    self.assertEqual(best_calls, [])
                    self.assertIsNone(result["best_checkpoint"])
                    self.assertIsNone(result["best_avg_return"])
                else:
                    self.assertEqual(len(best_calls), 1)
                    self.assertEqual(
                        best_calls[0].kwargs["metadata"]["progress_semantics"],
                        "reserved_transitions",
                    )
                    self.assertIs(
                        best_calls[0].kwargs["metadata"]["training_completed"], False
                    )
                    self.assertEqual(result["best_avg_return"], 100.0)
                    self.assertTrue(result["best_checkpoint"].endswith("best.pt"))

    def test_invalid_environment_spaces_close_inspection_env_and_run(self):
        for invalid_observation in (False, True):
            with self.subTest(invalid_observation=invalid_observation):
                env = StubEnv()
                if invalid_observation:
                    env.observation_space = spaces.Box(
                        -1.0, 1.0, shape=(2, 2), dtype=np.float32
                    )
                else:
                    env.action_space = spaces.Box(
                        -1.0, 1.0, shape=(2,), dtype=np.float32
                    )
                with self.assertRaises(ValueError):
                    self._run(env=env)
                self.assertTrue(env.close_called)
                self.assertTrue(self.wandb.run.finished)

    def test_process_start_failure_cleans_already_started_workers_and_queue(self):
        with self.assertRaisesRegex(RuntimeError, "start failed"):
            self._run(_make_config(num_workers=2), ctx=FakeContext(fail_start_worker=1))
        self.assertTrue(self.wandb.run.finished)
        self.assertTrue(self.ctx.processes[0].joined)
        self.assertTrue(self.ctx.result_queue.closed)
        self.assertTrue(self.ctx.result_queue.joined)

    def test_reported_worker_error_and_unreported_exit_are_not_silent_success(self):
        def report_error(worker_id, cfg, agent, counter, reports, stop):
            reports.put(
                {
                    "kind": "worker_error",
                    "worker_id": worker_id,
                    "error": "stub worker failed",
                    "exception_type": "RuntimeError",
                    "traceback": "stub worker failed",
                }
            )

        cases = (
            {"worker": report_error},
            {"ctx": FakeContext(silent_failure_worker=0)},
        )
        for kwargs in cases:
            with self.subTest(case=tuple(kwargs)):
                with self.assertRaises(RuntimeError):
                    self._run(**kwargs)
                self.assert_training_cleanup()

    def test_done_summary_followed_by_nonzero_exit_cannot_report_success(self):
        with self.assertRaises(RuntimeError):
            self._run(ctx=FakeContext(exitcode_after_target=3))
        self.assertEqual(self.ctx.processes[0].exitcode, 3)
        self.save.assert_not_called()
        self.assertFalse(
            any("progress/optimizer_steps" in data for data, _ in self.wandb.logged)
        )
        self.assert_training_cleanup()

    def test_done_summary_followed_by_forced_termination_cannot_report_success(self):
        with self.assertRaises(RuntimeError):
            self._run(ctx=FakeContext(alive_after_target=True))
        self.assertTrue(self.ctx.processes[0].terminated)
        self.save.assert_not_called()
        self.assertFalse(
            any("progress/optimizer_steps" in data for data, _ in self.wandb.logged)
        )
        self.assert_training_cleanup()

    def test_checkpoint_and_logging_errors_cleanup_all_resources(self):
        cases = (
            (
                _make_config(checkpoint_interval=1),
                {"save_error": RuntimeError("stub checkpoint failed")},
            ),
            (
                _make_config(log_interval=1),
                {"log_error": RuntimeError("stub log failed")},
            ),
        )
        for cfg, kwargs in cases:
            with self.subTest(case=tuple(kwargs)):
                with self.assertRaisesRegex(RuntimeError, "stub"):
                    self._run(cfg, **kwargs)
                self.assert_training_cleanup()

    def test_completed_transitions_not_reservations_determine_success(self):
        def incomplete(worker_id, cfg, agent, counter, reports, stop):
            counter.value = cfg.total_steps
            reports.put(
                {
                    "kind": "worker_done",
                    "worker_id": worker_id,
                    "collected_transitions": cfg.total_steps - 1,
                    "optimizer_steps": 0,
                    "global_step": counter.value,
                }
            )

        with self.assertRaises(RuntimeError):
            self._run(worker=incomplete)
        self.assert_training_cleanup()

    def test_demo_runs_with_stub_env_and_closes_on_checkpoint_error(self):
        for load_error in (None, RuntimeError("stub load failed")):
            with self.subTest(load_error=load_error):
                env = StubEnv()
                with (
                    patch(
                        "A3C.main.Config.from_yaml",
                        return_value=_make_config(episodes=2),
                    ),
                    patch("A3C.main.gym.make", return_value=env),
                    patch("A3C.main.load_checkpoint", side_effect=load_error) as load,
                    patch("A3C.main.setup_logger", return_value=MagicMock()),
                ):
                    if load_error is None:
                        a3c_main.demo(
                            config="unused.yaml", model_path="model.pt", episodes=2
                        )
                        self.assertEqual(env.step_calls, 4)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "load failed"):
                            a3c_main.demo(
                                config="unused.yaml", model_path="model.pt", episodes=2
                            )
                load.assert_called_once()
                self.assertTrue(env.close_called)

    def test_demo_never_done_environment_has_a_finite_step_cap(self):
        env = StubEnv(episode_length=10_000)
        with (
            patch(
                "A3C.main.Config.from_yaml",
                return_value=_make_config(max_steps_per_episode=3),
            ),
            patch("A3C.main.gym.make", return_value=env) as make,
            patch("A3C.main.load_checkpoint"),
            patch("A3C.main.setup_logger", return_value=MagicMock()),
        ):
            a3c_main.demo(config="unused.yaml", model_path="model.pt", episodes=2)
        self.assertEqual(env.step_calls, 6)
        self.assertIsNone(make.call_args.kwargs.get("render_mode"))
        self.assertTrue(env.close_called)
