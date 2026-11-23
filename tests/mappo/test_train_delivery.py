"""Vector delivery-trainer regression tests using fake workers and CPU models."""

import copy
import os
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from MAPPO.mappo.agent import MAPPOAgent
from MAPPO.mappo.config import Config
import MAPPO.train_delivery as trainer


def config(directory, total=6):
    cfg = Config()
    cfg.n_walkers = cfg.n_agents = 3
    cfg.num_envs = 2
    cfg.total_timesteps = total
    cfg.rollout_steps = 2
    cfg.minibatch_size = 4
    cfg.update_iterations = 1
    cfg.actor_hidden_sizes = cfg.critic_hidden_sizes = [4]
    cfg.device = "cpu"
    cfg.gamma = 0.5
    cfg.target_kl = 0.0
    cfg.anneal_lr = False
    cfg.normalize_inputs = False
    cfg.max_steps_per_episode = 3
    cfg.env_kwargs = {"max_cycles": 3}
    cfg.log_interval = 1
    cfg.checkpoint_interval = 100
    cfg.checkpoint_dir = directory
    return cfg


def learner(cfg):
    return MAPPOAgent(
        3,
        2,
        2,
        6,
        [4],
        [4],
        "tanh",
        0.001,
        0.2,
        0.01,
        0.5,
        0.5,
        "cpu",
        share_policy=True,
        action_type="continuous",
    )


class FakeWorkers:
    def __init__(self):
        self.obs_array = np.zeros((2, 3, 2), dtype=np.float32)
        self.step_calls = 0
        self.close = MagicMock()

    def step(self, actions):
        self.step_calls += 1
        if np.asarray(actions).shape != (2, 3, 2):
            raise AssertionError("actions must preserve E,N,A layout")
        sequence = self.step_calls
        timeout_endpoint = np.full((3, 2), 6.0 + sequence, dtype=np.float32)
        first = {
            "obs": np.zeros((3, 2), dtype=np.float32),
            "next_obs": timeout_endpoint,
            "rewards": np.full(3, -1.0, dtype=np.float32),
            "alive": np.ones(3),
            "terminations": np.zeros(3),
            "dones": np.ones(3),
            "episode": {
                "team_return": -1.0,
                "length": 1,
                "delivered": False,
                "progress_fraction": 0.0,
            },
        }
        if sequence % 2:
            endpoint = np.array([[0, 0], [2, 2], [2, 2]], dtype=np.float32)
            second = {
                "obs": endpoint.copy(),
                "next_obs": endpoint,
                "rewards": np.array([-1, 2, 3], dtype=np.float32),
                "alive": np.ones(3),
                "terminations": np.array([1, 0, 0]),
                "dones": np.array([1, 0, 0]),
                "episode": None,
            }
        else:
            endpoint = np.zeros((3, 2), dtype=np.float32)
            second = {
                "obs": endpoint.copy(),
                "next_obs": endpoint,
                "rewards": np.array([0, 2, 3], dtype=np.float32),
                "alive": np.array([0, 1, 1]),
                "terminations": np.ones(3),
                "dones": np.ones(3),
                "episode": {
                    "team_return": 3.0,
                    "length": 2,
                    "delivered": False,
                    "progress_fraction": 0.1,
                },
            }
        return [first, second]


class TestDeliveryTrainer(unittest.TestCase):
    def test_fake_workers_exact_budget_endpoint_gae_and_fresh_optimizer_warmstart(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            agent, workers, env = learner(cfg), FakeWorkers(), MagicMock()
            run = MagicMock()
            run.summary = {}
            original_gae = MAPPOAgent.compute_gae
            batches_seen = []
            original_update = agent.update

            def capture(batches):
                batches_seen.append(batches)
                return original_update(batches)

            selection = {"delivery_rate": 0.0, "mean_progress": 0.1, "mean_return": 1.0}
            with (
                patch.dict(os.environ, {"WANDB_API_KEY": ""}),
                patch.object(trainer.Config, "from_yaml", return_value=cfg),
                patch.object(trainer, "_make_env", return_value=env),
                patch.object(trainer, "_build_agent", return_value=agent),
                patch.object(
                    trainer, "ParallelMultiwalker", return_value=workers
                ) as factory,
                patch.object(trainer, "_logger", return_value=MagicMock()),
                patch.object(trainer.wandb, "init", return_value=run),
                patch.object(trainer.wandb, "log") as log,
                patch.object(trainer, "load_checkpoint") as load,
                patch.object(trainer, "save_checkpoint"),
                patch.object(
                    trainer,
                    "evaluate_candidate",
                    return_value=(selection, [], (0.0, 0.1, 1.0)),
                ),
                patch.object(
                    agent,
                    "values_many",
                    side_effect=lambda obs: np.asarray(obs)[:, :, 0],
                ),
                patch.object(agent, "update", side_effect=capture),
                patch.object(MAPPOAgent, "compute_gae", wraps=original_gae) as gae,
                patch(
                    "numpy.random.permutation",
                    side_effect=lambda count: np.arange(count),
                ),
            ):
                result = trainer.train(config="unused.yaml", init_checkpoint="old.pt")
            factory.assert_called_once_with(cfg, 2)
            load.assert_called_once_with("old.pt", agent.models, None)
            self.assertEqual(result["global_step"], 6)
            self.assertEqual(workers.step_calls, 3)
            self.assertEqual(gae.call_count, 2)
            first = gae.call_args_list[0]
            self.assertEqual(first.args[0].shape, (2, 6))
            np.testing.assert_array_equal(
                first.kwargs["next_value_sequence"],
                [[7, 7, 7, 0, 2, 2], [8, 8, 8, 0, 0, 0]],
            )
            np.testing.assert_array_equal(
                first.kwargs["terminations"], [[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 1, 1]]
            )
            np.testing.assert_array_equal(
                batches_seen[0][0].alive_mask.numpy(), [1, 1, 1, 0]
            )
            self.assertAlmostEqual(float(batches_seen[0][0].returns[0]), 2.5)
            self.assertAlmostEqual(float(batches_seen[0][0].returns[2]), 3.0)
            self.assertTrue(agent.optimizers[0].state)
            training_logs = [
                call.args[0]
                for call in log.call_args_list
                if "progress/global_step" in call.args[0]
            ]
            self.assertEqual(
                [data["progress/global_step"] for data in training_logs], [4, 6]
            )
            self.assertEqual(training_logs[-1]["progress/optimizer_steps"], 2)
            self.assertTrue(
                all(
                    np.isfinite(data["stats/return_target_mean"])
                    for data in training_logs
                )
            )
            env.close.assert_called_once()
            workers.close.assert_called_once()
            run.finish.assert_called_once()

    def test_worker_error_closes_pool_and_finishes_run(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            env, workers, run = MagicMock(), FakeWorkers(), MagicMock()
            run.summary = {}
            workers.step = MagicMock(side_effect=EOFError("worker EOF"))
            with (
                patch.dict(os.environ, {"WANDB_API_KEY": ""}),
                patch.object(trainer.Config, "from_yaml", return_value=cfg),
                patch.object(trainer, "_make_env", return_value=env),
                patch.object(trainer, "_build_agent", return_value=learner(cfg)),
                patch.object(trainer, "ParallelMultiwalker", return_value=workers),
                patch.object(trainer, "_logger", return_value=MagicMock()),
                patch.object(trainer.wandb, "init", return_value=run),
            ):
                with self.assertRaises(EOFError):
                    trainer.train(config="unused.yaml")
            workers.close.assert_called_once()
            env.close.assert_called_once()
            run.finish.assert_called_once()

    def test_auth_precedence_is_stubbed_and_public_training_config_omits_key(self):
        for cli, env_key, config_key, expected in (
            ("cli-test", "env-test", "cfg-test", "cli-test"),
            ("", "env-test", "cfg-test", "env-test"),
            ("", "", "cfg-test", "cfg-test"),
        ):
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                cfg = config(directory)
                cfg.wandb_key = config_key
                env, workers, run = MagicMock(), FakeWorkers(), MagicMock()
                workers.step = MagicMock(side_effect=EOFError("fake worker EOF"))
                with (
                    patch.dict(os.environ, {"WANDB_API_KEY": env_key}),
                    patch.object(trainer.Config, "from_yaml", return_value=cfg),
                    patch.object(trainer, "_make_env", return_value=env),
                    patch.object(trainer, "_build_agent", return_value=learner(cfg)),
                    patch.object(trainer, "ParallelMultiwalker", return_value=workers),
                    patch.object(trainer, "_logger", return_value=MagicMock()),
                    patch.object(trainer.wandb, "login") as login,
                    patch.object(trainer.wandb, "init", return_value=run) as init,
                ):
                    with self.assertRaises(EOFError):
                        trainer.train(config="unused.yaml", wandb_key=cli)
                login.assert_called_once_with(key=expected)
                self.assertNotIn("wandb_key", init.call_args.kwargs["config"])
                for secret in (cli, env_key, config_key):
                    if secret:
                        self.assertNotIn(secret, repr(init.call_args.kwargs["config"]))
                workers.close.assert_called_once()
                run.finish.assert_called_once()

    def test_all_rejected_updates_stop_remaining_epochs_recollect_and_log_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory)
            cfg.update_iterations = 3
            cfg.minibatch_size = 2
            agent, workers, env, run = (
                learner(cfg),
                FakeWorkers(),
                MagicMock(),
                MagicMock(),
            )
            run.summary = {}
            rejected = {
                "loss/policy": 0.0,
                "loss/value": 0.0,
                "loss/total": 0.0,
                "stats/entropy": 0.0,
                "stats/approx_kl": 0.0,
                "stats/skipped_updates": 1.0,
            }
            selection = {"delivery_rate": 0.0, "mean_progress": 0.1, "mean_return": 1.0}
            with (
                patch.dict(os.environ, {"WANDB_API_KEY": ""}),
                patch.object(trainer.Config, "from_yaml", return_value=cfg),
                patch.object(trainer, "_make_env", return_value=env),
                patch.object(trainer, "_build_agent", return_value=agent),
                patch.object(trainer, "ParallelMultiwalker", return_value=workers),
                patch.object(trainer, "_logger", return_value=MagicMock()),
                patch.object(trainer.wandb, "init", return_value=run),
                patch.object(trainer.wandb, "log") as log,
                patch.object(trainer, "save_checkpoint"),
                patch.object(
                    trainer,
                    "evaluate_candidate",
                    return_value=(selection, [], (0.0, 0.1, 1.0)),
                ),
                patch.object(agent, "update", return_value=(rejected, 0)) as update,
                patch.object(
                    agent, "update_normalizers", wraps=agent.update_normalizers
                ) as normalizers,
            ):
                result = trainer.train(config="unused.yaml")
            self.assertEqual(result["global_step"], 6)
            self.assertEqual(workers.step_calls, 3)
            self.assertEqual(
                update.call_count, 2
            )  # First rejection ends each rollout's PPO epochs.
            normalizers.assert_called_once()  # Fresh second rollout is collected, not stale batches retried.
            self.assertFalse(agent.optimizers[0].state)
            training_logs = [
                call.args[0]
                for call in log.call_args_list
                if "progress/global_step" in call.args[0]
            ]
            self.assertEqual(
                [data["progress/global_step"] for data in training_logs], [4, 6]
            )
            for data in training_logs:
                self.assertEqual(data["stats/skipped_updates"], 1.0)
                self.assertEqual(data["progress/optimizer_steps"], 0)
            workers.close.assert_called_once()
            run.finish.assert_called_once()

    def test_metric_window_retains_present_losses_and_sums_all_rejections(self):
        mixed = trainer.aggregate_metrics(
            [
                {"loss/value": 4.0, "stats/skipped_updates": 0.0},
                {"stats/skipped_updates": 1.0},
            ]
        )
        self.assertEqual(mixed, {"loss/value": 4.0, "stats/skipped_updates": 1.0})
        rejected = trainer.aggregate_metrics(
            [
                {"stats/skipped_updates": 2.0},
                {"stats/skipped_updates": 3.0},
            ]
        )
        self.assertEqual(rejected, {"stats/skipped_updates": 5.0})
        self.assertNotIn("loss/value", rejected)

    def test_nondivisible_vector_budget_is_rejected_before_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(directory, total=5)
            with (
                patch.object(trainer.Config, "from_yaml", return_value=cfg),
                patch.object(trainer.wandb, "init") as init,
                patch.object(trainer, "ParallelMultiwalker") as workers,
            ):
                with self.assertRaisesRegex(ValueError, "divisible"):
                    trainer.train(config="unused.yaml")
            init.assert_not_called()
            workers.assert_not_called()


class TestCandidateSelection(unittest.TestCase):
    def rng_state(self):
        return (
            random.getstate(),
            np.random.get_state(),
            torch.random.get_rng_state().clone(),
        )

    def assert_rng_equal(self, expected):
        self.assertEqual(random.getstate(), expected[0])
        actual = np.random.get_state()
        self.assertEqual(actual[0], expected[1][0])
        np.testing.assert_array_equal(actual[1], expected[1][1])
        self.assertEqual(actual[2:], expected[1][2:])
        torch.testing.assert_close(
            torch.random.get_rng_state(), expected[2], rtol=0, atol=0
        )

    def consume_rng(self):
        random.random()
        np.random.rand()
        torch.rand(1)

    def fixture(self):
        cfg = Config()
        cfg.max_steps_per_episode = 5000
        cfg.env_kwargs = {"max_cycles": 5000}
        source = SimpleNamespace(models=[torch.nn.Linear(2, 2)])
        candidate = SimpleNamespace(models=[torch.nn.Linear(2, 2)])
        return cfg, source, candidate

    def test_actual_delivery_outranks_stationary_full_horizon_despite_lower_return_and_restores_rng(
        self,
    ):
        cfg, source, candidate = self.fixture()
        scores = []
        for delivered, progress, reward in ((False, 0.0, 1000.0), (True, 1.01, -200.0)):
            env = MagicMock()
            record = {
                "delivered": delivered,
                "progress_fraction": progress,
                "package_displacement": progress * 79.333,
                "team_return": reward,
                "length": 5000,
                "survived_horizon": True,
                "frames": [],
            }

            def rollout(*args, **kwargs):
                self.consume_rng()
                return copy.deepcopy(record)

            before = self.rng_state()
            with (
                patch.object(trainer, "_make_env", return_value=env),
                patch.object(trainer, "_build_agent", return_value=candidate),
                patch.object(trainer, "rollout_episode", side_effect=rollout),
                patch.object(torch.cuda, "is_available", return_value=False),
            ):
                metrics, records, score = trainer.evaluate_candidate(
                    source, cfg, 180000, 2
                )
            self.assert_rng_equal(before)
            self.assertEqual(len(records), 2)
            self.assertEqual(metrics["delivery_rate"], float(delivered))
            env.close.assert_called_once()
            scores.append(score)
        self.assertGreater(scores[1], scores[0])

    def test_factory_rollout_and_close_errors_always_restore_training_rng(self):
        for failure in ("factory", "rollout", "close"):
            with self.subTest(failure=failure):
                cfg, source, candidate = self.fixture()
                env = MagicMock()
                record = {
                    "delivered": False,
                    "progress_fraction": 0.0,
                    "package_displacement": 0.0,
                    "team_return": 0.0,
                    "length": 1,
                    "frames": [],
                }

                def factory(*args, **kwargs):
                    self.consume_rng()
                    if failure == "factory":
                        raise RuntimeError("factory failure")
                    return env

                def rollout(*args, **kwargs):
                    self.consume_rng()
                    if failure == "rollout":
                        raise RuntimeError("rollout failure")
                    return copy.deepcopy(record)

                if failure == "close":
                    env.close.side_effect = RuntimeError("close failure")
                before = self.rng_state()
                with (
                    patch.object(trainer, "_make_env", side_effect=factory),
                    patch.object(trainer, "_build_agent", return_value=candidate),
                    patch.object(trainer, "rollout_episode", side_effect=rollout),
                    patch.object(torch.cuda, "is_available", return_value=False),
                ):
                    with self.assertRaisesRegex(RuntimeError, "failure"):
                        trainer.evaluate_candidate(source, cfg, 180000, 1)
                self.assert_rng_equal(before)
                if failure != "factory":
                    env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
