"""Tabular train/demo integration with deterministic stub environments and W&B."""

from contextlib import ExitStack, contextmanager
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from gymnasium import spaces

from Qlearning.ql.agent import QLearningAgent
from Qlearning.ql.config import Config
from Qlearning.ql.utils import load_checkpoint, save_checkpoint
import Qlearning.main as ql_main


class DummyEnv:
    def __init__(
        self, length=2, end="terminated", reward=1.0, rewards=None, fail_at=None
    ):
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self.length = length
        self.end = end
        self.reward = reward
        self.rewards = rewards
        self.fail_at = fail_at
        self.episode = -1
        self.steps = 0
        self.step_calls = 0
        self.close_calls = 0
        self.render_calls = 0
        self.reset_seeds = []
        self.actions = []

    def reset(self, seed=None):
        self.episode += 1
        self.reset_seeds.append(seed)
        self.steps = 0
        return 0, {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise AssertionError("action out of bounds")
        self.actions.append(int(action))
        self.step_calls += 1
        self.steps += 1
        if self.step_calls == self.fail_at:
            raise RuntimeError("stub step failure")
        done = self.steps >= self.length
        reward = self.reward if self.rewards is None else self.rewards[self.episode]
        info = (
            {"episode": {"r": float(reward * self.steps), "l": self.steps}}
            if done
            else {}
        )
        return (
            min(self.steps, 2),
            float(reward),
            done and self.end == "terminated",
            done and self.end == "truncated",
            info,
        )

    def render(self):
        self.render_calls += 1
        return "stub ANSI frame"

    def close(self):
        self.close_calls += 1


class DummyWandb:
    def __init__(self):
        self.login_keys = []
        self.logged = []
        self.init_kwargs = None
        self.run = MagicMock()
        self.run.summary = {}

    def login(self, key=None):
        self.login_keys.append(key)

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def log(self, data, step=None):
        self.logged.append((dict(data), step))


class DummyTqdm:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def tiny_config():
    cfg = Config()
    cfg.env_id = "FrozenLake-v1"  # Factory is patched; no real FrozenLake is created.
    cfg.device = "cpu"
    cfg.total_episodes = 5
    cfg.max_steps_per_episode = 3
    cfg.log_interval = 2
    cfg.checkpoint_interval = 2
    cfg.best_window = 3
    cfg.save_best = False
    cfg.alpha = 0.5
    cfg.gamma = 0.5
    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0
    cfg.log_to_console = False
    cfg.log_to_file = False
    cfg.wandb_key = ""
    return cfg


@contextmanager
def train_stubs(cfg, env, env_key=""):
    wandb, logger = DummyWandb(), MagicMock()
    with ExitStack() as stack:
        if cfg.checkpoint_dir == "Qlearning/checkpoints":
            cfg.checkpoint_dir = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(patch.dict(os.environ, {"WANDB_API_KEY": env_key}))
        stack.enter_context(patch("Qlearning.main.Config.from_yaml", return_value=cfg))
        factory = stack.enter_context(
            patch("Qlearning.main.make_env", return_value=env)
        )
        stack.enter_context(patch("Qlearning.main.setup_logger", return_value=logger))
        stack.enter_context(patch("Qlearning.main.wandb", wandb))
        stack.enter_context(patch("Qlearning.main.tqdm", DummyTqdm))
        yield wandb, factory, logger


class TestTrain(unittest.TestCase):
    def test_tabular_diagnostics_have_exact_reward_td_and_discounted_units(self):
        cfg, env = tiny_config(), DummyEnv()
        cfg.total_episodes = 1
        with train_stubs(cfg, env) as (wandb, _, _):
            result = ql_main.train(config="unused.yaml")
        payload, log_step = wandb.logged[0]
        expected = {
            "charts/avg_return": 2.0,
            "charts/avg_length": 2.0,
            "charts/success_rate": 1.0,
            "charts/positive_return_rate": 1.0,
            "charts/truncation_rate": 0.0,
            "charts/epsilon": 0.0,
            "stats/start_q": 0.0,
            "stats/discounted_return": 1.5,
            "stats/td_error": 1.0,
            "loss/td_mse": 1.0,
        }
        for key, value in expected.items():
            self.assertTrue(np.isfinite(payload[key]))
            self.assertAlmostEqual(payload[key], value)
        self.assertEqual(log_step, 2)
        self.assertEqual(payload["progress/steps"], 2)
        self.assertEqual(result["episodes"], 1)
        self.assertEqual(result["global_step"], 2)
        self.assertIsNone(wandb.run.summary["training/best_return"])

    def test_generic_negative_terminal_return_is_not_labeled_task_failure(self):
        cfg, env = tiny_config(), DummyEnv(length=1, reward=-1.0)
        cfg.env_id = "Generic-v0"
        cfg.total_episodes = 1
        original_step = env.step

        def successful_terminal(action):
            state, reward, terminated, truncated, info = original_step(action)
            info["is_success"] = True
            return state, reward, terminated, truncated, info

        with (
            train_stubs(cfg, env) as (wandb, _, _),
            patch.object(env, "step", side_effect=successful_terminal),
        ):
            ql_main.train(config="unused.yaml")
        payload = wandb.logged[-1][0]
        self.assertEqual(payload["charts/avg_return"], -1.0)
        self.assertEqual(payload["charts/positive_return_rate"], 0.0)
        self.assertNotIn("charts/success_rate", payload)
        self.assertEqual(env.close_calls, 1)

    def test_invalid_observation_and_action_spaces_are_rejected_with_cleanup(self):
        for field, space in (
            ("observation_space", spaces.Discrete(3, start=1)),
            ("action_space", spaces.Discrete(2, start=1)),
            ("observation_space", spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)),
        ):
            with self.subTest(field=field, space=space):
                cfg, env = tiny_config(), DummyEnv()
                setattr(env, field, space)
                with train_stubs(cfg, env) as (wandb, _, _):
                    with self.assertRaisesRegex(ValueError, "zero-based Discrete"):
                        ql_main.train(config="unused.yaml")
                self.assertEqual(env.step_calls, 0)
                self.assertEqual(env.close_calls, 1)
                wandb.run.finish.assert_called_once()

    def test_stub_loop_logs_requested_intervals_and_final_episode(self):
        cfg, env = tiny_config(), DummyEnv()
        with train_stubs(cfg, env) as (wandb, factory, _):
            ql_main.train(config="unused.yaml")
        factory.assert_called_once_with(
            cfg.env_id, cfg.seed, render_mode=None, env_kwargs=cfg.env_kwargs
        )
        self.assertEqual(
            [data["progress/episode"] for data, _ in wandb.logged], [2, 4, 5]
        )
        self.assertEqual(wandb.logged[-1][0]["charts/avg_return"], 2.0)
        self.assertEqual(wandb.logged[-1][0]["charts/avg_length"], 2.0)
        self.assertEqual(env.step_calls, 10)
        self.assertEqual(env.reset_seeds, list(range(cfg.seed, cfg.seed + 5)))
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_final_episode_on_log_boundary_is_not_logged_twice(self):
        cfg, env = tiny_config(), DummyEnv()
        cfg.total_episodes = 4
        with train_stubs(cfg, env) as (wandb, _, _):
            ql_main.train(config="unused.yaml")
        self.assertEqual([data["progress/episode"] for data, _ in wandb.logged], [2, 4])

    def test_wandb_cli_over_environment_over_config_and_secret_redaction(self):
        for cli_key, env_key, config_key, expected in (
            ("cli-test-key", "env-test-key", "config-test-key", "cli-test-key"),
            ("", "env-test-key", "config-test-key", "env-test-key"),
            ("", "", "config-test-key", "config-test-key"),
            ("", "", "", None),
        ):
            with self.subTest(expected=expected):
                cfg, env = tiny_config(), DummyEnv()
                cfg.total_episodes = 1
                cfg.wandb_key = config_key
                with train_stubs(cfg, env, env_key) as (wandb, _, logger):
                    ql_main.train(config="unused.yaml", wandb_key=cli_key)
                self.assertEqual(
                    wandb.login_keys, [] if expected is None else [expected]
                )
                self.assertNotIn("wandb_key", wandb.init_kwargs["config"])
                public = repr((wandb.init_kwargs, wandb.logged, logger.mock_calls))
                for key in (cli_key, env_key, config_key):
                    if key:
                        self.assertNotIn(key, public)

    def test_true_terminal_suppresses_bootstrap_but_timeout_and_cap_do_not(self):
        for end, length, expected_target in (
            ("terminated", 1, 1.0),
            ("truncated", 1, 3.0),
            ("terminated", 999, 3.0),
        ):
            with self.subTest(end=end, length=length):
                cfg, env = tiny_config(), DummyEnv(length=length, end=end)
                cfg.total_episodes = 1
                cfg.max_steps_per_episode = 1
                cfg.alpha = 1.0
                agent = QLearningAgent(3, 2, 1.0, 0.5, 0.0, 0.0, 0.0)
                agent.Q[1] = [4.0, 2.0]
                with (
                    train_stubs(cfg, env) as (wandb, _, _),
                    patch("Qlearning.main.QLearningAgent", return_value=agent),
                    patch.object(agent, "update", wraps=agent.update) as update,
                ):
                    ql_main.train(config="unused.yaml")
                update.assert_called_once()
                transition = update.call_args.args[0]
                self.assertEqual(transition.done, end == "terminated" and length == 1)
                self.assertEqual(agent.Q[0, transition.a], expected_target)
                self.assertEqual(agent.global_step, 1)
                self.assertEqual(env.step_calls, 1)
                self.assertEqual(env.close_calls, 1)
                self.assertEqual(
                    wandb.logged[-1][0]["charts/truncation_rate"],
                    float(not transition.done),
                )

    def test_best_window_and_periodic_final_checkpoints_are_persisted(self):
        cfg, env = tiny_config(), DummyEnv()
        cfg.save_best = True
        with tempfile.TemporaryDirectory() as directory:
            cfg.checkpoint_dir = directory
            with (
                train_stubs(cfg, env),
                patch("Qlearning.main.save_checkpoint", wraps=save_checkpoint) as save,
            ):
                ql_main.train(config="unused.yaml")
            best_calls = [
                call
                for call in save.call_args_list
                if os.path.basename(call.args[0]) == "best.pt"
            ]
            self.assertTrue(best_calls)
            self.assertEqual(best_calls[0].args[2], cfg.best_window)
            self.assertTrue(all(call.args[2] >= cfg.best_window for call in best_calls))
            periodic = [
                os.path.basename(call.args[0])
                for call in save.call_args_list
                if os.path.basename(call.args[0]).startswith("checkpoint_ep")
            ]
            self.assertEqual(
                periodic,
                ["checkpoint_ep2.pt", "checkpoint_ep4.pt", "checkpoint_ep5.pt"],
            )
            data = load_checkpoint(os.path.join(directory, "best.pt"))
            self.assertEqual(data["step"], 3)
            self.assertEqual(data["global_step"], 6)
            self.assertEqual(data["best_return"], 2.0)
            self.assertTrue(np.isfinite(data["q_table"].numpy()).all())
            self.assertNotIn("wandb_key", data["config"])
            for filename in periodic:
                self.assertTrue(os.path.isfile(os.path.join(directory, filename)))

    def test_no_best_before_complete_best_window_but_final_checkpoint_exists(self):
        cfg, env = tiny_config(), DummyEnv()
        cfg.total_episodes = 2
        cfg.best_window = 3
        cfg.checkpoint_interval = 100
        cfg.save_best = True
        with tempfile.TemporaryDirectory() as directory:
            cfg.checkpoint_dir = directory
            with train_stubs(cfg, env):
                ql_main.train(config="unused.yaml")
            self.assertFalse(os.path.exists(os.path.join(directory, "best.pt")))
            self.assertTrue(
                os.path.isfile(os.path.join(directory, "checkpoint_ep2.pt"))
            )

    def test_step_failure_closes_environment_and_finishes_run(self):
        cfg, env = tiny_config(), DummyEnv(fail_at=1)
        with train_stubs(cfg, env) as (wandb, _, _):
            with self.assertRaisesRegex(RuntimeError, "stub step failure"):
                ql_main.train(config="unused.yaml")
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_update_failure_closes_environment_and_finishes_run(self):
        cfg, env = tiny_config(), DummyEnv()
        with (
            train_stubs(cfg, env) as (wandb, _, _),
            patch.object(
                QLearningAgent,
                "update",
                side_effect=RuntimeError("stub update failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stub update failure"):
                ql_main.train(config="unused.yaml")
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_factory_failure_finishes_initialized_run(self):
        cfg, env = tiny_config(), DummyEnv()
        with train_stubs(cfg, env) as (wandb, factory, _):
            factory.side_effect = RuntimeError("stub factory failure")
            with self.assertRaisesRegex(RuntimeError, "stub factory failure"):
                ql_main.train(config="unused.yaml")
        wandb.run.finish.assert_called_once()

    def test_close_failure_still_finishes_run(self):
        cfg, env = tiny_config(), DummyEnv()
        with (
            train_stubs(cfg, env) as (wandb, _, _),
            patch.object(
                env, "close", side_effect=RuntimeError("stub close failure")
            ) as close,
        ):
            with self.assertRaisesRegex(RuntimeError, "stub close failure"):
                ql_main.train(config="unused.yaml")
        close.assert_called_once()
        wandb.run.finish.assert_called_once()


class TestDemo(unittest.TestCase):
    @contextmanager
    def _stubs(self, cfg, env):
        with (
            patch("Qlearning.main.Config.from_yaml", return_value=cfg),
            patch("Qlearning.main.make_env", return_value=env) as factory,
            patch("Qlearning.main.setup_logger", return_value=MagicMock()),
        ):
            yield factory

    def _checkpoint(self, directory):
        path = os.path.join(directory, "best.pt")
        table = np.array([[0.0, 3.0], [0.0, 3.0], [0.0, 3.0]], dtype=np.float32)
        save_checkpoint(path, table, 9, 1.0, global_step=18)
        return path

    def test_generic_negative_return_demo_does_not_claim_failure_rate(self):
        cfg, env = tiny_config(), DummyEnv(length=1, reward=-1.0)
        cfg.env_id = "Generic-v0"
        original_step = env.step

        def successful_terminal(action):
            state, reward, terminated, truncated, info = original_step(action)
            info["is_success"] = True
            return state, reward, terminated, truncated, info

        with tempfile.TemporaryDirectory() as directory:
            path = self._checkpoint(directory)
            with (
                self._stubs(cfg, env),
                patch.object(env, "step", side_effect=successful_terminal),
            ):
                result = ql_main.demo(config="unused.yaml", model_path=path, episodes=1)
        self.assertEqual(result["returns"], [-1.0])
        self.assertEqual(result["positive_return_rate"], 0.0)
        self.assertIsNone(result["success_rate"])
        self.assertEqual(env.close_calls, 1)

    def test_zero_demo_episodes_is_rejected_before_environment_creation(self):
        cfg, env = tiny_config(), DummyEnv()
        with self._stubs(cfg, env) as factory:
            with self.assertRaisesRegex(ValueError, "episodes must be positive"):
                ql_main.demo(config="unused.yaml", episodes=0)
        factory.assert_not_called()

    def test_invalid_demo_action_space_is_rejected_before_checkpoint_load(self):
        cfg, env = tiny_config(), DummyEnv()
        env.action_space = spaces.Discrete(2, start=1)
        with self._stubs(cfg, env), patch("Qlearning.main.load_checkpoint") as load:
            with self.assertRaisesRegex(ValueError, "zero-based Discrete"):
                ql_main.demo(config="unused.yaml", model_path="unused.pt")
        load.assert_not_called()
        self.assertEqual(env.close_calls, 1)

    def test_default_demo_loads_best_headlessly_with_deterministic_actions(self):
        cfg, env = tiny_config(), DummyEnv(length=1, rewards=[1.0, 0.0, 1.0])
        cfg.episodes = 3
        with tempfile.TemporaryDirectory() as directory:
            cfg.inference_model_path = self._checkpoint(directory)
            with (
                self._stubs(cfg, env) as factory,
                patch("Qlearning.main.load_checkpoint", wraps=load_checkpoint) as load,
            ):
                result = ql_main.demo(config="unused.yaml")
        load.assert_called_once_with(cfg.inference_model_path)
        self.assertIsNone(factory.call_args.kwargs["render_mode"])
        self.assertEqual(env.actions, [1, 1, 1])
        self.assertEqual(env.render_calls, 0)
        self.assertEqual(env.reset_seeds, [cfg.seed, cfg.seed + 1, cfg.seed + 2])
        self.assertEqual(result["returns"], [1.0, 0.0, 1.0])
        self.assertEqual(result["lengths"], [1, 1, 1])
        self.assertAlmostEqual(result["mean"], 2.0 / 3.0)
        self.assertAlmostEqual(result["std"], np.std([1.0, 0.0, 1.0]))
        self.assertAlmostEqual(result["success_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(result["positive_return_rate"], 2.0 / 3.0)
        self.assertEqual(env.close_calls, 1)

    def test_demo_calls_deterministic_agent_without_mutating_saved_counter(self):
        cfg, env = tiny_config(), DummyEnv(length=1)
        agent = QLearningAgent(3, 2, 0.5, 0.5, 1.0, 1.0, 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = self._checkpoint(directory)
            with (
                self._stubs(cfg, env),
                patch("Qlearning.main.QLearningAgent", return_value=agent),
                patch.object(agent, "act", wraps=agent.act) as act,
            ):
                ql_main.demo(config="unused.yaml", model_path=path, episodes=2)
        self.assertEqual(act.call_count, 2)
        self.assertTrue(
            all(call.kwargs["deterministic"] for call in act.call_args_list)
        )
        self.assertEqual(agent.global_step, 18)
        self.assertEqual(env.actions, [1, 1])

    def test_ansi_render_override_prints_frames_and_seed_episode_overrides_apply(self):
        cfg, env = tiny_config(), DummyEnv(length=1)
        with tempfile.TemporaryDirectory() as directory:
            path = self._checkpoint(directory)
            with self._stubs(cfg, env) as factory, patch("builtins.print") as printer:
                result = ql_main.demo(
                    config="unused.yaml",
                    model_path=path,
                    episodes=2,
                    seed=91,
                    render_mode="ansi",
                )
        self.assertEqual(factory.call_args.args[:2], (cfg.env_id, 91))
        self.assertEqual(factory.call_args.kwargs["render_mode"], "ansi")
        self.assertEqual(env.reset_seeds, [91, 92])
        self.assertGreaterEqual(env.render_calls, 2)
        self.assertTrue(
            any(call.args == ("stub ANSI frame",) for call in printer.call_args_list)
        )
        self.assertEqual(result["returns"], [1.0, 1.0])
        self.assertEqual(env.close_calls, 1)

    def test_render_boolean_preserves_human_render_compatibility(self):
        cfg, env = tiny_config(), DummyEnv(length=1)
        with tempfile.TemporaryDirectory() as directory:
            path = self._checkpoint(directory)
            with self._stubs(cfg, env) as factory:
                ql_main.demo(
                    config="unused.yaml", model_path=path, episodes=1, render=True
                )
        self.assertEqual(factory.call_args.kwargs["render_mode"], "human")
        self.assertGreaterEqual(env.render_calls, 1)
        self.assertEqual(env.close_calls, 1)

    def test_none_override_disables_configured_rendering(self):
        cfg, env = tiny_config(), DummyEnv(length=1)
        cfg.render_mode = "human"
        with tempfile.TemporaryDirectory() as directory:
            path = self._checkpoint(directory)
            with self._stubs(cfg, env) as factory:
                ql_main.demo(
                    config="unused.yaml",
                    model_path=path,
                    episodes=1,
                    render_mode="none",
                )
        self.assertIsNone(factory.call_args.kwargs["render_mode"])
        self.assertEqual(env.render_calls, 0)
        self.assertEqual(env.close_calls, 1)

    def test_checkpoint_failure_closes_demo_environment(self):
        cfg, env = tiny_config(), DummyEnv()
        with (
            self._stubs(cfg, env),
            patch(
                "Qlearning.main.load_checkpoint",
                side_effect=ValueError("bad checkpoint"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "bad checkpoint"):
                ql_main.demo(config="unused.yaml", model_path="unused.pt")
        self.assertEqual(env.close_calls, 1)

    def test_demo_step_failure_closes_environment(self):
        cfg, env = tiny_config(), DummyEnv(fail_at=1)
        with tempfile.TemporaryDirectory() as directory:
            path = self._checkpoint(directory)
            with self._stubs(cfg, env):
                with self.assertRaisesRegex(RuntimeError, "stub step failure"):
                    ql_main.demo(config="unused.yaml", model_path=path)
        self.assertEqual(env.close_calls, 1)

    def test_malformed_table_shape_closes_demo_environment(self):
        cfg, env = tiny_config(), DummyEnv()
        with (
            self._stubs(cfg, env),
            patch(
                "Qlearning.main.load_checkpoint",
                return_value={
                    "q_table": np.zeros((2, 2)),
                    "global_step": 0,
                },
            ),
        ):
            with self.assertRaises((ValueError, TypeError)):
                ql_main.demo(config="unused.yaml", model_path="unused.pt")
        self.assertEqual(env.close_calls, 1)

    def test_demo_cap_limits_interaction_without_requiring_terminal_flag(self):
        cfg, env = tiny_config(), DummyEnv(length=999)
        cfg.max_steps_per_episode = 2
        with tempfile.TemporaryDirectory() as directory:
            path = self._checkpoint(directory)
            with self._stubs(cfg, env):
                result = ql_main.demo(config="unused.yaml", model_path=path, episodes=1)
        self.assertEqual(result["lengths"], [2])
        self.assertEqual(result["returns"], [2.0])
        self.assertEqual(env.step_calls, 2)
        self.assertEqual(env.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
