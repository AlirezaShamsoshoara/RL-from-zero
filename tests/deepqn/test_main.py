"""DQN loop regressions with native rewards, mock environments, and local checkpoints."""

from contextlib import ExitStack, contextmanager
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from gymnasium import spaces

from deepQN.dqn.agent import DQNAgent
from deepQN.dqn.config import Config
from deepQN.dqn.replay_buffer import ReplayBuffer
from deepQN.dqn.utils import load_checkpoint, save_checkpoint
import deepQN.main as main


class DummyEnv:
    def __init__(self, length=2, end="terminated", final_key=None, fail_at=None):
        self.observation_space = spaces.Box(-10.0, 10.0, (2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.length, self.end, self.final_key, self.fail_at = (
            length,
            end,
            final_key,
            fail_at,
        )
        self.steps = self.step_calls = self.close_calls = 0
        self.reset_seeds, self.actions = [], []
        self.unwrapped = self
        self.goal_position, self.goal_velocity = 0.5, 0.0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise AssertionError("action out of bounds")
        self.actions.append(action)
        self.steps += 1
        self.step_calls += 1
        if self.step_calls == self.fail_at:
            raise RuntimeError("stub step failure")
        ended = self.steps >= self.length
        terminated = ended and self.end == "terminated"
        truncated = ended and self.end == "truncated"
        obs = np.array(
            [0.6 if terminated else self.steps / 10.0, 0.01], dtype=np.float32
        )
        info = {"episode": {"r": -float(self.steps), "l": self.steps}} if ended else {}
        if ended and self.final_key:
            info[self.final_key] = np.array([7.0, 0.02], dtype=np.float32)
            obs = np.zeros(2, dtype=np.float32)  # Simulate an autoreset observation.
        return obs, -1.0, terminated, truncated, info

    def close(self):
        self.close_calls += 1


class DummyWandb:
    def __init__(self):
        self.login_keys, self.logged = [], []
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


def config():
    cfg = Config()
    cfg.device = "cpu"
    cfg.hidden_sizes = [4]
    cfg.total_steps = 5
    cfg.learning_starts = 0
    cfg.train_freq = 1
    cfg.batch_size = 2
    cfg.buffer_size = 20
    cfg.gamma = 0.5
    cfg.target_update_interval = 3
    cfg.log_interval = 2
    cfg.checkpoint_interval = 2
    cfg.best_window = 3
    cfg.max_steps_per_episode = 10
    cfg.save_best = False
    cfg.epsilon_start = cfg.epsilon_end = 0.0
    cfg.log_to_console = cfg.log_to_file = False
    cfg.wandb_key = ""
    return cfg


def build_agent(cfg, env):
    return DQNAgent(
        env.observation_space,
        env.action_space,
        cfg.hidden_sizes,
        cfg.activation,
        cfg.lr,
        cfg.gamma,
        cfg.target_update_interval,
        cfg.max_grad_norm,
        cfg.double_dqn,
        cfg.device,
    )


@contextmanager
def train_stubs(cfg, env, env_key=""):
    wandb, logger = DummyWandb(), MagicMock()
    with ExitStack() as stack:
        if cfg.checkpoint_dir == "deepQN/checkpoints":
            cfg.checkpoint_dir = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(patch.dict(os.environ, {"WANDB_API_KEY": env_key}))
        stack.enter_context(patch("deepQN.main.Config.from_yaml", return_value=cfg))
        factory = stack.enter_context(patch("deepQN.main.make_env", return_value=env))
        stack.enter_context(patch("deepQN.main.wandb", wandb))
        stack.enter_context(patch("deepQN.main.setup_logger", return_value=logger))
        stack.enter_context(patch("deepQN.main.tqdm", DummyTqdm))
        yield wandb, factory, logger


class TestTrain(unittest.TestCase):
    def test_epsilon_by_step_bounds(self):
        cfg = config()
        cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_decay_steps = 1.0, 0.1, 10
        for step, expected in ((0, 1.0), (5, 0.55), (10, 0.1), (20, 0.1)):
            self.assertAlmostEqual(main._epsilon_by_step(step, cfg), expected)
        cfg.epsilon_decay_steps = 0
        self.assertAlmostEqual(main._epsilon_by_step(5, cfg), 0.1)

    def test_loop_logs_requested_steps_and_final_native_rewards(self):
        cfg, env = config(), DummyEnv()
        with train_stubs(cfg, env) as (wandb, factory, _):
            main.train(config="unused.yaml")
        factory.assert_called_once_with(
            cfg.env_id, cfg.seed, render_mode=None, env_kwargs=cfg.env_kwargs
        )
        self.assertEqual([data["progress/step"] for data, _ in wandb.logged], [2, 4, 5])
        self.assertEqual([step for _, step in wandb.logged], [2, 4, 5])
        self.assertEqual(wandb.logged[-1][0]["charts/avg_return"], -2.0)
        self.assertEqual(wandb.logged[-1][0]["charts/avg_length"], 2.0)
        for key in (
            "loss/q_loss",
            "stats/td_error",
            "stats/q_value",
            "stats/target_value",
            "stats/start_q",
            "stats/discounted_return",
        ):
            self.assertTrue(np.isfinite(wandb.logged[-1][0][key]))
        self.assertEqual(wandb.logged[-1][0]["stats/discounted_return"], -1.5)
        self.assertEqual(env.step_calls, 5)
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_generic_termination_is_logged_without_inventing_success(self):
        cfg, env = config(), DummyEnv(length=1)
        cfg.env_id = "Generic-v0"
        cfg.total_steps = 1
        with train_stubs(cfg, env) as (wandb, _, _):
            main.train(config="unused.yaml")
        payload = wandb.logged[-1][0]
        self.assertEqual(payload["charts/avg_return"], -1.0)
        self.assertEqual(payload["charts/termination_rate"], 1.0)
        self.assertNotIn("charts/success_rate", payload)

    def test_auth_precedence_and_secret_redaction_use_module_wandb(self):
        for cli, env_key, cfg_key, expected in (
            ("cli-key", "env-key", "cfg-key", "cli-key"),
            ("", "env-key", "cfg-key", "env-key"),
            ("", "", "cfg-key", "cfg-key"),
            ("", "", "", None),
        ):
            with self.subTest(expected=expected):
                cfg, env = config(), DummyEnv()
                cfg.total_steps = 1
                cfg.wandb_key = cfg_key
                with train_stubs(cfg, env, env_key) as (wandb, _, logger):
                    main.train(config="unused.yaml", wandb_key=cli)
                self.assertEqual(
                    wandb.login_keys, [] if expected is None else [expected]
                )
                self.assertNotIn("wandb_key", wandb.init_kwargs["config"])
                public = repr((wandb.init_kwargs, wandb.logged, logger.mock_calls))
                for key in (cli, env_key, cfg_key):
                    if key:
                        self.assertNotIn(key, public)

    def test_terminal_timeout_and_cap_use_only_true_termination_done_masks(self):
        for end, length in (("terminated", 1), ("truncated", 1), ("terminated", 999)):
            with self.subTest(end=end, length=length):
                cfg, env = config(), DummyEnv(length=length, end=end)
                cfg.total_steps = cfg.max_steps_per_episode = 1
                buffer = ReplayBuffer(2, 20, torch.device("cpu"))
                with (
                    train_stubs(cfg, env),
                    patch("deepQN.main.ReplayBuffer", return_value=buffer),
                ):
                    main.train(config="unused.yaml")
                self.assertEqual(
                    buffer.dones[0], float(end == "terminated" and length == 1)
                )
                self.assertEqual(buffer.rewards[0], -1.0)
                np.testing.assert_allclose(
                    buffer.next_obs[0],
                    [0.6 if length == 1 and end == "terminated" else 0.1, 0.01],
                )
                self.assertGreaterEqual(len(env.reset_seeds), 2)
                self.assertEqual(env.close_calls, 1)

    def test_final_observation_aliases_replace_autoreset_obs_for_ended_steps(self):
        for key in ("final_obs", "final_observation"):
            for end in ("terminated", "truncated"):
                with self.subTest(key=key, end=end):
                    cfg, env = config(), DummyEnv(length=1, end=end, final_key=key)
                    cfg.total_steps = 1
                    buffer = ReplayBuffer(2, 20, torch.device("cpu"))
                    with (
                        train_stubs(cfg, env),
                        patch("deepQN.main.ReplayBuffer", return_value=buffer),
                    ):
                        main.train(config="unused.yaml")
                    np.testing.assert_allclose(buffer.next_obs[0], [7.0, 0.02])
                    self.assertEqual(buffer.dones[0], float(end == "terminated"))
                    self.assertEqual(buffer.rewards[0], -1.0)

    def test_warmup_uses_agent_act_and_counts_every_environment_step(self):
        cfg, env = config(), DummyEnv()
        cfg.learning_starts = 3
        agent = build_agent(cfg, env)
        with (
            train_stubs(cfg, env),
            patch("deepQN.main.DQNAgent", return_value=agent),
            patch.object(agent, "act", wraps=agent.act) as act,
        ):
            main.train(config="unused.yaml")
        self.assertEqual(agent.step_count, cfg.total_steps)
        self.assertEqual(act.call_count, cfg.total_steps)
        for call in act.call_args_list[:3]:
            self.assertEqual(call.kwargs["epsilon"], 1.0)
        self.assertEqual(env.close_calls, 1)

    def test_best_gate_and_periodic_final_checkpoints_are_real(self):
        cfg, env = config(), DummyEnv(length=1)
        cfg.total_steps = 5
        cfg.save_best = True
        with tempfile.TemporaryDirectory() as directory:
            cfg.checkpoint_dir = directory
            with (
                train_stubs(cfg, env),
                patch("deepQN.main.save_checkpoint", wraps=save_checkpoint) as save,
            ):
                main.train(config="unused.yaml")
            best = [
                call
                for call in save.call_args_list
                if os.path.basename(call.args[0]) == "best.pt"
            ]
            self.assertTrue(best)
            self.assertEqual(best[0].args[2], 3)
            periodic = [
                os.path.basename(call.args[0])
                for call in save.call_args_list
                if os.path.basename(call.args[0]).startswith("checkpoint_")
            ]
            self.assertEqual(
                periodic, ["checkpoint_2.pt", "checkpoint_4.pt", "checkpoint_5.pt"]
            )
            restored = build_agent(cfg, env)
            data = load_checkpoint(os.path.join(directory, "best.pt"), restored)
            self.assertEqual(data["step"], 3)
            self.assertEqual(data["best_return"], -1.0)
            self.assertEqual(restored.step_count, 3)
            self.assertTrue(restored.optimizer.state_dict()["state"])
            self.assertNotIn("wandb_key", data["config"])

    def test_no_best_before_complete_window_or_any_episodes(self):
        for length in (1, 999):
            with (
                self.subTest(length=length),
                tempfile.TemporaryDirectory() as directory,
            ):
                cfg, env = config(), DummyEnv(length=length)
                cfg.total_steps = 2
                cfg.best_window = 3
                cfg.save_best = True
                cfg.checkpoint_dir = directory
                with train_stubs(cfg, env):
                    main.train(config="unused.yaml")
                self.assertFalse(os.path.exists(os.path.join(directory, "best.pt")))
                self.assertTrue(
                    os.path.isfile(os.path.join(directory, "checkpoint_2.pt"))
                )

    def test_step_update_factory_and_close_failures_finish_run(self):
        for failure in ("step", "update", "factory", "close"):
            with self.subTest(failure=failure):
                cfg, env = config(), DummyEnv(fail_at=1 if failure == "step" else None)
                with train_stubs(cfg, env) as (wandb, factory, _), ExitStack() as stack:
                    if failure == "factory":
                        factory.side_effect = RuntimeError("stub failure")
                    elif failure == "update":
                        stack.enter_context(
                            patch.object(
                                DQNAgent,
                                "update",
                                side_effect=RuntimeError("stub failure"),
                            )
                        )
                    elif failure == "close":
                        stack.enter_context(
                            patch.object(
                                env, "close", side_effect=RuntimeError("stub failure")
                            )
                        )
                    with self.assertRaises(RuntimeError):
                        main.train(config="unused.yaml")
                wandb.run.finish.assert_called_once()
                if failure in ("step", "update"):
                    self.assertEqual(env.close_calls, 1)


class TestDemo(unittest.TestCase):
    @contextmanager
    def stubs(self, cfg, env):
        with (
            patch("deepQN.main.Config.from_yaml", return_value=cfg),
            patch("deepQN.main.make_env", return_value=env) as factory,
            patch("deepQN.main.setup_logger", return_value=MagicMock()),
        ):
            yield factory

    def checkpoint(self, directory, cfg, env):
        agent = build_agent(cfg, env)
        with torch.no_grad():
            final = agent.q_network.model[-1]
            final.weight.zero_()
            final.bias.copy_(torch.tensor([0.0, 3.0]))
        agent.step_count = 17
        path = os.path.join(directory, "best.pt")
        save_checkpoint(path, agent, 17, -2.0)
        return path

    def test_demo_defaults_headless_deterministic_and_goal_success_ignores_negative_return(
        self,
    ):
        cfg, env = config(), DummyEnv()
        cfg.episodes = 2
        with tempfile.TemporaryDirectory() as directory:
            cfg.inference_model_path = self.checkpoint(directory, cfg, env)
            with (
                self.stubs(cfg, env) as factory,
                patch("deepQN.main.load_checkpoint", wraps=load_checkpoint) as load,
            ):
                result = main.demo(config="unused.yaml")
        load.assert_called_once()
        self.assertEqual(load.call_args.args[0], cfg.inference_model_path)
        self.assertIsNone(factory.call_args.kwargs["render_mode"])
        self.assertEqual(env.actions, [1, 1, 1, 1])
        self.assertEqual(result["returns"], [-2.0, -2.0])
        self.assertEqual(result["lengths"], [2, 2])
        self.assertEqual((result["mean"], result["std"]), (-2.0, 0.0))
        self.assertEqual(result["termination_rate"], 1.0)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertEqual(env.close_calls, 1)

    def test_generic_termination_has_unknown_success_not_reward_sign_inference(self):
        cfg, env = config(), DummyEnv(length=1)
        cfg.env_id = "Generic-v0"
        with tempfile.TemporaryDirectory() as directory:
            path = self.checkpoint(directory, cfg, env)
            with self.stubs(cfg, env):
                result = main.demo(config="unused.yaml", model_path=path, episodes=1)
        self.assertEqual(result["returns"], [-1.0])
        self.assertEqual(result["termination_rate"], 1.0)
        self.assertIsNone(result["success_rate"])
        self.assertEqual(env.close_calls, 1)

    def test_timeout_and_cap_stop_demo_without_claiming_goal_success(self):
        for length, end in ((1, "truncated"), (999, "terminated")):
            with self.subTest(length=length):
                cfg, env = config(), DummyEnv(length=length, end=end)
                cfg.max_steps_per_episode = 1
                with tempfile.TemporaryDirectory() as directory:
                    path = self.checkpoint(directory, cfg, env)
                    with self.stubs(cfg, env):
                        result = main.demo(
                            config="unused.yaml", model_path=path, episodes=1
                        )
                self.assertEqual(result["termination_rate"], 0.0)
                self.assertEqual(result["success_rate"], 0.0)
                self.assertEqual(result["lengths"], [1])

    def test_demo_seed_render_overrides_and_stochastic_path(self):
        cfg, env = config(), DummyEnv(length=1)
        with tempfile.TemporaryDirectory() as directory:
            path = self.checkpoint(directory, cfg, env)
            with (
                self.stubs(cfg, env) as factory,
                patch.object(env.action_space, "sample", return_value=0),
            ):
                cfg.eval_epsilon = 1.0
                result = main.demo(
                    config="unused.yaml",
                    model_path=path,
                    episodes=2,
                    deterministic=False,
                    render_mode="rgb_array",
                    seed=91,
                )
        self.assertEqual(factory.call_args.args[:2], (cfg.env_id, 91))
        self.assertEqual(factory.call_args.kwargs["render_mode"], "rgb_array")
        self.assertEqual(env.reset_seeds, [91, 92])
        self.assertEqual(env.actions, [0, 0])
        self.assertEqual(result["returns"], [-1.0, -1.0])

    def test_none_override_and_load_or_step_failure_cleanup(self):
        for failure in ("load", "step"):
            with self.subTest(failure=failure):
                cfg, env = config(), DummyEnv(fail_at=1 if failure == "step" else None)
                cfg.render_mode = "human"
                with tempfile.TemporaryDirectory() as directory:
                    path = self.checkpoint(directory, cfg, env)
                    with self.stubs(cfg, env) as factory, ExitStack() as stack:
                        if failure == "load":
                            stack.enter_context(
                                patch(
                                    "deepQN.main.load_checkpoint",
                                    side_effect=RuntimeError("bad checkpoint"),
                                )
                            )
                        with self.assertRaises(RuntimeError):
                            main.demo(
                                config="unused.yaml",
                                model_path=path,
                                render_mode="none",
                            )
                self.assertIsNone(factory.call_args.kwargs["render_mode"])
                self.assertEqual(env.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
