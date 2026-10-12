"""PPO entrypoint tests with stub environments and offline-only W&B doubles."""

from contextlib import ExitStack, contextmanager
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from gymnasium import spaces

from PPO.ppo.agent import PPOAgent
from PPO.ppo.config import Config
from PPO.ppo.networks import ActorCritic
from PPO.ppo.utils import load_checkpoint, save_checkpoint
import PPO.main as ppo_main


class DummyVecEnv:
    """Same-step autoreset stub, including final-observation and episode info."""

    def __init__(
        self,
        num_envs=1,
        episode_length=2,
        timeout=False,
        fail_at=None,
        final_key="final_obs",
    ):
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(
            -100.0, 100.0, (4,), dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(3)
        self.episode_length = episode_length
        self.timeout = timeout
        self.fail_at = fail_at
        self.final_key = final_key
        self.counts = np.zeros(num_envs, dtype=np.int32)
        self.episode_returns = np.zeros(num_envs, dtype=np.float32)
        self.actions = []
        self.reset_seeds = []
        self.close_calls = 0
        self.step_calls = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.counts[:] = 0
        self.episode_returns[:] = 0
        return np.zeros((self.num_envs, 4), dtype=np.float32), {}

    def step(self, actions):
        self.step_calls += 1
        if self.step_calls == self.fail_at:
            raise RuntimeError("stub step failure")
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,):
            raise AssertionError("vector action shape mismatch")
        if not ((actions >= 0) & (actions < self.single_action_space.n)).all():
            raise AssertionError("action outside discrete bounds")
        self.actions.append(actions.copy())
        self.counts += 1
        rewards = np.arange(1, self.num_envs + 1, dtype=np.float32)
        self.episode_returns += rewards
        done = self.counts >= self.episode_length
        terminated = done & (not self.timeout)
        truncated = done & self.timeout
        next_obs = np.repeat(self.counts[:, None], 4, axis=1).astype(np.float32)
        info = {}
        if done.any():
            final_obs = np.empty(self.num_envs, dtype=object)
            for index in range(self.num_envs):
                final_obs[index] = (
                    np.full(4, 7.0, dtype=np.float32) if done[index] else None
                )
            info = {
                self.final_key: final_obs,
                "_" + self.final_key: done.copy(),
                "episode": {"r": self.episode_returns.copy(), "l": self.counts.copy()},
                "_episode": done.copy(),
            }
            self.counts[done] = 0
            self.episode_returns[done] = 0
            next_obs[done] = 0.0
        return next_obs, rewards, terminated, truncated, info

    def close(self):
        self.close_calls += 1


class DummyWandb:
    class Run:
        def __init__(self):
            self.finish_calls = 0
            self.summary = {}

        def finish(self):
            self.finish_calls += 1

    def __init__(self):
        self.logged = []
        self.login_keys = []
        self.init_kwargs = None
        self.run = self.Run()

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
    cfg.env_id = "Stub-v0"
    cfg.device = "cpu"
    cfg.num_envs = 1
    cfg.hidden_sizes = [4]
    cfg.total_timesteps = 4
    cfg.rollout_steps = 2
    cfg.minibatch_size = 2
    cfg.update_iterations = 1
    cfg.log_interval = 1
    cfg.checkpoint_interval = 100
    cfg.save_best = False
    cfg.log_to_console = False
    cfg.log_to_file = False
    cfg.wandb_key = ""
    return cfg


@contextmanager
def train_stubs(cfg, env, env_key=""):
    wandb = DummyWandb()
    logger = MagicMock()
    with ExitStack() as stack:
        if cfg.checkpoint_dir == "PPO/checkpoints":
            # Final periodic saves must never touch a developer's real checkpoints.
            cfg.checkpoint_dir = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(patch.dict(os.environ, {"WANDB_API_KEY": env_key}))
        stack.enter_context(patch("PPO.main.Config.from_yaml", return_value=cfg))
        factory = stack.enter_context(patch("PPO.main.make_vec_env", return_value=env))
        stack.enter_context(patch("PPO.main.wandb", wandb))
        stack.enter_context(patch("PPO.main.setup_logger", return_value=logger))
        stack.enter_context(patch("PPO.main.tqdm", DummyTqdm))
        yield wandb, factory, logger


class TestTrain(unittest.TestCase):
    def test_train_rejects_nonzero_action_start_and_cleans_up(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv()
        env.single_action_space = spaces.Discrete(3, start=5)
        with (
            train_stubs(cfg, env) as (wandb, _, _),
            patch("PPO.main._build_agent") as build,
        ):
            with self.assertRaisesRegex(ValueError, "zero-based discrete actions"):
                ppo_main.train(config="unused.yaml")
        build.assert_not_called()
        self.assertEqual(env.step_calls, 0)
        self.assertEqual(env.close_calls, 1)
        self.assertEqual(wandb.run.finish_calls, 1)

    def test_train_collects_episode_metrics_and_finishes_resources(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv()
        with train_stubs(cfg, env) as (wandb, factory, _):
            ppo_main.train(config="unused.yaml")
        factory.assert_called_once_with(
            cfg.env_id, cfg.num_envs, cfg.seed, render_mode=None
        )
        self.assertEqual(env.step_calls, 4)
        self.assertEqual(env.close_calls, 1)
        self.assertEqual(wandb.run.finish_calls, 1)
        last, _ = wandb.logged[-1]
        self.assertEqual(last["charts/avg_return"], 2.0)
        self.assertEqual(last["charts/avg_length"], 2.0)
        self.assertEqual(last["progress/global_step"], 4)
        for name in (
            "loss/policy",
            "loss/value",
            "loss/total",
            "stats/value_mean",
            "stats/return_target_mean",
            "stats/explained_variance",
        ):
            self.assertTrue(np.isfinite(last[name]))
        self.assertEqual(wandb.run.summary["environment_steps"], 4)
        self.assertIsNone(wandb.run.summary["best_avg_return"])

    def test_wandb_cli_then_environment_then_config_precedence(self) -> None:
        cases = [
            ("cli-test-key", "env-test-key", "config-test-key", "cli-test-key"),
            ("", "env-test-key", "config-test-key", "env-test-key"),
            ("", "", "config-test-key", "config-test-key"),
            ("", "", "", None),
        ]
        for cli_key, env_key, config_key, expected in cases:
            with self.subTest(cli_key=cli_key, env_key=env_key, config_key=config_key):
                cfg, env = tiny_config(), DummyVecEnv()
                cfg.wandb_key = config_key
                with train_stubs(cfg, env, env_key) as (wandb, _, logger):
                    ppo_main.train(config="unused.yaml", wandb_key=cli_key)
                self.assertEqual(
                    wandb.login_keys, [] if expected is None else [expected]
                )
                self.assertNotIn("wandb_key", wandb.init_kwargs["config"])
                public_output = repr(
                    (wandb.init_kwargs, wandb.logged, logger.mock_calls)
                )
                for key in (cli_key, env_key, config_key):
                    if key:
                        self.assertNotIn(key, public_output)
                self.assertEqual(env.close_calls, 1)
                self.assertEqual(wandb.run.finish_calls, 1)

    def test_log_interval_is_updates_and_last_update_is_logged(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv()
        cfg.total_timesteps = 10  # five updates
        cfg.log_interval = 2
        with train_stubs(cfg, env) as (wandb, _, _):
            ppo_main.train(config="unused.yaml")
        self.assertEqual(
            [data["progress/global_step"] for data, _ in wandb.logged], [4, 8, 10]
        )

    def test_final_update_on_log_boundary_is_not_logged_twice(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv()
        cfg.total_timesteps = 8
        cfg.log_interval = 2
        with train_stubs(cfg, env) as (wandb, _, _):
            ppo_main.train(config="unused.yaml")
        self.assertEqual(
            [data["progress/global_step"] for data, _ in wandb.logged], [4, 8]
        )

    def test_best_and_periodic_checkpoints_are_real_and_loadable(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv(episode_length=1)
        cfg.total_timesteps = 12
        cfg.checkpoint_interval = 2
        cfg.save_best = True
        with tempfile.TemporaryDirectory() as directory:
            cfg.checkpoint_dir = directory
            with (
                train_stubs(cfg, env),
                patch("PPO.main.save_checkpoint", wraps=save_checkpoint) as save,
            ):
                ppo_main.train(config="unused.yaml")
            best_calls = [
                call
                for call in save.call_args_list
                if os.path.basename(call.args[0]) == "best.pt"
            ]
            self.assertTrue(best_calls)
            self.assertEqual(best_calls[0].args[3], 10)
            self.assertTrue(all(call.args[3] >= 10 for call in best_calls))
            periodic_steps = [
                call.args[3]
                for call in save.call_args_list
                if os.path.basename(call.args[0]).startswith("checkpoint_")
            ]
            self.assertEqual(periodic_steps, [4, 8, 12])
            restored = ActorCritic(4, 3, cfg.hidden_sizes, cfg.activation)
            optimizer = torch.optim.Adam(restored.parameters(), lr=cfg.learning_rate)
            data = load_checkpoint(
                os.path.join(directory, "best.pt"), restored, optimizer
            )
            self.assertEqual(data["step"], 10)
            self.assertEqual(data["best_return"], 1.0)
            self.assertTrue(optimizer.state_dict()["state"])
            for parameter in restored.parameters():
                self.assertTrue(torch.isfinite(parameter).all().item())
            for step in periodic_steps:
                self.assertTrue(
                    os.path.isfile(os.path.join(directory, f"checkpoint_{step}.pt"))
                )

    def test_no_best_checkpoint_before_ten_episodes_or_without_episodes(self) -> None:
        for episode_length in (1, 100):
            with (
                self.subTest(episode_length=episode_length),
                tempfile.TemporaryDirectory() as directory,
            ):
                cfg, env = tiny_config(), DummyVecEnv(episode_length=episode_length)
                cfg.total_timesteps = 8
                cfg.save_best = True
                cfg.checkpoint_dir = directory
                with (
                    train_stubs(cfg, env),
                    patch("PPO.main.save_checkpoint", wraps=save_checkpoint) as save,
                ):
                    ppo_main.train(config="unused.yaml")
                self.assertEqual(
                    [os.path.basename(call.args[0]) for call in save.call_args_list],
                    ["checkpoint_8.pt"],
                )
                self.assertFalse(os.path.exists(os.path.join(directory, "best.pt")))

    def test_final_short_rollout_uses_exact_timestep_budget(self) -> None:
        for num_envs, total_timesteps, expected_steps, logged_steps in (
            (1, 7, 7, [2, 4, 6, 7]),
            (2, 10, 5, [4, 8, 10]),
        ):
            with self.subTest(num_envs=num_envs, total_timesteps=total_timesteps):
                cfg, env = tiny_config(), DummyVecEnv(num_envs=num_envs)
                cfg.num_envs = num_envs
                cfg.total_timesteps = total_timesteps
                with (
                    train_stubs(cfg, env) as (wandb, _, _),
                    patch("PPO.main.save_checkpoint", wraps=save_checkpoint) as save,
                ):
                    result = ppo_main.train(config="unused.yaml")
                self.assertEqual(env.step_calls, expected_steps)
                self.assertEqual(result["global_step"], total_timesteps)
                self.assertEqual(
                    [data["progress/global_step"] for data, _ in wandb.logged],
                    logged_steps,
                )
                self.assertEqual([step for _, step in wandb.logged], logged_steps)
                self.assertEqual(
                    wandb.run.summary["environment_steps"], total_timesteps
                )
                self.assertEqual(save.call_args.args[3], total_timesteps)
                self.assertEqual(env.close_calls, 1)
                self.assertEqual(wandb.run.finish_calls, 1)
                for data, _ in wandb.logged:
                    self.assertTrue(np.isfinite(data["loss/total"]))

    def test_step_failure_closes_environment_and_finishes_run(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv(fail_at=1)
        with train_stubs(cfg, env) as (wandb, _, _):
            with self.assertRaisesRegex(RuntimeError, "stub step failure"):
                ppo_main.train(config="unused.yaml")
        self.assertEqual(env.close_calls, 1)
        self.assertEqual(wandb.run.finish_calls, 1)

    def test_update_failure_closes_environment_and_finishes_run(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv()
        with (
            train_stubs(cfg, env) as (wandb, _, _),
            patch.object(
                PPOAgent, "update", side_effect=RuntimeError("stub update failure")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stub update failure"):
                ppo_main.train(config="unused.yaml")
        self.assertEqual(env.close_calls, 1)
        self.assertEqual(wandb.run.finish_calls, 1)

    def test_environment_factory_failure_still_finishes_initialized_run(self) -> None:
        cfg = tiny_config()
        with train_stubs(cfg, DummyVecEnv()) as (wandb, factory, _):
            factory.side_effect = RuntimeError("stub factory failure")
            with self.assertRaisesRegex(RuntimeError, "stub factory failure"):
                ppo_main.train(config="unused.yaml")
        self.assertEqual(wandb.run.finish_calls, 1)

    def test_close_failure_still_finishes_wandb_run(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv()
        with (
            train_stubs(cfg, env) as (wandb, _, _),
            patch.object(
                env, "close", side_effect=RuntimeError("stub close failure")
            ) as close,
        ):
            with self.assertRaisesRegex(RuntimeError, "stub close failure"):
                ppo_main.train(config="unused.yaml")
        close.assert_called_once()
        self.assertEqual(wandb.run.finish_calls, 1)

    def test_missing_timeout_final_observation_fails_and_cleans_up(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv(episode_length=1, timeout=True)
        original_step = env.step

        def missing_final(actions):
            obs, rewards, terminated, truncated, info = original_step(actions)
            info.pop("final_obs")
            return obs, rewards, terminated, truncated, info

        with (
            train_stubs(cfg, env) as (wandb, _, _),
            patch.object(env, "step", side_effect=missing_final),
        ):
            with self.assertRaisesRegex(RuntimeError, "final observations"):
                ppo_main.train(config="unused.yaml")
        self.assertEqual(env.close_calls, 1)
        self.assertEqual(wandb.run.finish_calls, 1)

    def test_bootstrap_observation_mask_only_replaces_timeouts(self) -> None:
        next_obs = np.array([[1.0] * 4, [2.0] * 4, [3.0] * 4], dtype=np.float32)
        original = next_obs.copy()
        final_obs = np.empty(3, dtype=object)
        final_obs[:] = [None, np.full(4, 7.0, dtype=np.float32), None]
        actual = ppo_main._bootstrap_observations(
            next_obs,
            np.array([False, False, True]),
            np.array([False, True, True]),
            {"final_obs": final_obs, "_final_obs": np.array([False, True, False])},
        )
        np.testing.assert_array_equal(actual, [[1.0] * 4, [7.0] * 4, [3.0] * 4])
        np.testing.assert_array_equal(next_obs, original)

    def test_timeout_bootstrap_uses_final_observation_not_autoreset_observation(
        self,
    ) -> None:
        def observation_value(model, obs):
            logits = torch.zeros((obs.shape[0], 3), device=obs.device)
            return logits, obs[:, 0]

        for final_key in ("final_obs", "final_observation"):
            with self.subTest(final_key=final_key):
                cfg = tiny_config()
                env = DummyVecEnv(episode_length=1, timeout=True, final_key=final_key)
                cfg.total_timesteps = 2
                original_gae = PPOAgent.compute_gae
                with (
                    train_stubs(cfg, env),
                    patch.object(ActorCritic, "forward", observation_value),
                    patch.object(PPOAgent, "update", return_value={"loss/total": 0.0}),
                    patch.object(PPOAgent, "compute_gae", wraps=original_gae) as gae,
                ):
                    ppo_main.train(config="unused.yaml")
                self.assertEqual(gae.call_count, 1)
                call = gae.call_args
                np.testing.assert_array_equal(call.args[1], np.ones((2, 1)))
                np.testing.assert_array_equal(
                    call.kwargs["terminations"], np.zeros((2, 1))
                )
                np.testing.assert_array_equal(
                    call.kwargs["next_values"], np.full((2, 1), 7.0)
                )
                advantages, returns = original_gae(*call.args, **call.kwargs)
                np.testing.assert_allclose(
                    returns, np.full((2, 1), 1.0 + cfg.gamma * 7.0)
                )
                self.assertTrue(np.isfinite(advantages).all())


class TestDemo(unittest.TestCase):
    @contextmanager
    def _demo_stubs(self, cfg, env):
        logger = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(patch("PPO.main.Config.from_yaml", return_value=cfg))
            factory = stack.enter_context(
                patch("PPO.main.make_vec_env", return_value=env)
            )
            stack.enter_context(patch("PPO.main.setup_logger", return_value=logger))
            yield factory

    def _save_demo_model(self, directory, cfg):
        model = ActorCritic(4, 3, cfg.hidden_sizes, cfg.activation)
        with torch.no_grad():
            model.pi.weight.zero_()
            model.pi.bias.copy_(torch.tensor([-1.0, 3.0, 0.0]))
        path = os.path.join(directory, "best.pt")
        save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 10, 2.0)
        return path

    def test_demo_rejects_nonzero_action_start_and_cleans_up(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv()
        env.single_action_space = spaces.Discrete(3, start=5)
        with self._demo_stubs(cfg, env), patch("PPO.main.load_checkpoint") as load:
            with self.assertRaisesRegex(ValueError, "zero-based discrete actions"):
                ppo_main.demo(
                    config="unused.yaml", model_path="unused.pt", render_mode="none"
                )
        load.assert_not_called()
        self.assertEqual(env.step_calls, 0)
        self.assertEqual(env.close_calls, 1)

    def test_demo_loads_best_and_defaults_to_deterministic_seeded_rollouts(
        self,
    ) -> None:
        cfg, env = tiny_config(), DummyVecEnv()
        cfg.episodes = 3
        with tempfile.TemporaryDirectory() as directory:
            cfg.inference_model_path = self._save_demo_model(directory, cfg)
            with (
                self._demo_stubs(cfg, env),
                patch("torch.distributions.Categorical.sample") as sample,
                patch("PPO.main.load_checkpoint", wraps=load_checkpoint) as load,
            ):
                result = ppo_main.demo(config="unused.yaml", render_mode="none")
        sample.assert_not_called()
        load.assert_called_once()
        self.assertEqual(load.call_args.args[0], cfg.inference_model_path)
        self.assertIsInstance(load.call_args.args[1], ActorCritic)
        self.assertEqual(result["returns"], [2.0, 2.0, 2.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["std"], 0.0)
        for actions in env.actions:
            np.testing.assert_array_equal(actions, [1])
        # Factories may perform their own reset; this stub only records demo resets.
        self.assertEqual(env.reset_seeds, [cfg.seed, cfg.seed + 1, cfg.seed + 2])
        self.assertEqual(env.close_calls, 1)

    def test_demo_overrides_seed_episodes_and_render_mode(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv()
        with tempfile.TemporaryDirectory() as directory:
            path = self._save_demo_model(directory, cfg)
            with self._demo_stubs(cfg, env) as factory:
                result = ppo_main.demo(
                    config="unused.yaml",
                    model_path=path,
                    episodes=2,
                    seed=91,
                    render_mode="rgb_array",
                )
        factory.assert_called_once_with(cfg.env_id, 1, 91, render_mode="rgb_array")
        self.assertEqual(env.reset_seeds, [91, 92])
        self.assertEqual(result["returns"], [2.0, 2.0])
        self.assertEqual(env.close_calls, 1)

    def test_demo_reports_mean_and_population_std_of_all_returns(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv(episode_length=1)
        original_step = env.step

        def varying_reward(actions):
            obs, reward, terminated, truncated, info = original_step(actions)
            return obs, reward * len(env.reset_seeds), terminated, truncated, info

        with tempfile.TemporaryDirectory() as directory:
            path = self._save_demo_model(directory, cfg)
            with (
                self._demo_stubs(cfg, env),
                patch.object(env, "step", side_effect=varying_reward),
            ):
                result = ppo_main.demo(
                    config="unused.yaml",
                    model_path=path,
                    episodes=3,
                    render_mode="none",
                )
        self.assertEqual(result["returns"], [1.0, 2.0, 3.0])
        self.assertEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["std"], np.sqrt(2.0 / 3.0))
        self.assertEqual(env.close_calls, 1)

    def test_demo_rejects_zero_episodes_without_creating_environment(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv()
        with self._demo_stubs(cfg, env) as factory:
            with self.assertRaisesRegex(ValueError, "episodes must be positive"):
                ppo_main.demo(config="unused.yaml", episodes=0, render_mode="none")
        factory.assert_not_called()
        self.assertEqual(env.close_calls, 0)

    def test_demo_explicit_stochastic_mode_samples(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv(episode_length=1)
        with tempfile.TemporaryDirectory() as directory:
            path = self._save_demo_model(directory, cfg)
            with (
                self._demo_stubs(cfg, env),
                patch(
                    "torch.distributions.Categorical.sample",
                    return_value=torch.tensor([2]),
                ) as sample,
            ):
                result = ppo_main.demo(
                    config="unused.yaml",
                    model_path=path,
                    episodes=2,
                    deterministic=False,
                    render_mode="none",
                )
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(result["returns"], [1.0, 1.0])
        for actions in env.actions:
            np.testing.assert_array_equal(actions, [2])
        self.assertEqual(env.close_calls, 1)

    def test_demo_none_render_override_disables_configured_human_rendering(
        self,
    ) -> None:
        cfg, env = tiny_config(), DummyVecEnv(episode_length=1)
        cfg.render_mode = "human"
        with tempfile.TemporaryDirectory() as directory:
            path = self._save_demo_model(directory, cfg)
            with self._demo_stubs(cfg, env) as factory:
                ppo_main.demo(
                    config="unused.yaml",
                    model_path=path,
                    episodes=1,
                    render_mode="none",
                )
        self.assertIsNone(factory.call_args.kwargs["render_mode"])
        self.assertEqual(env.close_calls, 1)

    def test_demo_checkpoint_load_failure_closes_environment(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv()
        with (
            self._demo_stubs(cfg, env),
            patch(
                "PPO.main.load_checkpoint",
                side_effect=RuntimeError("stub load failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stub load failure"):
                ppo_main.demo(
                    config="unused.yaml", model_path="missing.pt", render_mode="none"
                )
        self.assertEqual(env.close_calls, 1)

    def test_demo_rollout_failure_closes_environment(self) -> None:
        cfg, env = tiny_config(), DummyVecEnv(fail_at=1)
        with tempfile.TemporaryDirectory() as directory:
            path = self._save_demo_model(directory, cfg)
            with self._demo_stubs(cfg, env):
                with self.assertRaisesRegex(RuntimeError, "stub step failure"):
                    ppo_main.demo(
                        config="unused.yaml", model_path=path, render_mode="none"
                    )
        self.assertEqual(env.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
