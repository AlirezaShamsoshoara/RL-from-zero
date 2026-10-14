"""TD3 main loop tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/td3/test_main.py
    >>> # Run with unittest
    >>> # python -m unittest tests.td3.test_main
"""

import logging
from contextlib import ExitStack
import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
from gymnasium import spaces

from TD3.td3.agent import TD3Stats
from TD3.td3.config import Config
import TD3.main as td3_main


class DummyEnv:
    """Single-environment stub used by demo() tests."""

    def __init__(self, obs_dim: int, act_dim: int, episode_length: int = 2):
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(act_dim,), dtype=np.float32
        )
        self.episode_length = episode_length
        self._step = 0
        self.close_called = False

    def reset(self, seed=None):
        self._step = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = np.full(self.observation_space.shape, float(self._step), dtype=np.float32)
        reward = 1.0
        terminated = self._step >= self.episode_length
        truncated = False
        info = {}
        if terminated or truncated:
            info["episode"] = {"r": float(self._step), "l": self._step}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.close_called = True


class DummyVecEnv:
    """Vectorized-environment stub mimicking SyncVectorEnv with SAME_STEP autoreset."""

    def __init__(
        self, obs_dim: int, act_dim: int, num_envs: int = 1, episode_length: int = 2
    ):
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(act_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_envs, obs_dim), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(num_envs, act_dim), dtype=np.float32
        )
        self.num_envs = num_envs
        self.episode_length = episode_length
        self._steps = np.zeros(num_envs, dtype=np.int32)
        self.close_called = False

    def reset(self, seed=None):
        self._steps[:] = 0
        obs = np.zeros(
            (self.num_envs,) + self.single_observation_space.shape, dtype=np.float32
        )
        return obs, {}

    def step(self, actions):
        self._steps += 1
        obs_dim = self.single_observation_space.shape[0]
        next_obs = np.zeros((self.num_envs, obs_dim), dtype=np.float32)
        rewards = np.ones(self.num_envs, dtype=np.float32)
        terminated = self._steps >= self.episode_length
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: dict = {}

        dones = terminated | truncated
        if dones.any():
            # SAME_STEP semantics: next_obs for done envs is the reset obs,
            # terminal obs goes into final_obs.
            final_obs_list = []
            final_info_r = np.zeros(self.num_envs, dtype=np.float32)
            final_info_l = np.zeros(self.num_envs, dtype=np.int32)
            final_info_ep_mask = np.zeros(self.num_envs, dtype=bool)

            for i in range(self.num_envs):
                terminal_obs = np.full(obs_dim, float(self._steps[i]), dtype=np.float32)
                if dones[i]:
                    final_obs_list.append(terminal_obs)
                    final_info_r[i] = float(self._steps[i])
                    final_info_l[i] = int(self._steps[i])
                    final_info_ep_mask[i] = True
                    # Reset this sub-env; next_obs is the reset obs
                    self._steps[i] = 0
                    next_obs[i] = np.zeros(obs_dim, dtype=np.float32)
                else:
                    final_obs_list.append(np.zeros(obs_dim, dtype=np.float32))
                    next_obs[i] = terminal_obs

            infos["final_obs"] = final_obs_list
            infos["_final_obs"] = dones.copy()
            infos["final_info"] = {
                "episode": {
                    "r": final_info_r,
                    "_r": final_info_ep_mask.copy(),
                    "l": final_info_l,
                    "_l": final_info_ep_mask.copy(),
                },
                "_episode": final_info_ep_mask.copy(),
            }
            infos["_final_info"] = dones.copy()
        else:
            for i in range(self.num_envs):
                next_obs[i] = np.full(obs_dim, float(self._steps[i]), dtype=np.float32)

        return next_obs, rewards, terminated, truncated, infos

    def close(self):
        self.close_called = True


class DummyWandb:
    class _Run:
        def __init__(self):
            self.finished = False

        def finish(self):
            self.finished = True

    def __init__(self):
        self.logged = []
        self.init_kwargs = None
        self.login_keys = []
        self.run = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.run = DummyWandb._Run()
        return self.run

    def log(self, data, step=None):
        self.logged.append((data, step))

    def login(self, key=None):
        self.login_keys.append(key)
        return None


class DummyTqdm:
    def __init__(self, total=None, desc=None, **kwargs):
        self.total = total
        self.desc = desc
        self.n = 0

    def update(self, n=1):
        self.n += n

    def set_postfix(self, *args, **kwargs):
        return None

    def close(self):
        return None


def _null_logger() -> logging.Logger:
    logger = logging.getLogger("td3-test")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


class TestMain(unittest.TestCase):
    def setUp(self):
        saving = patch("TD3.main.save_checkpoint")
        saving.start()
        self.addCleanup(saving.stop)

    def test_stats_to_dict_drops_none_actor_loss(self) -> None:
        stats = TD3Stats(critic_loss=1.0, actor_loss=None, q1_value=0.1, q2_value=0.2)
        data = td3_main._stats_to_dict(stats)

        self.assertIn("critic_loss", data)
        self.assertIn("q1_value", data)
        self.assertIn("q2_value", data)
        self.assertNotIn("actor_loss", data)

    def test_train_runs_with_stub_env_and_wandb(self) -> None:
        cfg = Config()
        cfg.total_steps = 3
        cfg.start_steps = 0
        cfg.batch_size = 2
        cfg.buffer_size = 10
        cfg.updates_per_step = 1
        cfg.num_envs = 1
        cfg.hidden_sizes = [8]
        cfg.activation = "relu"
        cfg.device = "cpu"
        cfg.checkpoint_interval = 100
        cfg.save_best = False
        cfg.log_interval = 1
        cfg.log_to_console = False
        cfg.log_to_file = False
        cfg.wandb_key = ""

        dummy_env = DummyVecEnv(obs_dim=3, act_dim=2, num_envs=1, episode_length=2)
        dummy_wandb = DummyWandb()

        with (
            patch("TD3.main.Config.from_yaml", return_value=cfg),
            patch("TD3.main.make_vec_env", return_value=dummy_env),
            patch("TD3.main.wandb", dummy_wandb),
            patch("TD3.main.setup_logger", return_value=_null_logger()),
            patch("TD3.main.tqdm", DummyTqdm),
        ):
            td3_main.train(config="unused.yaml", wandb_key="")

        self.assertGreaterEqual(len(dummy_wandb.logged), 1)
        self.assertTrue(dummy_env.close_called)

    def test_train_uses_env_wandb_key_when_cli_key_missing(self) -> None:
        cfg = Config()
        cfg.total_steps = 1
        cfg.start_steps = 1
        cfg.batch_size = 1
        cfg.buffer_size = 10
        cfg.updates_per_step = 1
        cfg.num_envs = 1
        cfg.hidden_sizes = [8]
        cfg.activation = "relu"
        cfg.device = "cpu"
        cfg.checkpoint_interval = 100
        cfg.save_best = False
        cfg.log_interval = 1
        cfg.log_to_console = False
        cfg.log_to_file = False
        cfg.wandb_key = ""

        dummy_env = DummyVecEnv(obs_dim=3, act_dim=2, num_envs=1, episode_length=1)
        dummy_wandb = DummyWandb()

        with (
            patch.dict(os.environ, {"WANDB_API_KEY": "env-key"}, clear=False),
            patch("TD3.main.Config.from_yaml", return_value=cfg),
            patch("TD3.main.make_vec_env", return_value=dummy_env),
            patch("TD3.main.wandb", dummy_wandb),
            patch("TD3.main.setup_logger", return_value=_null_logger()),
            patch("TD3.main.tqdm", DummyTqdm),
        ):
            td3_main.train(config="unused.yaml", wandb_key="")

        self.assertIn("env-key", dummy_wandb.login_keys)

    def test_demo_runs_with_stub_env(self) -> None:
        cfg = Config()
        cfg.episodes = 2
        cfg.hidden_sizes = [8]
        cfg.activation = "relu"
        cfg.device = "cpu"
        cfg.render_mode = None
        cfg.log_to_console = False
        cfg.log_to_file = False

        dummy_env = DummyEnv(obs_dim=3, act_dim=2, episode_length=1)

        with (
            patch("TD3.main.Config.from_yaml", return_value=cfg),
            patch("TD3.main.make_env", return_value=dummy_env),
            patch("TD3.main.load_checkpoint") as load_ckpt,
            patch("TD3.main.setup_logger", return_value=_null_logger()),
        ):
            td3_main.demo(config="unused.yaml", model_path="model.pt", episodes=2)

        self.assertTrue(load_ckpt.called)
        self.assertTrue(dummy_env.close_called)

    def test_train_can_force_random_action_pulses(self) -> None:
        cfg = Config()
        cfg.total_steps = 3
        cfg.start_steps = 0
        cfg.batch_size = 1
        cfg.buffer_size = 10
        cfg.updates_per_step = 1
        cfg.num_envs = 1
        cfg.hidden_sizes = [8]
        cfg.activation = "relu"
        cfg.device = "cpu"
        cfg.checkpoint_interval = 100
        cfg.save_best = False
        cfg.log_interval = 1
        cfg.log_to_console = False
        cfg.log_to_file = False
        cfg.wandb_key = ""
        cfg.random_action_prob = 1.0
        cfg.random_action_hold_min = 10
        cfg.random_action_hold_max = 10

        dummy_env = DummyVecEnv(obs_dim=3, act_dim=2, num_envs=1, episode_length=10)
        dummy_wandb = DummyWandb()

        with (
            patch("TD3.main.Config.from_yaml", return_value=cfg),
            patch("TD3.main.make_vec_env", return_value=dummy_env),
            patch("TD3.main.wandb", dummy_wandb),
            patch("TD3.main.setup_logger", return_value=_null_logger()),
            patch("TD3.main.tqdm", DummyTqdm),
        ):
            td3_main.train(config="unused.yaml", wandb_key="")

        self.assertTrue(dummy_env.close_called)

    def test_train_multi_env(self) -> None:
        cfg = Config()
        cfg.total_steps = 8
        cfg.start_steps = 0
        cfg.batch_size = 2
        cfg.buffer_size = 20
        cfg.updates_per_step = 1
        cfg.num_envs = 2
        cfg.hidden_sizes = [8]
        cfg.activation = "relu"
        cfg.device = "cpu"
        cfg.checkpoint_interval = 100
        cfg.save_best = False
        cfg.log_interval = 4
        cfg.log_to_console = False
        cfg.log_to_file = False
        cfg.wandb_key = ""

        dummy_env = DummyVecEnv(obs_dim=3, act_dim=2, num_envs=2, episode_length=3)
        dummy_wandb = DummyWandb()

        with (
            patch("TD3.main.Config.from_yaml", return_value=cfg),
            patch("TD3.main.make_vec_env", return_value=dummy_env),
            patch("TD3.main.wandb", dummy_wandb),
            patch("TD3.main.setup_logger", return_value=_null_logger()),
            patch("TD3.main.tqdm", DummyTqdm),
        ):
            td3_main.train(config="unused.yaml", wandb_key="")

        self.assertTrue(dummy_env.close_called)
        self.assertGreaterEqual(len(dummy_wandb.logged), 1)


class RecordingReplay:
    def __init__(self):
        self.rows = []
        self.size = 0

    def add_batch(self, *batch):
        self.rows.append(tuple(np.array(field, copy=True) for field in batch))
        self.size += len(batch[0])

    def can_sample(self, n):
        return self.size >= n

    def sample(self, n):
        return ("recorded-batch", n)


class RecordingAgent:
    def __init__(self):
        self.obs_dim = 3
        self.act_dim = 2
        self.device = "cpu"
        self.actor = Mock()
        self.act_calls = []
        self.updates = []

    def act(self, obs, noise=0.0, deterministic=False):
        self.act_calls.append((np.array(obs, copy=True), noise, deterministic))
        return (
            np.zeros((len(obs), 2), dtype=np.float32)
            if np.ndim(obs) == 2
            else np.zeros(2, dtype=np.float32)
        )

    def update(self, batch):
        self.updates.append(batch)
        return TD3Stats(1.0, None, 0.1, 0.2)


class RecordingVecEnv(DummyVecEnv):
    """SAME_STEP stub with no episode info and optional reused observation buffer."""

    def __init__(
        self, lengths=(2, 3), kinds=None, final_key="final_obs", mutable=False
    ):
        super().__init__(3, 2, num_envs=len(lengths))
        self.lengths = np.asarray(lengths)
        self.kinds = kinds or ["terminated"] * len(lengths)
        self.final_key = final_key
        self.mutable = mutable
        self.obs_buffer = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.actions = []
        self.reset_seeds = []
        self.step_error = None

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self._steps[:] = 0
        self.obs_buffer[:] = 0
        return self.obs_buffer, {}

    def step(self, actions):
        if self.step_error:
            raise self.step_error
        self.actions.append(np.array(actions, copy=True))
        self._steps += 1
        boundaries = self._steps >= self.lengths
        final = np.repeat(self._steps[:, None], 3, axis=1).astype(np.float32)
        next_obs = final.copy()
        next_obs[boundaries] = -50.0
        terminated = boundaries & np.array(
            [kind == "terminated" for kind in self.kinds]
        )
        truncated = boundaries & np.array([kind == "truncated" for kind in self.kinds])
        infos = {}
        if boundaries.any() and self.final_key:
            infos[self.final_key] = [
                row.copy() if done else None for row, done in zip(final, boundaries)
            ]
            infos["_" + self.final_key] = boundaries.copy()
        self._steps[boundaries] = 0
        if self.mutable:
            self.obs_buffer[:] = next_obs
            next_obs = self.obs_buffer
        return (
            next_obs,
            np.arange(1, self.num_envs + 1, dtype=np.float32),
            terminated,
            truncated,
            infos,
        )


class TestTrainingContracts(unittest.TestCase):
    def _config(self, **overrides):
        cfg = Config()
        values = dict(
            total_steps=8,
            start_steps=0,
            update_after=None,
            batch_size=2,
            buffer_size=40,
            updates_per_step=1,
            num_envs=2,
            hidden_sizes=[],
            device="cpu",
            checkpoint_interval=0,
            log_interval=2,
            save_best=False,
            log_to_console=False,
            log_to_file=False,
            wandb_key="",
            random_action_prob=0.0,
        )
        for key, value in {**values, **overrides}.items():
            setattr(cfg, key, value)
        return cfg

    def _run(
        self,
        cfg=None,
        env=None,
        wandb_key="",
        agent=None,
        wandb=None,
        save=None,
        env_key="",
    ):
        cfg = cfg or self._config()
        env = env or RecordingVecEnv(lengths=(2,) * cfg.num_envs)
        replay = RecordingReplay()
        agent = agent or RecordingAgent()
        wandb = wandb or DummyWandb()
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"WANDB_API_KEY": env_key}, clear=False)
            )
            stack.enter_context(patch("TD3.main.Config.from_yaml", return_value=cfg))
            factory = stack.enter_context(
                patch("TD3.main.make_vec_env", return_value=env)
            )
            stack.enter_context(patch("TD3.main.TD3Agent", return_value=agent))
            stack.enter_context(patch("TD3.main.ReplayBuffer", return_value=replay))
            stack.enter_context(patch("TD3.main.wandb", wandb))
            stack.enter_context(
                patch("TD3.main.setup_logger", return_value=_null_logger())
            )
            stack.enter_context(patch("TD3.main.tqdm", DummyTqdm))
            stack.enter_context(patch("TD3.main.set_seed"))
            saved = stack.enter_context(
                patch("TD3.main.save_checkpoint", side_effect=save)
            )
            result = td3_main.train(config="unused.yaml", wandb_key=wandb_key)
        return result, env, agent, replay, wandb, saved, factory

    def test_return_counters_and_optimizer_ratio_per_vector_step(self):
        result, env, agent, replay, wandb, *_ = self._run(
            self._config(updates_per_step=3), RecordingVecEnv(lengths=(10, 10))
        )
        self.assertEqual(result["global_step"], 8)
        self.assertEqual(result["environment_steps"], 8)
        self.assertEqual(result["vector_steps"], 4)
        self.assertEqual(result["optimizer_updates"], 12)
        self.assertEqual(len(agent.updates), 12)
        self.assertEqual(result["native_episodes"], 0)
        self.assertIsNone(result["best_avg_return"])
        self.assertEqual(len(env.actions), 4)
        self.assertEqual(len(replay.rows), 4)
        self.assertEqual(env.reset_seeds, [[42, 43]])
        final_payload = wandb.logged[-1][0]
        self.assertEqual(final_payload["stats/updates_per_transition"], 1.5)
        self.assertEqual(final_payload["progress/optimizer_updates"], 12)
        self.assertTrue(wandb.run.finished)

    def test_explicit_update_after_does_not_change_warmup_action_selection(self):
        cfg = self._config(start_steps=6, update_after=0, batch_size=2)
        result, env, agent, *_ = self._run(cfg)
        self.assertEqual(result["optimizer_updates"], 4)
        self.assertEqual(len(agent.act_calls), 1)
        self.assertEqual(len(env.actions), 4)

    def test_none_update_after_means_start_steps_and_can_delay_learning(self):
        cfg = self._config(start_steps=6, update_after=None)
        result, _, agent, *_ = self._run(cfg)
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertEqual(len(agent.act_calls), 1)
        result, _, agent, *_ = self._run(self._config(update_after=20))
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual(len(agent.act_calls), 4)
        result, _, agent, *_ = self._run(self._config(start_steps=3))
        self.assertEqual(result["optimizer_updates"], 3)
        self.assertEqual(len(agent.act_calls), 2)
        result, _, agent, *_ = self._run(self._config(start_steps=10))
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual(len(agent.act_calls), 0)

    def test_bad_vector_budget_fails_before_environment_creation(self):
        for values in [dict(total_steps=7), dict(total_steps=9)]:
            cfg = self._config(**values)
            with (
                patch("TD3.main.Config.from_yaml", return_value=cfg),
                patch("TD3.main.make_vec_env") as factory,
            ):
                with (
                    self.subTest(values=values),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    td3_main.train("unused.yaml")
                factory.assert_not_called()

    def test_bad_count_types_fail_before_environment_creation(self):
        for key in [
            "num_envs",
            "updates_per_step",
            "total_steps",
            "batch_size",
            "policy_delay",
        ]:
            for value in [True, 1.5, 0]:
                cfg = self._config(**{key: value})
                with (
                    patch("TD3.main.Config.from_yaml", return_value=cfg),
                    patch("TD3.main.make_vec_env") as factory,
                ):
                    with (
                        self.subTest(key=key, value=value),
                        self.assertRaises((ValueError, TypeError)),
                    ):
                        td3_main.train("unused.yaml")
                    factory.assert_not_called()

    def test_final_observations_and_hard_terminal_only_replay_mask(self):
        env = RecordingVecEnv(lengths=(1, 1), kinds=["terminated", "truncated"])
        _, _, _, replay, *_ = self._run(self._config(total_steps=2), env)
        obs, actions, rewards, next_obs, dones = replay.rows[0]
        np.testing.assert_array_equal(obs, np.zeros((2, 3)))
        np.testing.assert_array_equal(next_obs, np.ones((2, 3)))
        np.testing.assert_array_equal(np.asarray(dones).flatten(), [1.0, 0.0])

    def test_final_observation_alias_uses_actual_terminal_not_autoreset_state(self):
        env = RecordingVecEnv(
            lengths=(1, 1),
            kinds=["truncated", "truncated"],
            final_key="final_observation",
        )
        _, _, _, replay, *_ = self._run(self._config(total_steps=2), env)
        np.testing.assert_array_equal(replay.rows[0][3], np.ones((2, 3)))
        np.testing.assert_array_equal(
            np.asarray(replay.rows[0][4]).flatten(), [0.0, 0.0]
        )

    def test_missing_actual_truncation_final_observation_fails_closed(self):
        for key in [None, "final_obs", "final_observation"]:
            env = RecordingVecEnv(
                lengths=(1, 1), kinds=["truncated", "truncated"], final_key=key
            )
            if key:
                original = env.step

                def incomplete(actions, original=original, key=key):
                    obs, rewards, terminated, truncated, info = original(actions)
                    info[key][1] = None
                    return obs, rewards, terminated, truncated, info

                env.step = incomplete
            wandb = DummyWandb()
            with self.subTest(key=key), self.assertRaises((ValueError, RuntimeError)):
                self._run(self._config(total_steps=2), env, wandb=wandb)
            self.assertTrue(env.close_called)
            self.assertTrue(wandb.run.finished)

    def test_pre_step_observation_copy_survives_reused_environment_buffer(self):
        env = RecordingVecEnv(lengths=(10, 10), mutable=True)
        _, _, _, replay, *_ = self._run(self._config(total_steps=4), env)
        np.testing.assert_array_equal(replay.rows[0][0], np.zeros((2, 3)))
        np.testing.assert_array_equal(replay.rows[0][3], np.ones((2, 3)))
        np.testing.assert_array_equal(replay.rows[1][0], np.ones((2, 3)))
        np.testing.assert_array_equal(replay.rows[1][3], np.full((2, 3), 2))

    def test_native_episode_accumulators_work_without_final_info_per_environment(self):
        result, _, _, _, wandb, *_ = self._run(
            self._config(total_steps=12), RecordingVecEnv(lengths=(2, 3))
        )
        self.assertEqual(result["native_episodes"], 5)
        payload = [data for data, _ in wandb.logged if "charts/avg_return" in data][-1]
        self.assertAlmostEqual(payload["charts/avg_return"], (2 + 6 + 2 + 2 + 6) / 5)
        self.assertAlmostEqual(payload["charts/avg_length"], (2 + 3 + 2 + 2 + 3) / 5)

    def test_no_native_zero_curve_before_an_episode_has_completed(self):
        _, _, _, _, wandb, *_ = self._run(
            self._config(total_steps=4), RecordingVecEnv(lengths=(10, 10))
        )
        self.assertTrue(wandb.logged)
        for payload, _ in wandb.logged:
            self.assertNotIn("charts/avg_return", payload)
            self.assertNotIn("charts/avg_length", payload)

    def test_best_window_20_is_updated_at_boundaries_not_checkpoint_cadence(self):
        cfg = self._config(
            total_steps=40,
            num_envs=1,
            batch_size=1,
            save_best=True,
            checkpoint_interval=0,
            log_interval=0,
            best_window=20,
        )
        result, _, _, _, _, saved, *_ = self._run(cfg, RecordingVecEnv(lengths=(1,)))
        self.assertEqual(result["native_episodes"], 40)
        self.assertEqual(result["best_avg_return"], 1.0)
        best_calls = [
            call
            for call in saved.call_args_list
            if os.path.basename(call.args[0]) == "best.pt"
        ]
        self.assertEqual(len(best_calls), 1)
        self.assertEqual(best_calls[0].args[2], 20)
        result, _, _, _, _, saved, *_ = self._run(
            self._config(
                total_steps=19,
                num_envs=1,
                batch_size=1,
                save_best=True,
                checkpoint_interval=0,
                best_window=20,
            ),
            RecordingVecEnv(lengths=(1,)),
        )
        self.assertIsNone(result["best_avg_return"])
        self.assertFalse(
            any(
                os.path.basename(call.args[0]) == "best.pt"
                for call in saved.call_args_list
            )
        )
        self.assertTrue(
            saved.call_args_list[-1].kwargs["metadata"]["training_completed"]
        )
        # Both completions in one vector step must evaluate their own rolling window.
        env = RecordingVecEnv(lengths=(1, 1))
        original_step = env.step

        def varying_rewards(actions):
            obs, rewards, term, trunc, info = original_step(actions)
            rewards = (
                np.array([10.0, 10.0])
                if len(env.actions) <= 10
                else np.array([100.0, -100.0])
            )
            return obs, rewards, term, trunc, info

        env.step = varying_rewards
        result, _, _, _, _, saved, *_ = self._run(
            self._config(
                total_steps=22,
                save_best=True,
                checkpoint_interval=0,
                log_interval=0,
                best_window=20,
            ),
            env,
        )
        self.assertEqual(result["best_avg_return"], 14.5)
        scores = [
            call.args[3]
            for call in saved.call_args_list
            if os.path.basename(call.args[0]) == "best.pt"
        ]
        self.assertEqual(scores, [10.0, 14.5])

    def test_random_pulse_clears_and_retriggers_at_native_boundary(self):
        cfg = self._config(
            total_steps=2,
            num_envs=1,
            batch_size=1,
            random_action_prob=1.0,
            random_action_hold_min=10,
            random_action_hold_max=10,
        )
        env = RecordingVecEnv(lengths=(1,))
        with patch.object(
            env.single_action_space, "sample", side_effect=[np.ones(2), -np.ones(2)]
        ) as sample:
            self._run(cfg, env)
        self.assertEqual(sample.call_count, 2)
        np.testing.assert_array_equal(env.actions[0], np.ones((1, 2)))
        np.testing.assert_array_equal(env.actions[1], -np.ones((1, 2)))

    def test_random_pulse_samples_only_at_idle_trigger_then_holds_action(self):
        cfg = self._config(
            total_steps=4,
            num_envs=1,
            batch_size=1,
            random_action_prob=1.0,
            random_action_hold_min=3,
            random_action_hold_max=3,
        )
        env = RecordingVecEnv(lengths=(10,))
        with patch.object(
            env.single_action_space, "sample", side_effect=[np.ones(2), -np.ones(2)]
        ) as sample:
            self._run(cfg, env)
        self.assertEqual(sample.call_count, 2)
        for action in env.actions[:3]:
            np.testing.assert_array_equal(action, np.ones((1, 2)))
        np.testing.assert_array_equal(env.actions[3], -np.ones((1, 2)))

    def test_wandb_cli_overrides_yaml_and_yaml_used_without_other_keys(self):
        _, _, _, _, wandb, *_ = self._run(
            self._config(wandb_key="yaml-key"), wandb_key="cli-key", env_key="env-key"
        )
        self.assertEqual(wandb.login_keys, ["cli-key"])
        _, _, _, _, wandb, *_ = self._run(
            self._config(wandb_key="yaml-key"), env_key="env-key"
        )
        self.assertEqual(wandb.login_keys, ["env-key"])
        _, _, _, _, wandb, *_ = self._run(self._config(wandb_key="yaml-key"))
        self.assertEqual(wandb.login_keys, ["yaml-key"])

    def test_training_cleanup_covers_environment_update_and_checkpoint_errors(self):
        for where in ["step", "update", "save", "close"]:
            env = RecordingVecEnv(lengths=(1, 1))
            agent = RecordingAgent()
            wandb = DummyWandb()
            save = None
            cfg = self._config(total_steps=2, checkpoint_interval=2)
            if where == "step":
                env.step_error = RuntimeError("step failed")
            elif where == "update":
                agent.update = lambda batch: (_ for _ in ()).throw(
                    RuntimeError("update failed")
                )
            elif where == "save":
                save = lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError("save failed")
                )
            else:

                def failed_close():
                    env.close_called = True
                    raise RuntimeError("close failed")

                env.close = failed_close
            with self.subTest(where=where), self.assertRaises(RuntimeError):
                self._run(cfg, env, agent=agent, wandb=wandb, save=save)
            self.assertTrue(env.close_called)
            self.assertTrue(wandb.run.finished)

    def test_factory_failure_still_finishes_wandb_run(self):
        cfg = self._config()
        wandb = DummyWandb()
        with (
            patch("TD3.main.Config.from_yaml", return_value=cfg),
            patch("TD3.main.make_vec_env", side_effect=RuntimeError("factory failed")),
            patch("TD3.main.wandb", wandb),
            patch("TD3.main.setup_logger", return_value=_null_logger()),
        ):
            with self.assertRaises(RuntimeError):
                td3_main.train("unused.yaml")
        self.assertTrue(wandb.run.finished)


class TestDemoContracts(unittest.TestCase):
    def _config(self, **overrides):
        cfg = Config(
            total_steps=4,
            start_steps=0,
            batch_size=1,
            buffer_size=10,
            hidden_sizes=[],
            device="cpu",
            render_mode=None,
            max_steps_per_episode=2,
            log_to_console=False,
            log_to_file=False,
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def test_demo_episode_and_horizon_counts_fail_before_environment_creation(self):
        for episodes in [0, -1, True, 1.5]:
            with (
                patch("TD3.main.Config.from_yaml", return_value=self._config()),
                patch("TD3.main.make_env") as factory,
            ):
                with (
                    self.subTest(episodes=episodes),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    td3_main.demo("unused.yaml", episodes=episodes)
                factory.assert_not_called()
        for horizon in [0, -1, True, 1.5]:
            with (
                patch(
                    "TD3.main.Config.from_yaml",
                    return_value=self._config(max_steps_per_episode=horizon),
                ),
                patch("TD3.main.make_env") as factory,
            ):
                with (
                    self.subTest(horizon=horizon),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    td3_main.demo("unused.yaml")
                factory.assert_not_called()

    def test_demo_is_headless_returns_native_flags_and_closes_on_load_or_step_error(
        self,
    ):
        for boundary in ["cap", "terminated", "load-error", "step-error"]:
            env = DummyEnv(3, 2, episode_length=1 if boundary == "terminated" else 10)
            env.reset = Mock(wraps=env.reset)
            agent = RecordingAgent()
            load_error = (
                RuntimeError("load failed") if boundary == "load-error" else None
            )
            if boundary == "step-error":
                env.step = Mock(side_effect=RuntimeError("step failed"))
            with (
                patch("TD3.main.Config.from_yaml", return_value=self._config()),
                patch("TD3.main.make_env", return_value=env) as factory,
                patch("TD3.main.TD3Agent", return_value=agent),
                patch("TD3.main.load_checkpoint", side_effect=load_error),
                patch("TD3.main.setup_logger", return_value=_null_logger()),
                patch("TD3.main.set_seed"),
            ):
                if boundary.endswith("error"):
                    with self.assertRaises(RuntimeError):
                        td3_main.demo(
                            "unused.yaml", model_path="weights.pt", episodes=2
                        )
                else:
                    result = td3_main.demo(
                        "unused.yaml", model_path="weights.pt", episodes=2
                    )
                    length = 1 if boundary == "terminated" else 2
                    self.assertEqual(result["returns"], [float(length)] * 2)
                    self.assertEqual(result["lengths"], [length] * 2)
                    self.assertEqual(
                        result["terminated"], [boundary == "terminated"] * 2
                    )
                    self.assertEqual(result["truncated"], [False, False])
                    self.assertEqual(result["capped"], [boundary == "cap"] * 2)
                    self.assertNotIn("success", result)
                    self.assertEqual(env.reset.call_args_list[0].kwargs["seed"], 42)
                    self.assertEqual(env.reset.call_args_list[1].kwargs["seed"], 43)
                    self.assertTrue(all(call[2] for call in agent.act_calls))
                self.assertIsNone(factory.call_args.kwargs["render_mode"])
            self.assertTrue(env.close_called)


if __name__ == "__main__":
    unittest.main()
