"""SAC loops exercised only with native spaces and fully stubbed I/O."""

from contextlib import contextmanager, ExitStack
import logging
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from gymnasium import spaces

from SAC.sac.agent import SACStats
from SAC.sac.config import Config
import SAC.main as sac_main


class DummyEnv:
    def __init__(
        self,
        obs_dim=3,
        act_dim=2,
        episode_length=2,
        truncated=False,
        episode_info=False,
    ):
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-2.0, 2.0, (act_dim,), dtype=np.float32)
        self.action_space.sample = Mock(
            return_value=np.full(act_dim, 0.25, dtype=np.float32)
        )
        self.action_space.seed = Mock()
        self.episode_length = episode_length
        self.truncated = truncated
        self.episode_info = episode_info
        self._step = 0
        self.total_steps = 0
        self.reset_seeds = []
        self.close_called = False
        self.close_count = 0
        self.actions = []
        self._observation = np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self, seed=None):
        self._step = 0
        self.reset_seeds.append(seed)
        self._observation.fill(0.0)
        return self._observation, {}

    def step(self, action):
        self._step += 1
        self.total_steps += 1
        self.actions.append(np.array(action, copy=True))
        self._observation.fill(float(self._step))
        obs = self._observation
        done = self._step >= self.episode_length
        info = {"episode": {"r": 99999.0, "l": 99999}} if self.episode_info else {}
        return (
            obs,
            1.0,
            bool(done and not self.truncated),
            bool(done and self.truncated),
            info,
        )

    def close(self):
        self.close_called = True
        self.close_count += 1


class DummyAgent:
    def __init__(self, act_dim=2, obs_dim=3):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.device = "cpu"
        self.global_step = 0
        self.act_calls = []
        self.update_calls = []
        self.actor = Mock()

    def act(self, obs, deterministic=False):
        self.act_calls.append((np.array(obs, copy=True), deterministic))
        return np.full(self.act_dim, -0.25, dtype=np.float32)

    def update(self, batch):
        self.update_calls.append(batch)
        self.global_step += 1
        return SACStats(
            critic_loss=1.0,
            actor_loss=2.0,
            alpha_loss=0.0,
            alpha_value=1.0,
            log_prob=-1.0,
        )


class RecordingReplay:
    def __init__(self):
        self.entries = []

    def add(self, obs, action, reward, next_obs, done):
        self.entries.append(
            (
                np.array(obs, copy=True),
                np.array(action, copy=True),
                float(reward),
                np.array(next_obs, copy=True),
                float(done),
            )
        )

    def can_sample(self, batch_size):
        return len(self.entries) >= batch_size

    def sample(self, batch_size):
        return tuple(self.entries[-batch_size:])


class DummyWandb:
    class _Run:
        def __init__(self):
            self.finished = False
            self.finish_count = 0

        def finish(self):
            self.finished = True
            self.finish_count += 1

    def __init__(self):
        self.logged = []
        self.init_kwargs = None
        self.login_keys = []
        self.run = self._Run()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def log(self, data, step=None):
        self.logged.append((data, step))

    def login(self, key=None):
        self.login_keys.append(key)


class DummyTqdm:
    def __init__(self, iterable, desc=None):
        self.iterable = iterable
        self.desc = desc
        self.closed = False
        self.close_count = 0

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        return None

    def close(self):
        self.closed = True
        self.close_count += 1


def _null_logger():
    logger = logging.getLogger("sac-test")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _config(**kwargs):
    cfg = Config()
    values = dict(
        total_steps=3,
        start_steps=0,
        update_after=0,
        batch_size=2,
        buffer_size=10,
        updates_per_step=1,
        hidden_sizes=[8],
        activation="relu",
        device="cpu",
        checkpoint_interval=100,
        save_best=False,
        log_interval=1,
        log_to_console=False,
        log_to_file=False,
        wandb_key="",
    )
    values.update(kwargs)
    for key, value in values.items():
        setattr(cfg, key, value)
    return cfg


class TestMain(unittest.TestCase):
    @contextmanager
    def _context(self, cfg=None, env=None, wb=None):
        cfg = cfg or _config()
        env = env or DummyEnv()
        wb = wb or DummyWandb()
        agent = DummyAgent(env.action_space.shape[0], env.observation_space.shape[0])
        replay = RecordingReplay()
        pbar = DummyTqdm(range(1, cfg.total_steps + 1))
        with ExitStack() as stack:
            context = SimpleNamespace(
                cfg=cfg, env=env, wb=wb, agent=agent, replay=replay, pbar=pbar
            )
            context.load_config = stack.enter_context(
                patch("SAC.main.Config.from_yaml", return_value=cfg)
            )
            context.make_env = stack.enter_context(
                patch("SAC.main.make_env", return_value=env)
            )
            stack.enter_context(patch("SAC.main.wandb", wb))
            stack.enter_context(
                patch("SAC.main.setup_logger", return_value=_null_logger())
            )
            context.seed = stack.enter_context(patch("SAC.main.set_seed"))
            context.make_agent = stack.enter_context(
                patch("SAC.main.SACAgent", return_value=agent)
            )
            context.make_replay = stack.enter_context(
                patch("SAC.main.ReplayBuffer", return_value=replay)
            )
            context.make_pbar = stack.enter_context(
                patch("SAC.main.tqdm", return_value=pbar)
            )
            context.save = stack.enter_context(patch("SAC.main.save_checkpoint"))
            context.load = stack.enter_context(patch("SAC.main.load_checkpoint"))
            stack.enter_context(patch.dict(os.environ, {"WANDB_API_KEY": ""}))
            yield context

    def test_train_runs_with_stub_env_and_wandb(self) -> None:
        with self._context() as context:
            sac_main.train(config="unused.yaml", wandb_key="")
        self.assertGreaterEqual(len(context.wb.logged), 1)
        self.assertTrue(context.env.close_called)
        self.assertTrue(context.wb.run.finished)
        self.assertTrue(context.pbar.closed)
        self.assertEqual(context.agent.global_step, 2)

    def test_demo_runs_with_stub_env(self) -> None:
        with self._context(
            cfg=_config(episodes=2, render_mode=None), env=DummyEnv(episode_length=1)
        ) as context:
            sac_main.demo(config="unused.yaml", model_path="model.pt", episodes=2)
        context.load.assert_called_once()
        self.assertTrue(context.env.close_called)
        context.agent.actor.eval.assert_called_once()
        self.assertTrue(
            all(deterministic for _, deterministic in context.agent.act_calls)
        )

    def test_wandb_key_precedence_is_cli_then_environment_then_yaml(self) -> None:
        for cli, environment, expected in [
            ("cli-secret", "env-secret", "cli-secret"),
            ("", "env-secret", "env-secret"),
            ("", "", "yaml-secret"),
        ]:
            with (
                self.subTest(cli=bool(cli), environment=bool(environment)),
                self._context(cfg=_config(wandb_key="yaml-secret")) as context,
            ):
                with patch.dict(os.environ, {"WANDB_API_KEY": environment}):
                    sac_main.train("unused.yaml", wandb_key=cli)
                self.assertEqual(context.wb.login_keys, [expected])
                self.assertNotIn("wandb_key", context.wb.init_kwargs["config"])
                self.assertNotIn(expected, repr(context.wb.init_kwargs["config"]))
                self.assertTrue(context.wb.run.finished)

    def test_factory_failure_finishes_the_acquired_wandb_run(self) -> None:
        with self._context() as context:
            context.make_env.side_effect = RuntimeError("factory failure")
            with self.assertRaisesRegex(RuntimeError, "factory failure"):
                sac_main.train("unused.yaml")
        self.assertEqual(context.wb.run.finish_count, 1)
        context.make_agent.assert_not_called()
        context.make_pbar.assert_not_called()
        with self._context() as context:
            context.wb.init = Mock(side_effect=RuntimeError("init failed"))
            with self.assertRaisesRegex(RuntimeError, "init failed"):
                sac_main.train("unused.yaml")
        context.make_env.assert_not_called()
        self.assertEqual(context.wb.run.finish_count, 0)

    def test_setup_failures_close_environment_and_finish_run(self) -> None:
        for phase in ["agent", "replay", "reset", "pbar"]:
            with self.subTest(phase=phase), self._context() as context:
                target = {
                    "agent": context.make_agent,
                    "replay": context.make_replay,
                    "pbar": context.make_pbar,
                }.get(phase)
                if phase == "reset":
                    target = Mock(side_effect=RuntimeError(phase))
                    context.env.reset = target
                else:
                    target.side_effect = RuntimeError(phase)
                with self.assertRaisesRegex(RuntimeError, phase):
                    sac_main.train("unused.yaml")
                self.assertEqual(context.env.close_count, 1)
                self.assertEqual(context.wb.run.finish_count, 1)

    def test_loop_failures_close_all_acquired_resources(self) -> None:
        for phase in ["act", "step", "add", "sample", "update", "log", "checkpoint"]:
            with (
                self.subTest(phase=phase),
                self._context(cfg=_config(batch_size=1)) as context,
            ):
                owner = {
                    "act": context.agent,
                    "step": context.env,
                    "add": context.replay,
                    "sample": context.replay,
                    "update": context.agent,
                    "log": context.wb,
                }.get(phase)
                if phase == "checkpoint":
                    context.save.side_effect = RuntimeError(phase)
                else:
                    setattr(owner, phase, Mock(side_effect=RuntimeError(phase)))
                with self.assertRaisesRegex(RuntimeError, phase):
                    sac_main.train("unused.yaml")
                self.assertEqual(context.env.close_count, 1)
                self.assertEqual(context.pbar.close_count, 1)
                self.assertEqual(context.wb.run.finish_count, 1)

    def test_cleanup_errors_do_not_skip_remaining_resource_cleanup(self) -> None:
        for phase in ["env", "pbar", "finish"]:
            with self.subTest(phase=phase), self._context() as context:
                owner, attribute = {
                    "env": (context.env, "close"),
                    "pbar": (context.pbar, "close"),
                    "finish": (context.wb.run, "finish"),
                }[phase]
                original = getattr(owner, attribute)

                def fail_after_cleanup(original=original, phase=phase):
                    original()
                    raise RuntimeError(phase)

                setattr(owner, attribute, Mock(side_effect=fail_after_cleanup))
                with self.assertRaisesRegex(RuntimeError, phase):
                    sac_main.train("unused.yaml")
                self.assertEqual(context.env.close_count, 1)
                self.assertEqual(context.pbar.close_count, 1)
                self.assertEqual(context.wb.run.finish_count, 1)

    def test_random_warmup_and_update_after_are_independent(self) -> None:
        for update_after, expected_updates in [(None, 4), (0, 5), (5, 2)]:
            with (
                self.subTest(update_after=update_after),
                self._context(
                    cfg=_config(total_steps=6, start_steps=3, update_after=update_after)
                ) as context,
            ):
                result = sac_main.train("unused.yaml")
                self.assertEqual(context.env.action_space.sample.call_count, 3)
                self.assertEqual(len(context.agent.act_calls), 3)
                self.assertEqual(result["optimizer_updates"], expected_updates)
                self.assertEqual(context.agent.global_step, expected_updates)
                for action in context.env.actions[:3]:
                    np.testing.assert_array_equal(action, [0.25, 0.25])
                for action in context.env.actions[3:]:
                    np.testing.assert_array_equal(action, [-0.25, -0.25])

    def test_exact_environment_budget_and_counters_ignore_episode_length(self) -> None:
        for length in [1, 2, 10]:
            cfg = _config(total_steps=5, updates_per_step=2)
            with (
                self.subTest(length=length),
                self._context(cfg=cfg, env=DummyEnv(episode_length=length)) as context,
            ):
                result = sac_main.train("unused.yaml")
                self.assertEqual(context.env.total_steps, 5)
                self.assertEqual(len(context.replay.entries), 5)
                self.assertEqual(result["global_step"], 5)
                self.assertEqual(result["environment_steps"], 5)
                self.assertEqual(result["optimizer_updates"], 8)
                self.assertEqual(result["episodes"], 5 // length)
                self.assertEqual(
                    context.env.reset_seeds, [cfg.seed] + [None] * (5 // length)
                )
                context.env.action_space.seed.assert_not_called()
                context.seed.assert_called_once_with(cfg.seed)

    def test_termination_mask_regression_and_terminal_endpoint_precede_reset(
        self,
    ) -> None:
        for truncated, expected_mask in [(False, 1.0), (True, 0.0)]:
            with (
                self.subTest(truncated=truncated),
                self._context(
                    cfg=_config(total_steps=2),
                    env=DummyEnv(episode_length=1, truncated=truncated),
                ) as context,
            ):
                sac_main.train("unused.yaml")
                for obs, _, _, endpoint, mask in context.replay.entries:
                    self.assertEqual(mask, expected_mask)
                    np.testing.assert_array_equal(obs, np.zeros(3))
                    np.testing.assert_array_equal(endpoint, np.ones(3))
        with self._context(
            cfg=_config(total_steps=2, max_steps_per_episode=1),
            env=DummyEnv(episode_length=1000),
        ) as context:
            sac_main.train("unused.yaml")
            self.assertEqual([entry[4] for entry in context.replay.entries], [0.0, 0.0])

    def test_native_episode_accumulators_ignore_missing_or_misleading_info(
        self,
    ) -> None:
        for episode_info in [False, True]:
            with (
                self.subTest(episode_info=episode_info),
                self._context(
                    cfg=_config(total_steps=5),
                    env=DummyEnv(episode_length=2, episode_info=episode_info),
                ) as context,
            ):
                result = sac_main.train("unused.yaml")
                self.assertEqual(result["episodes"], 2)
                payload = context.wb.logged[-1][0]
                self.assertEqual(payload["charts/avg_return"], 2.0)
                self.assertEqual(payload["charts/avg_length"], 2.0)
                self.assertEqual(payload["progress/optimizer_updates"], 4)

    def test_best_checkpoint_requires_twenty_completed_native_episodes(self) -> None:
        for total, expected in [(19, 0), (20, 1)]:
            with (
                self.subTest(total=total),
                self._context(
                    cfg=_config(total_steps=total, save_best=True),
                    env=DummyEnv(episode_length=1),
                ) as context,
            ):
                result = sac_main.train("unused.yaml")
                best = [
                    call
                    for call in context.save.call_args_list
                    if os.path.basename(call.args[0]) == "best.pt"
                ]
                self.assertEqual(len(best), expected)
                self.assertEqual(result["best_avg_return"], 1.0 if expected else None)

    def test_best_checkpoint_uses_the_actual_completed_episode_window(self) -> None:
        cfg = _config(total_steps=21, save_best=True, best_window=20)
        env = DummyEnv(episode_length=1)
        original = env.step

        def increasing_rewards(action):
            obs, _, terminated, truncated, info = original(action)
            return obs, float(env.total_steps), terminated, truncated, info

        env.step = increasing_rewards
        with self._context(cfg=cfg, env=env) as context:
            result = sac_main.train("unused.yaml")
        best = [
            call
            for call in context.save.call_args_list
            if os.path.basename(call.args[0]) == "best.pt"
        ]
        self.assertEqual([call.args[3] for call in best], [10.5, 11.5])
        self.assertEqual(result["best_avg_return"], 11.5)

    def test_unmeasured_best_score_is_none_not_negative_infinity(self) -> None:
        for capped in [False, True]:
            cfg = _config(
                total_steps=20 if capped else 3,
                save_best=True,
                max_steps_per_episode=1 if capped else 200,
            )
            with (
                self.subTest(capped=capped),
                self._context(cfg=cfg, env=DummyEnv(episode_length=1000)) as context,
            ):
                result = sac_main.train("unused.yaml")
                self.assertIsNone(result["best_avg_return"])
                self.assertEqual(result["episodes"], 0)
                self.assertEqual(result["capped_episodes"], 20 if capped else 0)
                self.assertTrue(context.save.called)
                self.assertTrue(
                    all(call.args[3] is None for call in context.save.call_args_list)
                )
                self.assertFalse(
                    any(
                        os.path.basename(call.args[0]) == "best.pt"
                        for call in context.save.call_args_list
                    )
                )
                self.assertTrue(context.wb.logged)
                for payload, _ in context.wb.logged:
                    self.assertNotIn("charts/avg_return", payload)
                    self.assertNotIn("charts/avg_length", payload)
                self.assertEqual(
                    context.wb.logged[-1][0]["progress/optimizer_updates"],
                    result["optimizer_updates"],
                )

    def test_final_numbered_checkpoint_records_redacted_config_and_scoped_counters(
        self,
    ) -> None:
        with self._context(
            cfg=_config(total_steps=5, checkpoint_interval=2, wandb_key="hidden-key")
        ) as context:
            result = sac_main.train("unused.yaml")
        names = [os.path.basename(call.args[0]) for call in context.save.call_args_list]
        self.assertEqual(
            names, ["checkpoint_2.pt", "checkpoint_4.pt", "checkpoint_5.pt"]
        )
        final = context.save.call_args_list[-1]
        self.assertEqual(final.args[2], 5)
        self.assertNotIn("wandb_key", final.kwargs["config"])
        self.assertNotIn("hidden-key", repr(final.kwargs["config"]))
        self.assertEqual(
            final.kwargs["metadata"],
            {
                "environment_steps": 5,
                "optimizer_updates": result["optimizer_updates"],
                "training_completed": True,
            },
        )
        self.assertNotIn("replay", final.kwargs)
        self.assertNotIn("rng", final.kwargs)

    def test_zero_log_and_checkpoint_intervals_disable_periodic_work_but_keep_final(
        self,
    ) -> None:
        with self._context(
            cfg=_config(total_steps=5, log_interval=0, checkpoint_interval=0)
        ) as context:
            sac_main.train("unused.yaml")
        self.assertEqual(len(context.wb.logged), 1)
        self.assertEqual(context.wb.logged[0][1], 5)
        self.assertEqual(
            [os.path.basename(call.args[0]) for call in context.save.call_args_list],
            ["checkpoint_5.pt"],
        )

    def test_demo_defaults_headless_and_rejects_nonpositive_episode_counts(
        self,
    ) -> None:
        with self._context(
            cfg=_config(episodes=1, render_mode=None), env=DummyEnv(episode_length=1)
        ) as context:
            sac_main.demo("unused.yaml")
            self.assertIsNone(context.make_env.call_args.kwargs["render_mode"])
            self.assertEqual(context.agent.update_calls, [])
        for value in [0, -1, True, 1.5]:
            with self.subTest(episodes=value), self._context() as context:
                with self.assertRaises(ValueError):
                    sac_main.demo("unused.yaml", episodes=value)
                context.make_env.assert_not_called()

    def test_demo_native_returns_fixed_reset_seeds_and_default_horizon_flags(
        self,
    ) -> None:
        with self._context(
            cfg=_config(episodes=2), env=DummyEnv(episode_length=1000)
        ) as context:
            result = sac_main.demo("unused.yaml", seed=17)
        self.assertEqual(context.env.total_steps, 400)
        self.assertEqual(context.env.reset_seeds, [17, 18])
        self.assertEqual(result["returns"], [200.0, 200.0])
        self.assertEqual(result["lengths"], [200, 200])
        self.assertEqual(result["capped"], [True, True])
        self.assertEqual(result["terminated"], [False, False])
        self.assertEqual(result["truncated"], [False, False])
        self.assertEqual(result["mean"], 200.0)
        self.assertEqual(context.agent.update_calls, [])

    def test_demo_failure_after_acquisition_always_closes_environment(self) -> None:
        for phase in ["agent", "load", "reset", "act", "step"]:
            with self.subTest(phase=phase), self._context() as context:
                if phase in ["agent", "load"]:
                    target = context.make_agent if phase == "agent" else context.load
                    target.side_effect = RuntimeError(phase)
                else:
                    owner = context.agent if phase == "act" else context.env
                    setattr(owner, phase, Mock(side_effect=RuntimeError(phase)))
                with self.assertRaisesRegex(RuntimeError, phase):
                    sac_main.demo("unused.yaml", episodes=1)
                self.assertEqual(context.env.close_count, 1)
                self.assertEqual(context.agent.update_calls, [])


if __name__ == "__main__":
    unittest.main()
