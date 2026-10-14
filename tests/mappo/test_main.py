"""MAPPO parallel-loop tests without Box2D, real rollouts, or cloud W&B."""

from contextlib import ExitStack, contextmanager
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from gymnasium import spaces

from MAPPO.mappo.agent import MAPPOAgent
from MAPPO.mappo.config import Config
from MAPPO.mappo.utils import save_checkpoint, load_checkpoint
import MAPPO.main as main


class ParallelEnv:
    def __init__(self, discrete=False, length=2, mode="normal", fail=False):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents.copy()
        self.obs_space = spaces.Box(-20.0, 20.0, (2,), dtype=np.float32)
        self.act_space = (
            spaces.Discrete(3)
            if discrete
            else spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        )
        self.length, self.mode, self.fail = length, mode, fail
        self.steps = self.step_calls = self.close_calls = 0
        self.reset_seeds, self.action_keys = [], []

    def observation_space(self, name):
        return self.obs_space

    def action_space(self, name):
        return self.act_space

    def reset(self, seed=None):
        self.steps = 0
        self.agents = self.possible_agents.copy()
        self.reset_seeds.append(seed)
        return {name: np.zeros(2, dtype=np.float32) for name in self.agents}, {}

    def step(self, actions):
        if not actions or set(actions) != set(self.agents):
            raise AssertionError(
                "must step exactly active agents, never an empty dictionary"
            )
        for action in actions.values():
            if not self.act_space.contains(action):
                raise AssertionError("action outside environment bounds")
        self.action_keys.append(set(actions))
        self.steps += 1
        self.step_calls += 1
        if self.fail:
            raise RuntimeError("stub step failure")
        active = self.agents.copy()
        rewards = {
            name: float(index * 2 + 1)
            for index, name in enumerate(self.possible_agents)
            if name in active
        }
        terms = {name: False for name in active}
        truncs = {name: False for name in active}
        infos = {}
        next_obs = {
            name: np.full(2, float(self.steps), dtype=np.float32) for name in active
        }
        if self.mode == "removal":
            rewards = {
                "agent_1": 4.0
            }  # Active agent_0 has a missing reward, not a dead pre-step mask.
            if self.steps == 1:
                terms["agent_0"] = True
                next_obs.pop("agent_0")
                self.agents = ["agent_1"]
            else:
                terms["agent_1"] = True
                next_obs = {}
                self.agents = []
        elif self.mode.startswith("timeout") and self.steps >= self.length:
            truncs = {name: True for name in active}
            next_obs = {
                name: np.full(2, 7.0 + index * 2, dtype=np.float32)
                for index, name in enumerate(self.possible_agents)
            }
            if self.mode != "timeout_nonempty":
                if self.mode == "timeout_final":
                    infos = {
                        name: {"final_observation": value}
                        for name, value in next_obs.items()
                    }
                next_obs = {}
            self.agents = []
        elif self.steps >= self.length:
            terms = {name: True for name in active}
            next_obs = {} if self.mode != "terminal_nonempty" else next_obs
            self.agents = []
        return next_obs, rewards, terms, truncs, infos

    def close(self):
        self.close_calls += 1


class DummyWandb:
    def __init__(self):
        self.logged, self.login_keys = [], []
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


def config():
    cfg = Config()
    cfg.n_agents = cfg.n_walkers = 2
    cfg.device = "cpu"
    cfg.actor_hidden_sizes = cfg.critic_hidden_sizes = [4]
    cfg.total_timesteps = 5
    cfg.rollout_steps = 2
    cfg.update_iterations = 1
    cfg.minibatch_size = 2
    cfg.anneal_lr = False
    cfg.target_kl = 0.0
    cfg.gamma = 0.5
    cfg.best_window = 10
    cfg.max_steps_per_episode = 10
    cfg.log_interval = 2
    cfg.checkpoint_interval = 100
    cfg.save_best = False
    cfg.log_to_console = cfg.log_to_file = False
    cfg.wandb_key = ""
    return cfg


@contextmanager
def stubs(cfg, env, key=""):
    wandb, logger = DummyWandb(), MagicMock()
    with ExitStack() as stack:
        if cfg.checkpoint_dir == "MAPPO/checkpoints":
            cfg.checkpoint_dir = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(patch.dict(os.environ, {"WANDB_API_KEY": key}))
        stack.enter_context(patch("MAPPO.main.Config.from_yaml", return_value=cfg))
        factory = stack.enter_context(
            patch("MAPPO.main.make_multiwalker_env", return_value=env)
        )
        stack.enter_context(patch("MAPPO.main.wandb", wandb))
        stack.enter_context(patch("MAPPO.main.setup_logger", return_value=logger))
        stack.enter_context(patch("MAPPO.main.tqdm", DummyTqdm))
        yield wandb, factory, logger


class TestTrain(unittest.TestCase):
    def test_exact_parallel_budget_partial_rollout_logging_and_continuous_discrete_actions(
        self,
    ):
        for discrete in (False, True):
            with self.subTest(discrete=discrete):
                cfg, env = config(), ParallelEnv(discrete=discrete)
                cfg.discretize_actions = discrete
                with stubs(cfg, env) as (wandb, factory, _):
                    result = main.train(config="unused.yaml")
                self.assertEqual(result["global_step"], 5)
                self.assertEqual(result["agent_transitions"], 10)
                self.assertEqual(result["optimizer_steps"], 3)
                self.assertEqual(env.step_calls, 5)
                self.assertEqual(
                    [data["progress/global_step"] for data, _ in wandb.logged], [4, 5]
                )
                self.assertEqual([step for _, step in wandb.logged], [4, 5])
                self.assertEqual(wandb.logged[-1][0]["charts/avg_return"], 4.0)
                self.assertEqual(wandb.logged[-1][0]["charts/avg_length"], 2.0)
                self.assertEqual(factory.call_args.kwargs["env_kwargs"], cfg.env_kwargs)
                for key in (
                    "loss/policy",
                    "loss/value",
                    "loss/total",
                    "stats/value_mean",
                    "stats/return_target_mean",
                ):
                    self.assertTrue(np.isfinite(wandb.logged[-1][0][key]))
                self.assertEqual(env.close_calls, 1)
                wandb.run.finish.assert_called_once()

    def test_auth_precedence_and_secret_redaction(self):
        for cli, env_key, cfg_key, expected in (
            ("cli-key", "env-key", "cfg-key", "cli-key"),
            ("", "env-key", "cfg-key", "env-key"),
            ("", "", "cfg-key", "cfg-key"),
            ("", "", "", None),
        ):
            with self.subTest(expected=expected):
                cfg, env = config(), ParallelEnv()
                cfg.total_timesteps = 1
                cfg.wandb_key = cfg_key
                with stubs(cfg, env, env_key) as (wandb, _, logger):
                    main.train(config="unused.yaml", wandb_key=cli)
                self.assertEqual(
                    wandb.login_keys, [] if expected is None else [expected]
                )
                self.assertNotIn("wandb_key", wandb.init_kwargs["config"])
                public = repr((wandb.init_kwargs, wandb.logged, logger.mock_calls))
                for secret in (cli, env_key, cfg_key):
                    if secret:
                        self.assertNotIn(secret, public)

    def test_removal_masks_use_pre_step_activity_and_reward_uses_fixed_original_team(
        self,
    ):
        cfg, env = config(), ParallelEnv(mode="removal")
        cfg.total_timesteps = 2
        original = MAPPOAgent.update
        batches_seen = []

        def capture(agent, batches):
            batches_seen.append(batches)
            return original(agent, batches)

        with (
            stubs(cfg, env) as (wandb, _, _),
            patch.object(MAPPOAgent, "update", capture),
            patch(
                "numpy.random.permutation", side_effect=lambda length: np.arange(length)
            ),
        ):
            result = main.train(config="unused.yaml")
        self.assertEqual(result["agent_transitions"], 3)
        self.assertEqual(env.action_keys, [{"agent_0", "agent_1"}, {"agent_1"}])
        np.testing.assert_array_equal(batches_seen[0][0].alive_mask.numpy(), [1.0, 0.0])
        np.testing.assert_array_equal(batches_seen[0][1].alive_mask.numpy(), [1.0, 1.0])
        self.assertEqual(wandb.logged[-1][0]["charts/avg_return"], 4.0)
        self.assertEqual(wandb.logged[-1][0]["charts/agent_0_avg_return"], 0.0)
        self.assertEqual(wandb.logged[-1][0]["charts/agent_1_avg_return"], 8.0)

    def test_timeout_uses_pre_reset_endpoints_and_nonempty_flags_end_episode(self):
        for mode in ("timeout_nonempty", "timeout_final"):
            with self.subTest(mode=mode):
                cfg, env = config(), ParallelEnv(length=1, mode=mode)
                cfg.total_timesteps = 1
                original_gae = MAPPOAgent.compute_gae
                with (
                    stubs(cfg, env),
                    patch.object(
                        MAPPOAgent,
                        "values",
                        side_effect=lambda obs, state=None: np.asarray(obs)[:, 0],
                    ),
                    patch.object(MAPPOAgent, "compute_gae", wraps=original_gae) as gae,
                ):
                    main.train(config="unused.yaml")
                call = gae.call_args
                np.testing.assert_array_equal(call.args[1], [[1.0, 1.0]])
                np.testing.assert_array_equal(call.kwargs["terminations"], [[0.0, 0.0]])
                np.testing.assert_array_equal(
                    call.kwargs["next_value_sequence"], [[7.0, 9.0]]
                )
                _, returns = original_gae(*call.args, **call.kwargs)
                np.testing.assert_allclose(returns, [[4.5, 7.5]], atol=1e-6)
                self.assertEqual(env.step_calls, 1)
                self.assertGreaterEqual(len(env.reset_seeds), 2)
                self.assertEqual(env.close_calls, 1)

    def test_missing_timeout_final_observations_fail_with_cleanup(self):
        cfg, env = config(), ParallelEnv(length=1, mode="timeout_missing")
        with stubs(cfg, env) as (wandb, _, _):
            with self.assertRaisesRegex(RuntimeError, "final observation"):
                main.train(config="unused.yaml")
        self.assertEqual(env.close_calls, 1)
        wandb.run.finish.assert_called_once()

    def test_mixed_death_zeroes_terminal_endpoint_in_surviving_central_bootstrap(self):
        cfg, env = config(), ParallelEnv(mode="removal")
        cfg.total_timesteps = 1
        original_step = env.step

        def nonzero_dead_endpoint(actions):
            obs, rewards, terms, truncs, infos = original_step(actions)
            obs["agent_0"] = np.full(2, 9.0, dtype=np.float32)
            obs["agent_1"] = np.full(2, 3.0, dtype=np.float32)
            return obs, rewards, terms, truncs, infos

        original_gae = MAPPOAgent.compute_gae
        with (
            stubs(cfg, env),
            patch.object(env, "step", side_effect=nonzero_dead_endpoint),
            patch.object(
                MAPPOAgent,
                "values",
                side_effect=lambda obs, state=None: np.full(2, np.asarray(state).sum()),
            ),
            patch.object(MAPPOAgent, "compute_gae", wraps=original_gae) as gae,
        ):
            main.train(config="unused.yaml")
        call = gae.call_args
        # Surviving critic state must be [0,0,3,3], not [9,9,3,3].
        np.testing.assert_array_equal(call.kwargs["next_value_sequence"], [[6.0, 6.0]])
        np.testing.assert_array_equal(call.kwargs["terminations"], [[1.0, 0.0]])
        _, returns = original_gae(*call.args, **call.kwargs)
        np.testing.assert_allclose(returns, [[0.0, 7.0]], atol=1e-6)

    def test_empty_reported_agents_end_episode_with_nonempty_unflagged_observations(
        self,
    ):
        cfg, env = config(), ParallelEnv(length=999)
        cfg.total_timesteps = 2
        original_step = env.step

        def no_reported_agents(actions):
            result = original_step(actions)
            env.agents = []
            return result

        original_gae = MAPPOAgent.compute_gae
        with (
            stubs(cfg, env),
            patch.object(env, "step", side_effect=no_reported_agents),
            patch.object(MAPPOAgent, "compute_gae", wraps=original_gae) as gae,
        ):
            main.train(config="unused.yaml")
        np.testing.assert_array_equal(gae.call_args.args[1], np.ones((2, 2)))
        np.testing.assert_array_equal(
            gae.call_args.kwargs["terminations"], np.zeros((2, 2))
        )
        self.assertEqual(env.action_keys, [{"agent_0", "agent_1"}] * 2)
        self.assertEqual(len(env.reset_seeds), 3)

    def test_cap_resets_without_terminal_bootstrap_mask(self):
        cfg, env = config(), ParallelEnv(length=999)
        cfg.max_steps_per_episode = 1
        cfg.total_timesteps = 2
        original_gae = MAPPOAgent.compute_gae
        with (
            stubs(cfg, env),
            patch.object(MAPPOAgent, "compute_gae", wraps=original_gae) as gae,
        ):
            main.train(config="unused.yaml")
        np.testing.assert_array_equal(gae.call_args.args[1], np.ones((2, 2)))
        np.testing.assert_array_equal(
            gae.call_args.kwargs["terminations"], np.zeros((2, 2))
        )
        self.assertEqual(len(env.reset_seeds), 3)
        self.assertEqual(env.close_calls, 1)

    def test_best_gate_ten_completed_episodes_and_final_periodic_files(self):
        cfg, env = config(), ParallelEnv(length=1)
        cfg.total_timesteps = 11
        cfg.rollout_steps = 1
        cfg.checkpoint_interval = 5
        cfg.save_best = True
        with tempfile.TemporaryDirectory() as directory:
            cfg.checkpoint_dir = directory
            with (
                stubs(cfg, env),
                patch("MAPPO.main.save_checkpoint", wraps=save_checkpoint) as save,
            ):
                main.train(config="unused.yaml")
            best = [
                call
                for call in save.call_args_list
                if os.path.basename(call.args[0]) == "best.pt"
            ]
            self.assertTrue(best)
            self.assertEqual(best[0].args[3], 10)
            self.assertTrue(os.path.isfile(os.path.join(directory, "checkpoint_11.pt")))
            shared = best[0].args[1]
            data = load_checkpoint(os.path.join(directory, "best.pt"), shared, None)
            self.assertEqual(data["step"], 10)
            self.assertEqual(data["best_return"], 2.0)
            self.assertNotIn("wandb_key", data["config"])

    def test_no_placeholder_or_short_window_best_is_written(self):
        for length in (1, 999):
            with (
                self.subTest(length=length),
                tempfile.TemporaryDirectory() as directory,
            ):
                cfg, env = config(), ParallelEnv(length=length)
                cfg.checkpoint_dir = directory
                cfg.save_best = True
                with stubs(cfg, env):
                    main.train(config="unused.yaml")
                self.assertFalse(os.path.exists(os.path.join(directory, "best.pt")))
                self.assertTrue(
                    os.path.isfile(os.path.join(directory, "checkpoint_5.pt"))
                )

    def test_step_and_update_failures_close_env_and_finish_run(self):
        for failure in ("step", "update"):
            with self.subTest(failure=failure):
                cfg, env = config(), ParallelEnv(fail=failure == "step")
                with stubs(cfg, env) as (wandb, _, _), ExitStack() as stack:
                    if failure == "update":
                        stack.enter_context(
                            patch.object(
                                MAPPOAgent,
                                "update",
                                side_effect=RuntimeError("stub update failure"),
                            )
                        )
                    with self.assertRaises(RuntimeError):
                        main.train(config="unused.yaml")
                self.assertEqual(env.close_calls, 1)
                wandb.run.finish.assert_called_once()


class TestDemo(unittest.TestCase):
    def test_demo_returns_fixed_team_and_per_agent_rewards_with_removal(self):
        cfg, env = config(), ParallelEnv(mode="removal")
        with (
            stubs(cfg, env) as (_, factory, _),
            patch("MAPPO.main.load_checkpoint") as load,
            patch(
                "torch.distributions.Normal.sample",
                side_effect=AssertionError("RNG used"),
            ),
        ):
            result = main.demo(
                config="unused.yaml",
                model_path="unused.pt",
                episodes=2,
                seed=91,
                render_mode="none",
            )
        load.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["seed"], 91)
        self.assertIsNone(factory.call_args.kwargs["render_mode"])
        self.assertEqual(env.reset_seeds, [91, 92])
        self.assertEqual(result["returns"], [4.0, 4.0])
        self.assertEqual(result["per_agent_returns"], [[0.0, 8.0], [0.0, 8.0]])
        self.assertEqual(result["lengths"], [2, 2])
        self.assertEqual((result["mean"], result["std"]), (4.0, 0.0))
        self.assertNotIn("success_rate", result)
        self.assertEqual(env.close_calls, 1)

    def test_nonempty_terminal_and_timeout_flags_stop_demo(self):
        for mode in ("terminal_nonempty", "timeout_nonempty"):
            with self.subTest(mode=mode):
                cfg, env = config(), ParallelEnv(length=1, mode=mode)
                with stubs(cfg, env), patch("MAPPO.main.load_checkpoint"):
                    result = main.demo(config="unused.yaml", episodes=1)
                self.assertEqual(result["lengths"], [1])
                self.assertEqual(env.step_calls, 1)
                self.assertEqual(env.close_calls, 1)

    def test_demo_cap_and_stochastic_mode_and_failure_cleanup(self):
        cfg, env = config(), ParallelEnv(length=999)
        cfg.max_steps_per_episode = 1
        with (
            stubs(cfg, env),
            patch("MAPPO.main.load_checkpoint"),
            patch(
                "torch.distributions.Normal.sample",
                return_value=np_to_tensor([[0.0, 0.0], [0.0, 0.0]]),
            ) as sample,
        ):
            result = main.demo(config="unused.yaml", episodes=1, deterministic=False)
        sample.assert_called_once()
        self.assertEqual(result["lengths"], [1])
        self.assertEqual(env.close_calls, 1)
        cfg, env = config(), ParallelEnv()
        with (
            stubs(cfg, env),
            patch(
                "MAPPO.main.load_checkpoint", side_effect=RuntimeError("bad checkpoint")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "bad checkpoint"):
                main.demo(config="unused.yaml")
        self.assertEqual(env.close_calls, 1)


def np_to_tensor(value):
    import torch

    return torch.tensor(value, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
