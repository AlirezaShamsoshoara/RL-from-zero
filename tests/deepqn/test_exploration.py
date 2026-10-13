"""Bounded held-action exploration tests; inference and replay stay unchanged."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from gymnasium import spaces

from deepQN.dqn.agent import DQNAgent
from deepQN.dqn.config import Config
from deepQN.dqn.exploration import ActionHoldExplorer
from deepQN.dqn.replay_buffer import ReplayBuffer
import deepQN.main as main


def agent():
    return DQNAgent(
        spaces.Box(-10.0, 10.0, (2,), dtype=np.float32),
        spaces.Discrete(3),
        [],
        "relu",
        0.001,
        0.5,
        100,
        1.0,
        False,
        "cpu",
    )


class DummyTqdm:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass


class BoundaryEnv:
    def __init__(self, end):
        self.observation_space = spaces.Box(-10.0, 10.0, (2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.end = end
        self.actions = []
        self.close_calls = 0

    def reset(self, seed=None):
        return np.array([-0.5, 0.01], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action)
        return (
            np.array([-0.4, 0.02], dtype=np.float32),
            -1.0,
            self.end == "terminated",
            self.end == "truncated",
            {},
        )

    def close(self):
        self.close_calls += 1


class TestActionHoldExplorer(unittest.TestCase):
    def test_start_probability_targets_random_transition_fraction_not_pulse_fraction(
        self,
    ):
        explorer = ActionHoldExplorer(20, 60)
        for epsilon in (0.0, 0.02, 0.5, 1.0):
            with self.subTest(epsilon=epsilon):
                probability = explorer.start_probability(epsilon)
                self.assertAlmostEqual(
                    probability, epsilon / (40.0 * (1.0 - epsilon) + epsilon)
                )
                fraction = (
                    probability * 40.0 / ((1.0 - probability) + probability * 40.0)
                )
                self.assertAlmostEqual(fraction, epsilon)
        self.assertLess(explorer.start_probability(0.02), 0.001)
        for epsilon in (-0.1, 1.1, np.nan):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                explorer.start_probability(epsilon)

    def test_default_explorer_delegates_exactly_without_extra_rng(self):
        learner = MagicMock()
        learner.act.return_value = 2
        obs = np.array([0.1, 0.0], dtype=np.float32)
        with (
            patch("numpy.random.rand", side_effect=AssertionError("extra RNG")),
            patch("numpy.random.randint", side_effect=AssertionError("extra RNG")),
        ):
            result = ActionHoldExplorer().act(learner, obs, 0.2)
        self.assertEqual(result, (2, None))
        learner.act.assert_called_once_with(obs, epsilon=0.2)

    def test_held_actions_validate_observations_count_steps_and_reset_without_resampling(
        self,
    ):
        learner = agent()
        explorer = ActionHoldExplorer(3, 3)
        obs = np.array([0.1, 0.0], dtype=np.float32)
        with (
            patch("numpy.random.rand", return_value=0.0) as rand,
            patch("numpy.random.randint", return_value=3) as duration,
            patch.object(learner.act_space, "sample", side_effect=[2, 1]) as sample,
        ):
            self.assertEqual(explorer.act(learner, obs, 0.5), (2, True))
            self.assertEqual((learner.step_count, explorer.remaining), (1, 2))
            with self.assertRaises(ValueError):
                explorer.act(learner, np.array([np.nan, 0.0]), 0.5)
            self.assertEqual((learner.step_count, explorer.remaining), (1, 2))
            self.assertEqual(explorer.act(learner, obs, 0.5), (2, True))
            self.assertEqual(explorer.act(learner, obs, 0.5), (2, True))
            self.assertEqual((learner.step_count, explorer.remaining), (3, 0))
            self.assertEqual(sample.call_count, 1)
            self.assertEqual(
                rand.call_count, 2
            )  # Pulse start plus delegated random action.
            duration.assert_called_once_with(3, 4)
            explorer.reset()
            self.assertIsNone(explorer.action)
            self.assertEqual(explorer.remaining, 0)
            self.assertEqual(explorer.act(learner, obs, 0.5), (1, True))
        self.assertEqual(learner.step_count, 4)
        self.assertEqual(sample.call_count, 2)

    def test_hold_bounds_and_config_order_are_validated_with_baseline_defaults(self):
        cfg = Config()
        self.assertEqual((cfg.exploration_hold_min, cfg.exploration_hold_max), (1, 1))
        for low, high in ((0, 1), (-1, 3), (3, 2), (True, 3), (1.5, 3)):
            with self.subTest(low=low, high=high):
                with self.assertRaises(ValueError):
                    ActionHoldExplorer(low, high)
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "config.yaml")
                    with open(path, "w", encoding="utf-8") as stream:
                        stream.write(
                            f"exploration_hold_min: {low}\nexploration_hold_max: {high}\n"
                        )
                    with self.assertRaises((ValueError, TypeError)):
                        Config.from_yaml(path)

    def test_main_resets_pulses_on_terminal_timeout_and_cap_with_raw_replay(self):
        for end in ("terminated", "truncated", "cap"):
            with self.subTest(end=end):
                cfg = Config()
                cfg.env_id = "Generic-v0"
                cfg.device = "cpu"
                cfg.hidden_sizes = []
                cfg.total_steps = 3
                cfg.learning_starts = 0
                cfg.epsilon_start = cfg.epsilon_end = 1.0
                cfg.exploration_hold_min = cfg.exploration_hold_max = 3
                cfg.max_steps_per_episode = 1
                cfg.batch_size, cfg.buffer_size = 4, 20
                cfg.log_interval = 2
                cfg.checkpoint_interval = 100
                cfg.save_best = False
                cfg.log_to_console = cfg.log_to_file = False
                env, learner = BoundaryEnv(end), agent()
                buffer = ReplayBuffer(2, 20, torch.device("cpu"))
                wandb = MagicMock()
                wandb.init.return_value.summary = {}
                with (
                    patch.dict(os.environ, {"WANDB_API_KEY": ""}),
                    patch("deepQN.main.Config.from_yaml", return_value=cfg),
                    patch("deepQN.main.make_env", return_value=env),
                    patch("deepQN.main.DQNAgent", return_value=learner),
                    patch("deepQN.main.ReplayBuffer", return_value=buffer),
                    patch("deepQN.main.wandb", wandb),
                    patch("deepQN.main.setup_logger", return_value=MagicMock()),
                    patch("deepQN.main.tqdm", DummyTqdm),
                    patch("deepQN.main.save_checkpoint"),
                    patch("numpy.random.rand", return_value=0.0),
                    patch("numpy.random.randint", return_value=3),
                    patch.object(
                        learner.act_space, "sample", side_effect=[0, 1, 2]
                    ) as sample,
                ):
                    main.train(config="unused.yaml")
                self.assertEqual(env.actions, [0, 1, 2])
                self.assertEqual(sample.call_count, 3)
                self.assertEqual(learner.step_count, 3)
                np.testing.assert_allclose(buffer.obs[:3], [[-0.5, 0.01]] * 3)
                np.testing.assert_allclose(buffer.next_obs[:3], [[-0.4, 0.02]] * 3)
                np.testing.assert_array_equal(buffer.rewards[:3], [-1.0] * 3)
                np.testing.assert_array_equal(
                    buffer.dones[:3], [float(end == "terminated")] * 3
                )
                for call in wandb.log.call_args_list:
                    self.assertEqual(
                        call.args[0]["charts/exploratory_step_fraction"], 1.0
                    )
                self.assertEqual(env.close_calls, 1)
                wandb.init.return_value.finish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
