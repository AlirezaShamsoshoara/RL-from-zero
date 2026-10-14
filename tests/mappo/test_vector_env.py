"""Native vector-worker protocol tests with fake pipes and no child processes."""

import copy
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from gymnasium import spaces

from MAPPO.mappo.config import Config
import MAPPO.mappo.vector_env as vector


class WorkerEnv:
    def __init__(self, mode="mixed"):
        self.possible_agents = ["walker_0", "walker_1", "walker_2"]
        self.agents = self.possible_agents.copy()
        self.observation = spaces.Box(-20.0, 20.0, (2,), dtype=np.float32)
        self.action = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.engine = None
        self.mode = mode
        self.steps = self.close_calls = 0
        self.action_keys = []

    def observation_space(self, name):
        return self.observation

    def action_space(self, name):
        return self.action

    def reset(self, seed=None):
        self.steps = 0
        self.agents = self.possible_agents.copy()
        return {name: np.zeros(2, dtype=np.float32) for name in self.agents}, {}

    def step(self, actions):
        if set(actions) != set(self.agents) or not actions:
            raise AssertionError("invalid active-agent dispatch")
        self.action_keys.append(set(actions))
        self.steps += 1
        if self.mode == "error":
            raise RuntimeError("fake environment failure")
        active = self.agents.copy()
        rewards = {
            name: float(index + 1)
            for index, name in enumerate(self.possible_agents)
            if name in active
        }
        terms, truncs = dict.fromkeys(active, False), dict.fromkeys(active, False)
        obs = {name: np.full(2, self.steps, dtype=np.float32) for name in active}
        infos = {}
        if self.mode == "mixed":
            if self.steps == 1:
                rewards["walker_0"] = -10.0
                terms["walker_0"] = True
                obs["walker_0"] = np.full(2, 9.0, dtype=np.float32)
                self.agents = ["walker_1", "walker_2"]
            else:
                terms = dict.fromkeys(active, True)
                self.agents = []
        elif self.mode == "timeout":
            truncs = dict.fromkeys(active, True)
            infos = {
                name: {"final_obs": np.full(2, index + 7, dtype=np.float32)}
                for index, name in enumerate(self.possible_agents)
            }
            obs = {}
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def close(self):
        self.close_calls += 1


class TestVectorWorkerProtocol(unittest.TestCase):
    def run_worker(self, env, steps, cap=10):
        cfg = Config()
        cfg.n_agents = cfg.n_walkers = 3
        cfg.max_steps_per_episode = cap
        connection = MagicMock()
        connection.recv.side_effect = [
            {"operation": "step", "actions": np.zeros((3, 2), dtype=np.float32)}
            for _ in range(steps)
        ] + [EOFError("fake parent closed")]
        messages = []
        connection.send.side_effect = lambda message: messages.append(
            copy.deepcopy(message)
        )
        with patch.dict(os.environ), patch("MAPPO.main.torch.set_num_threads"):
            vector._worker(
                connection, 0, cfg, 17, lambda cfg, seed, render_mode=None: env
            )
        connection.close.assert_called_once()
        return messages

    def test_worker_removal_masks_stable_original_order_and_zeroed_dead_endpoints(self):
        env = WorkerEnv()
        messages = self.run_worker(env, 2)
        replies = [
            message["payload"] for message in messages if message["operation"] == "step"
        ]
        self.assertEqual(len(replies), 2)
        np.testing.assert_array_equal(replies[0]["alive"], [1, 1, 1])
        np.testing.assert_array_equal(replies[0]["terminations"], [1, 0, 0])
        np.testing.assert_array_equal(replies[0]["next_obs"][0], [0, 0])
        np.testing.assert_array_equal(replies[1]["alive"], [0, 1, 1])
        np.testing.assert_array_equal(replies[1]["terminations"], [1, 1, 1])
        self.assertEqual(
            env.action_keys,
            [{"walker_0", "walker_1", "walker_2"}, {"walker_1", "walker_2"}],
        )
        self.assertEqual(replies[1]["episode"]["agent_transitions"], 5)
        np.testing.assert_array_equal(
            replies[1]["episode"]["per_agent_returns"], [-10.0, 4.0, 6.0]
        )
        self.assertIsNone(replies[1]["episode"]["delivered"])
        self.assertEqual(env.close_calls, 1)

    def test_timeout_and_cap_keep_pre_reset_bootstrap_separate_from_next_interaction(
        self,
    ):
        for mode, cap in (("timeout", 10), ("cap", 1)):
            with self.subTest(mode=mode):
                env = WorkerEnv(mode)
                messages = self.run_worker(env, 1, cap=cap)
                reply = next(
                    message["payload"]
                    for message in messages
                    if message["operation"] == "step"
                )
                np.testing.assert_array_equal(reply["obs"], np.zeros((3, 2)))
                expected = (
                    np.repeat(np.arange(7, 10)[:, None], 2, axis=1)
                    if mode == "timeout"
                    else np.ones((3, 2))
                )
                np.testing.assert_array_equal(reply["next_obs"], expected)
                np.testing.assert_array_equal(reply["terminations"], [0, 0, 0])
                np.testing.assert_array_equal(reply["dones"], [1, 1, 1])
                self.assertEqual(reply["episode"]["truncated"], mode == "timeout")
                self.assertEqual(reply["episode"]["capped"], mode == "cap")
                self.assertEqual(env.close_calls, 1)

    def test_worker_eof_and_environment_failure_close_environment_without_reuse(self):
        for mode, steps in (("mixed", 0), ("error", 1)):
            with self.subTest(mode=mode):
                env = WorkerEnv(mode)
                messages = self.run_worker(env, steps)
                self.assertEqual(env.close_calls, 1)
                if mode == "error":
                    self.assertFalse(messages[-1]["ok"])
                    self.assertIn("fake environment failure", messages[-1]["error"])
                else:
                    self.assertEqual(len(messages), 1)
                    self.assertTrue(messages[0]["ok"])

    def test_parent_pool_eof_closes_all_fake_connections(self):
        pool = object.__new__(vector.ParallelMultiwalker)
        pool._closed = False
        pool._rpc_lock = threading.RLock()
        pool.num_envs = 2
        pool.step_timeout = 1.0
        first, second = MagicMock(), MagicMock()
        pool._connections = [first, second]
        pool._processes = [SimpleNamespace(exitcode=1), SimpleNamespace(exitcode=0)]
        pool.close = MagicMock()
        first.recv.side_effect = EOFError("fake worker EOF")
        second.recv.return_value = {
            "ok": True,
            "rank": 1,
            "operation": "step",
            "payload": {},
        }
        with patch.object(
            vector,
            "wait",
            side_effect=lambda connections, timeout=None: list(connections),
        ):
            with self.assertRaisesRegex(vector.VectorWorkerError, "EOFError"):
                pool._request_many("step", [{}, {}])
        pool.close.assert_called_once()
        for connection in (first, second):
            self.assertEqual(
                connection.send.call_args_list[0].args[0], {"operation": "step"}
            )
            self.assertEqual(
                connection.send.call_args_list[-1].args[0], {"operation": "close"}
            )
            self.assertEqual(connection.send.call_count, 2)


if __name__ == "__main__":
    unittest.main()
