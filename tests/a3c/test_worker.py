"""Worker integration with inline calls, stub environments, and no multiprocessing."""

from contextlib import ExitStack
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from A3C.a3c.worker import worker_process

from ._support import FakeEvent, FakeQueue, FakeValue, StubEnv, StubVecEnv
from .test_agent import _make_agent
from .test_main import _make_config


class EndpointValueModel(torch.nn.Module):
    """Value equals the first observation feature, making reset mistakes visible."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, obs):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        value = obs[:, 0] + self.weight
        return self.weight.expand(obs.shape[0], 3), value

    def act(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        count = logits.shape[0]
        return (
            torch.zeros(count, dtype=torch.long),
            self.weight.expand(count) - 0.5,
            self.weight.expand(count) * 0.0 + 0.1,
            value,
        )


class TestWorker(unittest.TestCase):
    def _run(self, env, cfg=None, *, agent=None, counter=None, event=None, worker_id=0):
        cfg = cfg or _make_config(total_steps=6, t_max=4, anneal_lr=False)
        self.agent = agent or _make_agent()
        self.counter = counter or FakeValue()
        self.event = event or FakeEvent()
        self.queue = FakeQueue()
        with ExitStack() as stack:
            self.factory = stack.enter_context(
                patch(
                    "A3C.a3c.worker.make_vec_env"
                    if cfg.num_envs > 1
                    else "A3C.a3c.worker.make_env",
                    return_value=env,
                )
            )
            stack.enter_context(patch("A3C.a3c.worker.torch.set_num_threads"))
            stack.enter_context(
                patch(
                    "A3C.a3c.worker.gym.spec",
                    return_value=SimpleNamespace(max_episode_steps=1000),
                )
            )
            try:
                worker_process(
                    worker_id, cfg, self.agent, self.counter, self.queue, self.event
                )
            finally:
                self.messages = self.queue.messages()
        return self.messages

    def test_worker_multi_env_collects_episodes_with_exact_single_and_vector_counts(
        self,
    ):
        self.assertEqual(len(inspect.signature(worker_process).parameters), 6)
        for width in (1, 2):
            with self.subTest(width=width):
                env = (
                    StubEnv(episode_length=3)
                    if width == 1
                    else StubVecEnv(episode_length=3)
                )
                messages = self._run(
                    env,
                    _make_config(
                        total_steps=6, num_envs=width, t_max=4, anneal_lr=False
                    ),
                )
                updates = [m for m in messages if m["kind"] == "update"]
                episodes = [m for m in messages if m["kind"] == "episode"]
                done = [m for m in messages if m["kind"] == "worker_done"]
                self.assertEqual(len(done), 1)
                self.assertGreater(len(episodes), 0)
                self.assertEqual(self.counter.value, 6)
                self.assertEqual(done[0]["collected_transitions"], 6)
                self.assertEqual(done[0]["reserved_transitions"], 6)
                self.assertEqual(done[0]["worker_rollout_steps"], env.step_calls)
                self.assertEqual(sum(m["rollout_steps"] for m in updates), 6)
                self.assertEqual(done[0]["optimizer_steps"], len(updates))
                self.assertTrue(all(m["update_applied"] for m in updates))
                self.assertFalse(self.event.is_set())
                self.assertTrue(env.close_called)

    def test_reservation_precedes_step_and_sibling_budget_exhaustion_flushes_rollout(
        self,
    ):
        counter = FakeValue()
        env = StubEnv(episode_length=100)

        def sibling_reserves_remaining_budget():
            self.assertEqual(counter.value, 1)
            counter.value = 4

        env.on_step = sibling_reserves_remaining_budget
        messages = self._run(
            env, _make_config(total_steps=4, t_max=4, anneal_lr=False), counter=counter
        )
        done = messages[-1]
        self.assertEqual(env.step_calls, 1)
        self.assertEqual(done["collected_transitions"], 1)
        self.assertEqual(done["reserved_transitions"], 1)
        self.assertEqual(done["optimizer_steps"], 1)
        self.assertTrue(done["budget_exhausted"])
        self.assertFalse(self.event.is_set())

    def test_worker_rank_uses_distinct_vector_seed_list_and_single_initial_reset(self):
        env = StubVecEnv(episode_length=100)
        cfg = _make_config(num_envs=2, num_workers=2, total_steps=2, anneal_lr=False)
        self._run(env, cfg, worker_id=1)
        self.assertEqual(self.factory.call_args.kwargs["seed"], cfg.seed + 2)
        self.assertEqual(env.reset_seeds, [[cfg.seed + 2, cfg.seed + 3]])

    def test_scalar_and_vector_episode_reports_separate_native_and_learning_returns(
        self,
    ):
        for width in (1, 2):
            with self.subTest(width=width):
                env = (
                    StubEnv(reward=0.01, native_reward=1.0)
                    if width == 1
                    else StubVecEnv(reward=0.01, native_reward=1.0)
                )
                messages = self._run(
                    env,
                    _make_config(
                        total_steps=4,
                        num_envs=width,
                        reward_scale=0.01,
                        anneal_lr=False,
                    ),
                )
                episodes = [
                    message for message in messages if message["kind"] == "episode"
                ]
                self.assertEqual(len(episodes), 2)
                for report in episodes:
                    self.assertEqual(report["episode_return"], 2.0)
                    self.assertEqual(report["native_episode_return"], 2.0)
                    self.assertAlmostEqual(
                        report["learning_episode_return"], 0.02, places=6
                    )
                    self.assertTrue(report["terminated"])
                    self.assertFalse(report["truncated"])
                    self.assertEqual(report["episode_length"], 2)

    def test_timeout_bootstrap_uses_actual_final_obs_not_samestep_reset_observation(
        self,
    ):
        cases = [
            (1, None, True),
            (2, "final_obs", True),
            (2, "final_observation", True),
            (2, "final_obs", False),
        ]
        for width, final_key, timeout in cases:
            with self.subTest(width=width, final_key=final_key, timeout=timeout):
                env = (
                    StubEnv(episode_length=1, reward=1.0, truncated=timeout)
                    if width == 1
                    else StubVecEnv(
                        episode_length=1,
                        reward=1.0,
                        truncated=timeout,
                        final_key=final_key,
                    )
                )
                agent = _make_agent()
                with (
                    patch.object(
                        agent, "new_local_model", return_value=EndpointValueModel()
                    ),
                    patch.object(agent, "apply_gradients", return_value=True),
                    patch.object(
                        agent, "compute_loss", wraps=agent.compute_loss
                    ) as loss,
                ):
                    self._run(
                        env,
                        _make_config(
                            total_steps=width,
                            num_envs=width,
                            t_max=1,
                            gamma=0.5,
                            anneal_lr=False,
                        ),
                        agent=agent,
                    )
                targets = loss.call_args.kwargs["returns"]
                torch.testing.assert_close(
                    targets, torch.full((width,), 1.5 if timeout else 1.0)
                )

    def test_reset_step_and_local_model_errors_close_env_and_report_failure(self):
        for phase in ("reset", "step", "model"):
            with self.subTest(phase=phase):
                env = StubEnv(fail_reset=phase == "reset", fail_step=phase == "step")
                agent = _make_agent()
                with ExitStack() as stack:
                    if phase == "model":
                        stack.enter_context(
                            patch.object(
                                agent,
                                "new_local_model",
                                side_effect=RuntimeError("stub local model failed"),
                            )
                        )
                    with self.assertRaisesRegex(RuntimeError, "stub"):
                        self._run(env, agent=agent)
                self.assertTrue(env.close_called)
                self.assertTrue(self.event.is_set())
                errors = [m for m in self.messages if m["kind"] == "worker_error"]
                self.assertEqual(len(errors), 1)
                self.assertIn("RuntimeError", errors[0]["traceback"])
                self.assertFalse(any(m["kind"] == "worker_done" for m in self.messages))

    def test_missing_timeout_endpoint_and_nonfinite_reward_are_explicit_errors(self):
        for missing_endpoint in (False, True):
            with self.subTest(missing_endpoint=missing_endpoint):
                if missing_endpoint:
                    env = StubVecEnv(
                        episode_length=1, truncated=True, final_key="not_an_endpoint"
                    )
                    cfg = _make_config(num_envs=2, total_steps=2, anneal_lr=False)
                else:
                    env = StubEnv(reward=float("nan"))
                    cfg = _make_config(total_steps=1, anneal_lr=False)
                with self.assertRaises((ValueError, RuntimeError)):
                    self._run(env, cfg)
                self.assertTrue(env.close_called)
                self.assertTrue(self.event.is_set())
                self.assertEqual(self.messages[-1]["kind"], "worker_error")
                self.assertEqual(self.messages[-1]["optimizer_steps"], 0)

    def test_rejected_gradient_update_is_not_counted_as_optimizer_success(self):
        agent = _make_agent()
        env = StubEnv(episode_length=100)
        with patch.object(agent, "apply_gradients", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "rejected"):
                self._run(
                    env, _make_config(total_steps=1, anneal_lr=False), agent=agent
                )
        self.assertTrue(env.close_called)
        self.assertEqual(self.messages[-1]["kind"], "worker_error")
        self.assertEqual(self.messages[-1]["collected_transitions"], 1)
        self.assertEqual(self.messages[-1]["optimizer_steps"], 0)
        self.assertFalse(any(m["kind"] == "update" for m in self.messages))
