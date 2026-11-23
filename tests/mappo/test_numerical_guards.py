"""Unsafe PPO minibatches must never mutate model or Adam state."""

import copy
import unittest
from unittest.mock import patch

import torch

from MAPPO.mappo.agent import Batch, MAPPOAgent


def learner(max_grad_norm=0.5, separate=True):
    return MAPPOAgent(
        2,
        2,
        2,
        4,
        [4],
        [4],
        "tanh",
        0.001,
        0.2,
        0.005,
        0.5,
        max_grad_norm,
        "cpu",
        share_policy=True,
        action_type="continuous",
        critic_learning_rate=0.002,
        separate_grad_clipping=separate,
    )


def batches(agent):
    result = []
    for index, model in enumerate(agent.models):
        obs = torch.tensor([[0.1, 0.2], [0.3, 0.4]]) + index * 0.1
        state = torch.ones((2, 4))
        with torch.no_grad():
            actions, logprobs, _ = model.act(obs, deterministic=True)
            values = model.get_value(state)
        advantages = torch.tensor([-1.0, 1.0])
        result.append(
            Batch(
                obs,
                state,
                actions.detach(),
                logprobs.detach(),
                values.detach() + advantages,
                advantages,
                values.detach(),
                torch.ones(2),
            )
        )
    return result


class TestNumericalUpdateGuards(unittest.TestCase):
    def assert_nested_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def assert_skipped_unchanged(self, agent, bad_batches):
        model_before = copy.deepcopy(agent.models[0].state_dict())
        adam_before = copy.deepcopy(agent.optimizers[0].state_dict())
        self.assertTrue(
            adam_before["state"], "prime Adam before testing unchanged moments"
        )
        with patch.object(
            agent.optimizers[0], "step", wraps=agent.optimizers[0].step
        ) as step:
            stats, count = agent.update(bad_batches)
        step.assert_not_called()
        self.assertEqual(count, 0)
        self.assertEqual(stats["stats/skipped_updates"], 1.0)
        self.assert_nested_equal(model_before, agent.models[0].state_dict())
        self.assert_nested_equal(adam_before, agent.optimizers[0].state_dict())
        for parameter in agent.models[0].parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_extreme_positive_advantage_log_ratio_is_rejected_before_exponentiation(
        self,
    ):
        agent = learner()
        agent.update(batches(agent))
        bad = batches(agent)
        # Only a positive-advantage row overflows, so the selected clipped forward
        # loss can remain finite while autograd sees 0 * inf in the other branch.
        ratio = torch.exp(torch.tensor(1000.0))
        clipped_loss = torch.maximum(-ratio, -ratio.clamp(0.8, 1.2))
        self.assertTrue(torch.isfinite(clipped_loss).item())
        bad[0].logprobs[1] = -1000.0
        original_exp = torch.exp

        def guarded_exp(value, *args, **kwargs):
            if torch.any(value.abs() > 20):
                raise AssertionError("unsafe log ratio reached exp")
            return original_exp(value, *args, **kwargs)

        with patch("torch.exp", side_effect=guarded_exp):
            self.assert_skipped_unchanged(agent, bad)

    def test_nonfinite_policy_logprob_or_critic_value_skips_without_adam_mutation(self):
        for failure in ("logprob", "value"):
            with self.subTest(failure=failure):
                agent = learner()
                agent.update(batches(agent))
                bad = batches(agent)
                model = agent.models[0]
                original_evaluate, original_value = (
                    model.evaluate_actions,
                    model.get_value,
                )

                def nonfinite_evaluate(obs, actions):
                    logprob, entropy = original_evaluate(obs, actions)
                    return logprob * float("nan"), entropy

                def nonfinite_value(state):
                    return original_value(state) * float("nan")

                target = "evaluate_actions" if failure == "logprob" else "get_value"
                callback = (
                    nonfinite_evaluate if failure == "logprob" else nonfinite_value
                )
                with patch.object(model, target, side_effect=callback):
                    self.assert_skipped_unchanged(agent, bad)

    def test_unrelated_clip_runtime_error_propagates_without_mutating_model_or_adam(
        self,
    ):
        agent = learner()
        agent.update(batches(agent))
        bad = batches(agent)
        model_before = copy.deepcopy(agent.models[0].state_dict())
        adam_before = copy.deepcopy(agent.optimizers[0].state_dict())
        with (
            patch(
                "torch.nn.utils.clip_grad_norm_",
                side_effect=RuntimeError("out of memory"),
            ),
            patch.object(
                agent.optimizers[0], "step", wraps=agent.optimizers[0].step
            ) as step,
        ):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                agent.update(bad)
        step.assert_not_called()
        self.assert_nested_equal(model_before, agent.models[0].state_dict())
        self.assert_nested_equal(adam_before, agent.optimizers[0].state_dict())
        for parameter in agent.models[0].parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_nan_gradient_hook_is_skipped_with_or_without_gradient_clipping(self):
        for maximum, separate in ((0.5, True), (0.5, False), (0.0, True)):
            with self.subTest(maximum=maximum, separate=separate):
                agent = learner(max_grad_norm=maximum, separate=separate)
                original_clip = torch.nn.utils.clip_grad_norm_
                with patch(
                    "torch.nn.utils.clip_grad_norm_", wraps=original_clip
                ) as clip:
                    agent.update(batches(agent))
                self.assertEqual(clip.call_count, 2 if separate else 1)
                for call in clip.call_args_list:
                    self.assertIs(call.kwargs["error_if_nonfinite"], True)
                    self.assertEqual(
                        call.args[1], maximum if maximum > 0 else float("inf")
                    )
                bad = batches(agent)
                handle = agent.models[0].actor.mu.bias.register_hook(
                    lambda gradient: torch.full_like(gradient, float("nan"))
                )
                try:
                    self.assert_skipped_unchanged(agent, bad)
                finally:
                    handle.remove()


if __name__ == "__main__":
    unittest.main()
