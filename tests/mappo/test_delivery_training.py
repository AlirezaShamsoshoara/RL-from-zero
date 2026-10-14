"""Focused batched-policy, normalization and optimizer-option regressions."""

import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from MAPPO.mappo.agent import Batch, MAPPOAgent
from MAPPO.mappo.config import Config
from MAPPO.mappo.normalization import RunningNorm
from MAPPO.mappo.utils import load_checkpoint, save_checkpoint


def learner(shared=True, centralized=True, **options):
    return MAPPOAgent(
        3,
        2,
        2,
        6,
        [4],
        [4],
        "tanh",
        0.001,
        0.2,
        0.01,
        0.5,
        0.5,
        "cpu",
        share_policy=shared,
        use_centralized_critic=centralized,
        action_type="continuous",
        **options,
    )


def rollout_batches(agent):
    results = []
    for index, model in enumerate(agent.models):
        obs = torch.ones((2, 2)) * (index + 1)
        state = torch.ones((2, 6))
        with torch.no_grad():
            actions, logprobs, _ = model.act(obs, deterministic=True)
            values = model.get_value(state if agent.use_centralized_critic else obs)
        advantages = torch.tensor([-1.0, 1.0])
        results.append(
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
    return results


class TestDeliveryTrainingOptions(unittest.TestCase):
    def test_batched_deterministic_actions_and_values_match_single_team_queries_without_rng(
        self,
    ):
        observations = np.arange(12, dtype=np.float32).reshape(2, 3, 2) / 10.0
        for shared in (True, False):
            for centralized in (True, False):
                with self.subTest(shared=shared, centralized=centralized):
                    agent = learner(shared, centralized, normalize_inputs=True)
                    agent.update_normalizers(
                        observations[None], np.ones((1, 2, 3), dtype=bool)
                    )
                    with (
                        patch(
                            "torch.distributions.Normal.sample",
                            side_effect=AssertionError("RNG used"),
                        ),
                        patch(
                            "torch.distributions.Normal.rsample",
                            side_effect=AssertionError("RNG used"),
                        ),
                    ):
                        batch = agent.act_many(observations, deterministic=True)
                        values = agent.values_many(observations)
                        singles = [
                            agent.act(obs, obs.reshape(-1), deterministic=True)
                            for obs in observations
                        ]
                    self.assertEqual(batch[0].shape, (2, 3, 2))
                    self.assertEqual(batch[1].shape, (2, 3))
                    self.assertEqual(values.shape, (2, 3))
                    for field in range(3):
                        np.testing.assert_allclose(
                            batch[field],
                            np.stack([result[field] for result in singles]),
                            rtol=1e-6,
                            atol=1e-6,
                        )
                    np.testing.assert_allclose(values, batch[2], rtol=1e-6, atol=1e-6)

    def test_running_normalizer_population_statistics_freeze_and_invalid_samples(self):
        norm = RunningNorm(2)
        prior = float(norm.count)
        samples = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
        norm.update(samples)
        total = prior + 2.0
        sample_mean = np.array([2.0, 4.0])
        sample_var = np.array([1.0, 4.0])
        expected_mean = sample_mean * 2.0 / total
        expected_var = (
            prior + 2.0 * sample_var + sample_mean**2 * prior * 2.0 / total
        ) / total
        np.testing.assert_allclose(
            norm.mean.numpy(), expected_mean, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(norm.var.numpy(), expected_var, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(float(norm.count), total)
        before = copy.deepcopy(norm.state_dict())
        norm.train()
        self.assertTrue(torch.isfinite(norm(torch.tensor([[1e6, -1e6]]))).all().item())
        norm.update(torch.empty((0, 2)))
        with self.assertRaises(ValueError):
            norm.update(torch.tensor([[float("nan"), 1.0]]))
        for key, value in before.items():
            torch.testing.assert_close(norm.state_dict()[key], value, rtol=0, atol=0)

    def test_normalizer_updates_use_alive_observations_and_queries_do_not_mutate_buffers(
        self,
    ):
        agent = learner(normalize_inputs=True)
        observations = np.array(
            [[[[1, 2], [3, 4], [0, 0]]], [[[5, 6], [0, 0], [7, 8]]]], dtype=np.float32
        )
        alive = np.array([[[1, 1, 0]], [[1, 0, 1]]], dtype=bool)
        agent.update_normalizers(observations, alive)
        model = agent.models[0]
        self.assertAlmostEqual(float(model.obs_norm.count), 4.0001)
        self.assertAlmostEqual(float(model.state_norm.count), 2.0001)
        np.testing.assert_allclose(
            model.obs_norm.mean.numpy(), [4.0, 5.0], atol=2e-4, rtol=0
        )
        before = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if "norm." in key
        }
        query = observations[0]
        agent.act_many(query, deterministic=True)
        agent.values_many(query)
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_normalizer_checkpoint_roundtrip_and_disabled_legacy_state_keys(self):
        normalized = learner(normalize_inputs=True)
        observations = np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2)
        normalized.update_normalizers(observations, np.ones((1, 2, 3), dtype=bool))
        restored = learner(normalize_inputs=True)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "normalized.pt")
            save_checkpoint(
                path, normalized.models, normalized.optimizers, 2, 1.0, True
            )
            load_checkpoint(path, restored.models, restored.optimizers)
        for key, value in normalized.models[0].state_dict().items():
            torch.testing.assert_close(
                restored.models[0].state_dict()[key], value, rtol=0, atol=0
            )
        np.testing.assert_allclose(
            normalized.values_many(observations[0]),
            restored.values_many(observations[0]),
        )
        raw = learner()
        self.assertIsInstance(raw.models[0].obs_norm, torch.nn.Identity)
        self.assertFalse(any("norm." in key for key in raw.models[0].state_dict()))
        learner().models[0].load_state_dict(raw.models[0].state_dict(), strict=True)

    def test_normalized_and_raw_checkpoint_modes_reject_each_other_before_weight_restore(
        self,
    ):
        for normalized in (True, False):
            with (
                self.subTest(normalized=normalized),
                tempfile.TemporaryDirectory() as directory,
            ):
                source = learner(normalize_inputs=normalized)
                destination = learner(normalize_inputs=not normalized)
                before = copy.deepcopy(destination.models[0].state_dict())
                path = os.path.join(directory, "mode.pt")
                save_checkpoint(path, source.models, source.optimizers, 1, 0.0, True)
                with self.assertRaisesRegex(ValueError, "normaliz"):
                    load_checkpoint(path, destination.models, destination.optimizers)
                for key, value in before.items():
                    torch.testing.assert_close(
                        destination.models[0].state_dict()[key], value, rtol=0, atol=0
                    )

    def test_actor_critic_learning_rates_have_disjoint_groups_without_extra_optimizers(
        self,
    ):
        for shared in (True, False):
            with self.subTest(shared=shared):
                agent = learner(shared=shared, critic_learning_rate=0.002)
                self.assertEqual(len(agent.optimizers), 1 if shared else 3)
                for model, optimizer in zip(agent.models, agent.optimizers):
                    self.assertEqual(
                        [group["lr"] for group in optimizer.param_groups],
                        [0.001, 0.002],
                    )
                    actor = {
                        id(parameter)
                        for parameter in optimizer.param_groups[0]["params"]
                    }
                    critic = {
                        id(parameter)
                        for parameter in optimizer.param_groups[1]["params"]
                    }
                    self.assertFalse(actor & critic)
                    self.assertEqual(
                        actor, {id(parameter) for parameter in model.actor.parameters()}
                    )
                    self.assertEqual(
                        critic,
                        {id(parameter) for parameter in model.critic.parameters()},
                    )
                    self.assertEqual(
                        actor | critic,
                        {id(parameter) for parameter in model.parameters()},
                    )

    def test_separate_gradient_clips_actor_and_critic_independently(self):
        agent = learner(critic_learning_rate=0.002, separate_grad_clipping=True)
        scopes = []

        def capture(parameters, maximum, **kwargs):
            self.assertIs(kwargs.get("error_if_nonfinite"), True)
            scopes.append(({id(parameter) for parameter in parameters}, maximum))
            return torch.tensor(0.0)

        with patch("torch.nn.utils.clip_grad_norm_", side_effect=capture):
            stats, steps = agent.update(rollout_batches(agent))
        self.assertEqual(steps, 1)
        self.assertEqual(
            scopes,
            [
                (
                    {id(parameter) for parameter in agent.models[0].actor.parameters()},
                    0.5,
                ),
                (
                    {
                        id(parameter)
                        for parameter in agent.models[0].critic.parameters()
                    },
                    0.5,
                ),
            ],
        )
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))

    def test_unclipped_value_loss_is_plain_half_mse_and_default_remains_clipped(self):
        for clipped, expected in ((False, 2.0), (True, 3.92)):
            with self.subTest(clipped=clipped):
                agent = learner(clip_value_loss=clipped)
                with torch.no_grad():
                    for parameter in agent.models[0].critic.parameters():
                        parameter.zero_()
                    agent.models[0].critic.v.bias.fill_(1.0)
                batches = rollout_batches(agent)
                for batch in batches:
                    batch.values.zero_()
                    batch.returns.fill_(3.0)
                stats, _ = agent.update(batches)
                self.assertAlmostEqual(stats["loss/value"], expected, places=5)
        self.assertTrue(learner().clip_value_loss)

    def test_new_config_flags_counts_and_critic_learning_rate_validation(self):
        defaults = Config()
        self.assertEqual(defaults.num_envs, 1)
        self.assertIsNone(defaults.critic_learning_rate)
        self.assertFalse(defaults.normalize_inputs)
        self.assertFalse(defaults.separate_grad_clipping)
        self.assertTrue(defaults.clip_value_loss)
        for text in (
            "num_envs: 0\n",
            "critic_learning_rate: 0\n",
            "normalize_inputs: 1\n",
            "separate_grad_clipping: 'true'\n",
            "clip_value_loss: 1\n",
        ):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "bad.yaml")
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write(text)
                with self.assertRaises(ValueError):
                    Config.from_yaml(path)


if __name__ == "__main__":
    unittest.main()
