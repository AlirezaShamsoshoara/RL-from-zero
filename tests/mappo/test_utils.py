"""MAPPO utility tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/mappo/test_utils.py
    >>> # Run with unittest
    >>> # python -m unittest tests.mappo.test_utils
"""

import os
import random
import sys
import tempfile
import types
import unittest

import numpy as np
import torch


def _ensure_pettingzoo_stub() -> None:
    try:
        import pettingzoo.sisl.multiwalker_v9  # noqa: F401
        return
    except Exception:
        pettingzoo = types.ModuleType("pettingzoo")
        sisl = types.ModuleType("pettingzoo.sisl")
        multiwalker_v9 = types.ModuleType("pettingzoo.sisl.multiwalker_v9")

        def _parallel_env(**kwargs):
            raise RuntimeError("pettingzoo not available in tests")

        multiwalker_v9.parallel_env = _parallel_env
        sisl.multiwalker_v9 = multiwalker_v9
        pettingzoo.sisl = sisl
        sys.modules["pettingzoo"] = pettingzoo
        sys.modules["pettingzoo.sisl"] = sisl
        sys.modules["pettingzoo.sisl.multiwalker_v9"] = multiwalker_v9


_ensure_pettingzoo_stub()

from MAPPO.mappo import utils as mappo_utils


class DummyObsSpace:
    def __init__(self, shape):
        self.shape = shape


class DummyActSpace:
    def __init__(self, n):
        self.n = n


class DummyEnv:
    def __init__(self, obs_dim: int, act_dim: int, n_agents: int):
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self._obs_dim = obs_dim
        self._act_dim = act_dim

    def observation_space(self, agent_id):
        return DummyObsSpace((self._obs_dim,))

    def action_space(self, agent_id):
        return DummyActSpace(self._act_dim)


class TestUtils(unittest.TestCase):
    def test_set_seed_reproducible(self) -> None:
        mappo_utils.set_seed(123)
        vals1 = (random.random(), np.random.rand(), float(torch.rand(1).item()))

        mappo_utils.set_seed(123)
        vals2 = (random.random(), np.random.rand(), float(torch.rand(1).item()))

        self.assertAlmostEqual(vals1[0], vals2[0], places=7)
        self.assertAlmostEqual(vals1[1], vals2[1], places=7)
        self.assertAlmostEqual(vals1[2], vals2[2], places=7)

    def test_get_space_dims(self) -> None:
        env = DummyEnv(obs_dim=4, act_dim=3, n_agents=2)
        obs_dim, act_dim, state_dim, n_agents = mappo_utils.get_space_dims(env)

        self.assertEqual(obs_dim, 4)
        self.assertEqual(act_dim, 3)
        self.assertEqual(state_dim, 8)
        self.assertEqual(n_agents, 2)

    def test_save_and_load_checkpoint_round_trip(self) -> None:
        torch.manual_seed(0)
        models = [torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)]
        optimizers = [
            torch.optim.Adam(models[0].parameters(), lr=1e-3),
            torch.optim.Adam(models[1].parameters(), lr=1e-3),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pt")
            mappo_utils.save_checkpoint(
                path, models, optimizers, step=5, best_return=1.23, share_policy=False
            )

            restored_models = [torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)]
            restored_optimizers = [
                torch.optim.Adam(restored_models[0].parameters(), lr=1e-3),
                torch.optim.Adam(restored_models[1].parameters(), lr=1e-3),
            ]
            data = mappo_utils.load_checkpoint(path, restored_models, restored_optimizers)

        self.assertEqual(data["step"], 5)
        self.assertAlmostEqual(float(data["best_return"]), 1.23)
        self.assertFalse(data["share_policy"])
        for model, restored in zip(models, restored_models):
            for param, restored_param in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.allclose(param, restored_param))

    def test_save_and_load_checkpoint_shared_policy(self) -> None:
        torch.manual_seed(1)
        shared_model = torch.nn.Linear(3, 2)
        models = [shared_model, shared_model]
        optimizers = [torch.optim.Adam(shared_model.parameters(), lr=1e-3)]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "shared.pt")
            mappo_utils.save_checkpoint(
                path, models, optimizers, step=7, best_return=2.5, share_policy=True
            )

            restored_models = [torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)]
            restored_optimizers = [
                torch.optim.Adam(restored_models[0].parameters(), lr=1e-3)
            ]
            data = mappo_utils.load_checkpoint(path, restored_models, restored_optimizers)

        self.assertEqual(data["step"], 7)
        self.assertAlmostEqual(float(data["best_return"]), 2.5)
        self.assertTrue(data["share_policy"])
        for param, restored_param in zip(shared_model.parameters(), restored_models[0].parameters()):
            self.assertTrue(torch.allclose(param, restored_param))
        for param0, param1 in zip(restored_models[0].parameters(), restored_models[1].parameters()):
            self.assertTrue(torch.allclose(param0, param1))


if __name__ == "__main__":
    unittest.main()
