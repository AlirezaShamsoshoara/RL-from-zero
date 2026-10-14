"""Vector entrypoint dispatch is lazy, single-call, and never starts training here."""

import unittest
from unittest.mock import MagicMock, patch

from MAPPO.mappo.config import Config
import MAPPO.main as main


class TestVectorEntrypointDispatch(unittest.TestCase):
    def test_vector_config_dispatches_once_and_cli_checkpoint_overrides_config(self):
        for configured, supplied, expected in (
            ("config-best.pt", None, "config-best.pt"),
            ("config-best.pt", "cli-best.pt", "cli-best.pt"),
            (None, None, None),
        ):
            with self.subTest(configured=configured, supplied=supplied):
                cfg = Config()
                cfg.num_envs = 4
                cfg.init_checkpoint = configured
                result = {"global_step": 12, "sentinel": True}
                with (
                    patch.object(main.Config, "from_yaml", return_value=cfg),
                    patch(
                        "MAPPO.train_delivery.train", return_value=result
                    ) as delivery_train,
                    patch.object(main, "make_multiwalker_env") as single_factory,
                    patch.object(main, "wandb", MagicMock()) as wandb,
                ):
                    actual = main.train(
                        config="vector-config.yaml",
                        wandb_key="cli-test-key",
                        init_checkpoint=supplied,
                    )
                self.assertIs(actual, result)
                delivery_train.assert_called_once_with(
                    config="vector-config.yaml",
                    init_checkpoint=expected,
                    wandb_key="cli-test-key",
                )
                single_factory.assert_not_called()
                wandb.init.assert_not_called()
                wandb.login.assert_not_called()


if __name__ == "__main__":
    unittest.main()
