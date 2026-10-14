"""Native package-delivery evidence uses geometry and failure precedence, not survival."""

from types import SimpleNamespace
import unittest

from MAPPO.mappo.diagnostics import TERRAIN_STEP, task_status


def native_env(x=20.0 * (14.0 / 30.0), terrain_length=200, done=False, **overrides):
    engine = SimpleNamespace(
        terrain_length=terrain_length,
        package=SimpleNamespace(position=SimpleNamespace(x=x)),
        game_over=False,
        package_drop=False,
        fallen_walkers={0: False, 1: False, 2: False},
        terminate_on_fall=True,
        frames=500,
        max_cycles=500,
        get_last_dones=lambda: {0: done, 1: done, 2: done},
    )
    for key, value in overrides.items():
        setattr(engine, key, value)
    return SimpleNamespace(engine=engine)


class TestNativeDeliveryDiagnostics(unittest.TestCase):
    def test_default_native_geometry_requires_actual_79_unit_delivery_distance(self):
        start = 20.0 * TERRAIN_STEP
        status = task_status(native_env(), initial_package_x=start)
        self.assertAlmostEqual(status["goal_x"], (200 - 10) * (14.0 / 30.0))
        self.assertAlmostEqual(status["package_x"], 9.333333333333334)
        self.assertAlmostEqual(status["remaining_distance"], 79.33333333333333)
        self.assertEqual(status["progress_fraction"], 0.0)
        self.assertFalse(status["delivered"])
        self.assertFalse(status["failure"])

    def test_delivery_requires_strict_native_crossing_and_all_true_native_dones(self):
        goal = (200 - 10) * TERRAIN_STEP
        for x, done, delivered in (
            (goal, True, None),
            (goal + 0.1, False, False),
            (goal + 0.1, True, True),
        ):
            with self.subTest(x=x, done=done):
                status = task_status(native_env(x=x, done=done))
                self.assertIs(status["delivered"], delivered)
        partial = native_env(
            x=goal + 0.1, get_last_dones=lambda: {0: True, 1: False, 2: True}
        )
        self.assertFalse(task_status(partial)["delivered"])

    def test_stationary_full_horizon_and_shorter_safety_cap_are_not_delivery(self):
        for frames in (20, 500):
            with self.subTest(frames=frames):
                env = native_env(frames=frames)
                # An outer time-limit flag cannot replace native delivery evidence.
                env.terminations = {
                    "walker_0": True,
                    "walker_1": True,
                    "walker_2": True,
                }
                status = task_status(
                    env, initial_package_x=env.engine.package.position.x
                )
                self.assertFalse(status["delivered"])
                self.assertFalse(status["native_done"])
                self.assertEqual(status["progress_fraction"], 0.0)

    def test_drop_fall_and_left_edge_failures_take_precedence_over_goal_or_done(self):
        goal = (200 - 10) * TERRAIN_STEP
        for overrides, x, reason in (
            ({"game_over": True}, goal + 1.0, "package_drop"),
            ({"package_drop": True}, goal + 1.0, "package_drop"),
            (
                {"fallen_walkers": {0: False, 1: True, 2: False}},
                goal + 1.0,
                "walker_fall",
            ),
            ({}, -0.1, "left_edge"),
        ):
            with self.subTest(reason=reason, overrides=overrides):
                status = task_status(native_env(x=x, done=True, **overrides))
                self.assertFalse(status["delivered"])
                self.assertTrue(status["failure"])
                self.assertEqual(status["reason"], reason)

    def test_shorter_terrain_changes_only_explicit_native_geometry_not_default_goal(
        self,
    ):
        shorter = task_status(native_env(x=50.0, terrain_length=100, done=True))
        default = task_status(native_env(x=50.0, terrain_length=200, done=True))
        self.assertAlmostEqual(shorter["goal_x"], 42.0)
        self.assertTrue(shorter["delivered"])
        self.assertAlmostEqual(default["goal_x"], 88.66666666666667)
        self.assertIsNone(default["delivered"])
        self.assertEqual(native_env().engine.terrain_length, 200)

    def test_missing_native_evidence_is_unknown_and_nonfinite_geometry_is_rejected(
        self,
    ):
        unknown = task_status(SimpleNamespace(engine=None))
        self.assertIsNone(unknown["delivered"])
        self.assertIsNone(unknown["failure"])
        self.assertIsNone(unknown["goal_x"])
        incomplete = native_env(x=100.0, done=True)
        del incomplete.engine.fallen_walkers
        self.assertIsNone(task_status(incomplete)["delivered"])
        for env in (
            native_env(x=float("nan")),
            native_env(terrain_length=float("inf")),
        ):
            with self.assertRaises(ValueError):
                task_status(env)


if __name__ == "__main__":
    unittest.main()
