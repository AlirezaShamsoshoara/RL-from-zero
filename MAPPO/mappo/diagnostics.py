"""Read-only native MultiWalker package-delivery and motion diagnostics.

Geometry and simulation constants mirror
pettingzoo/sisl/multiwalker/multiwalker_base.py: TERRAIN_STEP=14/30,
TERRAIN_GRASS=10, FPS=50. No PettingZoo or Box2D import occurs here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

TERRAIN_STEP = 14.0 / 30.0
TERRAIN_GRASS = 10
FPS = 50


def _finite(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def package_position_x(env) -> float | None:
    engine = getattr(env, "engine", None)
    package = getattr(engine, "package", None)
    position = getattr(package, "position", None)
    value = getattr(position, "x", None)
    return None if value is None else _finite(value, "Native package x position")


def _flags(value) -> list[bool] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = list(value.values())
    try:
        return [bool(item) for item in value]
    except TypeError:
        return [bool(value)]


def task_status(env, initial_package_x: float | None = None) -> dict:
    """Classify the native endpoint, with failure precedence over delivery.

    Delivery requires x strictly beyond the unchanged native goal and a nonempty
    all-true get_last_dones() result. Neither timeout nor parallel termination
    flags alone prove delivery. Missing engine/failure evidence stays unknown.
    """
    engine = getattr(env, "engine", None)
    x = package_position_x(env)
    result = {
        "package_x": x,
        "goal_x": None,
        "remaining_distance": None,
        "progress_fraction": None,
        "delivered": None,
        "failure": None,
        "native_done": None,
        "reason": "unknown",
        "terrain_step": TERRAIN_STEP,
        "terrain_grass": TERRAIN_GRASS,
    }
    if engine is None:
        return result
    terrain_length = getattr(engine, "terrain_length", None)
    if terrain_length is not None:
        terrain_length = _finite(terrain_length, "Native terrain length")
        if terrain_length <= TERRAIN_GRASS:
            raise ValueError("Native terrain length must exceed the grass segment")
        goal = (terrain_length - TERRAIN_GRASS) * TERRAIN_STEP
        result["goal_x"] = goal
        if x is not None:
            result["remaining_distance"] = goal - x
            if initial_package_x is not None:
                start = _finite(initial_package_x, "Initial package x position")
                if goal > start:
                    result["progress_fraction"] = (x - start) / (goal - start)
    game_over = getattr(engine, "game_over", None)
    dropped = getattr(engine, "package_drop", False)
    fallen = _flags(getattr(engine, "fallen_walkers", None))
    terminate_on_fall = getattr(engine, "terminate_on_fall", None)
    if game_over is not None and bool(game_over) or bool(dropped):
        result.update(delivered=False, failure=True, reason="package_drop")
        return result
    if (
        fallen is not None
        and any(fallen)
        and terminate_on_fall is not None
        and bool(terminate_on_fall)
    ):
        result.update(delivered=False, failure=True, reason="walker_fall")
        return result
    if x is not None and x < 0:
        result.update(delivered=False, failure=True, reason="left_edge")
        return result
    # Do not certify delivery from an incomplete fake or legacy engine state.
    if game_over is None or fallen is None or terminate_on_fall is None or x is None:
        return result
    result["failure"] = False
    get_dones = getattr(engine, "get_last_dones", None)
    dones = _flags(get_dones()) if callable(get_dones) else None
    result["native_done"] = bool(dones and all(dones)) if dones is not None else None
    if result["goal_x"] is None or result["native_done"] is None:
        return result
    if x > result["goal_x"] and result["native_done"]:
        result.update(delivered=True, reason="delivered")
    elif result["native_done"]:
        result["reason"] = "unknown"
    else:
        result.update(delivered=False, reason="running")
    return result


def motion_summary(
    samples: list[dict], parallel_steps: int, displacement: float | None
) -> dict:
    """Observed motion only; negative/backtracking samples are never suppressed."""
    seconds = parallel_steps / FPS
    windows = []
    for left, right in zip(samples, samples[1:]):
        delta_steps = right["parallel_step"] - left["parallel_step"]
        if delta_steps <= 0:
            raise ValueError("Package sample time indices must increase strictly")
        delta = right["package_x"] - left["package_x"]
        speed = delta / (delta_steps / FPS)
        windows.append(
            {
                "start_step": left["parallel_step"],
                "end_step": right["parallel_step"],
                "displacement": delta,
                "speed_per_second": speed,
                "stalled": abs(speed) < 0.01,
            }
        )
    moving_steps = sum(
        window["end_step"] - window["start_step"]
        for window in windows
        if not window["stalled"]
    )
    covered_steps = sum(window["end_step"] - window["start_step"] for window in windows)
    return {
        "simulation_fps": FPS,
        "simulation_seconds": seconds,
        "package_speed_per_parallel_step": displacement / parallel_steps
        if displacement is not None and parallel_steps > 0
        else None,
        "package_speed_per_second": displacement / seconds
        if displacement is not None and seconds > 0
        else None,
        "motion_fraction": moving_steps / covered_steps if covered_steps else None,
        "stall_speed_threshold": 0.01,
        "motion_windows": windows,
    }
