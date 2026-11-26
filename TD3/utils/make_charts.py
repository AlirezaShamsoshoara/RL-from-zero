"""Three offline TD3 charts from an exact local W&B datastore, without invented data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

from TD3.utils._common import (
    DEFAULT_CONFIG,
    ENV_ID,
    MAX_JSON_BYTES,
    Q_NOTE,
    config_metadata,
    positive_int,
    protect_inputs,
    read_config,
    sha256,
    write_json,
)

STEP_KEY = "progress/step"
STEP_ALIASES = (STEP_KEY, "progress/global_step")
Q_KEYS = ("stats/q1_value", "stats/q2_value", "stats/target_q_value")
COUNT_KEYS = (
    "progress/optimizer_updates",
    "progress/actor_updates",
    "stats/critic_updates",
    "stats/actor_updates",
)
GOAL_KEYS = ("charts/native_goal_rate", "charts/goal_rate")
METRICS = (
    "charts/avg_return",
    "charts/avg_length",
    *Q_KEYS,
    "stats/td_error",
    "stats/actor_updated",
    "loss/critic_loss",
    "loss/actor_loss",
    "progress/vector_steps",
    *COUNT_KEYS,
    *GOAL_KEYS,
)
REQUIRED = ("charts/avg_return", "charts/avg_length", "loss/critic_loss", *Q_KEYS)
BLUE, RED, GREEN, ORANGE = "#2563eb", "#dc2626", "#16a34a", "#ea580c"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Exact closed training run-<id>.wandb; no newest-run guessing",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Exact YAML that produced this training history",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("TD3/assets"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--title", default="TD3 MountainCarContinuous-v0")
    parser.add_argument("--smooth", type=positive_int, default=9)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        help="Optional evaluate.py JSON; all initial Q/MC points, different policy/horizon",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def step_value(row: dict) -> float:
    present = [row[key] for key in STEP_ALIASES if key in row]
    if not present or any(not _number(value) or value < 0 for value in present):
        raise ValueError(
            "Training metric row requires finite nonnegative progress/step or progress/global_step"
        )
    if any(value != present[0] for value in present):
        raise ValueError("Environment-transition step aliases disagree")
    return float(present[0])


def _metric(value, key):
    if value is None:
        return None
    if key == "stats/actor_updated" and isinstance(value, bool):
        return value
    if not _number(value):
        raise ValueError(f"Non-finite/nonnumeric training metric: {key}")
    if key in COUNT_KEYS or key == "progress/vector_steps":
        if value < 0 or int(value) != value:
            raise ValueError(f"Logged count must be a nonnegative integer: {key}")
    if key in GOAL_KEYS or key == "stats/actor_updated":
        if not 0 <= value <= 1:
            raise ValueError(f"Rate/fraction must be in [0, 1]: {key}")
    return value


def load_history(run_path: str | Path) -> list[dict]:
    """Local protobuf scan; nested_key wins over key; null delayed losses survive."""
    ds = datastore.DataStore()
    ds.open_for_scan(str(run_path))
    rows = []
    try:
        while True:
            data = ds.scan_data()
            if data is None:
                break
            record = pb.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue
            row = {}
            for item in record.history.item:
                nested = list(item.nested_key)
                key = "/".join(nested) if nested else item.key
                if key not in METRICS + STEP_ALIASES:
                    continue
                value = json.loads(item.value_json)
                if key in row and row[key] != value:
                    raise ValueError(f"Duplicate conflicting history key: {key}")
                row[key] = value
            if not any(key in row for key in METRICS):
                continue
            step = step_value(row)
            clean = {key: row[key] for key in STEP_ALIASES if key in row}
            clean[STEP_KEY] = step
            for key in METRICS:
                if key in row:
                    clean[key] = _metric(row[key], key)
            rows.append(clean)
    finally:
        ds.close()
    rows.sort(key=step_value)
    return rows


def series(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for row in rows:
        if metric not in row or row[metric] is None:
            continue
        value = _metric(row[metric], metric)
        points.append((step_value(row), float(value)))
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x, y = zip(*points)
    return np.asarray(x), np.asarray(y)


def smooth(values, window: int) -> np.ndarray:
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("smooth window must be a positive integer")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Only finite one-dimensional series can be smoothed")
    left, right = (window - 1) // 2, window // 2
    return np.asarray(
        [
            np.mean(values[max(0, i - left) : min(len(values), i + right + 1)])
            for i in range(len(values))
        ]
    )


def matched_q_rows(rows: list[dict]) -> list[dict]:
    # Never zip independently filtered Q series or join separate records by step.
    matched = [
        row
        for row in rows
        if all(key in row and row[key] is not None for key in Q_KEYS)
    ]
    if not matched:
        raise ValueError(
            "No single history rows with matching Q1, Q2 and target bootstrap Q"
        )
    for key in Q_KEYS:
        series(matched, key)
    return matched


def summarize(rows: list[dict], run_path: str | Path) -> dict:
    if not rows:
        raise ValueError("No usable training rows")
    metrics = {}
    for key in METRICS:
        steps, values = series(rows, key)
        metrics[key] = (
            None
            if not len(values)
            else {
                "count": len(values),
                "first_step": float(steps[0]),
                "last_step": float(steps[-1]),
                "first": float(values[0]),
                "last": float(values[-1]),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
    return {
        "algorithm": "TD3",
        "env_id": ENV_ID,
        "source": str(Path(run_path).resolve()),
        "source_sha256": sha256(run_path),
        "training_rows": len(rows),
        "step_key": STEP_KEY,
        "key_union": sorted(set().union(*(set(row) for row in rows))),
        "metrics": metrics,
        "missing_metrics": [key for key in METRICS if metrics[key] is None],
        "source_rows": rows,
        "units": {
            "charts/avg_return": "Raw native undiscounted noisy-training episode return, logged rolling mean",
            "charts/avg_length": "Native transitions per episode, logged rolling mean",
            "stats/q1_value": "Online Q1 on sampled replay state/action batch, logged mean",
            "stats/q2_value": "Online Q2 on same replay batch, logged mean",
            "stats/target_q_value": "Clipped smoothed twin-min NEXT-STATE bootstrap Q BEFORE reward and termination mask; NOT full Bellman y",
            "stats/td_error": "Logged replay TD diagnostic; producer definition, not native greedy calibration",
            "stats/actor_updated": "Logged actor-update indicator or logging-window fraction, NOT an update count",
            "loss/critic_loss": "Twin critic MSE loss from actual updates",
            "loss/actor_loss": "Delayed actor -mean Q1(s, Actor(s)); null/skipped updates are absent, not zero",
            "progress/optimizer_updates": "Actual cumulative critic optimizer updates if logged",
            "progress/actor_updates": "Actual cumulative actor optimizer updates if logged",
        },
        "q_interpretation": Q_NOTE,
        "matching_note": "Q series use complete single logging records, potentially logging-window aggregates, not independent calibration samples",
        "mean_return_benchmark": 90.0,
    }


def evaluation_pairs(path: str | Path, config: dict) -> tuple[list[dict], dict]:
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Evaluation JSON exceeds 256 MiB read guard")
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if (
        not isinstance(data, dict)
        or data.get("algorithm") != "TD3"
        or data.get("env_id") != ENV_ID
    ):
        raise ValueError("Expected raw TD3 MountainCarContinuous evaluation JSON")
    if data.get("gamma") != float(config["gamma"]) or data.get("native_horizon") != 999:
        raise ValueError("Evaluation gamma/native horizon mismatch")
    if (
        data.get("selection")
        != "none: all requested episodes, including every timeout/failure"
    ):
        raise ValueError(
            "Use unbiased all-episode evaluate.py JSON, not a selected showcase"
        )
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Evaluation JSON contains no episode records")
    pairs = []
    for record in episodes:
        if not record.get("complete_episode") or record.get("capped"):
            raise ValueError("Evaluation contains a capped or incomplete episode")
        target = record.get("native_discounted_return")
        if not _number(target):
            raise ValueError("Non-finite native evaluation target")
        for head in ("q1", "q2"):
            value = record.get("initial_" + head)
            if value is None:
                continue
            if not _number(value):
                raise ValueError("Non-finite initial Q diagnostic")
            pairs.append(
                {
                    "episode": record["episode"],
                    "seed": record["seed"],
                    "head": head,
                    "initial_q": value,
                    "native_discounted_reward_to_boundary": target,
                    "native_goal": record.get("native_goal"),
                    "timeout": record["timeout"],
                }
            )
    return pairs, {
        "source": str(path.resolve()),
        "source_sha256": sha256(path),
        "checkpoint": data.get("checkpoint"),
        "all_episode_count": len(episodes),
        "pairs": pairs,
        "interpretation": "Matched initial state/action Q versus deterministic native finite-horizon Monte Carlo, policy/horizon shift, NOT calibration; no outcome filtering",
    }


def _draw(ax, rows, key, label, color, window=1):
    x, values = series(rows, key)
    if not len(values):
        ax.text(
            0.5,
            0.5,
            f"{label}: not logged",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return False
    ax.plot(
        x / 1000,
        values,
        color=color,
        marker=".",
        alpha=0.5 if window > 1 else 1,
        label=label + " (logged)",
    )
    if window > 1:
        ax.plot(
            x / 1000,
            smooth(values, window),
            color=color,
            label=f"Display mean ({window} logged points)",
        )
    ax.set_xlabel("Environment transitions (thousands)")
    ax.legend(loc="best", fontsize=8)
    return True


def _title(ax, text):
    ax.set_title(textwrap.fill(text, 80), fontsize=10)


def _save(fig, path):
    try:
        fig.tight_layout(pad=1.6)
        fig.savefig(path)
    finally:
        plt.close(fig)


def make_charts(args: argparse.Namespace) -> dict:
    if (
        isinstance(args.smooth, bool)
        or not isinstance(args.smooth, int)
        or args.smooth <= 0
    ):
        raise ValueError("smooth must be a positive integer")
    config = read_config(args.config)
    run_path = Path(args.run)
    if run_path.suffix != ".wandb" or not run_path.is_file() or run_path.is_symlink():
        raise ValueError("Pass an exact closed regular local .wandb file")
    before = sha256(run_path)
    rows = load_history(run_path)
    if not rows:
        raise ValueError("Empty local training history")
    missing = [key for key in REQUIRED if not len(series(rows, key)[1])]
    if missing:
        raise ValueError(
            "Missing required training metrics; no backfill: " + ", ".join(missing)
        )
    matched = matched_q_rows(rows)
    pairs, eval_source = [], None
    evaluation = getattr(args, "evaluation", None)
    if evaluation is not None:
        pairs, eval_source = evaluation_pairs(evaluation, config)
    summary = summarize(rows, run_path)
    summary.update(
        config_provenance=config_metadata(args.config, config),
        matched_q_rows=matched,
        evaluation=eval_source,
        display_smoothing_points=args.smooth,
    )
    # Preserve null actor losses and every later valid match across the key union.
    summary["display_series"] = {}
    for key in ("charts/avg_return", "charts/avg_length"):
        x, values = series(rows, key)
        summary["display_series"][key] = {
            "steps": x.tolist(),
            "logged": values.tolist(),
            "smoothed": smooth(values, args.smooth).tolist(),
        }
    json.dumps(summary, allow_nan=False)
    out_dir = Path(args.out_dir)
    summary_path = getattr(args, "summary", None) or out_dir / "chart_metrics.json"
    paths = [out_dir / f"chart_{index:02d}.png" for index in range(1, 4)]
    overwrite = getattr(args, "overwrite", False)
    protect_inputs(
        paths + [summary_path], [args.config, run_path, evaluation], overwrite
    )
    if sha256(run_path) != before:
        raise ValueError(
            "Training datastore changed during read; use a closed immutable run"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.unicode_minus": False,
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    for key, label, color in zip(
        Q_KEYS,
        (
            "Replay online Q1",
            "Replay online Q2",
            "Smoothed target bootstrap Q (NOT Bellman y)",
        ),
        (BLUE, GREEN, ORANGE),
    ):
        _draw(axes[0], matched, key, label, color)
    axes[0].set_ylabel("Training Q means")
    _title(
        axes[0],
        f"Same-record batch/window Q diagnostics, not native calibration: {args.title}",
    )
    _draw(axes[1], rows, "stats/td_error", "Logged replay TD diagnostic", RED)
    _title(
        axes[1],
        "TD diagnostic from producer, not deterministic episode prediction error",
    )
    if pairs:
        for head, color in (("q1", BLUE), ("q2", ORANGE)):
            values = [point for point in pairs if point["head"] == head]
            if values:
                axes[2].scatter(
                    [point["native_discounted_reward_to_boundary"] for point in values],
                    [point["initial_q"] for point in values],
                    color=color,
                    s=15,
                    label=head,
                )
        axes[2].set_xlabel(
            "Observed deterministic native discounted reward to boundary"
        )
        axes[2].set_ylabel("Matched initial training Q")
        axes[2].legend(fontsize=8)
        _title(
            axes[2],
            "All available initial-state test points: policy/horizon shift, NOT exact greedy calibration",
        )
    else:
        _draw(axes[2], rows, "charts/avg_return", "Native noisy-training return", GREEN)
        axes[2].set_ylabel("Undiscounted native reward")
        _title(
            axes[2],
            "Separate episode-return panel: different policy, horizon and samples than replay Q",
        )
    _save(fig, paths[0])

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    _draw(axes[0], rows, "loss/critic_loss", "Twin critic loss", RED)
    _draw(
        axes[1],
        rows,
        "loss/actor_loss",
        "Actual delayed actor loss (null skipped)",
        ORANGE,
    )
    _title(
        axes[1],
        "Delayed actor updates: omitted/null is not zero; no interpolation at skipped records",
    )
    found = False
    for key, color in zip(COUNT_KEYS, (BLUE, GREEN, RED, ORANGE)):
        if len(series(rows, key)[1]):
            _draw(axes[2], rows, key, key + " (actual logged count)", color)
            found = True
    if not found:
        axes[2].text(
            0.5,
            0.65,
            "Critic/actor counts not logged; no inferred counts",
            ha="center",
            transform=axes[2].transAxes,
        )
        if len(series(rows, "stats/actor_updated")[1]):
            _draw(
                axes[2],
                rows,
                "stats/actor_updated",
                "Actor-updated fraction, NOT count",
                BLUE,
            )
    _title(axes[0], f"Actual critic/actor optimization diagnostics: {args.title}")
    _save(fig, paths[1])

    goal_key = next((key for key in GOAL_KEYS if len(series(rows, key)[1])), None)
    fig, axes = plt.subplots(
        3 if goal_key else 2, 1, figsize=(11, 10 if goal_key else 7)
    )
    _draw(
        axes[0],
        rows,
        "charts/avg_return",
        "Raw native episode return (logged rolling mean)",
        GREEN,
        args.smooth,
    )
    axes[0].axhline(
        90,
        color=BLUE,
        linestyle="--",
        label="Native MEAN return benchmark 90, NOT goal predicate",
    )
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Native undiscounted reward")
    _title(axes[0], f"Native noisy-training mean return: {args.title}")
    _draw(
        axes[1],
        rows,
        "charts/avg_length",
        "Raw native episode length",
        ORANGE,
        args.smooth,
    )
    axes[1].set_ylabel("Transitions per episode")
    if goal_key:
        _draw(
            axes[2], rows, goal_key, "Physical native goal rate (only as logged)", BLUE
        )
        axes[2].set_ylim(-0.03, 1.03)
    else:
        _title(
            axes[1],
            "Native goal rate was not logged; never estimated from positive return",
        )
    _save(fig, paths[2])
    if sha256(run_path) != before:
        raise ValueError("Training datastore changed while plotting")
    summary["outputs"] = [
        {"path": str(path.resolve()), "sha256": sha256(path)} for path in paths
    ]
    write_json(summary_path, summary, overwrite)
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    result = make_charts(args)
    print(
        f"Wrote 3 offline charts: {result['training_rows']} rows, {len(result['matched_q_rows'])} matched Q records"
    )
    print("Missing metrics: " + ", ".join(result["missing_metrics"]))


if __name__ == "__main__":
    main()
