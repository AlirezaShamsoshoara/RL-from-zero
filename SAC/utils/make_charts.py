"""Three honest SAC charts from explicitly selected local W&B history."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

from SAC.utils._common import (
    SOFT_Q_NOTE,
    config_metadata,
    finite,
    positive_int,
    protect_inputs,
    read_config,
    sha256,
    write_json,
)

STEP_KEY = "progress/step"
STEP_ALIASES = (STEP_KEY, "progress/global_step")
METRICS = (
    "charts/avg_return",
    "charts/avg_length",
    "loss/critic_loss",
    "loss/actor_loss",
    "loss/alpha_loss",
    "stats/alpha_value",
    "stats/log_prob",
    "stats/q1_mean",
    "stats/q2_mean",
    "stats/target_q_mean",
    "stats/entropy",
    "stats/td_error",
)
REQUIRED_METRICS = tuple(
    metric for metric in METRICS if metric not in ("stats/entropy", "stats/td_error")
)
Q_METRICS = ("stats/q1_mean", "stats/q2_mean", "stats/target_q_mean")
BLUE, RED, GREEN, ORANGE, PURPLE = "#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Explicit training run-<id>.wandb file; no discovery or online API",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("SAC/configs/pendulum_fast.yaml"),
        help="Exact producer training config for interpretation; not proof of legacy run units",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("SAC/assets"))
    parser.add_argument("--title", default="SAC Pendulum-v1")
    parser.add_argument(
        "--smooth",
        type=positive_int,
        default=9,
        help="Centered display smoothing in logged points, not environment steps",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Source JSON; defaults to out-dir/chart_metrics.json",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        help="Optional all-episode evaluation JSON: matched soft-Q/reward scatter, NOT calibration",
    )
    return parser.parse_args(argv)


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _clean_row(row: dict) -> dict | None:
    if not any(metric in row for metric in METRICS):
        return None
    keys = [key for key in STEP_ALIASES if key in row]
    if not keys:
        raise ValueError("SAC metrics lack progress/step or progress/global_step")
    for key in keys:
        if not _is_number(row[key]) or row[key] < 0 or int(row[key]) != row[key]:
            raise ValueError(f"Invalid environment step: {key}")
    if len(keys) == 2 and row[keys[0]] != row[keys[1]]:
        raise ValueError("progress/step and progress/global_step disagree")
    clean = {STEP_KEY: row[keys[0]]}
    if "progress/global_step" in row:
        clean["progress/global_step"] = row["progress/global_step"]
    for metric in METRICS:
        if metric not in row:
            continue
        value = row[metric]
        if not _is_number(value):
            raise ValueError(
                f"Non-finite or nonnumeric {metric} at step {clean[STEP_KEY]}"
            )
        clean[metric] = value
    if clean.get("stats/alpha_value", 1.0) <= 0:
        raise ValueError("SAC alpha must be positive")
    if clean.get("charts/avg_length", 0) < 0:
        raise ValueError("Negative episode length")
    return clean


def load_history(run_path: str | Path) -> list[dict]:
    """Read flat/nested_key protobuf items, exporting known numeric metrics only."""
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
                # Do not parse or export arbitrary credential-bearing history.
                if key in METRICS or key in STEP_ALIASES:
                    if key in row:
                        raise ValueError(f"Duplicate history key {key}")
                    row[key] = json.loads(item.value_json)
            clean = _clean_row(row)
            if clean is not None:
                rows.append(clean)
    finally:
        ds.close()
    rows.sort(key=lambda row: row[STEP_KEY])
    return rows


def series(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for row in rows:
        if metric not in row:
            continue
        clean = _clean_row(row)
        if clean is None:
            continue
        points.append((float(clean[STEP_KEY]), float(clean[metric])))
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x, y = zip(*points)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("Smoothing window must be positive")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cannot smooth non-finite values")
    left, right = (window - 1) // 2, window // 2
    return np.asarray(
        [
            np.mean(values[max(0, i - left) : min(len(values), i + right + 1)])
            for i in range(len(values))
        ],
        dtype=float,
    )


def matched_q_rows(rows: list[dict]) -> list[dict]:
    matched = [row for row in rows if all(key in row for key in Q_METRICS)]
    if not matched:
        raise ValueError(
            "No matched rows containing q1_mean, q2_mean, target_q_mean; legacy histories cannot supply invented Q curves"
        )
    for metric in Q_METRICS:
        series(matched, metric)
    return matched


def summarize(rows: list[dict], run_path: str | Path) -> dict:
    metrics = {}
    for metric in METRICS:
        steps, values = series(rows, metric)
        if len(values):
            metrics[metric] = {
                "count": len(values),
                "first_step": float(steps[0]),
                "last_step": float(steps[-1]),
                "first": float(values[0]),
                "last": float(values[-1]),
                "mean": finite(np.mean(values), metric),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return {
        "schema_version": 1,
        "algorithm": "SAC",
        "source": str(Path(run_path).resolve()),
        "source_sha256": sha256(run_path),
        "source_url": None,
        "source_id": None,
        "training_rows": len(rows),
        "step_key": STEP_KEY,
        "accepted_step_aliases": list(STEP_ALIASES),
        "metrics": metrics,
        "missing_metrics": [metric for metric in METRICS if metric not in metrics],
        "source_rows": rows,
        "units": {
            "charts/avg_return": "Producer's native undiscounted episode return rolling mean, includes stochastic training actions; never independent held-out evaluation",
            "charts/avg_length": "Producer's transitions per episode rolling mean; normal native Pendulum limit is 200",
            "stats/q1_mean": "Online soft-Q1 mean on sampled replay batches, averaged over logging interval, entropy-augmented training objective",
            "stats/q2_mean": "Online soft-Q2 mean on the same sampled replay batches/logging interval",
            "stats/target_q_mean": "Same-batch reward + gamma*(1-terminated)*(min target soft-Q - alpha*log_prob), averaged over logging interval",
            "loss/critic_loss": "Sum of twin-Q mean squared TD errors, not absolute episode-return error",
            "loss/actor_loss": "mean(alpha*log_prob - min online soft-Q) on stochastic reparameterized policy actions",
            "loss/alpha_loss": "-mean(log_alpha*(log_prob + target_entropy).detach())",
            "stats/alpha_value": "Positive automatic entropy temperature",
            "stats/log_prob": "Mean squashed, affine-corrected continuous log density, not discrete probability",
            "stats/entropy": "Monte Carlo differential entropy estimate -mean(log_prob), in nats; may be negative",
            "stats/td_error": "Producer-labelled TD diagnostic; inspect exact producer definition, not a native return calibration error",
        },
        "training_unit_provenance": "Caller-supplied producer YAML; not independently established for a legacy datastore without embedded config",
        "interpretation": SOFT_Q_NOTE
        + " Q1/Q2/TD target curves are same-source aggregate diagnostics, not sample-level independent calibration.",
    }


def evaluation_pairs(path: str | Path, config: dict) -> tuple[list[dict], dict]:
    before = sha256(path)
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if sha256(path) != before:
        raise ValueError("Evaluation source changed while reading")
    if (
        data.get("algorithm") != "SAC"
        or data.get("env_id") != "Pendulum-v1"
        or data.get("gamma") != float(config["gamma"])
    ):
        raise ValueError("Expected native SAC evaluation with matching env and gamma")
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Evaluation has no individual episode records")
    pairs = []
    for episode in episodes:
        points = episode.get("soft_q_reward_points")
        if not isinstance(points, list) or len(points) != episode.get("episode_length"):
            raise ValueError("Evaluation must retain every matched trajectory point")
        for point in points:
            for field in ("soft_q1", "soft_q2"):
                if point.get(field) is not None:
                    pairs.append(
                        {
                            "episode": episode["episode"],
                            "seed": episode["seed"],
                            "t": point["t"],
                            "head": field,
                            "soft_q": finite(point[field], field),
                            "native_reward_to_boundary": finite(
                                point["native_discounted_reward_to_boundary"],
                                "reward target",
                            ),
                            "terminated": episode["terminated"],
                            "truncated": episode["truncated"],
                            "capped": episode["capped"],
                        }
                    )
    if not pairs:
        raise ValueError(
            "Evaluation has no optional soft-Q heads; use actor-only evaluation without --evaluation scatter"
        )
    return pairs, {
        "source": str(Path(path).resolve()),
        "source_sha256": before,
        "checkpoint": data.get("checkpoint"),
        "pairs": pairs,
        "all_episode_count": len(episodes),
        "point_count": len(pairs),
        "interpretation": SOFT_Q_NOTE
        + " All available heads and episode points retained, including native TimeLimit boundaries; no calibration RMSE or equality claim.",
    }


def _save(fig, path: Path) -> None:
    try:
        fig.tight_layout(pad=1.7)
        fig.savefig(str(path))
    finally:
        plt.close(fig)


def _draw_series(ax, rows, metric, label, color, window=1):
    x, y = series(rows, metric)
    if not len(y):
        raise ValueError(f"Cannot plot empty {metric}")
    if window > 1:
        ax.plot(x / 1000, y, color=color, alpha=0.3, label=label + " (logged)")
    ax.plot(
        x / 1000,
        smooth(y, window),
        color=color,
        marker="." if len(y) < 4 else None,
        label=label + (" (display smoothed)" if window > 1 else ""),
    )
    ax.set_xlabel("Environment transitions (thousands)")
    ax.set_title(textwrap.fill(label, 64))
    ax.legend(loc="best", fontsize=9)


def make_charts(args: argparse.Namespace) -> dict:
    os.environ["WANDB_MODE"] = "offline"
    config = read_config(args.config)
    if (
        isinstance(args.smooth, bool)
        or not isinstance(args.smooth, int)
        or args.smooth <= 0
    ):
        raise ValueError("smooth must be a positive integer")
    before = sha256(args.run)
    rows = load_history(args.run)
    if sha256(args.run) != before:
        raise ValueError(
            "Training datastore changed while reading; use a closed immutable run"
        )
    if not rows:
        raise ValueError("No finite SAC training history rows")
    missing = [
        metric for metric in REQUIRED_METRICS if not len(series(rows, metric)[1])
    ]
    if missing:
        raise ValueError(
            "Missing required series (no synthetic replacement): " + ", ".join(missing)
        )
    matched = matched_q_rows(rows)
    evaluation = getattr(args, "evaluation", None)
    pairs, eval_source = (
        evaluation_pairs(evaluation, config) if evaluation else (None, None)
    )
    summary = summarize(rows, args.run)
    if summary["source_sha256"] != before:
        raise ValueError("Training datastore changed during analysis")
    summary["config_provenance"] = config_metadata(args.config, config)
    summary["matched_q_rows"] = matched
    summary["evaluation"] = eval_source
    summary["display_smoothing_points"] = args.smooth
    summary["display_series"] = {
        metric: {
            "steps": series(rows, metric)[0].tolist(),
            "logged_values": series(rows, metric)[1].tolist(),
            "smoothed_values": smooth(series(rows, metric)[1], args.smooth).tolist(),
        }
        for metric in ("charts/avg_return", "charts/avg_length")
    }
    json.dumps(summary, allow_nan=False)
    out_dir = Path(args.out_dir)
    summary_path = getattr(args, "summary", None) or out_dir / "chart_metrics.json"
    pngs = [out_dir / f"chart_{i:02d}.png" for i in range(1, 4)]
    protect_inputs([summary_path, *pngs], [args.config, args.run, evaluation])
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )

    fig, axes = plt.subplots(3 if pairs else 2, 1, figsize=(10, 11 if pairs else 8))
    for metric, label, color in zip(
        Q_METRICS,
        ("Soft-Q1 batch mean", "Soft-Q2 batch mean", "Matched soft TD target mean"),
        (BLUE, PURPLE, ORANGE),
    ):
        _draw_series(axes[0], matched, metric, label, color)
    axes[0].set_title(
        "SAC same-batch/logging-window soft-Q means, NOT native calibration"
    )
    axes[0].set_ylabel("Entropy-augmented training units")
    _draw_series(
        axes[1],
        rows,
        "charts/avg_return",
        "Native episodic training return (separate objective)",
        GREEN,
        args.smooth,
    )
    axes[1].set_ylabel("Undiscounted native reward")
    if pairs:
        for head, color in (("soft_q1", BLUE), ("soft_q2", PURPLE)):
            subset = [point for point in pairs if point["head"] == head]
            if subset:
                axes[2].scatter(
                    [point["native_reward_to_boundary"] for point in subset],
                    [point["soft_q"] for point in subset],
                    color=color,
                    s=7,
                    alpha=0.4,
                    label=head + " at the matched observation/action",
                )
        axes[2].set_xlabel(
            "Observed deterministic discounted native reward to boundary"
        )
        axes[2].set_ylabel("Continuing stochastic soft-Q, training units")
        axes[2].set_title(
            "Different objectives/horizons: matched points, NOT overestimation proof"
        )
        axes[2].legend(fontsize=9)
    fig.suptitle(textwrap.fill(args.title, 80))
    _save(fig, pngs[0])

    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    panels = (
        ("loss/critic_loss", "Twin soft-Q critic MSE", RED),
        ("loss/actor_loss", "Stochastic actor: alpha log density - min soft-Q", ORANGE),
        ("loss/alpha_loss", "Automatic temperature loss", PURPLE),
        ("stats/alpha_value", "Entropy temperature alpha", BLUE),
        ("stats/log_prob", "Continuous squashed-policy mean log density", GREEN),
        (
            "stats/entropy",
            "Differential entropy estimate (nats, can be negative)",
            GREEN,
        ),
    )
    for ax, (metric, label, color) in zip(axes.flat, panels):
        if len(series(rows, metric)[1]):
            _draw_series(ax, rows, metric, label, color)
            ax.set_ylabel(label)
        else:
            ax.set_axis_off()
            ax.text(
                0.05,
                0.5,
                "Optional stats/entropy not logged.\nNo invented entropy curve.",
                transform=ax.transAxes,
            )
    fig.suptitle(textwrap.fill("SAC losses and entropy temperature: " + args.title, 80))
    _save(fig, pngs[1])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    _draw_series(
        axes[0],
        rows,
        "charts/avg_return",
        "Native undiscounted training return, rolling mean",
        GREEN,
        args.smooth,
    )
    axes[0].set_ylabel("Native reward sum")
    _draw_series(
        axes[1],
        rows,
        "charts/avg_length",
        "Training episode length, rolling mean",
        ORANGE,
        args.smooth,
    )
    axes[1].set_ylabel("Environment transitions")
    axes[1].axhline(
        200,
        color="gray",
        ls="--",
        label="Registered 200-step TimeLimit, not solved threshold",
    )
    axes[1].legend(fontsize=9)
    fig.suptitle(
        textwrap.fill(
            "SAC native return and length; no official solved threshold: " + args.title,
            80,
        )
    )
    _save(fig, pngs[2])
    if sha256(args.run) != before or (
        eval_source and sha256(evaluation) != eval_source["source_sha256"]
    ):
        raise ValueError("Analysis source changed before export")
    summary["charts"] = [
        {"path": str(path.resolve()), "sha256": sha256(path)} for path in pngs
    ]
    write_json(summary_path, summary)
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    summary = make_charts(args)
    print(
        f"Wrote three charts and source JSON; {summary['training_rows']} logged source rows; soft-Q is not native calibration"
    )


if __name__ == "__main__":
    main()
