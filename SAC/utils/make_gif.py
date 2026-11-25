"""Render full native SAC episodes and label a top-return illustrative GIF."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from SAC.utils._common import (
    assert_checkpoint_unchanged,
    config_metadata,
    load_agent,
    make_native_env,
    native_source_metadata,
    positive_int,
    protect_inputs,
    read_config,
    resolve_checkpoint,
    rollout_episode,
    seed_int,
    sha256,
    validate_counts,
    write_json,
)
from SAC.utils.evaluate import (
    _scalar_metrics,
    initialize_offline_run,
    log_episode,
    summarize,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("SAC/configs/pendulum_fast.yaml")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trusted immutable snapshot; defaults to selected config inference_model_path",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional trusted stock root supplying SAC/sac/networks.py",
    )
    parser.add_argument("--out", type=Path, default=Path("SAC/assets/sac_pendulum.gif"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="All-episode records; defaults to GIF path with .json suffix",
    )
    parser.add_argument("--episodes", type=positive_int, default=8)
    parser.add_argument("--keep-top", type=positive_int, default=3)
    parser.add_argument("--seed", type=seed_int, default=280000)
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=20,
        help="Playback rate; native dt=0.05 gives 20 physical steps/s",
    )
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=1,
        help="Display-only full-span subsampling; final frame always retained",
    )
    parser.add_argument("--hold-frames", type=positive_int, default=20)
    args = parser.parse_args(argv)
    try:
        validate_gif_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_gif_args(args) -> None:
    validate_counts(args.seed, args.episodes)
    for name in ("keep_top", "fps", "stride", "hold_frames"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.keep_top > args.episodes:
        raise ValueError("keep_top must not exceed episodes")
    if args.fps > 100:
        raise ValueError(
            "GIF fps must be <=100 for a nonzero centisecond frame duration"
        )
    if Path(args.out).suffix.lower() != ".gif":
        raise ValueError("GIF output must use the .gif extension")


def select_top_episodes(records: list[dict], keep_top: int) -> list[int]:
    if (
        isinstance(keep_top, bool)
        or not isinstance(keep_top, int)
        or not 0 < keep_top <= len(records)
    ):
        raise ValueError("keep_top must be positive and not exceed episode count")
    # Stable ties prefer the original episode order, never an outcome filter.
    return sorted(
        range(len(records)), key=lambda i: records[i]["native_return"], reverse=True
    )[:keep_top]


def frame_indices(frame_count: int, stride: int) -> list[int]:
    if frame_count <= 0 or stride <= 0:
        raise ValueError("frame_count and stride must be positive")
    indices = list(range(0, frame_count, stride))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def _label_frame(frame, record: dict, rank: int, keep_top: int, count: int):
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    labelled = Image.new("RGB", (image.width, image.height + 68), "#111827")
    labelled.paste(image, (0, 68))
    draw = ImageDraw.Draw(labelled)
    status = "native TimeLimit (normal)" if record["timeout"] else "native terminated"
    draw.text(
        (8, 5),
        f"SAC tanh(mean) | SELECTED top {keep_top}/{count}, rank {rank}",
        fill="white",
    )
    draw.text(
        (8, 24),
        f"seed={record['seed']} return={record['native_return']:.1f} steps={record['episode_length']}",
        fill="white",
    )
    draw.text(
        (8, 43), f"FULL EPISODE: {status}; illustrative, not unbiased", fill="white"
    )
    return np.asarray(labelled)


def encoded_gif_metadata(path: str | Path) -> dict:
    """Reuse A3C's fixed pattern: inspect encoded Pillow frames and durations."""
    with Image.open(path) as image:
        if image.format != "GIF":
            raise ValueError("Encoded output is not a GIF")
        count = image.n_frames
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Encoded GIF has no valid frame count")
        duration_ms = 0.0
        for index in range(count):
            image.seek(index)
            duration = image.info.get("duration")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise ValueError(
                    f"Encoded GIF frame {index} lacks valid positive duration metadata"
                )
            duration_ms += float(duration)
        if not math.isfinite(duration_ms):
            raise ValueError("Non-finite total encoded GIF duration")
    return {
        "frames": count,
        "encoded_frames": count,
        "duration_ms": duration_ms,
        "encoder_note": "Identical consecutive frames can merge while retaining duration; GIF timing is quantized to centiseconds. Counts and duration are read from encoded bytes, not inferred from submitted frames.",
    }


def make_gif(args: argparse.Namespace) -> dict:
    validate_gif_args(args)
    config = read_config(args.config)
    checkpoint_path = resolve_checkpoint(config, getattr(args, "checkpoint", None))
    source_root = getattr(args, "source_root", None)
    networks_path = (
        Path(source_root) if source_root else Path(__file__).resolve().parents[2]
    ) / "SAC/sac/networks.py"
    summary_path = getattr(args, "summary", None) or Path(args.out).with_suffix(".json")
    protect_inputs(
        [args.out, summary_path], [args.config, checkpoint_path, networks_path]
    )
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["WANDB_MODE"] = "offline"
    provenance = config_metadata(args.config, config)
    env = make_native_env(config, render_mode="rgb_array")
    run = None
    try:
        model, checkpoint = load_agent(config, env, checkpoint_path, source_root)
        source = native_source_metadata(env)
        source["training_env_kwargs"] = config.get("env_kwargs", {})
        source["ignored_training_max_episode_steps"] = config.get("env_kwargs", {}).get(
            "max_episode_steps"
        )
        source["effective_evaluation_cap"] = 200
        run = initialize_offline_run(
            config,
            provenance,
            checkpoint,
            source,
            args.episodes,
            args.seed,
            "gif-rollouts",
        )
        records, retained = [], {}
        for index in range(args.episodes):
            record, frames = rollout_episode(
                model,
                env,
                config,
                args.seed + index,
                episode=index,
                capture_frames=True,
                source=source,
            )
            # A stock native showcase must contain all 200 transitions. Abort on
            # unexpected early termination instead of hiding or extending it.
            if not record["native_200_step_episode"] or len(frames) != 201:
                raise ValueError(
                    "GIF requires complete native 200-step episodes with initial plus final frame"
                )
            records.append(record)
            log_episode(run, record)
            retained[index] = frames
            wanted = set(select_top_episodes(records, min(args.keep_top, len(records))))
            retained = {i: value for i, value in retained.items() if i in wanted}
        selected = select_top_episodes(records, args.keep_top)
        output_frames, segments = [], []
        for rank, index in enumerate(selected, start=1):
            indices = frame_indices(len(retained[index]), args.stride)
            segment = [
                _label_frame(
                    retained[index][i],
                    records[index],
                    rank,
                    args.keep_top,
                    args.episodes,
                )
                for i in indices
            ]
            start = len(output_frames)
            output_frames.extend(segment)
            output_frames.extend([segment[-1]] * args.hold_frames)
            segments.append(
                {
                    "episode_index": index,
                    "source_frames": len(retained[index]),
                    "source_frame_indices": indices,
                    "initial_frame_included": indices[0] == 0,
                    "final_frame_included": indices[-1] == 200,
                    "submitted_start_index": start,
                    "submitted_frame_count": len(segment) + args.hold_frames,
                    "final_hold_submitted_frames": args.hold_frames,
                }
            )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # imageio's Pillow writer expects milliseconds, not seconds.
        imageio.mimsave(str(out), output_frames, duration=1000.0 / args.fps, loop=0)
        encoded = encoded_gif_metadata(out)
        summary = summarize(records)
        metrics = _scalar_metrics("eval/summary", summary)
        run.log(metrics, step=args.episodes)
        run.summary.update(metrics)
        assert_checkpoint_unchanged(checkpoint)
        result = {
            "schema_version": 1,
            "algorithm": "SAC",
            "env_id": "Pendulum-v1",
            "device": "cpu",
            "policy": "CPU native observation-only GaussianPolicy tanh(mean)",
            "selection": "Top-return selected illustrative showcase, not unbiased evaluation; every complete episode counted before selection",
            "source_url": None,
            "source_id": None,
            "config_provenance": provenance,
            "checkpoint": checkpoint,
            "native_environment": source,
            "source_root": str(Path(source_root).resolve()) if source_root else None,
            "gamma": float(config["gamma"]),
            "episodes": records,
            "summary_all_episodes": summary,
            "seeds": [record["seed"] for record in records],
            "returns": [record["native_return"] for record in records],
            "selected_episode_indices": selected,
            "selected_returns": [records[i]["native_return"] for i in selected],
            "selected_seeds": [records[i]["seed"] for i in selected],
            "fps": args.fps,
            "stride": args.stride,
            "hold_frames": args.hold_frames,
            "source_render_fps": getattr(env.unwrapped, "metadata", {}).get(
                "render_fps"
            ),
            "physics_dt_seconds": source["observed_physics"].get("dt"),
            "playback_note": "Default stride=1 and fps=20 follow native dt=0.05; changed stride/fps changes playback speed, not trajectory span",
            "segments": segments,
            "submitted_frames": len(output_frames),
            **encoded,
            "gif_path": str(out.resolve()),
            "gif_sha256": sha256(out),
            "wandb": {"mode": "offline", "run_id": run.id, "run_dir": run.dir},
        }
        write_json(summary_path, result)
        return result
    finally:
        try:
            env.close()
        finally:
            if run is not None:
                run.finish()


def main(argv=None) -> None:
    args = parse_args(argv)
    result = make_gif(args)
    print(
        json.dumps(
            {
                "all_returns": result["returns"],
                "selected_returns": result["selected_returns"],
                "selected_seeds": result["selected_seeds"],
                "selection": result["selection"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    print(
        f"Wrote full-span selected showcase {args.out}; submitted={result['submitted_frames']}, encoded={result['encoded_frames']}, duration_ms={result['duration_ms']}"
    )


if __name__ == "__main__":
    main()
