"""Render complete native A3C rollouts, then label a top-return GIF showcase."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import wandb

from A3C.utils._common import (
    config_metadata,
    load_agent,
    make_native_env,
    native_source_metadata,
    positive_int,
    read_config,
    rollout_episode,
    seed_int,
    validate_counts,
    write_json,
)
from A3C.utils.evaluate import _scalar_metrics, summarize


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("A3C/configs/lunarlander_greedy.yaml")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trusted snapshot; defaults to the selected config's inference_model_path",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Trusted stock repository supplying the checkpoint architecture",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("A3C/assets/a3c_lunarlander.gif")
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="All rollout data; defaults to output GIF path with .json suffix",
    )
    parser.add_argument("--episodes", type=positive_int, default=8)
    parser.add_argument("--keep-top", type=positive_int, default=3)
    parser.add_argument("--seed", type=seed_int, default=280000)
    parser.add_argument("--fps", type=positive_int, default=25)
    parser.add_argument("--stride", type=positive_int, default=2)
    parser.add_argument("--hold-frames", type=positive_int, default=15)
    args = parser.parse_args(argv)
    try:
        validate_counts(args.seed, args.episodes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.keep_top > args.episodes:
        parser.error("keep-top must not exceed episodes")
    if args.summary is not None and args.summary.resolve() == args.out.resolve():
        parser.error("summary must not overwrite the GIF")
    return args


def _label_frame(frame, record: dict, rank: int, keep_top: int, count: int):
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    labelled = Image.new("RGB", (image.width, image.height + 58), "#111827")
    labelled.paste(image, (0, 58))
    draw = ImageDraw.Draw(labelled)
    status = "terminated" if record["terminated"] else "native timeout"
    landing = (
        "unknown" if record["native_landing"] is None else str(record["native_landing"])
    )
    draw.text(
        (8, 5),
        f"A3C argmax | SELECTED SHOWCASE: top {keep_top}/{count}, rank {rank}",
        fill="white",
    )
    draw.text(
        (8, 22),
        f"seed={record['seed']} native return={record['native_return']:.1f} steps={record['episode_length']}",
        fill="white",
    )
    draw.text(
        (8, 39),
        f"FULL EPISODE: {status} | native landing={landing} | not unbiased evaluation",
        fill="white",
    )
    return np.asarray(labelled)


def encoded_gif_metadata(path: str | Path) -> dict:
    """Read actual encoded frames and total duration, never infer from inputs."""
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
                or duration < 0
            ):
                raise ValueError(
                    f"Encoded GIF frame {index} lacks valid duration metadata"
                )
            duration_ms += float(duration)
        if not math.isfinite(duration_ms):
            raise ValueError("Encoded GIF has non-finite total duration")
    return {
        "frames": count,
        "encoded_frames": count,
        "duration_ms": duration_ms,
        "encoder_note": "Identical consecutive submitted frames may merge into fewer encoded frames while retaining their duration.",
    }


def make_gif(args: argparse.Namespace) -> dict:
    validate_counts(args.seed, args.episodes)
    for name in ("keep_top", "fps", "stride", "hold_frames"):
        number = getattr(args, name)
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.keep_top > args.episodes:
        raise ValueError("keep_top must not exceed episodes")
    summary_path = getattr(args, "summary", None) or Path(args.out).with_suffix(".json")
    if Path(summary_path).resolve() == Path(args.out).resolve():
        raise ValueError("summary must not overwrite GIF")
    config = read_config(args.config)
    if getattr(args, "checkpoint", None) is None:
        checkpoint_path = config.get("inference_model_path")
        if not checkpoint_path:
            raise ValueError(
                "Selected config must specify inference_model_path when --checkpoint is omitted"
            )
        args.checkpoint = Path(checkpoint_path)
    inputs = {Path(args.config).resolve(), Path(args.checkpoint).resolve()}
    if Path(args.out).resolve() in inputs or Path(summary_path).resolve() in inputs:
        raise ValueError("GIF outputs must not overwrite config or checkpoint")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["WANDB_MODE"] = "offline"
    provenance = config_metadata(args.config, config)
    env = make_native_env(config, render_mode="rgb_array")
    run = None
    try:
        model, checkpoint = load_agent(
            config, env, args.checkpoint, getattr(args, "source_root", None)
        )
        source = native_source_metadata(env)
        source["training_env_kwargs"] = config.get("env_kwargs", {})
        source["ignored_training_max_episode_steps"] = config.get("env_kwargs", {}).get(
            "max_episode_steps"
        )
        source["effective_evaluation_cap"] = None
        run = wandb.init(
            project=config.get("project", "rl-practice"),
            mode="offline",
            reinit=True,
            name=f"{config.get('run_name', 'a3c-lunarlander')}-gif-rollouts",
            config={
                "algorithm": "A3C",
                "checkpoint": checkpoint,
                "config_provenance": provenance,
                "native_environment": source,
                "episodes": args.episodes,
                "seed": args.seed,
                "selection": "roll out every complete episode, then choose top native returns",
            },
        )
        records, rollout_frames = [], []
        for episode in range(args.episodes):
            record, frames = rollout_episode(
                model,
                env,
                config,
                args.seed + episode,
                episode=episode,
                capture_frames=True,
                source=source,
            )
            records.append(record)
            rollout_frames.append(frames)
            run.log(_scalar_metrics("eval", record), step=episode)
        summary = summarize(records)
        metrics = _scalar_metrics("eval/summary", summary)
        run.log(metrics, step=args.episodes)
        run.summary.update(metrics)
        selected = sorted(
            range(len(records)), key=lambda i: records[i]["native_return"], reverse=True
        )[: args.keep_top]
        output_frames = []
        for rank, index in enumerate(selected, start=1):
            episode_frames = rollout_frames[index]
            # Striding never removes the true final post-transition frame.
            indices = list(range(0, len(episode_frames), args.stride))
            if indices[-1] != len(episode_frames) - 1:
                indices.append(len(episode_frames) - 1)
            segment = [
                _label_frame(
                    episode_frames[i],
                    records[index],
                    rank,
                    args.keep_top,
                    args.episodes,
                )
                for i in indices
            ]
            output_frames.extend(segment)
            output_frames.extend([segment[-1]] * args.hold_frames)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out), output_frames, duration=1000.0 / args.fps, loop=0)
        encoded = encoded_gif_metadata(out)
        result = {
            "algorithm": "A3C",
            "policy": "CPU native observation-only categorical argmax",
            "selection": "Top-return selected showcase, not unbiased evaluation",
            "config_provenance": provenance,
            "checkpoint": checkpoint,
            "native_environment": source,
            "source_root": str(Path(args.source_root).resolve())
            if getattr(args, "source_root", None)
            else None,
            "episodes": records,
            "summary_all_episodes": summary,
            "returns": [record["native_return"] for record in records],
            "selected_episode_indices": selected,
            "selected_returns": [records[i]["native_return"] for i in selected],
            "selected_seeds": [records[i]["seed"] for i in selected],
            "fps": args.fps,
            "stride": args.stride,
            "hold_frames": args.hold_frames,
            "source_fps": getattr(env.unwrapped, "metadata", {}).get("render_fps"),
            "submitted_frames": len(output_frames),
            **encoded,
            "gif_path": str(out.resolve()),
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
            },
            indent=2,
            allow_nan=False,
        )
    )
    print(
        f"Wrote selected showcase {args.out}, including the final state of every selected episode"
    )


if __name__ == "__main__":
    main()
