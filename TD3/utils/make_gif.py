"""Render full native TD3 episodes as a top-return selected showcase, offline."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from TD3.utils._common import (
    DEFAULT_CONFIG,
    assert_checkpoint_unchanged,
    config_metadata,
    finite,
    load_agent,
    make_native_env,
    network_path,
    positive_int,
    protect_inputs,
    rollout_episode,
    seed_int,
    sha256,
    validate_counts,
    write_json,
)
from TD3.utils.evaluate import (
    _scalar_metrics,
    environment_metadata,
    evaluation_context,
    initialize_offline_run,
    log_episode,
    result_metadata,
    summarize,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Omitted uses selected config inference_model_path, with no baseline fallback",
    )
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path, default=Path("TD3/assets/td3_mountaincar_continuous.gif")
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--episodes", type=positive_int, default=8)
    parser.add_argument("--keep-top", type=positive_int, default=3)
    parser.add_argument("--seed", type=seed_int, default=280000)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--stride", type=positive_int, default=2)
    parser.add_argument("--hold-frames", type=positive_int, default=12)
    parser.add_argument("--overwrite", action="store_true")
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
        raise ValueError("GIF fps must be <=100 for positive centisecond duration")
    if Path(args.out).suffix.lower() != ".gif":
        raise ValueError("Output must use .gif extension")


def select_top_episodes(records: list[dict], keep_top: int) -> list[int]:
    if (
        isinstance(keep_top, bool)
        or not isinstance(keep_top, int)
        or not 0 < keep_top <= len(records)
    ):
        raise ValueError("keep_top must be positive and not exceed record count")
    for record in records:
        finite(record["native_return"], "ranking return")
    # Stable ties keep original order; never filter out failure or timeout records.
    return sorted(
        range(len(records)), key=lambda i: records[i]["native_return"], reverse=True
    )[:keep_top]


def frame_indices(frame_count: int, stride: int) -> list[int]:
    for value in (frame_count, stride):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("frame_count and stride must be positive integers")
    indices = list(range(0, frame_count, stride))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def _label_frame(frame, record: dict, rank: int, keep_top: int, count: int):
    image = Image.fromarray(frame)
    labelled = Image.new("RGB", (image.width, image.height + 68), "#111827")
    labelled.paste(image, (0, 68))
    draw = ImageDraw.Draw(labelled)
    goal = (
        "unknown"
        if record["native_goal"] is None
        else str(record["native_goal"]).lower()
    )
    draw.text(
        (8, 5),
        f"TD3 Actor | SELECTED top {keep_top}/{count}, rank {rank}",
        fill="white",
    )
    draw.text(
        (8, 24),
        f"seed={record['seed']} return={record['native_return']:.2f} steps={record['episode_length']}",
        fill="white",
    )
    draw.text(
        (8, 43),
        f"FULL SPAN goal={goal} timeout={record['timeout']}; showcase, not unbiased",
        fill="white",
    )
    return np.asarray(labelled)


def encoded_gif_metadata(path: str | Path) -> dict:
    """Read actual encoded count and duration, not submitted-frame estimates."""
    with Image.open(path) as image:
        if image.format != "GIF" or image.n_frames <= 0:
            raise ValueError("Encoded output has no GIF frames")
        count, duration_ms = image.n_frames, 0.0
        for index in range(count):
            image.seek(index)
            duration = image.info.get("duration")
            if duration is None or finite(duration, "encoded frame duration") <= 0:
                raise ValueError(f"GIF frame {index} lacks positive duration")
            duration_ms += float(duration)
    if not math.isfinite(duration_ms):
        raise ValueError("Non-finite total encoded duration")
    return {
        "frames": count,
        "encoded_frames": count,
        "encoded_frame_count": count,
        "duration_ms": duration_ms,
        "encoder_note": "Pillow readback: identical frames can merge, timing is quantized to centiseconds; counts and duration describe actual bytes",
    }


def make_gif(args: argparse.Namespace) -> dict:
    validate_gif_args(args)
    config, checkpoint_path, source_root = evaluation_context(args)
    summary_path = getattr(args, "summary", None) or Path(args.out).with_suffix(".json")
    overwrite = getattr(args, "overwrite", False)
    protect_inputs(
        [args.out, summary_path],
        [args.config, checkpoint_path, network_path(source_root)],
        overwrite,
    )
    # These changes apply only when a caller deliberately runs this rendering tool.
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["WANDB_MODE"] = "offline"
    provenance = config_metadata(args.config, config)
    env, run = make_native_env(config, render_mode="rgb_array"), None
    try:
        model, checkpoint = load_agent(config, env, checkpoint_path, source_root)
        source = environment_metadata(env, config)
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
            records.append(record)
            log_episode(run, record)
            if (
                not record["complete_episode"]
                or record["capped"]
                or len(frames) != record["episode_length"] + 1
            ):
                raise ValueError(
                    "GIF must retain complete reset-through-final-state spans"
                )
            retained[index] = frames
            wanted = set(select_top_episodes(records, min(args.keep_top, len(records))))
            retained = {i: value for i, value in retained.items() if i in wanted}
        selected = select_top_episodes(records, args.keep_top)
        output_frames, segments = [], []
        for rank, index in enumerate(selected, start=1):
            record = records[index]
            indices = frame_indices(len(retained[index]), args.stride)
            segment = [
                _label_frame(
                    retained[index][i], record, rank, args.keep_top, args.episodes
                )
                for i in indices
            ]
            start = len(output_frames)
            output_frames.extend(segment)
            output_frames.extend([segment[-1]] * args.hold_frames)
            segments.append(
                {
                    "episode_index": index,
                    "seed": record["seed"],
                    "native_goal": record["native_goal"],
                    "timeout": record["timeout"],
                    "source_frames": len(retained[index]),
                    "source_frame_indices": indices,
                    "initial_frame_included": indices[0] == 0,
                    "final_frame_included": indices[-1] == record["episode_length"],
                    "submitted_start_index": start,
                    "submitted_frame_count": len(segment) + args.hold_frames,
                    "final_hold_submitted_frames": args.hold_frames,
                }
            )
        assert_checkpoint_unchanged(checkpoint)
        out = Path(args.out)
        # Recheck immediately before an encoder that otherwise silently overwrites.
        protect_inputs(
            [out, summary_path],
            [args.config, checkpoint_path, network_path(source_root)],
            overwrite,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out), output_frames, duration=1000.0 / args.fps, loop=0)
        encoded = encoded_gif_metadata(out)
        summary = summarize(records)
        metrics = _scalar_metrics("eval/summary", summary)
        run.log(metrics, step=args.episodes)
        run.summary.update(metrics)
        assert_checkpoint_unchanged(checkpoint)
        result = result_metadata(config, provenance, checkpoint, source, run, records)
        result.update(
            selection="Top-return selected showcase, not unbiased evaluation; all complete episodes retained in JSON and offline W&B",
            summary_all_episodes=summary,
            selected_episode_indices=selected,
            selected_returns=[records[i]["native_return"] for i in selected],
            selected_seeds=[records[i]["seed"] for i in selected],
            fps=args.fps,
            stride=args.stride,
            hold_frames=args.hold_frames,
            source_render_fps=getattr(env.unwrapped, "metadata", {}).get("render_fps"),
            playback_note="Only full-span stride/fps and final hold alter presentation; no failure segment removal or trajectory modification",
            segments=segments,
            submitted_frames=len(output_frames),
            gif_path=str(out.resolve()),
            gif_sha256=sha256(out),
            **encoded,
        )
        write_json(summary_path, result, overwrite)
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
        f"Wrote {args.out}: encoded_frames={result['encoded_frames']}, duration_ms={result['duration_ms']}"
    )


if __name__ == "__main__":
    main()
