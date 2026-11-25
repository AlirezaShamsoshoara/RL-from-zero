"""Select an unchanged A3C snapshot by complete native development-set returns."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import torch
import wandb

from A3C.utils._common import (
    config_metadata,
    finite,
    load_agent,
    make_native_env,
    native_source_metadata,
    positive_int,
    read_config,
    redact,
    rollout_episode,
    seed_int,
    sha256,
    validate_counts,
    write_json,
)
from A3C.utils.evaluate import _scalar_metrics, summarize


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="One exact training profile per selection call",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Defaults to the selected config's checkpoint_dir",
    )
    parser.add_argument("--episodes", type=positive_int, default=20)
    parser.add_argument(
        "--seed",
        type=seed_int,
        default=90000,
        help="Development selection seeds, not independent held-out test seeds",
    )
    parser.add_argument(
        "--max-candidates",
        type=positive_int,
        default=100,
        help="Fail rather than silently omit candidates above this bound",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Unchanged snapshot alias; defaults to checkpoint-dir/best_greedy.pt",
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("A3C/logs/checkpoint-selection.json")
    )
    args = parser.parse_args(argv)
    try:
        validate_counts(args.seed, args.episodes)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _regular_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular, non-symlink file: {path}")


def enumerate_candidates(
    checkpoint_dir: str | Path, max_candidates: int = 100
) -> list[Path]:
    """Inspect only checkpoint_*.pt and best.pt immediately inside one directory."""
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates <= 0
    ):
        raise ValueError("max_candidates must be a positive integer")
    directory = Path(checkpoint_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint_dir must be an explicit non-symlink directory")
    candidates = []
    # This is a single-directory glob, never rglob or recursive repo discovery.
    for path in directory.glob("checkpoint_*.pt"):
        _regular_file(path, "Candidate")
        candidates.append(path)
        if len(candidates) > max_candidates:
            raise ValueError(
                "Too many candidates; increase --max-candidates explicitly"
            )
    best = directory / "best.pt"
    if best.exists() or best.is_symlink():
        _regular_file(best, "Candidate")
        candidates.append(best)
    if not candidates:
        raise ValueError(f"No checkpoint_*.pt or best.pt candidates in {directory}")
    if len(candidates) > max_candidates:
        raise ValueError("Too many candidates; increase --max-candidates explicitly")
    return sorted(candidates, key=lambda path: path.name)


def validate_paths(
    config_path: str | Path,
    candidates: list[Path],
    out: str | Path,
    summary: str | Path,
) -> None:
    out, summary = Path(out), Path(summary)
    if out.suffix != ".pt" or summary.suffix != ".json":
        raise ValueError("Snapshot output must use .pt and summary must use .json")
    if not candidates:
        raise ValueError("Selection requires at least one candidate")
    protected = {Path(config_path).resolve()} | {
        Path(path).resolve() for path in candidates
    }
    if (
        out.resolve() in protected
        or summary.resolve() in protected
        or out.resolve() == summary.resolve()
    ):
        raise ValueError(
            "Outputs must not overwrite any candidate, config, or each other"
        )
    for path in (out, summary):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(
                "Outputs must be regular files, not symlinks or directories"
            )
        for source in [Path(config_path), *map(Path, candidates)]:
            if path.exists() and source.exists() and path.samefile(source):
                raise ValueError("Outputs must not alias an input file")


def choose_candidate(results: list[dict]) -> dict:
    """Maximize all-episode native mean, including negative means; ties keep first."""
    if not results:
        raise ValueError("No scored candidates")
    scores = []
    expected_count = None
    expected_seeds = None
    for candidate in results:
        summary = candidate.get("summary")
        episodes = candidate.get("episodes")
        if (
            not isinstance(summary, dict)
            or not isinstance(episodes, list)
            or not episodes
        ):
            raise ValueError(
                "Every candidate must have complete scored episode records"
            )
        if summary.get("count") != len(episodes):
            raise ValueError("Candidate summary count does not match all episodes")
        if expected_count is None:
            expected_count = len(episodes)
        elif len(episodes) != expected_count:
            raise ValueError("Every candidate must use the same episode count")
        if all("seed" in episode for episode in episodes):
            seeds = [episode["seed"] for episode in episodes]
            if expected_seeds is None:
                expected_seeds = seeds
            elif seeds != expected_seeds:
                raise ValueError("Every candidate must use the same development seeds")
        if any(
            not episode.get("complete_episode") or episode.get("capped")
            for episode in episodes
        ):
            raise ValueError(
                "Development selection requires complete native episodes, not caller caps"
            )
        mean = finite(summary["native_return"]["mean"], "candidate native mean")
        actual = finite(
            sum(finite(ep["native_return"], "episode native return") for ep in episodes)
            / len(episodes),
            "all-episode native mean",
        )
        if not math.isclose(mean, actual, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                "Candidate score must be the mean over every native episode"
            )
        scores.append(mean)
    return results[max(range(len(scores)), key=scores.__getitem__)]


def atomic_copy_checkpoint(
    source: str | Path, out: str | Path, expected_sha256: str
) -> None:
    """Copy source bytes unchanged, verify hashes, then atomically replace alias."""
    source, out = Path(source), Path(out)
    _regular_file(source, "Chosen checkpoint")
    if source.resolve() == out.resolve() or (out.exists() and source.samefile(out)):
        raise ValueError("Snapshot alias must not overwrite its source")
    if out.is_symlink() or (out.exists() and not out.is_file()):
        raise ValueError("Snapshot alias must be a regular file")
    if sha256(source) != expected_sha256:
        raise ValueError(
            "Chosen checkpoint changed after scoring; use frozen snapshots"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{out.name}.", suffix=".tmp", dir=str(out.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as destination, source.open("rb") as original:
            shutil.copyfileobj(original, destination)
            destination.flush()
            os.fsync(destination.fileno())
        if sha256(temporary) != expected_sha256 or sha256(source) != expected_sha256:
            raise ValueError(
                "Chosen checkpoint changed while copying; alias was not replaced"
            )
        os.replace(temporary, out)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metadata_json(value):
    """Keep primitive snapshot metadata, including False, with finite JSON values."""
    if isinstance(value, dict):
        return {str(key): _metadata_json(item) for key, item in redact(value).items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise ValueError("Checkpoint selection metadata must contain JSON-safe primitives")


def _snapshot_metadata(path: Path, expected_sha256: str) -> dict:
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if sha256(path) != expected_sha256:
        raise ValueError("Checkpoint changed while reading selection provenance")
    # No model or optimizer tensors are exported. Source bytes remain unchanged.
    metadata = {
        key: payload[key]
        for key in ("metadata", "model_tag", "snapshot", "training_complete")
        if key in payload
    }
    return _metadata_json(metadata)


def select_checkpoint(args: argparse.Namespace) -> dict:
    validate_counts(args.seed, args.episodes)
    config = read_config(args.config)
    directory = getattr(args, "checkpoint_dir", None) or config.get("checkpoint_dir")
    if not directory:
        raise ValueError(
            "Selected config must specify checkpoint_dir when --checkpoint-dir is omitted"
        )
    directory = Path(directory)
    out = Path(getattr(args, "out", None) or directory / "best_greedy.pt")
    summary_path = Path(
        getattr(args, "summary", None) or "A3C/logs/checkpoint-selection.json"
    )
    candidates = enumerate_candidates(directory, getattr(args, "max_candidates", 100))
    validate_paths(args.config, candidates, out, summary_path)
    provenance = config_metadata(args.config, config)
    os.environ["WANDB_MODE"] = "offline"
    env = make_native_env(config)
    run = None
    result = None
    try:
        source = native_source_metadata(env)
        source["training_env_kwargs"] = config.get("env_kwargs", {})
        source["ignored_training_max_episode_steps"] = config.get("env_kwargs", {}).get(
            "max_episode_steps"
        )
        source["effective_evaluation_cap"] = None
        if source["native_horizon"] != 1000:
            raise ValueError(
                "Native LunarLander selection requires the registered 1000-step horizon"
            )
        run = wandb.init(
            project=config.get("project", "rl-practice"),
            name=f"{config.get('run_name', 'a3c-lunarlander')}-development-selection",
            mode="offline",
            reinit=True,
            config={
                "algorithm": "A3C",
                "phase": "development checkpoint selection, not held-out evaluation",
                "selection_metric": "mean native undiscounted return over all episodes",
                "config_provenance": provenance,
                "native_environment": source,
                "candidate_paths": [str(path.resolve()) for path in candidates],
                "episodes_per_candidate": args.episodes,
                "seed": args.seed,
            },
        )
        result = {
            "algorithm": "A3C",
            "phase": "development checkpoint selection, not held-out evaluation",
            "policy": "CPU categorical argmax, native observation only",
            "selection_metric": "maximum mean native undiscounted return over every requested episode",
            "tie_break": "first candidate in deterministic filename order",
            "development_seeds": list(range(args.seed, args.seed + args.episodes)),
            "episodes_per_candidate": args.episodes,
            "candidate_count": len(candidates),
            "config_provenance": provenance,
            "native_environment": source,
            "candidates": [],
            "chosen_source_path": None,
            "chosen_source_sha256": None,
            "out_path": str(out.resolve()),
            "alias_status": "not copied",
            "snapshot_semantics": "byte-identical inference alias; embedded training config and optimizer state are unchanged. This is not a new training result or exact-resume guarantee.",
            "independent_test_note": "Freeze the alias, then evaluate on a disjoint seed range. Selection returns are development scores, not held-out performance.",
            "wandb": {"mode": "offline", "run_id": run.id, "run_dir": run.dir},
        }
        log_step = 0
        for index, path in enumerate(candidates):
            candidate = {
                "candidate_index": index,
                "source_path": str(path.resolve()),
                "checkpoint": None,
                "snapshot_metadata": None,
                "episodes": [],
                "summary": None,
            }
            result["candidates"].append(candidate)
            model, checkpoint = load_agent(config, env, path)
            candidate["checkpoint"] = checkpoint
            candidate["snapshot_metadata"] = _snapshot_metadata(
                path, checkpoint["sha256"]
            )
            for episode in range(args.episodes):
                record, _ = rollout_episode(
                    model,
                    env,
                    config,
                    args.seed + episode,
                    episode=episode,
                    source=source,
                )
                candidate["episodes"].append(record)
                metrics = _scalar_metrics("selection/episode", record)
                metrics["selection/candidate_index"] = index
                metrics["selection/native_rewards"] = record["native_rewards"]
                run.log(metrics, step=log_step)
                log_step += 1
            candidate["summary"] = summarize(candidate["episodes"])
            metrics = _scalar_metrics(
                "selection/candidate_summary", candidate["summary"]
            )
            metrics["selection/candidate_index"] = index
            run.log(metrics, step=log_step)
            log_step += 1
            del model
        chosen = choose_candidate(result["candidates"])
        result["chosen_source_path"] = chosen["source_path"]
        result["chosen_source_sha256"] = chosen["checkpoint"]["sha256"]
        result["chosen_checkpoint"] = chosen["checkpoint"]
        result["chosen_snapshot_metadata"] = chosen["snapshot_metadata"]
        result["chosen_summary"] = chosen["summary"]
        result["alias_status"] = "scores complete; copy pending"
        # Persist every candidate's full scores before creating/replacing the alias.
        write_json(summary_path, result)
        validate_paths(args.config, candidates, out, summary_path)
        atomic_copy_checkpoint(
            chosen["source_path"], out, result["chosen_source_sha256"]
        )
        result["alias_status"] = "copied byte-identically"
        result["out_sha256"] = result["chosen_source_sha256"]
        write_json(summary_path, result)
        metrics = _scalar_metrics("selection/chosen_summary", chosen["summary"])
        metrics["selection/chosen_candidate_index"] = chosen["candidate_index"]
        run.log(metrics, step=log_step)
        run.summary.update(metrics)
        run.summary.update(
            {
                "selection/chosen_source_path": result["chosen_source_path"],
                "selection/chosen_source_sha256": result["chosen_source_sha256"],
                "selection/chosen_checkpoint": chosen["checkpoint"],
                "selection/chosen_snapshot_metadata": chosen["snapshot_metadata"],
                "selection/config_redacted_sha256": provenance["redacted_sha256"],
                "selection/out_path": result["out_path"],
            }
        )
        return result
    except Exception as exc:
        if result is not None:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            validate_paths(args.config, candidates, out, summary_path)
            write_json(summary_path, result)
        raise
    finally:
        try:
            env.close()
        finally:
            if run is not None:
                run.finish()


def main(argv=None) -> None:
    result = select_checkpoint(parse_args(argv))
    print(
        json.dumps(
            {
                "chosen_source_path": result["chosen_source_path"],
                "chosen_source_sha256": result["chosen_source_sha256"],
                "development_native_mean": result["chosen_summary"]["native_return"][
                    "mean"
                ],
                "out_path": result["out_path"],
                "phase": result["phase"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
