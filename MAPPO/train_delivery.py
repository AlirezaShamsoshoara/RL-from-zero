"""Full-geometry MultiWalker locomotion training with batched independent teams."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from MAPPO.main import _build_agent, _logger, _make_env
from MAPPO.mappo.agent import Batch, MAPPOAgent
from MAPPO.mappo.config import Config
from MAPPO.mappo.utils import load_checkpoint, save_checkpoint, set_seed
from MAPPO.mappo.vector_env import ParallelMultiwalker
from MAPPO.utils._common import rollout_episode


def aggregate_metrics(rows):
    """Retain successful losses even if the last update was entirely rejected."""
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    return {
        key: float(sum(row[key] for row in rows if key in row))
        if key == "stats/skipped_updates"
        else float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
    }


def evaluate_candidate(agent, cfg, seed, episodes):
    """Development-seed selection only; preserve the training random streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.device = "cpu"
    env = None
    try:
        env = _make_env(eval_cfg, seed)
        candidate = _build_agent(eval_cfg, env)
        for source, target in zip(agent.models, candidate.models):
            target.load_state_dict(
                {
                    key: value.detach().cpu()
                    for key, value in source.state_dict().items()
                }
            )
            target.eval()
        records = [
            rollout_episode(
                env,
                eval_cfg,
                candidate,
                seed + i,
                cfg.max_steps_per_episode,
                cfg.env_kwargs.get("max_cycles", 500),
                deterministic=True,
            )
            for i in range(episodes)
        ]
        for record in records:
            record.pop("frames", None)
        delivered = [bool(record.get("delivered", False)) for record in records]
        progress = [float(record.get("progress_fraction") or 0.0) for record in records]
        displacement = [
            float(record.get("package_displacement") or 0.0) for record in records
        ]
        stats = {
            "delivery_rate": float(np.mean(delivered)),
            "mean_progress": float(np.mean(progress)),
            "median_progress": float(np.median(progress)),
            "mean_displacement": float(np.mean(displacement)),
            "mean_return": float(
                np.mean([record["team_return"] for record in records])
            ),
            "mean_length": float(np.mean([record["length"] for record in records])),
        }
        # Completion outranks distance. Standing at the time limit cannot win by survival.
        score = (stats["delivery_rate"], stats["mean_progress"], stats["mean_return"])
        return stats, records, score
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def train(
    config="MAPPO/configs/multiwalker_delivery.yaml",
    init_checkpoint=None,
    evaluation_interval=25,
    evaluation_episodes=6,
    evaluation_seed=180000,
    run_name=None,
    wandb_key="",
):
    cfg = Config.from_yaml(config)
    init_checkpoint = init_checkpoint or cfg.init_checkpoint
    env_key = os.getenv("WANDB_API_KEY", "")
    if wandb_key:
        cfg.wandb_key = wandb_key
    elif env_key:
        cfg.wandb_key = env_key
    if cfg.total_timesteps % cfg.num_envs:
        raise ValueError("total_timesteps must be divisible by num_envs")
    if evaluation_interval <= 0 or evaluation_episodes <= 0:
        raise ValueError("evaluation interval and episodes must be positive")
    if cfg.env_kwargs.get("max_cycles", 500) != cfg.max_steps_per_episode:
        raise ValueError("native and training delivery horizons must explicitly match")
    if run_name:
        cfg.run_name = run_name
    set_seed(cfg.seed)
    logger = _logger(cfg)
    if cfg.wandb_key:
        wandb.login(key=cfg.wandb_key)
    # Evaluation is always offline; this run follows the explicitly chosen W&B mode.
    run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.run_name,
        config={
            **cfg.to_dict(),
            "init_checkpoint": init_checkpoint,
            "selection_seed": evaluation_seed,
            "selection_episodes": evaluation_episodes,
            "selection_rule": "delivery_rate, mean_progress, native_return",
        },
    )
    workers = None
    best_score = (-1.0, -1.0, -np.inf)
    best_metrics = None
    start_time = time.monotonic()
    env = None
    try:
        env = _make_env(cfg, cfg.seed)
        agent = _build_agent(cfg, env)
        if init_checkpoint:
            # Intentional fine-tuning: preserve model function, initialize a fresh optimizer.
            load_checkpoint(init_checkpoint, agent.models, None)
            logger.info(
                "Loaded model initialization %s; optimizer is intentionally fresh",
                init_checkpoint,
            )
        env.close()
        env = None
        workers = ParallelMultiwalker(cfg, cfg.num_envs)
        observations = workers.obs_array.copy()
        total_steps = optimizer_steps = update = 0
        num_updates = (cfg.total_timesteps + cfg.num_envs * cfg.rollout_steps - 1) // (
            cfg.num_envs * cfg.rollout_steps
        )
        recent_episodes = []
        previous_obs = previous_alive = None
        log_window = []
        log_dir = Path(cfg.checkpoint_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        selection_history = []
        while total_steps < cfg.total_timesteps:
            if previous_obs is not None:
                agent.update_normalizers(previous_obs, previous_alive)
            length = min(
                cfg.rollout_steps, (cfg.total_timesteps - total_steps) // cfg.num_envs
            )
            e, n, o = cfg.num_envs, agent.n_agents, agent.obs_dim
            obs_buf = np.zeros((length, e, n, o), np.float32)
            shape = (length, e, n)
            actions_shape = (
                (*shape, agent.act_dim) if agent.action_type == "continuous" else shape
            )
            action_buf = np.zeros(
                actions_shape,
                np.float32 if agent.action_type == "continuous" else np.int64,
            )
            (
                logp_buf,
                reward_buf,
                value_buf,
                nextv_buf,
                done_buf,
                term_buf,
                alive_buf,
            ) = [np.zeros(shape, np.float32) for _ in range(7)]
            if cfg.anneal_lr:
                frac = 1 - update / max(1, num_updates)
                for optimizer in agent.optimizers:
                    optimizer.param_groups[0]["lr"] = cfg.learning_rate * frac
                    if len(optimizer.param_groups) > 1:
                        optimizer.param_groups[1]["lr"] = (
                            cfg.critic_learning_rate or cfg.learning_rate
                        ) * frac
            for t in range(length):
                actions, logps, values = agent.act_many(observations)
                obs_buf[t], action_buf[t], logp_buf[t], value_buf[t] = (
                    observations,
                    actions,
                    logps,
                    values,
                )
                replies = workers.step(actions)
                next_observations = np.stack([r["next_obs"] for r in replies])
                observations = np.stack([r["obs"] for r in replies])
                rewards = np.stack([r["rewards"] for r in replies])
                alive = np.stack([r["alive"] for r in replies])
                reward_buf[t], alive_buf[t] = rewards, alive
                value_buf[t] *= alive
                nextv_buf[t] = agent.values_many(next_observations) * alive
                done_buf[t] = np.stack([r["dones"] for r in replies])
                term_buf[t] = np.stack([r["terminations"] for r in replies])
                total_steps += e
                for reply in replies:
                    if reply.get("episode") is not None:
                        recent_episodes.append(reply["episode"])
                recent_episodes = recent_episodes[-100:]
            adv, ret = MAPPOAgent.compute_gae(
                reward_buf.reshape(length, e * n),
                done_buf.reshape(length, e * n),
                value_buf.reshape(length, e * n),
                nextv_buf[-1].reshape(-1),
                cfg.gamma,
                cfg.gae_lambda,
                next_value_sequence=nextv_buf.reshape(length, e * n),
                terminations=term_buf.reshape(length, e * n),
            )
            adv, ret = adv.reshape(shape), ret.reshape(shape)
            samples = length * e
            joint = obs_buf.reshape(samples, -1)
            batches = [
                Batch(
                    torch.as_tensor(
                        obs_buf[:, :, i].reshape(samples, o), device=agent.device
                    ),
                    torch.as_tensor(joint, device=agent.device),
                    torch.as_tensor(
                        action_buf[:, :, i].reshape(
                            (samples, agent.act_dim)
                            if agent.action_type == "continuous"
                            else (samples,)
                        ),
                        device=agent.device,
                    ),
                    torch.as_tensor(logp_buf[:, :, i].reshape(-1), device=agent.device),
                    torch.as_tensor(ret[:, :, i].reshape(-1), device=agent.device),
                    torch.as_tensor(adv[:, :, i].reshape(-1), device=agent.device),
                    torch.as_tensor(
                        value_buf[:, :, i].reshape(-1), device=agent.device
                    ),
                    torch.as_tensor(
                        alive_buf[:, :, i].reshape(-1), device=agent.device
                    ),
                )
                for i in range(n)
            ]
            all_stats = []
            rejected_steps = 0
            early_stop = False
            for epoch in range(cfg.update_iterations):
                permutation = np.random.permutation(samples)
                for s in range(0, samples, cfg.minibatch_size):
                    indices = permutation[s : s + cfg.minibatch_size]
                    minibatches = [
                        Batch(
                            **{
                                field: getattr(b, field)[indices]
                                for field in Batch.__dataclass_fields__
                            }
                        )
                        for b in batches
                    ]
                    metrics, count = agent.update(minibatches)
                    optimizer_steps += count
                    rejected_steps += int(metrics.get("stats/skipped_updates", 0))
                    if count:
                        all_stats.append(metrics)
                    if metrics.get("stats/skipped_updates", 0) > 0:
                        # Recollect fresh data rather than retrying numerically
                        # invalid importance ratios from this stale rollout.
                        early_stop = True
                        break
                    if (
                        count
                        and cfg.target_kl
                        and metrics["stats/approx_kl"] > cfg.target_kl
                    ):
                        early_stop = True
                        break
                if early_stop:
                    break
            previous_obs, previous_alive = obs_buf.copy(), alive_buf.copy()
            mask = alive_buf > 0.5
            metrics = (
                {
                    key: float(np.mean([m[key] for m in all_stats]))
                    for key in all_stats[0]
                }
                if all_stats
                else {}
            )
            metrics["stats/skipped_updates"] = float(rejected_steps)
            variance = float(np.var(ret[mask]))
            metrics.update(
                {
                    "stats/value_mean": float(np.mean(value_buf[mask])),
                    "stats/return_target_mean": float(np.mean(ret[mask])),
                    "stats/explained_variance": 1
                    - float(np.var(ret[mask] - value_buf[mask])) / variance
                    if variance > 1e-8
                    else 0.0,
                }
            )
            log_window.append(metrics)
            update += 1
            if update % cfg.log_interval == 0 or total_steps == cfg.total_timesteps:
                payload = aggregate_metrics(log_window)
                payload.update(
                    {
                        "progress/global_step": total_steps,
                        "progress/update": update,
                        "progress/optimizer_steps": optimizer_steps,
                        "charts/avg_return": float(
                            np.mean([r["team_return"] for r in recent_episodes[-20:]])
                        )
                        if recent_episodes
                        else 0.0,
                        "charts/avg_length": float(
                            np.mean([r["length"] for r in recent_episodes[-20:]])
                        )
                        if recent_episodes
                        else 0.0,
                        "charts/delivery_rate": float(
                            np.mean(
                                [
                                    bool(r.get("delivered"))
                                    for r in recent_episodes[-20:]
                                ]
                            )
                        )
                        if recent_episodes
                        else 0.0,
                        "charts/mean_progress": float(
                            np.mean(
                                [
                                    float(r.get("progress_fraction") or 0.0)
                                    for r in recent_episodes[-20:]
                                ]
                            )
                        )
                        if recent_episodes
                        else 0.0,
                        "charts/steps_per_second": total_steps
                        / (time.monotonic() - start_time),
                    }
                )
                wandb.log(payload, step=total_steps)
                logger.info(
                    "Step %d/%d: sampled return %.2f, delivery %.1f%%, progress %.1f%%, throughput %.0f steps/s",
                    total_steps,
                    cfg.total_timesteps,
                    payload["charts/avg_return"],
                    100 * payload["charts/delivery_rate"],
                    100 * payload["charts/mean_progress"],
                    payload["charts/steps_per_second"],
                )
                log_window.clear()
            if (
                update % cfg.checkpoint_interval == 0
                or total_steps == cfg.total_timesteps
            ):
                save_checkpoint(
                    str(log_dir / f"checkpoint_{total_steps}.pt"),
                    agent.models,
                    agent.optimizers,
                    total_steps,
                    best_metrics["mean_return"] if best_metrics else -np.inf,
                    cfg.share_policy,
                    config=cfg.to_dict(),
                )
            if update % evaluation_interval == 0 or total_steps == cfg.total_timesteps:
                selection, records, score = evaluate_candidate(
                    agent, cfg, evaluation_seed, evaluation_episodes
                )
                selection_history.append(
                    {"step": total_steps, "metrics": selection, "episodes": records}
                )
                (log_dir / "development-evaluations.json").write_text(
                    json.dumps(selection_history, indent=2)
                )
                wandb.log(
                    {f"development/{k}": v for k, v in selection.items()},
                    step=total_steps,
                )
                logger.info(
                    "Development evaluation at %d: delivery %.1f%%, progress %.1f%%, return %.2f",
                    total_steps,
                    100 * selection["delivery_rate"],
                    100 * selection["mean_progress"],
                    selection["mean_return"],
                )
                if score > best_score:
                    best_score, best_metrics = score, selection
                    save_checkpoint(
                        str(log_dir / "best.pt"),
                        agent.models,
                        agent.optimizers,
                        total_steps,
                        selection["mean_return"],
                        cfg.share_policy,
                        config=cfg.to_dict(),
                    )
                    (log_dir / "selection.json").write_text(
                        json.dumps(
                            {
                                "step": total_steps,
                                "selection_seed": evaluation_seed,
                                "score": score,
                                "metrics": selection,
                            },
                            indent=2,
                        )
                    )
        run.summary.update(
            {
                "best_development_delivery_rate": best_metrics["delivery_rate"],
                "best_development_progress": best_metrics["mean_progress"],
                "environment_steps": total_steps,
            }
        )
        return {"global_step": total_steps, "best_development": best_metrics}
    finally:
        try:
            if workers is not None:
                workers.close()
            if env is not None:
                env.close()
        finally:
            run.finish()


if __name__ == "__main__":
    import fire

    fire.Fire({"train": train})
