from __future__ import annotations

import math
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import wandb
from gymnasium import spaces
from tqdm import tqdm

from TD3.td3.agent import TD3Agent, TD3Stats
from TD3.td3.config import Config
from TD3.td3.logging_utils import setup_logger
from TD3.td3.replay_buffer import ReplayBuffer
from TD3.td3.utils import (
    load_checkpoint,
    make_env,
    make_vec_env,
    save_checkpoint,
    set_seed,
)


def _stats_to_dict(stats: TD3Stats):
    return {key: value for key, value in asdict(stats).items() if value is not None}


def _logger(cfg):
    return setup_logger(
        name="td3",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )


def _agent(cfg, obs_space, act_space):
    if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) != 1:
        raise ValueError("TD3 requires flat Box observations")
    if not isinstance(act_space, spaces.Box) or len(act_space.shape) != 1:
        raise ValueError("TD3 requires flat Box actions")
    return TD3Agent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        policy_delay=cfg.policy_delay,
        target_noise=cfg.target_noise,
        noise_clip=cfg.noise_clip,
        device=cfg.device,
        max_grad_norm=cfg.max_grad_norm,
    )


def _endpoints(next_obs, infos, boundaries):
    endpoints = np.asarray(next_obs, dtype=np.float32).copy()
    for index in np.flatnonzero(boundaries):
        found = False
        for key in ("final_obs", "final_observation"):
            if key not in infos:
                continue
            mask = infos.get("_" + key)
            if mask is not None and not bool(mask[index]):
                continue
            value = infos[key][index]
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float32)
            if value.shape != endpoints[index].shape or not np.isfinite(value).all():
                raise ValueError("invalid vector final observation")
            endpoints[index] = value
            found = True
            break
        if not found:
            raise ValueError(
                "SAME_STEP boundary is missing its actual final observation"
            )
    return endpoints


def train(config: str = "TD3/configs/mountaincar_balanced.yaml", wandb_key: str = ""):
    cfg = Config.from_yaml(config).validate()
    key = os.getenv("WANDB_API_KEY", "")
    if wandb_key:
        cfg.wandb_key = wandb_key
    elif key:
        cfg.wandb_key = key
    logger = _logger(cfg)
    set_seed(cfg.seed)
    if cfg.wandb_key:
        wandb.login(key=cfg.wandb_key)
    run = wandb.init(
        project=cfg.project, entity=cfg.entity, name=cfg.run_name, config=cfg.to_dict()
    )
    env = pbar = None
    try:
        env = make_vec_env(
            cfg.env_id,
            cfg.num_envs,
            cfg.seed,
            render_mode=None,
            env_kwargs=cfg.env_kwargs,
        )
        agent = _agent(cfg, env.single_observation_space, env.single_action_space)
        buffer = ReplayBuffer(
            agent.obs_dim, agent.act_dim, cfg.buffer_size, agent.device
        )
        obs, _ = env.reset(seed=[cfg.seed + rank for rank in range(cfg.num_envs)])
        ep_returns = np.zeros(cfg.num_envs, np.float64)
        ep_lengths = np.zeros(cfg.num_envs, np.int64)
        episode_returns, episode_lengths = [], []
        pulse_actions = np.zeros((cfg.num_envs, agent.act_dim), np.float32)
        pulse_remaining = np.zeros(cfg.num_envs, np.int64)
        total_steps = vector_steps = optimizer_updates = actor_updates = (
            pulse_transitions
        ) = 0
        best_avg_return = None
        update_metrics = []
        next_log = cfg.log_interval if cfg.log_interval else None
        next_checkpoint = cfg.checkpoint_interval if cfg.checkpoint_interval else None
        pbar = tqdm(total=cfg.total_steps, desc="TD3 environment transitions")
        while total_steps < cfg.total_steps:
            state = np.asarray(obs, dtype=np.float32).copy()
            if total_steps < cfg.start_steps:
                actions = env.action_space.sample()
            else:
                actions = agent.act(
                    state, noise=cfg.exploration_noise, deterministic=False
                )
                if cfg.random_action_prob > 0:
                    expired = pulse_remaining <= 0
                    starts = expired & (
                        np.random.rand(cfg.num_envs) < cfg.random_action_prob
                    )
                    for rank in np.flatnonzero(starts):
                        pulse_actions[rank] = env.single_action_space.sample()
                    if starts.any():
                        pulse_remaining[starts] = np.random.randint(
                            cfg.random_action_hold_min,
                            cfg.random_action_hold_max + 1,
                            size=int(starts.sum()),
                        )
                    pulsing = pulse_remaining > 0
                    actions[pulsing] = pulse_actions[pulsing]
                    pulse_transitions += int(pulsing.sum())
                    pulse_remaining[pulsing] -= 1
            actions = np.asarray(actions, dtype=np.float32).copy()
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            terminals = np.asarray(terminated, dtype=bool)
            boundaries = terminals | np.asarray(truncated, dtype=bool)
            endpoints = _endpoints(next_obs, infos, boundaries)
            buffer.add_batch(
                state, actions, rewards, endpoints, terminals.astype(np.float32)
            )
            ep_returns += np.asarray(rewards, dtype=np.float64)
            ep_lengths += 1
            candidate_score = None
            for rank in np.flatnonzero(boundaries):
                episode_returns.append(float(ep_returns[rank]))
                episode_lengths.append(int(ep_lengths[rank]))
                if cfg.save_best and len(episode_returns) >= cfg.best_window:
                    score = float(np.mean(episode_returns[-cfg.best_window :]))
                    if candidate_score is None or score > candidate_score:
                        candidate_score = score
                ep_returns[rank] = 0.0
                ep_lengths[rank] = 0
            pulse_actions[boundaries] = 0
            pulse_remaining[boundaries] = 0
            obs = np.asarray(next_obs, dtype=np.float32).copy()
            total_steps += cfg.num_envs
            vector_steps += 1
            pbar.update(cfg.num_envs)
            if total_steps >= cfg.update_after and buffer.can_sample(cfg.batch_size):
                # updates_per_step is per vector step, not secretly per interaction.
                for _ in range(cfg.updates_per_step):
                    stats = agent.update(buffer.sample(cfg.batch_size))
                    update_metrics.append(_stats_to_dict(stats))
                    optimizer_updates += 1
                    actor_updates += int(stats.actor_loss is not None)
            metadata = {
                "environment_steps": total_steps,
                "vector_steps": vector_steps,
                "optimizer_updates": optimizer_updates,
                "actor_updates": actor_updates,
                "training_completed": False,
            }
            if candidate_score is not None and (
                best_avg_return is None or candidate_score > best_avg_return
            ):
                best_avg_return = candidate_score
                save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, "best.pt"),
                    agent,
                    total_steps,
                    candidate_score,
                    config=cfg.to_dict(),
                    metadata=metadata,
                )
            if (
                next_log is not None and total_steps >= next_log
            ) or total_steps == cfg.total_steps:
                payload = {
                    "progress/step": total_steps,
                    "progress/global_step": total_steps,
                    "progress/vector_steps": vector_steps,
                    "progress/optimizer_updates": optimizer_updates,
                    "progress/actor_updates": actor_updates,
                    "stats/updates_per_transition": optimizer_updates / total_steps,
                    "stats/pulse_transitions": pulse_transitions,
                }
                if episode_returns:
                    payload["charts/avg_return"] = float(
                        np.mean(episode_returns[-cfg.best_window :])
                    )
                    payload["charts/avg_length"] = float(
                        np.mean(episode_lengths[-cfg.best_window :])
                    )
                keys = (
                    set().union(*(row.keys() for row in update_metrics))
                    if update_metrics
                    else set()
                )
                for key in keys:
                    values = [row[key] for row in update_metrics if key in row]
                    payload[f"{'loss' if 'loss' in key else 'stats'}/{key}"] = float(
                        np.mean(values)
                    )
                update_metrics.clear()
                wandb.log(payload, step=total_steps)
                measured = payload.get("charts/avg_return")
                pbar.set_postfix(
                    {
                        "avgR": f"{measured:.1f}"
                        if measured is not None
                        else "unmeasured"
                    }
                )
                logger.info(
                    "step=%d native_return=%s updates=%d",
                    total_steps,
                    measured if measured is not None else "unmeasured",
                    optimizer_updates,
                )
                if cfg.log_interval:
                    next_log = (total_steps // cfg.log_interval + 1) * cfg.log_interval
            if next_checkpoint is not None and total_steps >= next_checkpoint:
                save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, f"checkpoint_{total_steps}.pt"),
                    agent,
                    total_steps,
                    best_avg_return,
                    config=cfg.to_dict(),
                    metadata=metadata,
                )
                next_checkpoint = (
                    total_steps // cfg.checkpoint_interval + 1
                ) * cfg.checkpoint_interval
        metadata["training_completed"] = True
        save_checkpoint(
            os.path.join(cfg.checkpoint_dir, f"checkpoint_{total_steps}.pt"),
            agent,
            total_steps,
            best_avg_return,
            config=cfg.to_dict(),
            metadata=metadata,
        )
        return {
            "global_step": total_steps,
            "environment_steps": total_steps,
            "vector_steps": vector_steps,
            "optimizer_updates": optimizer_updates,
            "actor_updates": actor_updates,
            "native_episodes": len(episode_returns),
            "best_avg_return": best_avg_return,
        }
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            try:
                if pbar is not None:
                    pbar.close()
            finally:
                run.finish()


def demo(
    config: str = "TD3/configs/mountaincar_balanced.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
    exploration_noise: float = 0.0,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
):
    cfg = Config.from_yaml(config).validate()
    episodes = cfg.episodes if episodes is None else episodes
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if not math.isfinite(exploration_noise) or exploration_noise < 0:
        raise ValueError("exploration_noise must be finite and nonnegative")
    seed = cfg.seed if seed is None else seed
    render_mode = cfg.render_mode if render_mode is None else render_mode
    if render_mode == "none":
        render_mode = None
    set_seed(seed)
    logger = _logger(cfg)
    env = make_env(cfg.env_id, seed, render_mode=render_mode, env_kwargs=cfg.env_kwargs)
    try:
        agent = _agent(cfg, env.observation_space, env.action_space)
        load_checkpoint(model_path or cfg.inference_model_path, agent)
        agent.actor.eval()
        returns, lengths, terminal_flags, truncation_flags, cap_flags = (
            [],
            [],
            [],
            [],
            [],
        )
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total = 0.0
            terminated = truncated = False
            for step in range(cfg.max_steps_per_episode):
                action = agent.act(
                    obs, noise=exploration_noise, deterministic=exploration_noise == 0
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(step + 1)
            terminal_flags.append(bool(terminated))
            truncation_flags.append(bool(truncated))
            cap_flags.append(not (terminated or truncated))
            logger.info(
                "Episode %d native_return=%.2f length=%d", episode + 1, total, step + 1
            )
        return {
            "returns": returns,
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "lengths": lengths,
            "terminated": terminal_flags,
            "truncated": truncation_flags,
            "capped": cap_flags,
        }
    finally:
        env.close()


if __name__ == "__main__":
    import fire

    fire.Fire({"train": train, "demo": demo})
