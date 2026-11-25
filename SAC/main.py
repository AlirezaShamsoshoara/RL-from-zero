from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import wandb
from gymnasium import spaces
from tqdm import tqdm

from SAC.sac.agent import SACAgent
from SAC.sac.config import Config
from SAC.sac.logging_utils import setup_logger
from SAC.sac.replay_buffer import ReplayBuffer
from SAC.sac.utils import load_checkpoint, make_env, save_checkpoint, set_seed


def _logger(cfg):
    return setup_logger(
        name="sac",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )


def _agent(cfg, env):
    if (
        not isinstance(env.observation_space, spaces.Box)
        or len(env.observation_space.shape) != 1
    ):
        raise ValueError("SAC requires flat Box observations")
    if not isinstance(env.action_space, spaces.Box) or len(env.action_space.shape) != 1:
        raise ValueError("SAC requires flat Box actions")
    return SACAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=cfg.alpha_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        target_entropy_scale=cfg.target_entropy_scale,
        device=cfg.device,
        max_grad_norm=cfg.max_grad_norm,
    )


def train(config: str = "SAC/configs/pendulum_fast.yaml", wandb_key: str = ""):
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
        env = make_env(
            cfg.env_id, cfg.seed, render_mode=None, env_kwargs=cfg.env_kwargs
        )
        agent = _agent(cfg, env)
        buffer = ReplayBuffer(
            agent.obs_dim, agent.act_dim, cfg.buffer_size, agent.device
        )
        obs, _ = env.reset(seed=cfg.seed)
        episode_returns, episode_lengths = [], []
        capped_episodes = 0
        ep_return, ep_length = 0.0, 0
        update_metrics = []
        best_avg_return = None
        update_steps = 0
        pbar = tqdm(range(1, cfg.total_steps + 1), desc="SAC environment transitions")
        for step in pbar:
            # Snapshot before step: Gym environments can reuse observation arrays.
            state = np.asarray(obs, dtype=np.float32).copy()
            action = (
                env.action_space.sample()
                if step <= cfg.start_steps
                else agent.act(state, deterministic=False)
            )
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            capped = ep_length >= cfg.max_steps_per_episode and not (
                terminated or truncated
            )
            boundary = bool(terminated or truncated or capped)
            endpoint = next_obs
            if boundary and isinstance(info, dict):
                endpoint = info.get(
                    "final_observation", info.get("final_obs", next_obs)
                )
            # Only true terminals suppress the continuing soft-value bootstrap.
            buffer.add(state, action, reward, endpoint, float(terminated))
            obs = next_obs
            if boundary:
                native_completed = bool(terminated or truncated)
                if native_completed:
                    episode_returns.append(ep_return)
                    episode_lengths.append(ep_length)
                else:
                    capped_episodes += 1
                ep_return, ep_length = 0.0, 0
                obs, _ = env.reset()
                if (
                    native_completed
                    and cfg.save_best
                    and len(episode_returns) >= cfg.best_window
                ):
                    mean = float(np.mean(episode_returns[-cfg.best_window :]))
                    if best_avg_return is None or mean > best_avg_return:
                        best_avg_return = mean
                        save_checkpoint(
                            os.path.join(cfg.checkpoint_dir, "best.pt"),
                            agent,
                            step,
                            best_avg_return,
                            config=cfg.to_dict(),
                            metadata={
                                "environment_steps": step,
                                "optimizer_updates": update_steps,
                                "training_completed": False,
                            },
                        )
            if step >= cfg.update_after and buffer.can_sample(cfg.batch_size):
                for _ in range(cfg.updates_per_step):
                    update_metrics.append(
                        asdict(agent.update(buffer.sample(cfg.batch_size)))
                    )
                    update_steps += 1
            if (
                cfg.log_interval and step % cfg.log_interval == 0
            ) or step == cfg.total_steps:
                payload = {
                    "progress/step": step,
                    "progress/global_step": step,
                    "progress/optimizer_updates": update_steps,
                    "progress/capped_episodes": capped_episodes,
                }
                if episode_returns:
                    payload["charts/avg_return"] = float(
                        np.mean(episode_returns[-cfg.best_window :])
                    )
                    payload["charts/avg_length"] = float(
                        np.mean(episode_lengths[-cfg.best_window :])
                    )
                if update_metrics:
                    keys = update_metrics[0].keys()
                    means = {
                        name: float(
                            np.mean([record[name] for record in update_metrics])
                        )
                        for name in keys
                    }
                    payload.update(
                        {
                            f"{'loss' if 'loss' in name else 'stats'}/{name}": value
                            for name, value in means.items()
                        }
                    )
                    update_metrics.clear()
                wandb.log(payload, step=step)
                pbar.set_postfix(
                    {
                        "avgR": f"{payload['charts/avg_return']:.1f}"
                        if episode_returns
                        else "unmeasured"
                    }
                )
                logger.info(
                    "step=%d native_return=%s length=%s updates=%d",
                    step,
                    payload.get("charts/avg_return", "unmeasured"),
                    payload.get("charts/avg_length", "unmeasured"),
                    update_steps,
                )
            if cfg.checkpoint_interval and step % cfg.checkpoint_interval == 0:
                save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, f"checkpoint_{step}.pt"),
                    agent,
                    step,
                    best_avg_return,
                    config=cfg.to_dict(),
                    metadata={
                        "environment_steps": step,
                        "optimizer_updates": update_steps,
                        "training_completed": step == cfg.total_steps,
                    },
                )
        save_checkpoint(
            os.path.join(cfg.checkpoint_dir, f"checkpoint_{cfg.total_steps}.pt"),
            agent,
            cfg.total_steps,
            best_avg_return,
            config=cfg.to_dict(),
            metadata={
                "environment_steps": cfg.total_steps,
                "optimizer_updates": update_steps,
                "training_completed": True,
            },
        )
        return {
            "global_step": cfg.total_steps,
            "environment_steps": cfg.total_steps,
            "optimizer_updates": update_steps,
            "episodes": len(episode_returns),
            "capped_episodes": capped_episodes,
            "best_avg_return": best_avg_return,
        }
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            try:
                if pbar is not None:
                    close = getattr(pbar, "close", None)
                    if close is not None:
                        close()
            finally:
                run.finish()


def demo(
    config: str = "SAC/configs/pendulum_fast.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
):
    cfg = Config.from_yaml(config).validate()
    episodes = cfg.episodes if episodes is None else episodes
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    seed = cfg.seed if seed is None else seed
    render_mode = cfg.render_mode if render_mode is None else render_mode
    if render_mode == "none":
        render_mode = None
    set_seed(seed)
    logger = _logger(cfg)
    env = make_env(cfg.env_id, seed, render_mode=render_mode, env_kwargs=cfg.env_kwargs)
    try:
        agent = _agent(cfg, env)
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
                obs, reward, terminated, truncated, _ = env.step(
                    agent.act(obs, deterministic=True)
                )
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(step + 1)
            terminal_flags.append(bool(terminated))
            truncation_flags.append(bool(truncated))
            cap_flags.append(not (terminated or truncated))
            logger.info(
                "Episode %d: native_return=%.2f length=%d", episode + 1, total, step + 1
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
