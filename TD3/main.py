from __future__ import annotations
import os
from dataclasses import asdict
from typing import Optional, List, Dict
import numpy as np
import torch
from tqdm import tqdm
import wandb
from gymnasium import spaces

from TD3.td3.config import Config
from TD3.td3.utils import set_seed, make_env, save_checkpoint, load_checkpoint
from TD3.td3.agent import TD3Agent, TD3Stats
from TD3.td3.replay_buffer import ReplayBuffer
from TD3.td3.logging_utils import setup_logger


def _stats_to_dict(stats: TD3Stats) -> Dict[str, float]:
    data = asdict(stats)
    if data["actor_loss"] is None:
        data.pop("actor_loss")
    return data


def train(config: str = "TD3/configs/pendulum.yaml", wandb_key: str = ""):
    cfg = Config.from_yaml(config)
    if wandb_key:
        cfg.wandb_key = wandb_key

    logger = setup_logger(
        name="td3",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )

    set_seed(cfg.seed)

    if getattr(cfg, "wandb_key", ""):
        import wandb as _wandb
        _wandb.login(key=cfg.wandb_key)

    logger.info(f"Initializing wandb run={cfg.run_name}")
    run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.run_name,
        config=cfg.to_dict(),
    )

    env = make_env(cfg.env_id, cfg.seed, render_mode=None, env_kwargs=cfg.env_kwargs)
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1
    assert isinstance(act_space, spaces.Box) and len(act_space.shape) == 1

    agent = TD3Agent(
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
    )

    buffer = ReplayBuffer(
        obs_dim=int(np.prod(obs_space.shape)),
        act_dim=int(np.prod(act_space.shape)),
        capacity=cfg.buffer_size,
        device=torch.device(cfg.device),
    )

    obs, _ = env.reset(seed=cfg.seed)
    best_avg_return = -np.inf
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    update_metrics: List[Dict[str, float]] = []

    pbar = tqdm(range(1, cfg.total_steps + 1), desc="TD3 Steps")
    for step in pbar:
        if step <= cfg.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, noise=cfg.exploration_noise, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        buffer_done = float(terminated)
        buffer.add(obs, action, reward, next_obs, buffer_done)
        obs = next_obs

        if done:
            obs, _ = env.reset()

        if step >= cfg.start_steps and buffer.can_sample(cfg.batch_size):
            for _ in range(cfg.updates_per_step):
                stats = agent.update(buffer.sample(cfg.batch_size))
                stat_dict = _stats_to_dict(stats)
                update_metrics.append(stat_dict)

        if isinstance(info, dict) and "episode" in info:
            ep_r = float(info["episode"]["r"])
            ep_l = int(info["episode"]["l"])
            episode_returns.append(ep_r)
            episode_lengths.append(ep_l)

        if step % cfg.log_interval == 0:
            avg_return = float(np.mean(episode_returns[-10:])) if episode_returns else 0.0
            avg_length = float(np.mean(episode_lengths[-10:])) if episode_lengths else 0.0
            log_payload: Dict[str, float] = {
                "charts/avg_return": avg_return,
                "charts/avg_length": avg_length,
                "progress/step": step,
            }
            if update_metrics:
                keys = update_metrics[0].keys()
                for key in keys:
                    values = [m[key] for m in update_metrics if key in m]
                    if values:
                        prefix = "loss" if "loss" in key else "stats"
                        log_payload[f"{prefix}/{key}"] = float(np.mean(values))
                update_metrics.clear()
            wandb.log(log_payload, step=step)
            pbar.set_postfix({"avgR": f"{avg_return:.1f}"})

        if step % cfg.checkpoint_interval == 0:
            path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{step}.pt")
            save_checkpoint(path, agent, step, best_avg_return)
            logger.info(f"Saved checkpoint: {path}")

        if cfg.save_best and len(episode_returns) >= 5:
            avg_recent = float(np.mean(episode_returns[-5:]))
            if avg_recent > best_avg_return:
                best_avg_return = avg_recent
                best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                save_checkpoint(best_path, agent, step, best_avg_return)
                logger.info(
                    f"New best avg return {best_avg_return:.2f}; saved {best_path}"
                )

    env.close()
    run.finish()
    logger.info(
        f"Training finished. Best 5-ep avg return: {best_avg_return:.2f}"
    )


def demo(
    config: str = "TD3/configs/pendulum.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
    exploration_noise: float = 0.0,
):
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="td3",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )
    model_path = model_path or cfg.inference_model_path
    episodes = episodes or cfg.episodes

    set_seed(cfg.seed)
    env = make_env(
        cfg.env_id,
        cfg.seed,
        render_mode=cfg.render_mode or "human",
        env_kwargs=cfg.env_kwargs,
    )
    obs_space = env.observation_space
    act_space = env.action_space
    assert isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1
    assert isinstance(act_space, spaces.Box) and len(act_space.shape) == 1

    agent = TD3Agent(
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
    )
    load_checkpoint(model_path, agent)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(
                obs,
                noise=exploration_noise,
                deterministic=exploration_noise <= 0.0,
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
        logger.info(f"Episode {ep + 1}: return={ep_ret:.2f}")
        returns.append(ep_ret)

    avg_return = float(np.mean(returns)) if returns else 0.0
    logger.info(f"Average return over {episodes} episodes: {avg_return:.2f}")
    env.close()


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "train": train,
            "demo": demo,
        }
    )
