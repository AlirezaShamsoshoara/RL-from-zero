from __future__ import annotations
import os
from dataclasses import asdict
from typing import Optional, List, Dict
import numpy as np
import torch
from tqdm import tqdm
import wandb
from gymnasium import spaces

from SAC.sac.config import Config
from SAC.sac.utils import set_seed, make_env, save_checkpoint, load_checkpoint
from SAC.sac.agent import SACAgent
from SAC.sac.replay_buffer import ReplayBuffer
from SAC.sac.logging_utils import setup_logger


def train(config: str = "SAC/configs/pendulum.yaml", wandb_key: str = ""):
    cfg = Config.from_yaml(config)
    env_wandb_key = os.getenv("WANDB_API_KEY", "")
    if wandb_key:
        cfg.wandb_key = wandb_key
    elif env_wandb_key:
        cfg.wandb_key = env_wandb_key

    logger = setup_logger(
        name="sac",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )

    set_seed(cfg.seed)

    if getattr(cfg, "wandb_key", ""):
        wandb.login(key=cfg.wandb_key)

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

    agent = SACAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=cfg.alpha_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        target_entropy_scale=cfg.target_entropy_scale,
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

    pbar = tqdm(range(1, cfg.total_steps + 1), desc="SAC Steps")
    for step in pbar:
        if step <= cfg.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        buffer_done = float(terminated)
        buffer.add(obs, action, reward, next_obs, buffer_done)
        obs = next_obs

        if done:
            obs, _ = env.reset()

        if buffer.can_sample(cfg.batch_size):
            for _ in range(cfg.updates_per_step):
                stats = agent.update(buffer.sample(cfg.batch_size))
                update_metrics.append(asdict(stats))

        if isinstance(info, dict) and "episode" in info:
            ep_r = float(info["episode"]["r"])
            ep_l = int(info["episode"]["l"])
            episode_returns.append(ep_r)
            episode_lengths.append(ep_l)

        if step % cfg.log_interval == 0:
            avg_return = float(np.mean(episode_returns[-10:])) if episode_returns else 0.0
            avg_length = float(np.mean(episode_lengths[-10:])) if episode_lengths else 0.0
            log_payload = {
                "charts/avg_return": avg_return,
                "charts/avg_length": avg_length,
                "progress/step": step,
            }
            if update_metrics:
                means = {
                    key: float(np.mean([m[key] for m in update_metrics]))
                    for key in update_metrics[0].keys()
                }
                update_metrics.clear()
                for key, value in means.items():
                    prefix = "loss" if "loss" in key else "stats"
                    log_payload[f"{prefix}/{key}"] = value
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
    config: str = "SAC/configs/pendulum.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
):
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="sac",
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

    agent = SACAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=cfg.alpha_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        target_entropy_scale=cfg.target_entropy_scale,
        device=cfg.device,
    )
    load_checkpoint(model_path, agent)
    agent.actor.eval()

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
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
