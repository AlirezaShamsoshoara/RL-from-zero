"""DQN entry points.

Examples:
    >>> # Train on MountainCar-v0
    >>> # python -m deepQN.main train --config deepQN/configs/mountaincar.yaml
    >>> # Run a demo with a saved checkpoint
    >>> # python -m deepQN.main demo --config deepQN/configs/mountaincar.yaml --model_path deepQN/checkpoints/best.pt
"""

from __future__ import annotations
import os
from typing import Optional, List, Dict
import numpy as np
import torch
from tqdm import tqdm
import wandb
from gymnasium import spaces

from deepQN.dqn.config import Config
from deepQN.dqn.utils import set_seed, make_env, save_checkpoint, load_checkpoint
from deepQN.dqn.agent import DQNAgent
from deepQN.dqn.replay_buffer import ReplayBuffer
from deepQN.dqn.logging_utils import setup_logger


def _epsilon_by_step(step: int, cfg: Config) -> float:
    if cfg.epsilon_decay_steps <= 0:
        return cfg.epsilon_end
    ratio = min(1.0, max(0.0, step) / float(cfg.epsilon_decay_steps))
    return cfg.epsilon_start + ratio * (cfg.epsilon_end - cfg.epsilon_start)


def train(config: str = "deepQN/configs/mountaincar.yaml", wandb_key: str = ""):
    cfg = Config.from_yaml(config)
    if wandb_key:
        cfg.wandb_key = wandb_key

    logger = setup_logger(
        name="deepqn",
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
    assert isinstance(act_space, spaces.Discrete)

    device = torch.device(cfg.device)
    agent = DQNAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        lr=cfg.lr,
        gamma=cfg.gamma,
        target_update_interval=cfg.target_update_interval,
        max_grad_norm=cfg.max_grad_norm,
        double_dqn=cfg.double_dqn,
        device=cfg.device,
    )

    buffer = ReplayBuffer(
        obs_dim=int(np.prod(obs_space.shape)),
        capacity=cfg.buffer_size,
        device=device,
    )

    obs, _ = env.reset(seed=cfg.seed)
    ep_return = 0.0
    ep_length = 0
    best_avg_return = -np.inf
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    update_metrics: List[Dict[str, float]] = []

    pbar = tqdm(range(1, cfg.total_steps + 1), desc="DQN Steps")
    for step in pbar:
        if step <= cfg.learning_starts:
            epsilon = cfg.epsilon_start
            action = act_space.sample()
        else:
            epsilon = _epsilon_by_step(step - cfg.learning_starts, cfg)
            action = agent.act(obs, epsilon=epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        buffer_done = float(terminated)
        buffer.add(obs, action, reward, next_obs, buffer_done)

        ep_return += float(reward)
        ep_length += 1
        obs = next_obs

        if step >= cfg.learning_starts and step % cfg.train_freq == 0 and buffer.can_sample(cfg.batch_size):
            stats = agent.update(buffer.sample(cfg.batch_size))
            update_metrics.append({"loss": stats.loss, "td_error": stats.td_error})

        if done:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            obs, _ = env.reset()
            ep_return = 0.0
            ep_length = 0

        if step % cfg.log_interval == 0:
            avg_return = float(np.mean(episode_returns[-10:])) if episode_returns else 0.0
            avg_length = float(np.mean(episode_lengths[-10:])) if episode_lengths else 0.0
            log_payload = {
                "charts/avg_return": avg_return,
                "charts/avg_length": avg_length,
                "charts/epsilon": epsilon,
                "charts/buffer_size": len(buffer),
                "progress/step": step,
            }
            if update_metrics:
                loss_mean = float(np.mean([m["loss"] for m in update_metrics]))
                td_mean = float(np.mean([m["td_error"] for m in update_metrics]))
                log_payload["loss/q_loss"] = loss_mean
                log_payload["stats/td_error"] = td_mean
                update_metrics.clear()
            wandb.log(log_payload, step=step)
            pbar.set_postfix({"avgR": f"{avg_return:.1f}", "eps": f"{epsilon:.2f}"})

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
    config: str = "deepQN/configs/mountaincar.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
):
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="deepqn",
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
    assert isinstance(act_space, spaces.Discrete)

    agent = DQNAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        lr=cfg.lr,
        gamma=cfg.gamma,
        target_update_interval=cfg.target_update_interval,
        max_grad_norm=cfg.max_grad_norm,
        double_dqn=cfg.double_dqn,
        device=cfg.device,
    )
    load_checkpoint(model_path, agent)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(obs, epsilon=cfg.eval_epsilon, deterministic=True)
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
