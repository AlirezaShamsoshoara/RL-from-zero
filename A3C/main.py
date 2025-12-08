from __future__ import annotations
import os
import queue
from collections import deque
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
import gymnasium as gym
from gymnasium import spaces

from A3C.a3c.config import Config
from A3C.a3c.agent import A3CAgent
from A3C.a3c.logging_utils import setup_logger
from A3C.a3c.utils import set_seed, save_checkpoint, load_checkpoint
from A3C.a3c.worker import worker_process


def _log_mean(stats: list[Dict[str, float]]) -> Dict[str, float]:
    if not stats:
        return {}
    keys = stats[0].keys()
    return {key: float(np.mean([item[key] for item in stats])) for key in keys}


def train(config: str = "A3C/configs/cartpole.yaml", wandb_key: str = ""):
    cfg = Config.from_yaml(config)
    if wandb_key:
        cfg.wandb_key = wandb_key

    logger = setup_logger(
        name="a3c",
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

    # Inspect environment spaces once on the main process
    env = gym.make(cfg.env_id)
    env.reset(seed=cfg.seed)
    obs_space = env.observation_space
    act_space = env.action_space
    env.close()

    if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) != 1:
        raise ValueError("A3C supports only flat Box observation spaces")
    if not isinstance(act_space, spaces.Discrete):
        raise ValueError("A3C supports only discrete action spaces")

    agent = A3CAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        learning_rate=cfg.learning_rate,
        entropy_coef=cfg.entropy_coef,
        value_loss_coef=cfg.value_loss_coef,
        max_grad_norm=cfg.max_grad_norm,
        device=cfg.device,
    )

    logger.info(
        f"Env={cfg.env_id} | obs_dim={agent.obs_dim} | act_dim={agent.act_dim} | workers={cfg.num_workers}"
    )

    ctx = mp.get_context("spawn")
    global_step = ctx.Value("i", 0)
    result_queue: mp.Queue = ctx.Queue()  # type: ignore[assignment]
    stop_event = ctx.Event()

    processes: list[mp.Process] = []
    for worker_id in range(cfg.num_workers):
        proc = ctx.Process(
            target=worker_process,
            args=(worker_id, cfg, agent, global_step, result_queue, stop_event),
            daemon=True,
        )
        proc.start()
        processes.append(proc)

    episode_returns = deque(maxlen=200)
    episode_lengths = deque(maxlen=200)
    update_stats: list[Dict[str, float]] = []
    best_avg_return = -np.inf
    next_log_step = cfg.log_interval if cfg.log_interval > 0 else None
    next_checkpoint = cfg.checkpoint_interval if cfg.checkpoint_interval > 0 else None
    latest_step = 0

    try:
        while True:
            alive = any(p.is_alive() for p in processes)
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                if not alive:
                    break
                continue

            latest_step = max(latest_step, int(message.get("global_step", latest_step)))

            if message.get("kind") == "episode":
                episode_returns.append(float(message["episode_return"]))
                episode_lengths.append(int(message["episode_length"]))
                if cfg.save_best and len(episode_returns) >= 5:
                    recent = list(episode_returns)[-10:]
                    avg_recent = float(np.mean(recent))
                    if avg_recent > best_avg_return:
                        best_avg_return = avg_recent
                        best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                        save_checkpoint(best_path, agent.model, agent.optimizer, latest_step, best_avg_return)
                        logger.info(
                            f"New best avg return {best_avg_return:.2f}; saved {best_path}"
                        )
            elif message.get("kind") == "update":
                update_stats.append(message["stats"])

            if next_log_step is not None and latest_step >= next_log_step:
                avg_return = float(np.mean(list(episode_returns)[-10:])) if episode_returns else 0.0
                avg_length = float(np.mean(list(episode_lengths)[-10:])) if episode_lengths else 0.0
                log_payload: Dict[str, Any] = {
                    "charts/avg_return": avg_return,
                    "charts/avg_length": avg_length,
                    "progress/global_step": latest_step,
                }
                mean_stats = _log_mean(update_stats)
                if mean_stats:
                    for key, value in mean_stats.items():
                        prefix = "loss" if "loss" in key else "stats"
                        log_payload[f"{prefix}/{key}"] = value
                    update_stats.clear()
                wandb.log(log_payload, step=latest_step)
                next_log_step += cfg.log_interval

            if next_checkpoint is not None and latest_step >= next_checkpoint:
                checkpoint_path = os.path.join(
                    cfg.checkpoint_dir, f"checkpoint_{latest_step}.pt"
                )
                save_checkpoint(checkpoint_path, agent.model, agent.optimizer, latest_step, best_avg_return)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
                next_checkpoint += cfg.checkpoint_interval

            if not alive and result_queue.empty():
                break
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    finally:
        stop_event.set()
        for proc in processes:
            proc.join()

    run.finish()
    logger.info(
        f"Training finished. Best 10-ep avg return: {best_avg_return:.2f}"
    )


def demo(
    config: str = "A3C/configs/cartpole.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
):
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="a3c",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )
    model_path = model_path or cfg.inference_model_path
    episodes = episodes or cfg.episodes

    set_seed(cfg.seed)

    env = gym.make(cfg.env_id, render_mode=cfg.render_mode or "human")
    env.reset(seed=cfg.seed)
    obs_space = env.observation_space
    act_space = env.action_space

    agent = A3CAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        learning_rate=cfg.learning_rate,
        entropy_coef=cfg.entropy_coef,
        value_loss_coef=cfg.value_loss_coef,
        max_grad_norm=cfg.max_grad_norm,
        device=cfg.device,
    )
    load_checkpoint(model_path, agent.model)
    agent.model.eval()
    logger.info(f"Loaded model from {model_path}")

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                logits, _ = agent.model.forward(obs_t)
                action = int(torch.argmax(logits).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
        logger.info(f"Episode {ep + 1}: return={ep_return:.2f}")
        returns.append(ep_return)

    avg_return = float(np.mean(returns)) if returns else 0.0
    logger.info(f"Average return over {episodes} episodes: {avg_return:.2f}")
    env.close()


if __name__ == "__main__":
    import torch
    import fire

    fire.Fire(
        {
            "train": train,
            "demo": demo,
        }
    )
