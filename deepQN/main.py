from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm

from deepQN.dqn.agent import DQNAgent
from deepQN.dqn.config import Config
from deepQN.dqn.exploration import ActionHoldExplorer
from deepQN.dqn.logging_utils import setup_logger
from deepQN.dqn.replay_buffer import ReplayBuffer
from deepQN.dqn.utils import load_checkpoint, make_env, save_checkpoint, set_seed


def _epsilon_by_step(step: int, cfg: Config) -> float:
    if cfg.epsilon_decay_steps <= 0:
        return cfg.epsilon_end
    ratio = min(1.0, max(0.0, step) / float(cfg.epsilon_decay_steps))
    return cfg.epsilon_start + ratio * (cfg.epsilon_end - cfg.epsilon_start)


def _agent(cfg: Config, env) -> DQNAgent:
    return DQNAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        lr=cfg.lr,
        gamma=cfg.gamma,
        target_update_interval=cfg.target_update_interval,
        max_grad_norm=cfg.max_grad_norm,
        double_dqn=cfg.double_dqn,
        device=cfg.device,
        normalize_observations=cfg.normalize_observations,
    )


def _logger(cfg: Config):
    return setup_logger(
        name="deepqn",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )


def train(
    config: str = "deepQN/configs/mountaincar_exploration.yaml", wandb_key: str = ""
):
    """Train DQN on native rewards; evaluate saved policies separately."""
    cfg = Config.from_yaml(config)
    env_key = os.getenv("WANDB_API_KEY", "")
    if wandb_key:
        cfg.wandb_key = wandb_key
    elif env_key:
        cfg.wandb_key = env_key
    logger = _logger(cfg)
    set_seed(cfg.seed)
    if cfg.wandb_key:
        wandb.login(key=cfg.wandb_key)
    run = wandb.init(
        project=cfg.project, entity=cfg.entity, name=cfg.run_name, config=cfg.to_dict()
    )
    env = None
    try:
        env = make_env(
            cfg.env_id, cfg.seed, render_mode=None, env_kwargs=cfg.env_kwargs
        )
        agent = _agent(cfg, env)
        explorer = ActionHoldExplorer(
            cfg.exploration_hold_min, cfg.exploration_hold_max
        )
        exploratory_steps = exploration_window_steps = 0
        buffer = ReplayBuffer(agent.obs_dim, cfg.buffer_size, torch.device(cfg.device))
        obs, _ = env.reset(seed=cfg.seed)
        start_q = agent.q_value(obs)
        mountaincar = cfg.env_id == "MountainCar-v0"
        max_position = float(obs[0]) if mountaincar else None
        ep_return = ep_discounted = 0.0
        ep_length = 0
        episode_returns, episode_lengths = [], []
        episode_discounted, episode_start_q = [], []
        episode_terminations, episode_positions = [], []
        update_metrics = []
        best_avg_return = -np.inf
        pbar = tqdm(range(1, cfg.total_steps + 1), desc="DQN Steps")
        for step in pbar:
            epsilon = (
                1.0
                if step <= cfg.learning_starts
                else _epsilon_by_step(step - cfg.learning_starts, cfg)
            )
            action, exploratory = explorer.act(agent, obs, epsilon)
            if exploratory is not None:
                exploratory_steps += int(exploratory)
                exploration_window_steps += 1
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_length += 1
            capped = ep_length >= cfg.max_steps_per_episode and not (
                terminated or truncated
            )
            ended = bool(terminated or truncated or capped)
            # This is a manually reset single env. If a wrapper supplies a final
            # observation, retain that actual transition endpoint in replay.
            transition_next_obs = next_obs
            if ended and isinstance(info, dict):
                final = info.get("final_observation", info.get("final_obs"))
                if final is not None:
                    transition_next_obs = final
            buffer.add(
                obs, action, float(reward), transition_next_obs, float(terminated)
            )
            ep_return += float(reward)
            ep_discounted += cfg.gamma ** (ep_length - 1) * float(reward)
            if mountaincar:
                max_position = max(max_position, float(transition_next_obs[0]))
            obs = next_obs
            if (
                step >= cfg.learning_starts
                and step % cfg.train_freq == 0
                and buffer.can_sample(cfg.batch_size)
            ):
                update_metrics.append(
                    asdict(agent.update(buffer.sample(cfg.batch_size)))
                )
            if ended:
                episode_returns.append(ep_return)
                episode_lengths.append(ep_length)
                episode_discounted.append(ep_discounted)
                episode_start_q.append(start_q)
                episode_terminations.append(float(terminated))
                if mountaincar:
                    episode_positions.append(max_position)
                obs, _ = env.reset()
                explorer.reset()
                start_q = agent.q_value(obs)
                max_position = float(obs[0]) if mountaincar else None
                ep_return = ep_discounted = 0.0
                ep_length = 0
            if step % cfg.log_interval == 0 or step == cfg.total_steps:
                payload = {
                    "charts/avg_return": float(np.mean(episode_returns[-10:]))
                    if episode_returns
                    else 0.0,
                    "charts/avg_length": float(np.mean(episode_lengths[-10:]))
                    if episode_lengths
                    else 0.0,
                    "charts/termination_rate": float(
                        np.mean(episode_terminations[-10:])
                    )
                    if episode_terminations
                    else 0.0,
                    "charts/episodes": len(episode_returns),
                    "charts/epsilon": epsilon,
                    "charts/buffer_size": len(buffer),
                    "progress/step": step,
                }
                if episode_returns:
                    payload["stats/start_q"] = float(np.mean(episode_start_q[-10:]))
                    payload["stats/discounted_return"] = float(
                        np.mean(episode_discounted[-10:])
                    )
                if mountaincar:
                    payload["charts/success_rate"] = payload["charts/termination_rate"]
                    payload["charts/max_position"] = (
                        float(np.mean(episode_positions[-10:]))
                        if episode_positions
                        else float(obs[0])
                    )
                if update_metrics:
                    for name, key in (
                        ("loss", "loss/q_loss"),
                        ("td_error", "stats/td_error"),
                        ("q_value", "stats/q_value"),
                        ("target_value", "stats/target_value"),
                    ):
                        payload[key] = float(
                            np.mean([row[name] for row in update_metrics])
                        )
                    update_metrics.clear()
                if exploration_window_steps:
                    payload["charts/exploratory_step_fraction"] = (
                        exploratory_steps / exploration_window_steps
                    )
                wandb.log(payload, step=step)
                exploratory_steps = exploration_window_steps = 0
                pbar.set_postfix(
                    {
                        "avgR": f"{payload['charts/avg_return']:.1f}",
                        "eps": f"{epsilon:.2f}",
                    }
                )
            if cfg.save_best and len(episode_returns) >= cfg.best_window:
                avg_recent = float(np.mean(episode_returns[-cfg.best_window :]))
                if avg_recent > best_avg_return:
                    best_avg_return = avg_recent
                    path = os.path.join(cfg.checkpoint_dir, "best.pt")
                    save_checkpoint(
                        path, agent, step, best_avg_return, config=cfg.to_dict()
                    )
                    logger.info(
                        "New best %d-episode return %.2f; saved %s",
                        cfg.best_window,
                        best_avg_return,
                        path,
                    )
            if step % cfg.checkpoint_interval == 0 or step == cfg.total_steps:
                path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{step}.pt")
                save_checkpoint(
                    path, agent, step, best_avg_return, config=cfg.to_dict()
                )
                logger.info("Saved checkpoint: %s", path)
        if hasattr(run, "summary"):
            run.summary.update(
                {
                    "training/environment_steps": cfg.total_steps,
                    "training/gradient_updates": agent.train_steps,
                    "training/best_avg_return": best_avg_return
                    if np.isfinite(best_avg_return)
                    else None,
                }
            )
        logger.info(
            "Training finished. Best %d-episode avg return: %.2f",
            cfg.best_window,
            best_avg_return,
        )
        return {
            "global_step": cfg.total_steps,
            "gradient_updates": agent.train_steps,
            "best_avg_return": best_avg_return,
        }
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            run.finish()


def demo(
    config: str = "deepQN/configs/mountaincar_exploration.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
    deterministic: bool = True,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Saved-policy rollouts; native rewards and headless rendering by default."""
    cfg = Config.from_yaml(config)
    logger = _logger(cfg)
    episodes = cfg.episodes if episodes is None else episodes
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    seed = cfg.seed if seed is None else seed
    mode = cfg.render_mode if render_mode is None else render_mode
    if isinstance(mode, str) and mode.lower() in {"none", "null"}:
        mode = None
    set_seed(seed)
    env = make_env(cfg.env_id, seed, render_mode=mode, env_kwargs=cfg.env_kwargs)
    try:
        agent = _agent(cfg, env)
        load_checkpoint(model_path or cfg.inference_model_path, agent)
        agent.q_network.eval()
        agent.target_q_network.eval()
        returns, lengths, terminations = [], [], []
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            terminated = False
            for step in range(cfg.max_steps_per_episode):
                action = agent.act(
                    obs, epsilon=cfg.eval_epsilon, deterministic=deterministic
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(step + 1)
            terminations.append(bool(terminated))
            logger.info(
                "Episode %d: return=%.2f, length=%d",
                episode + 1,
                episode_return,
                step + 1,
            )
        termination_rate = float(np.mean(terminations))
        result = {
            "returns": returns,
            "lengths": lengths,
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "termination_rate": termination_rate,
            "success_rate": termination_rate
            if cfg.env_id == "MountainCar-v0"
            else None,
        }
        logger.info("Average return over %d episodes: %.2f", episodes, result["mean"])
        return result
    finally:
        env.close()


if __name__ == "__main__":
    import fire

    fire.Fire({"train": train, "demo": demo})
