from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm

from PPO.ppo.agent import Batch, PPOAgent
from PPO.ppo.config import Config
from PPO.ppo.logging_utils import setup_logger
from PPO.ppo.utils import load_checkpoint, make_vec_env, save_checkpoint, set_seed


def _to_numpy_safe(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    try:
        return tensor.numpy()
    except RuntimeError:
        return np.asarray(tensor.tolist())


def _build_agent(cfg: Config, obs_dim: int, act_dim: int) -> PPOAgent:
    return PPOAgent(
        obs_dim,
        act_dim,
        cfg.hidden_sizes,
        cfg.activation,
        cfg.learning_rate,
        cfg.clip_coef,
        cfg.ent_coef,
        cfg.vf_coef,
        cfg.max_grad_norm,
        cfg.device,
    )


def _bootstrap_observations(next_obs, terminated, truncated, infos):
    """Use the pre-reset final observation when a time limit ends an episode."""
    observations = np.asarray(next_obs, dtype=np.float32).copy()
    timeouts = np.logical_and(truncated, np.logical_not(terminated))
    if not np.any(timeouts):
        return observations
    final_obs = infos.get("final_obs", infos.get("final_observation"))
    final_mask = infos.get("_final_obs", infos.get("_final_observation"))
    for index in np.flatnonzero(timeouts):
        if (
            final_obs is None
            or (final_mask is not None and not final_mask[index])
            or final_obs[index] is None
        ):
            raise RuntimeError(
                "Time-limit truncation requires final observations from "
                "same-step autoreset."
            )
        observations[index] = final_obs[index]
    return observations


def train(config: str = "PPO/configs/cartpole_tuned.yaml", wandb_key: str = ""):
    """Train a discrete PPO policy and write metrics plus best/periodic models."""
    cfg = Config.from_yaml(config)
    env_wandb_key = os.getenv("WANDB_API_KEY", "")
    if wandb_key:
        cfg.wandb_key = wandb_key
    elif env_wandb_key:
        cfg.wandb_key = env_wandb_key
    logger = setup_logger(
        name="ppo",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )
    set_seed(cfg.seed)
    if cfg.wandb_key:
        wandb.login(key=cfg.wandb_key)
    run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.run_name,
        config=cfg.to_dict(),
    )
    vec_env = None
    try:
        vec_env = make_vec_env(cfg.env_id, cfg.num_envs, cfg.seed, render_mode=None)
        obs_space = vec_env.single_observation_space
        act_space = vec_env.single_action_space
        if (
            len(obs_space.shape) != 1
            or not hasattr(act_space, "n")
            or getattr(act_space, "start", 0) != 0
        ):
            raise ValueError(
                "PPO requires flat observations and zero-based discrete actions"
            )
        obs_dim, act_dim = obs_space.shape[0], act_space.n
        agent = _build_agent(cfg, obs_dim, act_dim)
        num_updates = (cfg.total_timesteps + cfg.rollout_steps * cfg.num_envs - 1) // (
            cfg.rollout_steps * cfg.num_envs
        )
        logger.info(
            f"Env={cfg.env_id} | device={cfg.device} | num_envs={cfg.num_envs} | "
            f"steps={cfg.total_timesteps} | updates={num_updates}"
        )
        shape = (cfg.rollout_steps, cfg.num_envs)
        obs_buf = np.zeros((*shape, obs_dim), dtype=np.float32)
        actions_buf = np.zeros(shape, dtype=np.int64)
        logprobs_buf = np.zeros(shape, dtype=np.float32)
        rewards_buf = np.zeros(shape, dtype=np.float32)
        dones_buf = np.zeros(shape, dtype=np.float32)
        terminations_buf = np.zeros(shape, dtype=np.float32)
        values_buf = np.zeros(shape, dtype=np.float32)
        next_values_buf = np.zeros(shape, dtype=np.float32)
        episode_returns, episode_lengths = [], []
        per_env_returns = np.zeros(cfg.num_envs, dtype=np.float32)
        per_env_lengths = np.zeros(cfg.num_envs, dtype=np.int32)
        best_avg_return = -np.inf
        global_step = 0
        log_window = []
        obs, _ = vec_env.reset(seed=cfg.seed)
        pbar = tqdm(range(num_updates), desc="PPO Updates")
        for update in pbar:
            rollout_length = min(
                cfg.rollout_steps, (cfg.total_timesteps - global_step) // cfg.num_envs
            )
            for step in range(rollout_length):
                obs_buf[step] = obs
                actions, logprobs, values = agent.act(obs)
                actions_buf[step], logprobs_buf[step], values_buf[step] = (
                    actions,
                    logprobs,
                    values,
                )
                next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
                dones = np.logical_or(terminated, truncated)
                rewards_buf[step] = rewards
                dones_buf[step] = dones
                terminations_buf[step] = terminated
                bootstrap_obs = _bootstrap_observations(
                    next_obs, terminated, truncated, infos
                )
                with torch.no_grad():
                    _, bootstrap_value = agent.model(
                        torch.as_tensor(
                            bootstrap_obs, dtype=torch.float32, device=agent.device
                        )
                    )
                next_values_buf[step] = _to_numpy_safe(bootstrap_value)
                per_env_returns += rewards
                per_env_lengths += 1
                for index in np.flatnonzero(dones):
                    episode_returns.append(float(per_env_returns[index]))
                    episode_lengths.append(int(per_env_lengths[index]))
                    per_env_returns[index] = 0.0
                    per_env_lengths[index] = 0
                obs = next_obs
                global_step += cfg.num_envs
            advantages, returns = PPOAgent.compute_gae(
                rewards_buf[:rollout_length],
                dones_buf[:rollout_length],
                values_buf[:rollout_length],
                next_values_buf[rollout_length - 1],
                cfg.gamma,
                cfg.gae_lambda,
                next_values=next_values_buf[:rollout_length],
                terminations=terminations_buf[:rollout_length],
            )
            b_obs = torch.as_tensor(
                obs_buf[:rollout_length].reshape(-1, obs_dim),
                dtype=torch.float32,
                device=agent.device,
            )
            b_actions = torch.as_tensor(
                actions_buf[:rollout_length].reshape(-1),
                dtype=torch.int64,
                device=agent.device,
            )
            b_logprobs = torch.as_tensor(
                logprobs_buf[:rollout_length].reshape(-1),
                dtype=torch.float32,
                device=agent.device,
            )
            b_returns = torch.as_tensor(
                returns.reshape(-1), dtype=torch.float32, device=agent.device
            )
            b_advantages = torch.as_tensor(
                advantages.reshape(-1), dtype=torch.float32, device=agent.device
            )
            b_values = torch.as_tensor(
                values_buf[:rollout_length].reshape(-1),
                dtype=torch.float32,
                device=agent.device,
            )
            losses = []
            for _ in range(cfg.update_iterations):
                indices = torch.randperm(len(b_obs), device=agent.device)
                for start in range(0, len(b_obs), cfg.minibatch_size):
                    selected = indices[start : start + cfg.minibatch_size]
                    batch = Batch(
                        b_obs[selected],
                        b_actions[selected],
                        b_logprobs[selected],
                        b_returns[selected],
                        b_advantages[selected],
                        b_values[selected],
                    )
                    losses.append(agent.update(batch))
            mean_losses = {
                key: float(np.mean([item[key] for item in losses])) for key in losses[0]
            }
            prediction = values_buf[:rollout_length]
            target_var = float(np.var(returns))
            log_window.append(
                {
                    **mean_losses,
                    "stats/value_mean": float(np.mean(prediction)),
                    "stats/return_target_mean": float(np.mean(returns)),
                    "stats/explained_variance": (
                        1.0 - float(np.var(returns - prediction)) / target_var
                        if target_var > 1e-8
                        else 0.0
                    ),
                }
            )
            avg_return = (
                float(np.mean(episode_returns[-10:])) if episode_returns else 0.0
            )
            avg_length = (
                float(np.mean(episode_lengths[-10:])) if episode_lengths else 0.0
            )
            if (update + 1) % cfg.log_interval == 0 or update + 1 == num_updates:
                payload = {
                    key: float(np.mean([item[key] for item in log_window]))
                    for key in log_window[0]
                }
                wandb.log(
                    {
                        "charts/avg_return": avg_return,
                        "charts/avg_length": avg_length,
                        "charts/episodes": len(episode_returns),
                        "progress/global_step": global_step,
                        "progress/update": update + 1,
                        **payload,
                    },
                    step=global_step,
                )
                log_window.clear()
            pbar.set_postfix(
                {
                    "avgR": f"{avg_return:.1f}",
                    "loss": f"{mean_losses['loss/total']:.3f}",
                }
            )
            if mean_losses.get("stats/approx_kl", 0.0) > 0.03:
                logger.warning(
                    f"High KL detected: {mean_losses['stats/approx_kl']:.4f}"
                )
            # Do not select an untrained model from an empty return history.
            if (
                cfg.save_best
                and len(episode_returns) >= 10
                and avg_return > best_avg_return
            ):
                best_avg_return = avg_return
                path = os.path.join(cfg.checkpoint_dir, "best.pt")
                save_checkpoint(
                    path, agent.model, agent.optimizer, global_step, best_avg_return
                )
                logger.info(
                    f"New best 10-episode avg return {best_avg_return:.2f}; saved {path}"
                )
            if (update + 1) % cfg.checkpoint_interval == 0 or update + 1 == num_updates:
                path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{global_step}.pt")
                save_checkpoint(
                    path, agent.model, agent.optimizer, global_step, best_avg_return
                )
                logger.info(f"Saved checkpoint: {path}")
        if hasattr(run, "summary"):
            run.summary["best_avg_return"] = (
                best_avg_return if np.isfinite(best_avg_return) else None
            )
            run.summary["environment_steps"] = global_step
        logger.info(
            f"Training finished. Best 10-episode avg return: {best_avg_return:.2f}"
        )
        return {"global_step": global_step, "best_avg_return": best_avg_return}
    finally:
        try:
            if vec_env is not None:
                vec_env.close()
        finally:
            run.finish()


def demo(
    config: str = "PPO/configs/cartpole_tuned.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
    deterministic: bool = True,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Evaluate a saved policy; use render_mode='none' for headless rollouts."""
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="ppo",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )
    model_path = model_path or cfg.inference_model_path
    episodes = cfg.episodes if episodes is None else episodes
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    seed = cfg.seed if seed is None else seed
    mode = cfg.render_mode or "human" if render_mode is None else render_mode
    if isinstance(mode, str) and mode.lower() in {"none", "null"}:
        mode = None
    set_seed(seed)
    env = make_vec_env(cfg.env_id, 1, seed, render_mode=mode)
    try:
        obs_space = env.single_observation_space
        act_space = env.single_action_space
        if (
            len(obs_space.shape) != 1
            or not hasattr(act_space, "n")
            or getattr(act_space, "start", 0) != 0
        ):
            raise ValueError(
                "PPO requires flat observations and zero-based discrete actions"
            )
        agent = _build_agent(cfg, obs_space.shape[0], act_space.n)
        load_checkpoint(model_path, agent.model)
        agent.model.eval()
        logger.info(f"Loaded model from {model_path}")
        returns = []
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0
            while not done:
                action, _, _ = agent.act(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated[0] or truncated[0])
                episode_return += float(reward[0])
            returns.append(episode_return)
            logger.info(f"Episode {episode + 1}: return={episode_return:.2f}")
        result = {
            "returns": returns,
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
        }
        logger.info(f"Average return over {episodes} episodes: {result['mean']:.2f}")
        return result
    finally:
        env.close()


if __name__ == "__main__":
    import fire

    fire.Fire({"train": train, "demo": demo})
