from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch
import wandb
from gymnasium import spaces
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
from tqdm import tqdm


def _stats_to_dict(stats: TD3Stats) -> Dict[str, float]:
    data = asdict(stats)
    if data["actor_loss"] is None:
        data.pop("actor_loss")
    return data


def train(config: str = "TD3/configs/mountaincar_continuous.yaml", wandb_key: str = ""):
    cfg = Config.from_yaml(config)
    env_wandb_key = os.getenv("WANDB_API_KEY", "")
    if wandb_key:
        cfg.wandb_key = wandb_key
    elif env_wandb_key:
        cfg.wandb_key = env_wandb_key

    logger = setup_logger(
        name="td3",
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

    env = make_vec_env(
        cfg.env_id, cfg.num_envs, cfg.seed, render_mode=None, env_kwargs=cfg.env_kwargs
    )
    obs_space = env.single_observation_space
    act_space = env.single_action_space
    num_envs = cfg.num_envs
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
    random_action_prob = float(
        max(0.0, min(1.0, getattr(cfg, "random_action_prob", 0.0)))
    )
    hold_min = max(1, int(getattr(cfg, "random_action_hold_min", 1)))
    hold_max = max(hold_min, int(getattr(cfg, "random_action_hold_max", hold_min)))

    act_dim = int(np.prod(act_space.shape))
    pulse_actions = np.zeros((num_envs, act_dim), dtype=np.float32)
    pulse_steps_remaining = np.zeros(num_envs, dtype=np.int32)

    total_transitions = 0
    pbar = tqdm(total=cfg.total_steps, desc="TD3 Steps")

    while total_transitions < cfg.total_steps:
        # --- Action selection ---
        if total_transitions < cfg.start_steps:
            actions = env.action_space.sample()
        else:
            actions = agent.act(obs, noise=cfg.exploration_noise, deterministic=False)

            if random_action_prob > 0.0:
                expired = pulse_steps_remaining <= 0
                if expired.any():
                    roll = np.random.rand(num_envs) < random_action_prob
                    start_pulse = expired & roll
                    if start_pulse.any():
                        for i in np.where(start_pulse)[0]:
                            pulse_actions[i] = env.single_action_space.sample()
                        pulse_steps_remaining[start_pulse] = np.random.randint(
                            hold_min, hold_max + 1, size=int(start_pulse.sum())
                        )

                pulsing = pulse_steps_remaining > 0
                if pulsing.any():
                    actions[pulsing] = pulse_actions[pulsing]
                    pulse_steps_remaining[pulsing] -= 1

        # --- Step the vectorized environment ---
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        dones = np.logical_or(terminated, truncated)

        # Under SAME_STEP autoreset: when done[i] is True, next_obs[i] is the
        # reset obs of the new episode. The terminal obs is infos["final_obs"][i].
        buffer_next_obs = next_obs.copy()
        buffer_dones = terminated.astype(np.float32)

        if dones.any():
            final_obs_mask = infos.get("_final_obs", np.zeros(num_envs, dtype=bool))
            if np.any(final_obs_mask):
                final_obs = infos["final_obs"]
                for i in np.where(final_obs_mask)[0]:
                    buffer_next_obs[i] = final_obs[i]

            pulse_actions[dones] = 0.0
            pulse_steps_remaining[dones] = 0

        buffer.add_batch(obs, actions, rewards, buffer_next_obs, buffer_dones)
        obs = next_obs
        total_transitions += num_envs
        pbar.update(num_envs)

        # --- Gradient updates ---
        if total_transitions >= cfg.start_steps and buffer.can_sample(cfg.batch_size):
            for _ in range(cfg.updates_per_step):
                stats = agent.update(buffer.sample(cfg.batch_size))
                stat_dict = _stats_to_dict(stats)
                update_metrics.append(stat_dict)

        # --- Episode logging from final_info ---
        final_info_mask = infos.get("_final_info", np.zeros(num_envs, dtype=bool))
        if np.any(final_info_mask):
            fi = infos["final_info"]
            if isinstance(fi, dict) and "episode" in fi:
                ep_mask = fi.get("_episode", np.zeros(num_envs, dtype=bool))
                for i in np.where(final_info_mask & ep_mask)[0]:
                    episode_returns.append(float(fi["episode"]["r"][i]))
                    episode_lengths.append(int(fi["episode"]["l"][i]))

        # --- Periodic logging ---
        if total_transitions % cfg.log_interval < num_envs:
            avg_return = (
                float(np.mean(episode_returns[-10:])) if episode_returns else 0.0
            )
            avg_length = (
                float(np.mean(episode_lengths[-10:])) if episode_lengths else 0.0
            )
            log_payload: Dict[str, float] = {
                "charts/avg_return": avg_return,
                "charts/avg_length": avg_length,
                "progress/step": total_transitions,
            }
            if update_metrics:
                keys = update_metrics[0].keys()
                for key in keys:
                    values = [m[key] for m in update_metrics if key in m]
                    if values:
                        prefix = "loss" if "loss" in key else "stats"
                        log_payload[f"{prefix}/{key}"] = float(np.mean(values))
                update_metrics.clear()
            wandb.log(log_payload, step=total_transitions)
            pbar.set_postfix({"avgR": f"{avg_return:.1f}"})

        # --- Periodic checkpointing + best model ---
        if total_transitions % cfg.checkpoint_interval < num_envs:
            path = os.path.join(
                cfg.checkpoint_dir, f"checkpoint_{total_transitions}.pt"
            )
            save_checkpoint(path, agent, total_transitions, best_avg_return)
            logger.info(f"Saved checkpoint: {path}")

            if cfg.save_best and len(episode_returns) >= 5:
                avg_recent = float(np.mean(episode_returns[-5:]))
                if avg_recent > best_avg_return:
                    best_avg_return = avg_recent
                    best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                    save_checkpoint(
                        best_path, agent, total_transitions, best_avg_return
                    )
                    logger.info(
                        f"New best avg return {best_avg_return:.2f}; saved {best_path}"
                    )

    pbar.close()
    env.close()
    run.finish()
    logger.info(f"Training finished. Best 5-ep avg return: {best_avg_return:.2f}")


def demo(
    config: str = "TD3/configs/mountaincar_continuous.yaml",
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
