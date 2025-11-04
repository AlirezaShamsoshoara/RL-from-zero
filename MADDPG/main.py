from __future__ import annotations
import os
from dataclasses import asdict
from typing import Optional, List, Dict
import numpy as np
import torch
from tqdm import tqdm
import wandb

from MADDPG.maddpg.config import Config
from MADDPG.maddpg.utils import (
    set_seed,
    make_env,
    get_space_info,
    save_checkpoint,
    load_checkpoint,
)
from MADDPG.maddpg.agent import MADDPGAgent, MADDPGStats
from MADDPG.maddpg.replay_buffer import MultiAgentReplayBuffer
from MADDPG.maddpg.logging_utils import setup_logger


def _stats_to_dict(stats: MADDPGStats) -> Dict[str, float]:
    """Convert stats dataclass to dictionary."""
    return asdict(stats)


def train(config: str = "MADDPG/configs/simple_spread.yaml", wandb_key: str = ""):
    """
    Train MADDPG agents on a multi-agent environment.

    Args:
        config: Path to YAML configuration file
        wandb_key: Weights & Biases API key for logging
    """
    cfg = Config.from_yaml(config)
    if wandb_key:
        cfg.wandb_key = wandb_key

    logger = setup_logger(
        name="maddpg",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )

    set_seed(cfg.seed)

    # Login to WandB if key provided
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

    # Create environment
    env = make_env(
        cfg.env_id,
        cfg.seed,
        n_agents=cfg.n_agents,
        max_cycles=cfg.max_cycles,
        render_mode=None,
    )
    n_agents, obs_dims, act_dims, action_lows, action_highs = get_space_info(env)

    logger.info(
        f"Env={cfg.env_id} | n_agents={n_agents} | obs_dims={obs_dims} | act_dims={act_dims}"
    )

    # Create MADDPG agent
    agent = MADDPGAgent(
        n_agents=n_agents,
        obs_dims=obs_dims,
        act_dims=act_dims,
        action_lows=action_lows,
        action_highs=action_highs,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        target_policy_noise=cfg.target_policy_noise,
        target_noise_clip=cfg.target_noise_clip,
        device=cfg.device,
    )

    # Create replay buffer
    buffer = MultiAgentReplayBuffer(
        n_agents=n_agents,
        obs_dims=obs_dims,
        act_dims=act_dims,
        capacity=cfg.buffer_size,
        device=torch.device(cfg.device),
    )

    # Initialize environment
    observations, _ = env.reset(seed=cfg.seed)
    agent_ids = list(observations.keys())

    # Training tracking
    best_avg_return = -np.inf
    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    update_metrics: List[Dict[str, float]] = []
    episode_return = 0.0
    episode_length = 0

    pbar = tqdm(range(1, cfg.total_steps + 1), desc="MADDPG Steps")

    for step in pbar:
        # Get observations as list
        obs_list = [observations[agent_id] for agent_id in agent_ids]

        # Select actions
        if step <= cfg.start_steps:
            # Random exploration
            actions_list = [env.action_space(agent_id).sample() for agent_id in agent_ids]
        else:
            # Policy with exploration noise
            actions_list = agent.act(
                obs_list,
                noise=cfg.exploration_noise,
                deterministic=cfg.exploration_noise <= 0.0,
            )

        # Convert actions to dict for environment
        actions_dict = {agent_id: actions_list[i] for i, agent_id in enumerate(agent_ids)}

        # Step environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions_dict)

        # Extract data for all agents
        next_obs_list = [next_observations.get(agent_id, np.zeros(obs_dims[i])) for i, agent_id in enumerate(agent_ids)]
        rewards_list = [rewards.get(agent_id, 0.0) for agent_id in agent_ids]
        dones_list = [
            bool(terminations.get(agent_id, False) or truncations.get(agent_id, False))
            for agent_id in agent_ids
        ]

        # Store transition in replay buffer
        buffer.add(obs_list, actions_list, rewards_list, next_obs_list, dones_list)

        # Track episode statistics (use first agent's reward as representative)
        episode_return += rewards_list[0]
        episode_length += 1

        # Check if episode ended
        episode_done = not next_observations or len(next_observations) == 0 or all(dones_list)
        if episode_done:
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            episode_return = 0.0
            episode_length = 0
            observations, _ = env.reset()
            agent_ids = list(observations.keys())
        else:
            observations = next_observations

        # Training updates
        if step >= cfg.start_steps and buffer.can_sample(cfg.batch_size):
            for _ in range(cfg.updates_per_step):
                batch = buffer.sample(cfg.batch_size)
                stats = agent.update(batch)
                update_metrics.append(_stats_to_dict(stats))

        # Logging
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

        # Checkpointing
        if step % cfg.checkpoint_interval == 0:
            path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{step}.pt")
            save_checkpoint(path, agent, step, best_avg_return)
            logger.info(f"Saved checkpoint: {path}")

        # Save best model
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
    logger.info(f"Training finished. Best 5-ep avg return: {best_avg_return:.2f}")


def demo(
    config: str = "MADDPG/configs/simple_spread.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
    exploration_noise: float = 0.0,
):
    """
    Run inference with trained MADDPG agents.

    Args:
        config: Path to YAML configuration file
        model_path: Path to model checkpoint (optional, uses config default if not provided)
        episodes: Number of episodes to run (optional, uses config default if not provided)
        exploration_noise: Exploration noise std (0.0 for deterministic)
    """
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="maddpg",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )
    model_path = model_path or cfg.inference_model_path
    episodes = episodes or cfg.episodes

    set_seed(cfg.seed)

    # Create environment with rendering
    env = make_env(
        cfg.env_id,
        cfg.seed,
        n_agents=cfg.n_agents,
        max_cycles=cfg.max_cycles,
        render_mode=cfg.render_mode or "human",
    )
    n_agents, obs_dims, act_dims, action_lows, action_highs = get_space_info(env)

    logger.info(
        f"Demo: Env={cfg.env_id} | n_agents={n_agents} | obs_dims={obs_dims} | act_dims={act_dims}"
    )

    # Create agent
    agent = MADDPGAgent(
        n_agents=n_agents,
        obs_dims=obs_dims,
        act_dims=act_dims,
        action_lows=action_lows,
        action_highs=action_highs,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        target_policy_noise=cfg.target_policy_noise,
        target_noise_clip=cfg.target_noise_clip,
        device=cfg.device,
    )

    # Load checkpoint
    load_checkpoint(model_path, agent)
    for actor in agent.actors:
        actor.eval()
    logger.info(f"Loaded model from {model_path}")

    # Run episodes
    returns = []
    for ep in range(episodes):
        observations, _ = env.reset(seed=cfg.seed + ep)
        agent_ids = list(observations.keys())
        ep_ret = 0.0
        done = False
        steps = 0

        while not done and observations and steps < cfg.max_cycles:
            obs_list = [observations.get(agent_id, np.zeros(obs_dims[i])) for i, agent_id in enumerate(agent_ids)]

            actions_list = agent.act(
                obs_list,
                noise=exploration_noise,
                deterministic=exploration_noise <= 0.0,
            )

            actions_dict = {agent_id: actions_list[i] for i, agent_id in enumerate(agent_ids)}
            observations, rewards, terminations, truncations, infos = env.step(actions_dict)

            # Accumulate reward from first agent
            if agent_ids[0] in rewards:
                ep_ret += rewards[agent_ids[0]]

            # Check if episode ended
            if not observations or len(observations) == 0:
                done = True

            steps += 1

        logger.info(f"Episode {ep + 1}: return={ep_ret:.2f}, steps={steps}")
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
