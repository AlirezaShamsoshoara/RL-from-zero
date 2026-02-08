from __future__ import annotations
import os
import time
from typing import Optional, Dict
import numpy as np
import torch
from tqdm import tqdm
import wandb
from MAPPO.mappo.config import Config
from MAPPO.mappo.utils import (
    set_seed,
    make_multiwalker_env,
    get_space_dims,
    save_checkpoint,
    load_checkpoint,
)
from MAPPO.mappo.agent import MAPPOAgent, Batch
from MAPPO.mappo.logging_utils import setup_logger


def train(config: str = "MAPPO/configs/multiwalker.yaml", wandb_key: str = ""):
    cfg = Config.from_yaml(config)
    env_wandb_key = os.getenv("WANDB_API_KEY", "")
    # Allow providing the key via CLI arg or env var; default remains blank
    if wandb_key:
        cfg.wandb_key = wandb_key
    elif env_wandb_key:
        cfg.wandb_key = env_wandb_key
    logger = setup_logger(
        name="mappo",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )
    set_seed(cfg.seed)

    # Login to Weights & Biases if a key is provided
    if getattr(cfg, "wandb_key", ""):
        wandb.login(key=cfg.wandb_key)

    logger.info(f"Initializing wandb run={cfg.run_name}")
    run = wandb.init(
        project=cfg.project, entity=cfg.entity, name=cfg.run_name, config=cfg.to_dict()
    )

    # Create environment
    env = make_multiwalker_env(
        n_walkers=cfg.n_walkers,
        seed=cfg.seed,
        render_mode=None,
        discretize_actions=cfg.discretize_actions,
        action_bins=cfg.action_bins,
    )
    obs_dim, act_dim, state_dim, n_agents, act_space = get_space_dims(
        env, return_action_space=True
    )
    action_type = "discrete" if hasattr(act_space, "n") else "continuous"
    action_low = None
    action_high = None
    if action_type == "continuous":
        action_low = np.asarray(act_space.low, dtype=np.float32)
        action_high = np.asarray(act_space.high, dtype=np.float32)

    # Verify n_agents matches config
    if n_agents != cfg.n_agents:
        logger.warning(
            f"Config n_agents={cfg.n_agents} but env has {n_agents} agents. Using env value."
        )
        cfg.n_agents = n_agents

    logger.info(
        f"Env={cfg.env_id} | n_agents={n_agents} | obs_dim={obs_dim} | act_dim={act_dim} | state_dim={state_dim} | action_type={action_type}"
    )

    # Create MAPPO agent
    agent = MAPPOAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        state_dim=state_dim,
        actor_hidden_sizes=cfg.actor_hidden_sizes,
        critic_hidden_sizes=cfg.critic_hidden_sizes,
        activation=cfg.activation,
        lr=cfg.learning_rate,
        clip_coef=cfg.clip_coef,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        device=cfg.device,
        share_policy=cfg.share_policy,
        use_centralized_critic=cfg.use_centralized_critic,
        action_type=action_type,
        action_low=action_low,
        action_high=action_high,
    )

    num_updates = cfg.total_timesteps // cfg.rollout_steps
    global_step = 0
    logger.info(f"Training for {cfg.total_timesteps} steps -> {num_updates} updates")

    # Episode tracking
    ep_returns_hist = []
    ep_lengths_hist = []

    # Rollout buffers for each agent
    obs_bufs = [
        np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32) for _ in range(n_agents)
    ]
    state_bufs = np.zeros((cfg.rollout_steps, state_dim), dtype=np.float32)
    if action_type == "continuous":
        actions_bufs = [
            np.zeros((cfg.rollout_steps, act_dim), dtype=np.float32)
            for _ in range(n_agents)
        ]
    else:
        actions_bufs = [
            np.zeros(cfg.rollout_steps, dtype=np.int64) for _ in range(n_agents)
        ]
    logprobs_bufs = [
        np.zeros(cfg.rollout_steps, dtype=np.float32) for _ in range(n_agents)
    ]
    rewards_bufs = [
        np.zeros(cfg.rollout_steps, dtype=np.float32) for _ in range(n_agents)
    ]
    dones_bufs = [
        np.zeros(cfg.rollout_steps, dtype=np.float32) for _ in range(n_agents)
    ]
    values_bufs = [
        np.zeros(cfg.rollout_steps, dtype=np.float32) for _ in range(n_agents)
    ]
    alive_bufs = [
        np.zeros(cfg.rollout_steps, dtype=np.float32) for _ in range(n_agents)
    ]

    # Reset environment
    observations, infos = env.reset(seed=cfg.seed)
    agent_order = list(env.possible_agents)
    zero_obs = np.zeros(obs_dim, dtype=np.float32)

    # Episode tracking variables
    episode_return = 0.0
    episode_length = 0

    pbar = tqdm(range(num_updates), desc="MAPPO Updates")
    best_avg_return = -np.inf

    for update in pbar:
        # Collect rollouts
        for step in range(cfg.rollout_steps):
            # Convert observations to list and create state
            active_agents = set(observations.keys())
            obs_list = [
                observations.get(agent_id, zero_obs) for agent_id in agent_order
            ]
            state = np.concatenate(obs_list, axis=0)

            # Get actions from agents
            actions_array, logprobs_array, values_array = agent.act(obs_list, state)

            # Store in buffers
            for i in range(n_agents):
                obs_bufs[i][step] = obs_list[i]
                actions_bufs[i][step] = actions_array[i]
                logprobs_bufs[i][step] = logprobs_array[i]
                values_bufs[i][step] = values_array[i]
            state_bufs[step] = state

            # Convert actions to dict for environment
            actions_dict = {}
            for i, agent_id in enumerate(agent_order):
                if agent_id not in active_agents:
                    continue
                action = actions_array[i]
                if action_type == "discrete":
                    action = int(action)
                actions_dict[agent_id] = action

            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions_dict)

            # Store rewards and dones
            for i, agent_id in enumerate(agent_order):
                if agent_id in rewards:
                    rewards_bufs[i][step] = rewards[agent_id]
                    dones_bufs[i][step] = float(terminations.get(agent_id, False) or truncations.get(agent_id, False))
                    alive_bufs[i][step] = 1.0
                else:
                    # Agent removed from environment (e.g., fell in multiwalker)
                    rewards_bufs[i][step] = 0.0
                    dones_bufs[i][step] = 1.0
                    alive_bufs[i][step] = 0.0

            # Track episode statistics
            if rewards:
                episode_return += float(np.mean(list(rewards.values())))
            episode_length += 1

            # Check if episode ended
            if not next_observations or len(next_observations) == 0:
                # Episode ended, record stats and reset
                ep_returns_hist.append(episode_return)
                ep_lengths_hist.append(episode_length)
                episode_return = 0.0
                episode_length = 0
                next_observations, infos = env.reset()

            observations = next_observations
            global_step += 1

        # Bootstrap value for next state
        with torch.no_grad():
            next_obs_list = [
                observations.get(agent_id, zero_obs) for agent_id in agent_order
            ]
            next_state = np.concatenate(next_obs_list, axis=0)
            _, _, next_values = agent.act(next_obs_list, next_state)

        # Compute advantages and returns for each agent
        batches = []
        for i in range(n_agents):
            # Reshape to [T, 1] for compute_gae
            rewards_2d = rewards_bufs[i].reshape(-1, 1)
            dones_2d = dones_bufs[i].reshape(-1, 1)
            values_2d = values_bufs[i].reshape(-1, 1)
            alive_2d = alive_bufs[i].reshape(-1, 1)
            next_values_1d = next_values[i:i+1] * alive_2d[-1]

            rewards_2d = rewards_2d * alive_2d
            values_2d = values_2d * alive_2d
            dones_2d = np.where(alive_2d > 0.5, dones_2d, 1.0)

            advantages, returns = MAPPOAgent.compute_gae(
                rewards_2d, dones_2d, values_2d, next_values_1d, cfg.gamma, cfg.gae_lambda
            )

            # Flatten
            advantages_flat = advantages.reshape(-1)
            returns_flat = returns.reshape(-1)

            # Create batch for this agent
            action_dtype = torch.float32 if action_type == "continuous" else torch.int64
            batch = Batch(
                obs=torch.as_tensor(obs_bufs[i], dtype=torch.float32, device=agent.device),
                state=torch.as_tensor(state_bufs, dtype=torch.float32, device=agent.device),
                actions=torch.as_tensor(actions_bufs[i], dtype=action_dtype, device=agent.device),
                logprobs=torch.as_tensor(logprobs_bufs[i], dtype=torch.float32, device=agent.device),
                returns=torch.as_tensor(returns_flat, dtype=torch.float32, device=agent.device),
                advantages=torch.as_tensor(advantages_flat, dtype=torch.float32, device=agent.device),
                values=torch.as_tensor(values_bufs[i], dtype=torch.float32, device=agent.device),
                alive_mask=torch.as_tensor(alive_bufs[i], dtype=torch.float32, device=agent.device),
            )
            batches.append(batch)

        # Optimize for K epochs
        total_samples = cfg.rollout_steps
        stats_accum = {
            "loss/policy": 0.0,
            "loss/value": 0.0,
            "loss/total": 0.0,
            "stats/entropy": 0.0,
            "stats/approx_kl": 0.0,
        }
        stats_count = 0
        for epoch in range(cfg.update_iterations):
            # Shuffle indices
            idxs = np.arange(total_samples)
            np.random.shuffle(idxs)

            # Mini-batch updates
            for start in range(0, total_samples, cfg.minibatch_size):
                end = min(start + cfg.minibatch_size, total_samples)
                mb_idx = idxs[start:end]

                # Create mini-batches for each agent
                mini_batches = []
                for batch in batches:
                    mini_batch = Batch(
                        obs=batch.obs[mb_idx],
                        state=batch.state[mb_idx],
                        actions=batch.actions[mb_idx],
                        logprobs=batch.logprobs[mb_idx],
                        returns=batch.returns[mb_idx],
                        advantages=batch.advantages[mb_idx],
                        values=batch.values[mb_idx],
                        alive_mask=batch.alive_mask[mb_idx],
                    )
                    mini_batches.append(mini_batch)

                stats, update_steps = agent.update(mini_batches)
                if update_steps > 0:
                    for key, value in stats.items():
                        stats_accum[key] += value * update_steps
                    stats_count += update_steps

        if stats_count > 0:
            stats = {k: v / stats_count for k, v in stats_accum.items()}
        else:
            stats = stats_accum

        # Logging
        if len(ep_returns_hist) > 0:
            avg_return = float(np.mean(ep_returns_hist[-10:]))
            avg_length = float(np.mean(ep_lengths_hist[-10:])) if len(ep_lengths_hist) > 0 else 0
        else:
            avg_return = 0.0
            avg_length = 0.0

        wandb.log(
            {
                "charts/avg_return": avg_return,
                "charts/avg_length": avg_length,
                "loss/policy": stats["loss/policy"],
                "loss/value": stats["loss/value"],
                "loss/total": stats["loss/total"],
                "stats/entropy": stats["stats/entropy"],
                "stats/approx_kl": stats["stats/approx_kl"],
                "progress/global_step": global_step,
                "progress/update": update,
            }
        )
        pbar.set_postfix(
            {
                "avgR": f"{avg_return:.1f}",
                "loss": f"{stats['loss/total']:.3f}",
            }
        )

        if stats["stats/approx_kl"] > 0.03:
            logger.warning(f"High KL detected: {stats['stats/approx_kl']:.4f}")

        # Checkpointing
        if (update + 1) % cfg.checkpoint_interval == 0:
            path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{global_step}.pt")
            save_checkpoint(
                path, agent.models, agent.optimizers, global_step, best_avg_return, cfg.share_policy
            )
            logger.info(f"Saved checkpoint: {path}")

        if cfg.save_best and avg_return > best_avg_return:
            best_avg_return = avg_return
            best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
            save_checkpoint(
                best_path, agent.models, agent.optimizers, global_step, best_avg_return, cfg.share_policy
            )
            logger.info(f"New best avg return {best_avg_return:.2f}; saved {best_path}")

    env.close()
    run.finish()
    logger.info(f"Training finished. Best avg return: {best_avg_return:.2f}")


def demo(
    config: str = "MAPPO/configs/multiwalker.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
):
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="mappo",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )
    model_path = model_path or cfg.inference_model_path
    episodes = episodes or cfg.episodes

    set_seed(cfg.seed)

    # Create environment with rendering
    env = make_multiwalker_env(
        n_walkers=cfg.n_walkers,
        seed=cfg.seed,
        render_mode=cfg.render_mode or "human",
        discretize_actions=cfg.discretize_actions,
        action_bins=cfg.action_bins,
    )
    obs_dim, act_dim, state_dim, n_agents, act_space = get_space_dims(
        env, return_action_space=True
    )
    action_type = "discrete" if hasattr(act_space, "n") else "continuous"
    action_low = None
    action_high = None
    if action_type == "continuous":
        action_low = np.asarray(act_space.low, dtype=np.float32)
        action_high = np.asarray(act_space.high, dtype=np.float32)

    logger.info(
        f"Demo: Env={cfg.env_id} | n_agents={n_agents} | obs_dim={obs_dim} | act_dim={act_dim} | action_type={action_type}"
    )

    # Create agent
    agent = MAPPOAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        state_dim=state_dim,
        actor_hidden_sizes=cfg.actor_hidden_sizes,
        critic_hidden_sizes=cfg.critic_hidden_sizes,
        activation=cfg.activation,
        lr=cfg.learning_rate,
        clip_coef=cfg.clip_coef,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        device=cfg.device,
        share_policy=cfg.share_policy,
        use_centralized_critic=cfg.use_centralized_critic,
        action_type=action_type,
        action_low=action_low,
        action_high=action_high,
    )

    # Load checkpoint
    load_checkpoint(model_path, agent.models, None)
    for model in agent.models:
        model.eval()
    logger.info(f"Loaded model from {model_path}")

    returns = []
    agent_order = list(env.possible_agents)
    zero_obs = np.zeros(obs_dim, dtype=np.float32)
    for ep in range(episodes):
        observations, _ = env.reset(seed=cfg.seed + ep)
        ep_ret = 0.0
        done = False

        while not done and observations:
            active_agents = set(observations.keys())
            obs_list = [observations.get(agent_id, zero_obs) for agent_id in agent_order]
            state = np.concatenate(obs_list, axis=0)

            with torch.no_grad():
                actions_array, _, _ = agent.act(obs_list, state, deterministic=True)

            actions_dict = {}
            for i, agent_id in enumerate(agent_order):
                if agent_id not in active_agents:
                    continue
                action = actions_array[i]
                if action_type == "discrete":
                    action = int(action)
                actions_dict[agent_id] = action
            observations, rewards, terminations, truncations, infos = env.step(actions_dict)

            # Accumulate reward
            if rewards:
                ep_ret += float(np.mean(list(rewards.values())))

            # Check if episode ended
            if not observations or len(observations) == 0:
                done = True

        logger.info(f"Episode {ep+1}: return={ep_ret:.2f}")
        returns.append(ep_ret)

    logger.info(f"Average return over {episodes} episodes: {np.mean(returns):.2f}")
    env.close()


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "train": train,
            "demo": demo,
        }
    )
