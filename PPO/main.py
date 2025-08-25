from __future__ import annotations
import os
import time
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
import wandb
from PPO.ppo.config import Config
from PPO.ppo.utils import set_seed, make_vec_env, save_checkpoint
from PPO.ppo.agent import PPOAgent, Batch
from PPO.ppo.logging_utils import setup_logger


def train(config: str = "PPO/configs/cartpole.yaml"):
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="ppo",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )
    set_seed(cfg.seed)

    logger.info(f"Initializing wandb run={cfg.run_name}")
    run = wandb.init(project=cfg.project, entity=cfg.entity, name=cfg.run_name, config=cfg.to_dict())

    # Env
    vec_env = make_vec_env(cfg.env_id, cfg.num_envs, cfg.seed, render_mode=None)
    obs_space = vec_env.single_observation_space
    act_space = vec_env.single_action_space
    assert len(obs_space.shape) == 1, "Only flat observation spaces are supported"
    assert hasattr(act_space, "n"), "Only discrete action spaces are supported"

    obs_dim = obs_space.shape[0]
    act_dim = act_space.n
    logger.info(f"Env={cfg.env_id} | obs_dim={obs_dim} | act_dim={act_dim} | num_envs={cfg.num_envs}")

    agent = PPOAgent(obs_dim, act_dim, cfg.hidden_sizes, cfg.activation, cfg.learning_rate, cfg.clip_coef, cfg.ent_coef, cfg.vf_coef, cfg.max_grad_norm, cfg.device)

    num_updates = cfg.total_timesteps // (cfg.rollout_steps * cfg.num_envs)
    global_step = 0
    logger.info(f"Training for {cfg.total_timesteps} steps -> {num_updates} updates")

    ep_returns_hist = []
    ep_lengths_hist = []
    per_env_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    per_env_lengths = np.zeros(cfg.num_envs, dtype=np.int32)

    # Buffers
    obs_buf = np.zeros((cfg.rollout_steps, cfg.num_envs, obs_dim), dtype=np.float32)
    actions_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.int64)
    logprobs_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    rewards_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    dones_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    values_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)

    # Reset
    obs, _ = vec_env.reset(seed=cfg.seed)

    pbar = tqdm(range(num_updates), desc="PPO Updates")
    best_avg_return = -np.inf

    for update in pbar:
        for step in range(cfg.rollout_steps):
            obs_buf[step] = obs
            actions, logprobs, values = agent.act(obs)
            actions_buf[step] = actions
            logprobs_buf[step] = logprobs
            values_buf[step] = values

            next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
            dones = np.logical_or(terminated, truncated).astype(np.float32)
            rewards_buf[step] = rewards
            dones_buf[step] = dones

            per_env_returns += rewards
            per_env_lengths += 1
            for i in range(cfg.num_envs):
                if dones[i]:
                    ep_returns_hist.append(per_env_returns[i])
                    ep_lengths_hist.append(per_env_lengths[i])
                    per_env_returns[i] = 0.0
                    per_env_lengths[i] = 0

            obs = next_obs
            global_step += cfg.num_envs

        # Bootstrap value
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            _, next_value = agent.model.forward(obs_t)
            next_value = next_value.cpu().numpy()

        advantages, returns = PPOAgent.compute_gae(rewards_buf, dones_buf, values_buf, next_value, cfg.gamma, cfg.gae_lambda)

        # Flatten
        b_obs = torch.as_tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=agent.device)
        b_actions = torch.as_tensor(actions_buf.reshape(-1), dtype=torch.int64, device=agent.device)
        b_logprobs = torch.as_tensor(logprobs_buf.reshape(-1), dtype=torch.float32, device=agent.device)
        b_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=agent.device)
        b_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=agent.device)
        b_values = torch.as_tensor(values_buf.reshape(-1), dtype=torch.float32, device=agent.device)

        batch = Batch(b_obs, b_actions, b_logprobs, b_returns, b_advantages, b_values)

        # Optimize policy for K epochs
        num_samples = b_obs.shape[0]
        idxs = np.arange(num_samples)
        losses = []
        for epoch in range(cfg.update_iterations):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = idxs[start:end]
                mini_batch = Batch(
                    obs=b_obs[mb_idx],
                    actions=b_actions[mb_idx],
                    logprobs=b_logprobs[mb_idx],
                    returns=b_returns[mb_idx],
                    advantages=b_advantages[mb_idx],
                    values=b_values[mb_idx],
                )
                stats = agent.update(mini_batch)
                losses.append(stats)

        # Logging
        if len(ep_returns_hist) > 0:
            avg_return = float(np.mean(ep_returns_hist[-10:]))
            avg_length = float(np.mean(ep_lengths_hist[-10:])) if len(ep_lengths_hist) > 0 else 0
        else:
            avg_return = 0.0
            avg_length = 0.0

        mean_losses = {}
        if losses:
            for k in losses[0].keys():
                mean_losses[k] = float(np.mean([x[k] for x in losses]))
        wandb.log({
            "charts/avg_return": avg_return,
            "charts/avg_length": avg_length,
            **mean_losses,
            "progress/global_step": global_step,
            "progress/update": update,
        })
        pbar.set_postfix({"avgR": f"{avg_return:.1f}", "loss": f"{mean_losses.get('loss/total', 0):.3f}"})

        if mean_losses.get("stats/approx_kl", 0) > 0.03:
            logger.warning(f"High KL detected: {mean_losses['stats/approx_kl']:.4f}")

        # Checkpointing
        if (update + 1) % cfg.checkpoint_interval == 0:
            path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{global_step}.pt")
            save_checkpoint(path, agent.model, agent.optimizer, global_step, best_avg_return)
            logger.info(f"Saved checkpoint: {path}")
        if cfg.save_best and avg_return > best_avg_return:
            best_avg_return = avg_return
            best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
            save_checkpoint(best_path, agent.model, agent.optimizer, global_step, best_avg_return)
            logger.info(f"New best avg return {best_avg_return:.2f}; saved {best_path}")

    run.finish()
    logger.info(f"Training finished. Best avg return: {best_avg_return:.2f}")


def demo(config: str = "PPO/configs/cartpole.yaml", model_path: Optional[str] = None, episodes: Optional[int] = None):
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="ppo",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )
    model_path = model_path or cfg.inference_model_path
    episodes = episodes or cfg.episodes

    set_seed(cfg.seed)

    # Make a single env with rendering
    env = make_vec_env(cfg.env_id, 1, cfg.seed, render_mode=cfg.render_mode or "human")
    obs_space = env.single_observation_space
    act_space = env.single_action_space
    obs_dim = obs_space.shape[0]
    act_dim = act_space.n

    agent = PPOAgent(obs_dim, act_dim, cfg.hidden_sizes, cfg.activation, cfg.learning_rate, cfg.clip_coef, cfg.ent_coef, cfg.vf_coef, cfg.max_grad_norm, cfg.device)

    # Load checkpoint
    data = torch.load(model_path, map_location=cfg.device)
    agent.model.load_state_dict(data["model_state_dict"])
    agent.model.eval()
    logger.info(f"Loaded model from {model_path}")

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = np.array([False])
        ep_ret = 0.0
        while not done[0]:
            with torch.no_grad():
                action, _, _ = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = np.logical_or(terminated, truncated)
            ep_ret += float(reward[0])
            # Render is handled by env when render_mode="human"
        logger.info(f"Episode {ep+1}: return={ep_ret:.2f}")
        returns.append(ep_ret)

    logger.info(f"Average return over {episodes} episodes: {np.mean(returns):.2f}")


if __name__ == "__main__":
    import fire
    fire.Fire({
        "train": train,
        "demo": demo,
    })
