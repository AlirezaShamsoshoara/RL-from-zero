from __future__ import annotations
import os
from typing import List, Optional
import numpy as np
import torch
from tqdm import tqdm
import wandb
from gym import spaces

from TRPO.trpo.config import Config
from TRPO.trpo.utils import (
    set_seed,
    make_vec_env,
    save_checkpoint,
    load_checkpoint,
)
from TRPO.trpo.agent import TRPOAgent, Batch
from TRPO.trpo.logging_utils import setup_logger


def train(config: str = "TRPO/configs/pendulum.yaml", wandb_key: str = ""):
    cfg = Config.from_yaml(config)
    if wandb_key:
        cfg.wandb_key = wandb_key

    logger = setup_logger(
        name="trpo",
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

    vec_env = make_vec_env(
        cfg.env_id,
        cfg.num_envs,
        cfg.seed,
        render_mode=None,
        env_kwargs=cfg.env_kwargs,
    )
    obs_space = vec_env.single_observation_space
    act_space = vec_env.single_action_space
    assert isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1
    assert isinstance(act_space, spaces.Box) and len(act_space.shape) == 1

    agent = TRPOAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        max_kl=cfg.max_kl,
        cg_iters=cfg.cg_iters,
        cg_damping=cfg.cg_damping,
        line_search_coef=cfg.line_search_coef,
        line_search_steps=cfg.line_search_steps,
        vf_lr=cfg.vf_lr,
        vf_iters=cfg.vf_iters,
        entropy_coef=cfg.entropy_coef,
        normalize_advantages=cfg.normalize_advantages,
        device=cfg.device,
    )

    obs_dim = int(np.prod(obs_space.shape))
    act_dim = int(np.prod(act_space.shape))
    logger.info(
        f"Env={cfg.env_id} | obs_dim={obs_dim} | act_dim={act_dim} | num_envs={cfg.num_envs}"
    )

    num_updates = cfg.total_timesteps // (cfg.rollout_steps * cfg.num_envs)
    global_step = 0
    best_avg_return = -np.inf

    ep_returns_hist: List[float] = []
    ep_lengths_hist: List[int] = []
    per_env_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    per_env_lengths = np.zeros(cfg.num_envs, dtype=np.int32)

    obs_buf = np.zeros((cfg.rollout_steps, cfg.num_envs, obs_dim), dtype=np.float32)
    actions_buf = np.zeros((cfg.rollout_steps, cfg.num_envs, act_dim), dtype=np.float32)
    logprobs_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    rewards_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    dones_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)
    values_buf = np.zeros((cfg.rollout_steps, cfg.num_envs), dtype=np.float32)

    obs, _ = vec_env.reset(seed=cfg.seed)
    pbar = tqdm(range(num_updates), desc="TRPO Updates")

    for update in pbar:
        for step in range(cfg.rollout_steps):
            obs_buf[step] = obs
            actions, log_probs, values = agent.act(obs)
            actions_buf[step] = actions
            logprobs_buf[step] = log_probs
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

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            value_t = agent.value_fn(obs_t)
            next_value = value_t.cpu().numpy()

        advantages, returns = TRPOAgent.compute_gae(
            rewards_buf,
            dones_buf,
            values_buf,
            next_value,
            cfg.gamma,
            cfg.gae_lambda,
        )

        b_obs = torch.as_tensor(
            obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=agent.device
        )
        b_actions = torch.as_tensor(
            actions_buf.reshape(-1, act_dim), dtype=torch.float32, device=agent.device
        )
        b_logprobs = torch.as_tensor(
            logprobs_buf.reshape(-1), dtype=torch.float32, device=agent.device
        )
        b_adv = torch.as_tensor(
            advantages.reshape(-1), dtype=torch.float32, device=agent.device
        )
        b_returns = torch.as_tensor(
            returns.reshape(-1), dtype=torch.float32, device=agent.device
        )
        b_values = torch.as_tensor(
            values_buf.reshape(-1), dtype=torch.float32, device=agent.device
        )

        batch = Batch(
            obs=b_obs,
            actions=b_actions,
            logprobs=b_logprobs,
            returns=b_returns,
            advantages=b_adv,
            values=b_values,
        )
        stats = agent.update(batch)

        if ep_returns_hist:
            avg_return = float(np.mean(ep_returns_hist[-10:]))
            avg_length = float(np.mean(ep_lengths_hist[-10:])) if ep_lengths_hist else 0.0
        else:
            avg_return = 0.0
            avg_length = 0.0

        wandb.log(
            {
                "charts/avg_return": avg_return,
                "charts/avg_length": avg_length,
                "loss/policy": stats["loss/policy"],
                "loss/value": stats["loss/value"],
                "stats/kl": stats["stats/kl"],
                "stats/entropy": stats["stats/entropy"],
                "stats/line_search_steps": stats["stats/line_search_steps"],
                "stats/line_search_success": stats["stats/line_search_success"],
                "progress/global_step": global_step,
                "progress/update": update,
            }
        )
        pbar.set_postfix({"avgR": f"{avg_return:.1f}", "KL": f"{stats['stats/kl']:.4f}"})

        if (update + 1) % cfg.checkpoint_interval == 0:
            path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{global_step}.pt")
            save_checkpoint(path, agent, global_step, best_avg_return)
            logger.info(f"Saved checkpoint: {path}")

        if cfg.save_best and avg_return > best_avg_return:
            best_avg_return = avg_return
            best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
            save_checkpoint(best_path, agent, global_step, best_avg_return)
            logger.info(
                f"New best avg return {best_avg_return:.2f}; saved checkpoint to {best_path}"
            )

    vec_env.close()
    run.finish()
    logger.info(f"Training finished. Best avg return: {best_avg_return:.2f}")


def demo(
    config: str = "TRPO/configs/pendulum.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
):
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="trpo",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )
    model_path = model_path or cfg.inference_model_path
    episodes = episodes or cfg.episodes

    set_seed(cfg.seed)

    env = make_vec_env(
        cfg.env_id,
        1,
        cfg.seed,
        render_mode=cfg.render_mode or "human",
        env_kwargs=cfg.env_kwargs,
    )
    obs_space = env.single_observation_space
    act_space = env.single_action_space
    assert isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1
    assert isinstance(act_space, spaces.Box) and len(act_space.shape) == 1

    agent = TRPOAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        max_kl=cfg.max_kl,
        cg_iters=cfg.cg_iters,
        cg_damping=cfg.cg_damping,
        line_search_coef=cfg.line_search_coef,
        line_search_steps=cfg.line_search_steps,
        vf_lr=cfg.vf_lr,
        vf_iters=cfg.vf_iters,
        entropy_coef=cfg.entropy_coef,
        normalize_advantages=cfg.normalize_advantages,
        device=cfg.device,
    )
    load_checkpoint(model_path, agent)
    logger.info(f"Loaded model from {model_path}")

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = np.array([False])
        ep_ret = 0.0
        while not done[0]:
            actions, _, _ = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(actions)
            done = np.logical_or(terminated, truncated)
            ep_ret += float(reward[0])
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
