from __future__ import annotations

import os
from typing import Optional

import gymnasium as gym
import numpy as np
import wandb
from tqdm import tqdm

from Qlearning.ql.agent import QLearningAgent, Transition
from Qlearning.ql.config import Config
from Qlearning.ql.logging_utils import setup_logger
from Qlearning.ql.utils import load_checkpoint, make_env, save_checkpoint, set_seed


def _agent(cfg: Config, env) -> QLearningAgent:
    for space in (env.observation_space, env.action_space):
        if not isinstance(space, gym.spaces.Discrete) or space.start != 0:
            raise ValueError(
                "Q-learning requires zero-based Discrete states and actions"
            )
    return QLearningAgent(
        n_states=int(env.observation_space.n),
        n_actions=int(env.action_space.n),
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
    )


def _logger(cfg: Config):
    return setup_logger(
        name="qlearning",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )


def train(config: str = "Qlearning/configs/frozenlake.yaml", wandb_key: str = ""):
    """Train a tabular learner with diagnostics, best checkpoints and cleanup."""
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
        returns = []
        log_window = []
        best_avg_return = -np.inf
        pbar = tqdm(range(cfg.total_episodes), desc="Q-learning Episodes")
        for episode in pbar:
            state, _ = env.reset(seed=cfg.seed + episode)
            state = int(state)
            start_q = float(np.max(agent.Q[state]))
            episode_return = discounted_return = td_abs = td_square = 0.0
            cutoff = False
            for step in range(cfg.max_steps_per_episode):
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = int(next_state)
                # Only a true terminal state suppresses the value bootstrap.
                # A timeout/cap ends interaction, but not the underlying MDP.
                stats = agent.update(
                    Transition(
                        state, action, float(reward), next_state, bool(terminated)
                    )
                )
                episode_return += float(reward)
                discounted_return += cfg.gamma**step * float(reward)
                td_abs += abs(stats.td_error)
                td_square += stats.td_error**2
                state = next_state
                cutoff = bool(
                    truncated
                    or (step + 1 == cfg.max_steps_per_episode and not terminated)
                )
                if terminated or truncated:
                    break
            length = step + 1
            returns.append(episode_return)
            log_window.append(
                {
                    "charts/avg_return": episode_return,
                    "charts/avg_length": length,
                    "charts/positive_return_rate": float(episode_return > 0.0),
                    "charts/truncation_rate": float(cutoff),
                    "stats/start_q": start_q,
                    "stats/discounted_return": discounted_return,
                    "stats/td_error": td_abs / length,
                    "loss/td_mse": td_square / length,
                }
            )
            if (
                episode + 1
            ) % cfg.log_interval == 0 or episode + 1 == cfg.total_episodes:
                payload = {
                    key: float(np.mean([row[key] for row in log_window]))
                    for key in log_window[0]
                }
                if cfg.env_id.startswith("FrozenLake"):
                    payload["charts/success_rate"] = payload[
                        "charts/positive_return_rate"
                    ]
                wandb.log(
                    {
                        **payload,
                        "charts/epsilon": agent.epsilon(),
                        "progress/episode": episode + 1,
                        "progress/steps": agent.global_step,
                    },
                    step=agent.global_step,
                )
                pbar.set_postfix(
                    {
                        "avgR": f"{payload['charts/avg_return']:.2f}",
                        "eps": f"{agent.epsilon():.2f}",
                    }
                )
                log_window.clear()
            if cfg.save_best and len(returns) >= cfg.best_window:
                avg_recent = float(np.mean(returns[-cfg.best_window :]))
                if avg_recent > best_avg_return:
                    best_avg_return = avg_recent
                    path = os.path.join(cfg.checkpoint_dir, "best.pt")
                    save_checkpoint(
                        path,
                        agent.Q,
                        episode + 1,
                        best_avg_return,
                        global_step=agent.global_step,
                        config=cfg.to_dict(),
                    )
                    logger.info(
                        "New best %d-episode return %.3f; saved %s",
                        cfg.best_window,
                        best_avg_return,
                        path,
                    )
            if (
                episode + 1
            ) % cfg.checkpoint_interval == 0 or episode + 1 == cfg.total_episodes:
                path = os.path.join(
                    cfg.checkpoint_dir, f"checkpoint_ep{episode + 1}.pt"
                )
                save_checkpoint(
                    path,
                    agent.Q,
                    episode + 1,
                    best_avg_return,
                    global_step=agent.global_step,
                    config=cfg.to_dict(),
                )
                logger.info("Saved checkpoint: %s", path)
        if hasattr(run, "summary"):
            run.summary.update(
                {
                    "training/episodes": cfg.total_episodes,
                    "training/steps": agent.global_step,
                    "training/best_return": best_avg_return
                    if np.isfinite(best_avg_return)
                    else None,
                }
            )
        logger.info(
            "Training finished. Best %d-episode avg return: %.3f",
            cfg.best_window,
            best_avg_return,
        )
        return {
            "episodes": cfg.total_episodes,
            "global_step": agent.global_step,
            "best_avg_return": best_avg_return,
        }
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            run.finish()


def demo(
    config: str = "Qlearning/configs/frozenlake.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
    render: bool = False,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
):
    """Greedy demo; use render_mode='ansi', 'human', or 'none' explicitly."""
    cfg = Config.from_yaml(config)
    logger = _logger(cfg)
    episodes = cfg.episodes if episodes is None else episodes
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    seed = cfg.seed if seed is None else seed
    mode = (
        ("human" if render else cfg.render_mode) if render_mode is None else render_mode
    )
    if isinstance(mode, str) and mode.lower() in {"none", "null"}:
        mode = None
    set_seed(seed)
    env = make_env(cfg.env_id, seed, render_mode=mode, env_kwargs=cfg.env_kwargs)
    try:
        agent = _agent(cfg, env)
        agent.load_state_dict(load_checkpoint(model_path or cfg.inference_model_path))
        returns, lengths = [], []
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            if mode == "ansi":
                print(env.render())
            elif mode == "human":
                env.render()
            for step in range(cfg.max_steps_per_episode):
                action = agent.act(int(state), deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                if mode == "ansi":
                    print(env.render())
                elif mode == "human":
                    env.render()
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(step + 1)
            logger.info(
                "Episode %d: return=%.3f, length=%d",
                episode + 1,
                episode_return,
                step + 1,
            )
        result = {
            "returns": returns,
            "lengths": lengths,
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "positive_return_rate": float(np.mean(np.asarray(returns) > 0.0)),
            "success_rate": (
                float(np.mean(np.asarray(returns) > 0.0))
                if cfg.env_id.startswith("FrozenLake")
                else None
            ),
        }
        logger.info("Average return over %d episodes: %.3f", episodes, result["mean"])
        return result
    finally:
        env.close()


if __name__ == "__main__":
    import fire

    fire.Fire({"train": train, "demo": demo})
