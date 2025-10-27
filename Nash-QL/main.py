from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm

from nash_ql.agent import NashQLearningAgent, Transition
from nash_ql.config import Config
from nash_ql.logging_utils import setup_logger
from nash_ql.utils import load_checkpoint, make_env, save_checkpoint, set_seed


def train(
    config: str = "Nash-QL/configs/line_world.yaml",
    wandb_key: str = "",
) -> None:
    """
    Train Nash Q-learning agents on a multi-agent environment.

    Args:
        config: Path to YAML configuration file
        wandb_key: Weights & Biases API key for logging
    """
    cfg = Config.from_yaml(config)
    if wandb_key:
        cfg.wandb_key = wandb_key

    logger = setup_logger(
        name="nash-ql",
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

    env = make_env(cfg.env_id, cfg.seed, cfg.env_kwargs)
    n_agents = env.n_agents
    n_states = env.n_states
    n_actions = env.n_actions
    logger.info(
        "Env=%s | agents=%d | states=%d | actions=%d",
        cfg.env_id,
        n_agents,
        n_states,
        n_actions,
    )

    agent = NashQLearningAgent(
        n_agents=n_agents,
        n_states=n_states,
        n_actions=n_actions,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
    )

    ep_returns: list[np.ndarray] = []
    best_avg_return = -np.inf
    pbar = tqdm(range(cfg.total_episodes), desc="Nash Q-learning")

    for ep in pbar:
        states = env.reset(seed=cfg.seed + ep)
        episode_returns = np.zeros(n_agents, dtype=np.float32)
        steps = 0

        while steps < cfg.max_steps_per_episode:
            actions = agent.act(states)
            step_result = env.step(actions)
            next_states = step_result.observations

            # Create transitions with joint actions
            transitions = []
            joint_action = tuple(actions)
            for idx in range(n_agents):
                done_flag = bool(step_result.terminated[idx] or step_result.truncated)
                transitions.append(
                    Transition(
                        agent=idx,
                        state=states[idx],
                        joint_action=joint_action,
                        reward=float(step_result.rewards[idx]),
                        next_state=next_states[idx],
                        done=done_flag,
                    )
                )
                episode_returns[idx] += float(step_result.rewards[idx])

            agent.update(transitions)
            states = next_states
            steps += 1

            if all(step_result.terminated) or step_result.truncated:
                break

        ep_returns.append(episode_returns)

        # Logging
        if (ep + 1) % cfg.log_interval == 0:
            recent = np.stack(ep_returns[-cfg.log_interval :])
            mean_per_agent = recent.mean(axis=0)
            log_data = {
                "charts/mean_return": float(mean_per_agent.mean()),
                "charts/epsilon": agent.epsilon(),
                "progress/episode": ep + 1,
                "progress/steps": agent.global_step,
            }
            for idx, value in enumerate(mean_per_agent):
                log_data[f"charts/agent{idx}_mean_return"] = float(value)
            wandb.log(log_data)
            pbar.set_postfix(
                {
                    "avgReturn": f"{mean_per_agent.mean():.3f}",
                    "eps": f"{agent.epsilon():.3f}",
                }
            )

        # Periodic checkpoint
        if (ep + 1) % cfg.checkpoint_interval == 0:
            path = os.path.join(cfg.checkpoint_dir, f"checkpoint_ep{ep+1}.pt")
            save_checkpoint(path, agent.Q, ep + 1, best_avg_return)
            logger.info("Saved checkpoint: %s", path)

        # Save best model
        if len(ep_returns) >= 10:
            recent_stack = np.stack(ep_returns[-10:])
            avg_last = float(recent_stack.mean())
            if cfg.save_best and avg_last > best_avg_return:
                best_avg_return = avg_last
                best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                save_checkpoint(best_path, agent.Q, ep + 1, best_avg_return)
                logger.info(
                    "New best mean return %.3f; saved %s",
                    best_avg_return,
                    best_path,
                )

    run.finish()
    logger.info(
        "Training finished. Best 10-episode mean return across agents: %.3f",
        best_avg_return,
    )


def demo(
    config: str = "Nash-QL/configs/line_world.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
) -> None:
    """
    Demonstrate trained Nash Q-learning agents.

    Args:
        config: Path to YAML configuration file
        model_path: Path to model checkpoint (overrides config)
        episodes: Number of episodes to run (overrides config)
    """
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="nash-ql-demo",
        level=cfg.log_level,
        to_console=True,
        to_file=False,
    )
    model_path = model_path or cfg.inference_model_path
    episodes = episodes or cfg.episodes

    env = make_env(cfg.env_id, cfg.seed, cfg.env_kwargs)
    n_agents = env.n_agents
    n_states = env.n_states
    n_actions = env.n_actions

    # Load checkpoint
    data = load_checkpoint(model_path)
    q_tables = data["q_tables"]
    if isinstance(q_tables, torch.Tensor):
        q_tables = q_tables.cpu().numpy()
    q_tables = np.asarray(q_tables, dtype=np.float32)

    # Verify shape: [n_agents, n_states, n_actions, n_actions]
    expected_shape = (n_agents, n_states, n_actions, n_actions)
    if q_tables.shape != expected_shape:
        raise ValueError(
            f"Q-table shape mismatch: expected {expected_shape}, got {q_tables.shape}"
        )

    agent = NashQLearningAgent(
        n_agents=n_agents,
        n_states=n_states,
        n_actions=n_actions,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
    )
    agent.Q = q_tables.copy()

    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Running {episodes} demo episodes...")

    for ep in range(episodes):
        states = env.reset(seed=cfg.seed + ep)
        episode_returns = np.zeros(n_agents, dtype=np.float32)
        steps = 0

        while steps < cfg.max_steps_per_episode:
            actions = agent.greedy_actions(states)
            step_result = env.step(actions)
            states = step_result.observations
            episode_returns += np.asarray(step_result.rewards, dtype=np.float32)
            steps += 1

            if all(step_result.terminated) or step_result.truncated:
                break

        logger.info(
            "Episode %d | returns=%s | steps=%d",
            ep + 1,
            np.array2string(episode_returns, precision=3),
            steps,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "train": train,
            "demo": demo,
        }
    )
