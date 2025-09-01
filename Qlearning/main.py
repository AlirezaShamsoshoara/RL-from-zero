from __future__ import annotations
from typing import Optional
import os
import numpy as np
import torch
from tqdm import tqdm
import wandb
from Qlearning.ql.config import Config
from Qlearning.ql.utils import set_seed, make_env, save_checkpoint, load_checkpoint
from Qlearning.ql.agent import QLearningAgent, Transition
from Qlearning.ql.logging_utils import setup_logger


def train(config: str = "Qlearning/configs/frozenlake.yaml", wandb_key: str = ""):
    cfg = Config.from_yaml(config)
    if wandb_key:
        cfg.wandb_key = wandb_key
    logger = setup_logger(
        name="qlearning",
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
        project=cfg.project, entity=cfg.entity, name=cfg.run_name, config=cfg.to_dict()
    )

    # Environment
    env = make_env(cfg.env_id, cfg.seed, render_mode=None, env_kwargs=cfg.env_kwargs)
    obs_space = env.observation_space
    act_space = env.action_space
    assert hasattr(obs_space, "n"), "Q-learning requires discrete observation spaces"
    assert hasattr(act_space, "n"), "Q-learning requires discrete action spaces"
    n_states, n_actions = int(obs_space.n), int(act_space.n)
    logger.info(
        f"Env={cfg.env_id} | n_states={n_states} | n_actions={n_actions}"
    )

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
    )

    ep_returns = []
    best_avg_return = -np.inf
    pbar = tqdm(range(cfg.total_episodes), desc="Q-learning Episodes")

    for ep in pbar:
        state, _ = env.reset(seed=cfg.seed + ep)
        state = int(state)
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < cfg.max_steps_per_episode:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            next_state = int(next_state)

            agent.update(Transition(state, action, float(reward), next_state, done))
            state = next_state
            ep_ret += float(reward)
            steps += 1

        ep_returns.append(ep_ret)

        if (ep + 1) % cfg.log_interval == 0:
            avg_return = float(np.mean(ep_returns[-cfg.log_interval:]))
            wandb.log(
                {
                    "charts/avg_return": avg_return,
                    "charts/epsilon": agent.epsilon(),
                    "progress/episode": ep + 1,
                    "progress/steps": agent.global_step,
                }
            )
            pbar.set_postfix({"avgR": f"{avg_return:.2f}", "eps": f"{agent.epsilon():.2f}"})

        if (ep + 1) % cfg.checkpoint_interval == 0:
            path = os.path.join(cfg.checkpoint_dir, f"checkpoint_ep{ep+1}.pt")
            save_checkpoint(path, agent.Q, ep + 1, best_avg_return)
            logger.info(f"Saved checkpoint: {path}")

        # Save best based on moving average over last 100 episodes
        if len(ep_returns) >= 10:
            avg_last = float(np.mean(ep_returns[-10:]))
            if cfg.save_best and avg_last > best_avg_return:
                best_avg_return = avg_last
                best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                save_checkpoint(best_path, agent.Q, ep + 1, best_avg_return)
                logger.info(
                    f"New best avg return {best_avg_return:.2f}; saved {best_path}"
                )

    run.finish()
    logger.info(
        f"Training finished. Best 10-ep avg return: {best_avg_return:.2f}"
    )


def demo(
    config: str = "Qlearning/configs/frozenlake.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
):
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="qlearning",
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
        render_mode=cfg.render_mode or None,
        env_kwargs=cfg.env_kwargs,
    )
    obs_space = env.observation_space
    act_space = env.action_space
    n_states, n_actions = int(obs_space.n), int(act_space.n)

    # Load Q-table
    data = load_checkpoint(model_path)
    q_table = data["q_table"].cpu().numpy() if isinstance(data["q_table"], torch.Tensor) else data["q_table"]
    assert q_table.shape == (n_states, n_actions)
    logger.info(f"Loaded Q-table from {model_path}")

    returns = []
    for ep in range(episodes):
        state, _ = env.reset(seed=cfg.seed + ep)
        state = int(state)
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < cfg.max_steps_per_episode:
            # Greedy action from Q-table
            q_vals = q_table[state]
            action = int(np.argmax(q_vals))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            state = int(next_state)
            ep_ret += float(reward)
            steps += 1
        logger.info(f"Episode {ep+1}: return={ep_ret:.2f}")
        returns.append(ep_ret)

    logger.info(f"Average return over {episodes} episodes: {np.mean(returns):.2f}")


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "train": train,
            "demo": demo,
        }
    )

