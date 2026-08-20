from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm

from nash_ql.agent import NashQLearningAgent, Transition
from nash_ql.config import Config
from nash_ql.exact_solver import (
    enumerate_transitions,
    exploitability,
    head_to_head_vs_exact,
    initial_state_indices,
    load_or_solve,
)
from nash_ql.logging_utils import setup_logger
from nash_ql.utils import (
    evaluate_vs_random,
    load_checkpoint,
    make_env,
    save_checkpoint,
    set_seed,
)


def train(
    config: str = "Nash-QL/configs/grid_soccer.yaml",
    wandb_key: str = "",
) -> None:
    """
    Train Nash Q-learning agents on a multi-agent environment.

    Args:
        config: Path to YAML configuration file
        wandb_key: Weights & Biases API key for logging
    """
    cfg = Config.from_yaml(config)
    env_wandb_key = os.getenv("WANDB_API_KEY", "")
    if wandb_key:
        cfg.wandb_key = wandb_key
    elif env_wandb_key:
        cfg.wandb_key = env_wandb_key

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

    # Exact minimax solution of the TRUE (unshaped) zero-sum game. Cached to
    # disk so subsequent runs load in milliseconds. Enabled only when the env
    # supports it (grid_soccer exposes simulate/decode/encode).
    exact_soln = None
    T_true = None
    init_states = None
    if getattr(cfg, "exact_eval", True) and hasattr(env, "simulate"):
        logger.info("Loading/solving exact Nash equilibrium for %s...", cfg.env_id)
        exact_soln = load_or_solve(
            env, cfg.env_kwargs, gamma=cfg.gamma,
            cache_dir=cfg.checkpoint_dir, shaping=0.0, tol=1e-6, verbose=True,
        )
        T_true = enumerate_transitions(env, shaping=0.0)
        init_states = list(initial_state_indices(env))
        logger.info(
            "Exact solved: V*(start ball=0)=%.4f  V*(start ball=1)=%.4f  iters=%d",
            exact_soln.V[init_states[0]], exact_soln.V[init_states[1]], exact_soln.iters,
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

    # Rolling episode outcomes (agent 0 win / agent 1 win / draw). On this
    # zero-sum game the raw self-play return is ~0 at equilibrium, so outcome
    # rates and the eval-vs-random win rate are the informative signals.
    outcomes: list[int] = []  # +1 agent0 scored, -1 agent1 scored, 0 draw
    best_win_rate = -np.inf
    best_exploit = np.inf  # lower is better; used only when exact_soln is available
    pbar = tqdm(range(cfg.total_episodes), desc="Nash Q-learning (grid soccer)")

    for ep in pbar:
        states = env.reset(seed=cfg.seed + ep)
        episode_returns = np.zeros(n_agents, dtype=np.float32)
        steps = 0
        scorer = None

        while steps < cfg.max_steps_per_episode:
            actions = agent.act(states)
            step_result = env.step(actions)
            next_states = step_result.observations

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
                scorer = step_result.info.get("scorer") if isinstance(step_result.info, dict) else None
                break

        outcomes.append(0 if scorer is None else (1 if scorer == 0 else -1))

        # Logging: self-play win / loss / draw rates over the recent window.
        if (ep + 1) % cfg.log_interval == 0:
            recent = np.array(outcomes[-cfg.log_interval:])
            win0 = float(np.mean(recent == 1))
            win1 = float(np.mean(recent == -1))
            draw = float(np.mean(recent == 0))
            wandb.log(
                {
                    "charts/agent0_win_rate": win0,
                    "charts/agent1_win_rate": win1,
                    "charts/draw_rate": draw,
                    "charts/decisive_rate": win0 + win1,
                    "charts/epsilon": agent.epsilon(),
                    "progress/episode": ep + 1,
                    "progress/steps": agent.global_step,
                }
            )
            pbar.set_postfix(
                {"win0": f"{win0:.2f}", "win1": f"{win1:.2f}", "draw": f"{draw:.2f}",
                 "eps": f"{agent.epsilon():.2f}"}
            )

        # Periodic checkpoint.
        if (ep + 1) % cfg.checkpoint_interval == 0:
            path = os.path.join(cfg.checkpoint_dir, f"checkpoint_ep{ep+1}.pt")
            save_checkpoint(path, agent.Q, ep + 1, best_win_rate)
            logger.info("Saved checkpoint: %s", path)

        # Periodic evaluation: vs-random (retained), plus exploitability +
        # head-to-head against the analytical Nash opponent when available.
        if cfg.eval_interval > 0 and (ep + 1) % cfg.eval_interval == 0:
            win_rate, draw_rate = evaluate_vs_random(
                agent, env, cfg.eval_episodes, cfg.seed + 100000,
                max_steps=cfg.max_steps_per_episode,
            )
            log_data = {
                "eval/win_rate_vs_random": win_rate,
                "eval/draw_rate_vs_random": draw_rate,
                "progress/episode": ep + 1,
            }

            new_best = False
            if exact_soln is not None and T_true is not None:
                expl = exploitability(
                    agent.Q[0], T_true, exact_soln, cfg.gamma,
                    initial_states=init_states,
                )
                h2h = head_to_head_vs_exact(
                    agent.Q[0], env, exact_soln,
                    n_episodes=cfg.eval_episodes,
                    seed=cfg.seed + 200000 + ep,
                    max_steps=cfg.max_steps_per_episode,
                )
                log_data.update({f"exact/{k}": v for k, v in expl.items()})
                log_data.update({f"exact/{k}": v for k, v in h2h.items()})
                log_data["exact/V_star_start"] = float(
                    0.5 * (exact_soln.V[init_states[0]] + exact_soln.V[init_states[1]])
                )
                logger.info(
                    "Eval ep=%d | exploit(start)=%.4f mean=%.4f | h2h vs exact: "
                    "win=%.2f draw=%.2f loss=%.2f meanR0=%.3f (V*=%.3f) | "
                    "win_vs_random=%.3f",
                    ep + 1, expl["exploit_start"], expl["exploit_mean"],
                    h2h["h2h_win"], h2h["h2h_draw"], h2h["h2h_loss"],
                    h2h["h2h_mean_r0"], log_data["exact/V_star_start"],
                    win_rate,
                )
                if cfg.save_best and expl["exploit_start"] < best_exploit:
                    best_exploit = expl["exploit_start"]
                    new_best = True
            else:
                logger.info(
                    "Eval ep=%d | win_rate_vs_random=%.3f (draw=%.3f)",
                    ep + 1, win_rate, draw_rate,
                )
                if cfg.save_best and win_rate > best_win_rate:
                    best_win_rate = win_rate
                    new_best = True

            wandb.log(log_data)
            if new_best:
                best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                metric = best_exploit if exact_soln is not None else best_win_rate
                save_checkpoint(best_path, agent.Q, ep + 1, metric)
                logger.info(
                    "New best: %s=%.4f; saved %s",
                    "exploit_start" if exact_soln is not None else "win_rate_vs_random",
                    metric, best_path,
                )

    run.finish()
    if exact_soln is not None:
        logger.info("Training finished. Best exploitability at start states: %.4f",
                    best_exploit)
    else:
        logger.info("Training finished. Best win rate vs random: %.3f", best_win_rate)


def demo(
    config: str = "Nash-QL/configs/grid_soccer.yaml",
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
    logger.info("Running %d demo episodes (learned agent 0 vs a random agent 1)...", episodes)

    import random as _random

    wins = 0
    for ep in range(episodes):
        states = env.reset(seed=cfg.seed + ep)
        steps = 0
        scorer = None

        while steps < cfg.max_steps_per_episode:
            a0 = agent.best_response_action(states[0], agent=0)
            a1 = _random.randint(0, n_actions - 1)
            step_result = env.step([a0, a1])
            states = step_result.observations
            steps += 1

            if all(step_result.terminated) or step_result.truncated:
                scorer = step_result.info.get("scorer") if isinstance(step_result.info, dict) else None
                break

        if scorer == 0:
            wins += 1
        outcome = "draw" if scorer is None else f"agent {scorer} scored"
        logger.info("Episode %d | %s | steps=%d", ep + 1, outcome, steps)

    logger.info("Learned agent 0 won %d/%d games vs the random opponent.", wins, episodes)


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "train": train,
            "demo": demo,
        }
    )
