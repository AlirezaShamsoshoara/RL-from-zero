from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import wandb
from gymnasium import spaces
from tqdm import tqdm

from MAPPO.mappo.diagnostics import task_status
from MAPPO.mappo.agent import Batch, MAPPOAgent
from MAPPO.mappo.config import Config
from MAPPO.mappo.logging_utils import setup_logger
from MAPPO.mappo.utils import (
    get_space_dims,
    load_checkpoint,
    make_multiwalker_env,
    save_checkpoint,
    set_seed,
)


def _logger(cfg):
    return setup_logger(
        name="mappo",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )


def _make_env(cfg, seed, render_mode=None):
    return make_multiwalker_env(
        n_walkers=cfg.n_walkers,
        seed=seed,
        render_mode=render_mode,
        discretize_actions=cfg.discretize_actions,
        action_bins=cfg.action_bins,
        env_kwargs=cfg.env_kwargs,
    )


def _build_agent(cfg, env):
    obs_dim, act_dim, state_dim, n_agents, action_space = get_space_dims(
        env, return_action_space=True
    )
    if n_agents != cfg.n_agents:
        raise ValueError("configured agent count does not match environment")
    continuous = isinstance(action_space, spaces.Box)
    return MAPPOAgent(
        n_agents,
        obs_dim,
        act_dim,
        state_dim,
        cfg.actor_hidden_sizes,
        cfg.critic_hidden_sizes,
        cfg.activation,
        cfg.learning_rate,
        cfg.clip_coef,
        cfg.ent_coef,
        cfg.vf_coef,
        cfg.max_grad_norm,
        cfg.device,
        cfg.share_policy,
        cfg.use_centralized_critic,
        action_type="continuous" if continuous else "discrete",
        action_low=action_space.low if continuous else None,
        action_high=action_space.high if continuous else None,
        critic_learning_rate=cfg.critic_learning_rate,
        separate_grad_clipping=cfg.separate_grad_clipping,
        clip_value_loss=cfg.clip_value_loss,
        normalize_inputs=cfg.normalize_inputs,
    )


def _observations(observations, order, obs_dim):
    if set(observations) - set(order):
        raise ValueError("environment returned unknown agent IDs")
    result = np.stack(
        [
            np.asarray(observations.get(name, np.zeros(obs_dim)), dtype=np.float32)
            for name in order
        ]
    )
    if result.shape != (len(order), obs_dim) or not np.isfinite(result).all():
        raise ValueError("agent observations have invalid shape or non-finite values")
    return result


def _actions(actions, observations, order, continuous):
    return {
        name: actions[i] if continuous else int(actions[i])
        for i, name in enumerate(order)
        if name in observations
    }


def _transition(
    next_observations,
    rewards,
    terminations,
    truncations,
    infos,
    active,
    order,
    obs_dim,
    capped,
):
    """Preserve transition endpoints for bootstrap before filtering/resetting agents."""
    endpoints = dict(next_observations)
    reward_array = np.array(
        [float(rewards.get(name, 0.0)) if name in active else 0.0 for name in order],
        dtype=np.float32,
    )
    terminal = np.ones(len(order), dtype=np.float32)
    done = np.ones(len(order), dtype=np.float32)
    for i, name in enumerate(order):
        if name not in active:
            continue
        terminal[i] = float(terminations.get(name, False))
        timeout = bool(truncations.get(name, False))
        done[i] = float(terminal[i] or timeout or capped)
        info = infos.get(name, {}) if isinstance(infos, dict) else {}
        if isinstance(info, dict):
            final = info.get("final_observation", info.get("final_obs"))
            if final is not None:
                endpoints[name] = final
        if name not in endpoints and not terminal[i]:
            raise RuntimeError(
                "a nonterminal/timeout transition requires its final observation"
            )
    for i, name in enumerate(order):
        if terminal[i]:
            # Survivors must bootstrap from the same zero-padded dead-agent
            # representation they will use on the following rollout transition.
            endpoints[name] = np.zeros(obs_dim, dtype=np.float32)
    if not np.isfinite(reward_array).all():
        raise ValueError("environment rewards must be finite")
    next_active = {
        name: obs
        for name, obs in next_observations.items()
        if name in active
        and not (terminations.get(name, False) or truncations.get(name, False))
    }
    ended = capped or not next_active
    return (
        _observations(endpoints, order, obs_dim),
        reward_array,
        terminal,
        done,
        next_active,
        ended,
    )


def train(
    config: str = "MAPPO/configs/multiwalker_delivery_long_credit.yaml",
    wandb_key: str = "",
    init_checkpoint: Optional[str] = None,
):
    cfg = Config.from_yaml(config)
    if cfg.num_envs > 1:
        from MAPPO.train_delivery import train as train_parallel

        return train_parallel(
            config=config,
            init_checkpoint=init_checkpoint or cfg.init_checkpoint,
            wandb_key=wandb_key,
        )
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
        env = _make_env(cfg, cfg.seed)
        agent = _build_agent(cfg, env)
        if init_checkpoint or cfg.init_checkpoint:
            load_checkpoint(init_checkpoint or cfg.init_checkpoint, agent.models, None)
        order = list(env.possible_agents)
        n_agents, obs_dim, state_dim = agent.n_agents, agent.obs_dim, agent.state_dim
        continuous = agent.action_type == "continuous"
        observations, _ = env.reset(seed=cfg.seed)
        episode_returns, episode_lengths, per_agent_returns, survival = [], [], [], []
        ep_rewards = np.zeros(n_agents, dtype=np.float64)
        ep_length = 0
        ep_had_termination = False
        best_avg_return = -np.inf
        global_step = optimizer_steps = agent_transitions = 0
        num_updates = (cfg.total_timesteps + cfg.rollout_steps - 1) // cfg.rollout_steps
        log_window = []
        pbar = tqdm(range(num_updates), desc="MAPPO Updates")
        for update in pbar:
            current_lr = (
                cfg.learning_rate * (1 - update / num_updates)
                if cfg.anneal_lr
                else cfg.learning_rate
            )
            for optimizer in agent.optimizers:
                for index, group in enumerate(optimizer.param_groups):
                    group["lr"] = (
                        (
                            (cfg.critic_learning_rate or cfg.learning_rate)
                            if index
                            else cfg.learning_rate
                        )
                        * (1 - update / num_updates)
                        if cfg.anneal_lr
                        else (
                            (cfg.critic_learning_rate or cfg.learning_rate)
                            if index
                            else cfg.learning_rate
                        )
                    )
            length = min(cfg.rollout_steps, cfg.total_timesteps - global_step)
            obs_buf = np.zeros((length, n_agents, obs_dim), np.float32)
            state_buf = np.zeros((length, state_dim), np.float32)
            action_shape = (
                (length, n_agents, agent.act_dim) if continuous else (length, n_agents)
            )
            action_buf = np.zeros(action_shape, np.float32 if continuous else np.int64)
            logprob_buf = np.zeros((length, n_agents), np.float32)
            reward_buf, done_buf, term_buf, value_buf, next_value_buf, alive_buf = [
                np.zeros((length, n_agents), np.float32) for _ in range(6)
            ]
            for t in range(length):
                if not observations:
                    raise RuntimeError("cannot step an empty active-agent set")
                active = set(observations)
                obs_array = _observations(observations, order, obs_dim)
                joint_state = obs_array.reshape(-1)
                actions, logprobs, values = agent.act(obs_array, joint_state)
                obs_buf[t], state_buf[t], action_buf[t], logprob_buf[t] = (
                    obs_array,
                    joint_state,
                    actions,
                    logprobs,
                )
                alive = np.array([name in active for name in order], np.float32)
                alive_buf[t], value_buf[t] = alive, values * alive
                result = env.step(_actions(actions, observations, order, continuous))
                next_obs, rewards, terms, truncs, infos = result
                ep_length += 1
                capped = ep_length >= cfg.max_steps_per_episode
                endpoints, reward_array, terminal, dones, next_active, ended = (
                    _transition(
                        next_obs,
                        rewards,
                        terms,
                        truncs,
                        infos,
                        active,
                        order,
                        obs_dim,
                        capped,
                    )
                )
                if hasattr(env, "agents") and not env.agents:
                    ended = True
                    dones[alive > 0.5] = 1.0
                reward_buf[t], term_buf[t], done_buf[t] = reward_array, terminal, dones
                next_value_buf[t] = (
                    agent.values(endpoints, endpoints.reshape(-1)) * alive
                )
                ep_rewards += reward_array
                ep_had_termination |= any(
                    bool(terms.get(name, False)) for name in active
                )
                global_step += 1
                agent_transitions += len(active)
                if ended:
                    # Fixed original-team denominator avoids score changes when agents disappear.
                    episode_returns.append(float(np.mean(ep_rewards)))
                    per_agent_returns.append(ep_rewards.copy())
                    episode_lengths.append(ep_length)
                    horizon = cfg.env_kwargs.get("max_cycles", 500)
                    survival.append(
                        float(ep_length >= horizon and not ep_had_termination)
                    )
                    observations, _ = env.reset()
                    ep_rewards[:] = 0.0
                    ep_length = 0
                    ep_had_termination = False
                else:
                    observations = next_active
            advantages, returns = MAPPOAgent.compute_gae(
                reward_buf,
                done_buf,
                value_buf,
                next_value_buf[-1],
                cfg.gamma,
                cfg.gae_lambda,
                next_value_sequence=next_value_buf,
                terminations=term_buf,
            )
            action_dtype = torch.float32 if continuous else torch.int64
            batches = [
                Batch(
                    torch.as_tensor(obs_buf[:, i], device=agent.device),
                    torch.as_tensor(state_buf, device=agent.device),
                    torch.as_tensor(
                        action_buf[:, i], dtype=action_dtype, device=agent.device
                    ),
                    torch.as_tensor(logprob_buf[:, i], device=agent.device),
                    torch.as_tensor(returns[:, i], device=agent.device),
                    torch.as_tensor(advantages[:, i], device=agent.device),
                    torch.as_tensor(value_buf[:, i], device=agent.device),
                    torch.as_tensor(alive_buf[:, i], device=agent.device),
                )
                for i in range(n_agents)
            ]
            all_stats = []
            rejected_steps = 0
            early_stop = False
            for _ in range(cfg.update_iterations):
                indices = np.random.permutation(length)
                for start in range(0, length, cfg.minibatch_size):
                    selected = indices[start : start + cfg.minibatch_size]
                    minibatches = [
                        Batch(
                            **{
                                field: getattr(batch, field)[selected]
                                for field in Batch.__dataclass_fields__
                            }
                        )
                        for batch in batches
                    ]
                    stats, steps = agent.update(minibatches)
                    optimizer_steps += steps
                    rejected_steps += int(stats.get("stats/skipped_updates", 0))
                    if steps:
                        all_stats.append((stats, steps))
                    if stats.get("stats/skipped_updates", 0) > 0:
                        early_stop = True
                        break
                    if (
                        steps
                        and cfg.target_kl > 0
                        and stats["stats/approx_kl"] > cfg.target_kl
                    ):
                        early_stop = True
                        break
                if early_stop:
                    break
            stats = (
                {
                    key: float(
                        sum(s[key] * count for s, count in all_stats)
                        / sum(count for _, count in all_stats)
                    )
                    for key in all_stats[0][0]
                }
                if all_stats
                else dict.fromkeys(
                    (
                        "loss/policy",
                        "loss/value",
                        "loss/total",
                        "stats/entropy",
                        "stats/approx_kl",
                    ),
                    0.0,
                )
            )
            stats["stats/skipped_updates"] = float(rejected_steps)
            mask = alive_buf > 0.5
            targets = returns[mask]
            predicted = value_buf[mask]
            target_variance = float(np.var(targets))
            log_window.append(
                {
                    **stats,
                    "stats/value_mean": float(np.mean(predicted)),
                    "stats/return_target_mean": float(np.mean(targets)),
                    "stats/explained_variance": 1
                    - float(np.var(targets - predicted)) / target_variance
                    if target_variance > 1e-8
                    else 0.0,
                    "stats/early_stop_kl": float(early_stop),
                }
            )
            avg_return = (
                float(np.mean(episode_returns[-10:])) if episode_returns else 0.0
            )
            if (update + 1) % cfg.log_interval == 0 or update + 1 == num_updates:
                payload = {
                    key: float(np.mean([row[key] for row in log_window]))
                    for key in log_window[0]
                }
                payload.update(
                    {
                        "charts/avg_return": avg_return,
                        "charts/avg_length": float(np.mean(episode_lengths[-10:]))
                        if episode_lengths
                        else 0.0,
                        "charts/survival_rate": float(np.mean(survival[-10:]))
                        if survival
                        else 0.0,
                        "charts/episodes": len(episode_returns),
                        "charts/learning_rate": current_lr,
                        "progress/global_step": global_step,
                        "progress/update": update + 1,
                        "progress/agent_transitions": agent_transitions,
                        "progress/optimizer_steps": optimizer_steps,
                    }
                )
                if per_agent_returns:
                    for i, value in enumerate(np.mean(per_agent_returns[-10:], axis=0)):
                        payload[f"charts/agent_{i}_avg_return"] = float(value)
                wandb.log(payload, step=global_step)
                log_window.clear()
            pbar.set_postfix(
                {"avgR": f"{avg_return:.1f}", "loss": f"{stats['loss/total']:.3f}"}
            )
            if cfg.save_best and len(episode_returns) >= cfg.best_window:
                score = float(np.mean(episode_returns[-cfg.best_window :]))
                if score > best_avg_return:
                    best_avg_return = score
                    save_checkpoint(
                        os.path.join(cfg.checkpoint_dir, "best.pt"),
                        agent.models,
                        agent.optimizers,
                        global_step,
                        score,
                        cfg.share_policy,
                        config=cfg.to_dict(),
                    )
                    logger.info(
                        "New best %d-episode mean return %.2f", cfg.best_window, score
                    )
            if (update + 1) % cfg.checkpoint_interval == 0 or update + 1 == num_updates:
                save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, f"checkpoint_{global_step}.pt"),
                    agent.models,
                    agent.optimizers,
                    global_step,
                    best_avg_return,
                    cfg.share_policy,
                    config=cfg.to_dict(),
                )
        if hasattr(run, "summary"):
            run.summary.update(
                {
                    "training/environment_steps": global_step,
                    "training/agent_transitions": agent_transitions,
                    "training/best_return": best_avg_return
                    if np.isfinite(best_avg_return)
                    else None,
                }
            )
        logger.info(
            "Training finished. Best completed-episode return %.2f", best_avg_return
        )
        return {
            "global_step": global_step,
            "agent_transitions": agent_transitions,
            "optimizer_steps": optimizer_steps,
            "best_avg_return": best_avg_return,
        }
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            run.finish()


def demo(
    config: str = "MAPPO/configs/multiwalker_delivery_long_credit.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
    deterministic: bool = True,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
):
    cfg = Config.from_yaml(config)
    logger = _logger(cfg)
    episodes = cfg.episodes if episodes is None else episodes
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    seed = cfg.seed if seed is None else seed
    mode = cfg.render_mode if render_mode is None else render_mode
    if isinstance(mode, str) and mode.lower() in {"none", "null"}:
        mode = None
    set_seed(seed)
    env = _make_env(cfg, seed, mode)
    try:
        agent = _build_agent(cfg, env)
        load_checkpoint(model_path or cfg.inference_model_path, agent.models, None)
        for model in agent.models:
            model.eval()
        order = list(env.possible_agents)
        returns, lengths, individual, delivery_flags, task_results = [], [], [], [], []
        for episode in range(episodes):
            observations, _ = env.reset(seed=seed + episode)
            initial_x = task_status(env).get("package_x")
            total = np.zeros(agent.n_agents, np.float64)
            for t in range(cfg.max_steps_per_episode):
                obs = _observations(observations, order, agent.obs_dim)
                actions, _, _ = agent.act(
                    obs, obs.reshape(-1), deterministic=deterministic
                )
                next_obs, rewards, terms, truncs, _ = env.step(
                    _actions(
                        actions, observations, order, agent.action_type == "continuous"
                    )
                )
                total += np.array(
                    [
                        float(rewards.get(name, 0.0)) if name in observations else 0.0
                        for name in order
                    ]
                )
                observations = {
                    name: value
                    for name, value in next_obs.items()
                    if name in observations
                    and not (terms.get(name, False) or truncs.get(name, False))
                }
                if not observations or (hasattr(env, "agents") and not env.agents):
                    break
            returns.append(float(np.mean(total)))
            lengths.append(t + 1)
            individual.append(total.tolist())
            outcome = task_status(env, initial_x)
            delivery_flags.append(outcome["delivered"])
            task_results.append(outcome)
            logger.info(
                "Episode %d: return %.2f, length %d, delivered=%s, remaining distance=%s",
                episode + 1,
                returns[-1],
                lengths[-1],
                outcome["delivered"],
                outcome["remaining_distance"],
            )
        return {
            "returns": returns,
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "lengths": lengths,
            "per_agent_returns": individual,
            "delivered": delivery_flags,
            "delivery_rate": sum(value is True for value in delivery_flags)
            / len(delivery_flags)
            if any(value is not None for value in delivery_flags)
            else None,
            "task_results": task_results,
        }
    finally:
        env.close()


if __name__ == "__main__":
    import fire

    fire.Fire({"train": train, "demo": demo})
