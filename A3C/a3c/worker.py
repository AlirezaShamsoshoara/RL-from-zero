from __future__ import annotations
from dataclasses import asdict
import numpy as np
import torch
from .utils import make_env, make_vec_env, compute_returns


def worker_process(
    worker_id: int,
    cfg,
    shared_agent,
    global_step,
    result_queue,
    stop_event,
):
    num_envs = getattr(cfg, "num_envs", 1)
    use_vec = num_envs > 1

    if use_vec:
        env = make_vec_env(
            cfg.env_id,
            num_envs=num_envs,
            seed=cfg.seed + worker_id * num_envs,
        )
    else:
        env = make_env(cfg.env_id, seed=cfg.seed + worker_id)

    local_model = shared_agent.new_local_model()
    device = shared_agent.device

    obs, _ = env.reset(seed=cfg.seed + worker_id)

    if use_vec:
        ep_returns = np.zeros(num_envs, dtype=np.float64)
        ep_lengths = np.zeros(num_envs, dtype=np.int64)
    else:
        episode_return = 0.0
        episode_length = 0

    current_step = 0

    while not stop_event.is_set():
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list = []
        dones: list = []
        rollout_steps = 0

        for _ in range(cfg.t_max):
            if stop_event.is_set():
                break

            state_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, log_prob, entropy, value = local_model.act(state_t)

            if use_vec:
                next_obs, reward, terminated, truncated, _infos = env.step(
                    action.numpy()
                )
                done = np.logical_or(terminated, truncated)

                log_probs.append(log_prob.squeeze())
                values.append(value.squeeze())
                entropies.append(entropy.squeeze())
                rewards.append(
                    torch.tensor(reward, dtype=torch.float32, device=device)
                )
                dones.append(
                    torch.tensor(done, dtype=torch.float32, device=device)
                )

                ep_returns += reward
                ep_lengths += 1

                with global_step.get_lock():
                    global_step.value += num_envs
                    current_step = global_step.value
                if current_step >= cfg.total_steps:
                    stop_event.set()

                for i in range(num_envs):
                    if done[i]:
                        result_queue.put(
                            {
                                "kind": "episode",
                                "worker_id": worker_id,
                                "episode_return": float(ep_returns[i]),
                                "episode_length": int(ep_lengths[i]),
                                "global_step": current_step,
                            }
                        )
                        ep_returns[i] = 0.0
                        ep_lengths[i] = 0

                obs = next_obs
                rollout_steps += 1

            else:
                # --- original single-env path (unchanged behaviour) ---
                next_obs, reward, terminated, truncated, _ = env.step(
                    int(action.item())
                )
                done = bool(terminated or truncated)

                log_probs.append(log_prob.squeeze())
                values.append(value.squeeze())
                entropies.append(entropy.squeeze())
                rewards.append(float(reward))
                dones.append(1.0 if done else 0.0)
                episode_return += float(reward)
                episode_length += 1
                rollout_steps += 1

                with global_step.get_lock():
                    global_step.value += 1
                    current_step = global_step.value
                if current_step >= cfg.total_steps:
                    stop_event.set()

                obs = next_obs
                if done:
                    result_queue.put(
                        {
                            "kind": "episode",
                            "worker_id": worker_id,
                            "episode_return": episode_return,
                            "episode_length": episode_length,
                            "global_step": current_step,
                        }
                    )
                    obs, _ = env.reset()
                    episode_return = 0.0
                    episode_length = 0
                    break

        if rollout_steps == 0:
            continue

        # --- bootstrap value for n-step returns ---
        with torch.no_grad():
            if use_vec:
                next_state = torch.as_tensor(
                    obs, dtype=torch.float32, device=device
                )
                _, next_value_t = local_model.forward(next_state)
                next_value = next_value_t.squeeze()
            else:
                if dones[-1]:
                    next_value = torch.zeros(1, dtype=torch.float32, device=device)
                else:
                    next_state = torch.as_tensor(
                        obs, dtype=torch.float32, device=device
                    )
                    _, next_value_t = local_model.forward(next_state)
                    next_value = next_value_t.squeeze()

        # --- compute returns & advantages ---
        if use_vec:
            rewards_t = torch.stack(rewards)      # (steps, num_envs)
            dones_t = torch.stack(dones)           # (steps, num_envs)
        else:
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

        values_t = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        returns_t = compute_returns(rewards_t, dones_t, next_value, cfg.gamma)
        advantages = returns_t - values_t

        # flatten for loss when vectorized
        if use_vec:
            log_probs_t = log_probs_t.reshape(-1)
            values_t = values_t.reshape(-1)
            entropies_t = entropies_t.reshape(-1)
            returns_t = returns_t.reshape(-1)
            advantages = advantages.reshape(-1)

        total_loss, stats = shared_agent.compute_loss(
            advantages=advantages,
            log_probs=log_probs_t,
            values=values_t,
            returns=returns_t,
            entropies=entropies_t,
        )

        local_model.zero_grad()
        total_loss.backward()
        shared_agent.apply_gradients(local_model)
        shared_agent.sync_local(local_model)

        result_queue.put(
            {
                "kind": "update",
                "worker_id": worker_id,
                "stats": asdict(stats),
                "global_step": current_step,
                "rollout_steps": rollout_steps * num_envs,
            }
        )

    env.close()
