from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from numbers import Integral
import traceback

import gymnasium as gym
import numpy as np
import torch

from .utils import compute_returns, make_env, make_vec_env, set_seed


def _masked_info_value(infos, key: str, index: int):
    """Read a vector-info field only when its per-environment mask allows it."""
    if not isinstance(infos, Mapping) or key not in infos:
        return None
    mask = infos.get(f"_{key}")
    if mask is not None and not bool(mask[index]):
        return None
    return infos[key][index]


def _transition_info_value(infos, key: str, index: int, boundary: bool):
    """SAME_STEP final_info may be a batched dict or an array of dicts."""
    if boundary and isinstance(infos, Mapping) and "final_info" in infos:
        mask = infos.get("_final_info")
        if mask is None or bool(mask[index]):
            final_infos = infos["final_info"]
            if isinstance(final_infos, Mapping):
                value = _masked_info_value(final_infos, key, index)
            else:
                final_info = final_infos[index]
                value = final_info.get(key) if isinstance(final_info, Mapping) else None
            if value is not None:
                return value
    return _masked_info_value(infos, key, index)


def _final_observation(infos, index: int, expected_shape):
    for key in ("final_obs", "final_observation"):
        observation = _masked_info_value(infos, key, index)
        if observation is not None:
            observation = np.asarray(observation, dtype=np.float32)
            if (
                observation.shape != expected_shape
                or not np.isfinite(observation).all()
            ):
                raise ValueError(
                    f"Invalid SAME_STEP timeout final observation for env {index}"
                )
            return observation
    raise RuntimeError(
        f"SAME_STEP timeout for env {index} is missing its actual final observation; "
        "refusing to bootstrap from the reset observation"
    )


def _values_for_observations(local_model, observations, device):
    with torch.no_grad():
        states = torch.as_tensor(observations, dtype=torch.float32, device=device)
        _, values = local_model.forward(states)
        return values.detach().reshape(-1)


def worker_process(
    worker_id: int,
    cfg,
    shared_agent,
    global_step,
    result_queue,
    stop_event,
):
    """Collect reserved transitions and flush even the last partial rollout.

    global_step is a reservation counter, not a live completion counter. On a
    successful joined run, the sum of worker_done collected_transitions equals
    it exactly. Budget exhaustion does not set stop_event: other workers must
    finish the transitions they already reserved and their final gradients.
    """
    env = None
    failure = None
    cleanup_failure = None
    collected_transitions = 0
    reserved_transitions = 0
    worker_rollout_steps = 0
    optimizer_steps = 0
    updates_attempted = 0
    episodes = 0
    current_step = 0
    budget_exhausted = False

    try:
        torch.set_num_threads(1)
        num_envs = getattr(cfg, "num_envs", 1)
        if (
            isinstance(num_envs, bool)
            or not isinstance(num_envs, Integral)
            or num_envs < 1
        ):
            raise ValueError("num_envs must be a positive integer")
        total_steps = cfg.total_steps
        if (
            isinstance(total_steps, bool)
            or not isinstance(total_steps, Integral)
            or total_steps < 0
        ):
            raise ValueError("total_steps must be a nonnegative integer")
        if total_steps % num_envs:
            raise ValueError("total_steps must be divisible by num_envs")
        if (
            isinstance(cfg.t_max, bool)
            or not isinstance(cfg.t_max, Integral)
            or cfg.t_max < 1
        ):
            raise ValueError("t_max must be a positive integer")

        use_vec = num_envs > 1
        base_seed = cfg.seed + worker_id * num_envs
        set_seed(base_seed)
        reward_shaping = getattr(cfg, "reward_shaping", False)
        reward_scale = getattr(cfg, "reward_scale", 1.0)
        env_kwargs = dict(getattr(cfg, "env_kwargs", None) or {})
        effective_episode_limit = None
        safety_cap = getattr(cfg, "max_steps_per_episode", None)
        if safety_cap is not None:
            if (
                isinstance(safety_cap, bool)
                or not isinstance(safety_cap, Integral)
                or safety_cap < 1
            ):
                raise ValueError("max_steps_per_episode must be a positive integer")
            # Let TimeLimit produce a genuine truncation inside SAME_STEP, so
            # final observations and native statistics are preserved. Never
            # lengthen a registered native limit or an explicit shorter limit.
            limits = [int(safety_cap)]
            native_limit = gym.spec(cfg.env_id).max_episode_steps
            explicit_limit = env_kwargs.get("max_episode_steps")
            for limit in (native_limit, explicit_limit):
                if (
                    isinstance(limit, Integral)
                    and not isinstance(limit, bool)
                    and limit > 0
                ):
                    limits.append(int(limit))
            effective_episode_limit = min(limits)
            env_kwargs["max_episode_steps"] = effective_episode_limit
        if use_vec:
            env = make_vec_env(
                cfg.env_id,
                num_envs=num_envs,
                seed=base_seed,
                reward_shaping=reward_shaping,
                gamma=cfg.gamma,
                reward_scale=reward_scale,
                env_kwargs=env_kwargs,
            )
            reset_seed = [base_seed + index for index in range(num_envs)]
        else:
            env = make_env(
                cfg.env_id,
                seed=base_seed,
                reward_shaping=reward_shaping,
                gamma=cfg.gamma,
                reward_scale=reward_scale,
                env_kwargs=env_kwargs,
            )
            reset_seed = base_seed
        obs, _ = env.reset(seed=reset_seed)
        local_model = shared_agent.new_local_model()
        device = shared_agent.device
        anneal_lr = getattr(cfg, "anneal_lr", False)
        initial_lr = cfg.learning_rate
        anneal_entropy = getattr(cfg, "anneal_entropy", False)
        entropy_coef_start = cfg.entropy_coef
        entropy_coef_end = getattr(cfg, "entropy_coef_end", cfg.entropy_coef)
        normalize_advantages = getattr(cfg, "normalize_advantages", False)
        native_returns = np.zeros(num_envs, dtype=np.float64)
        learning_returns = np.zeros(num_envs, dtype=np.float64)
        episode_lengths = np.zeros(num_envs, dtype=np.int64)

        with global_step.get_lock():
            current_step = global_step.value

        while not stop_event.is_set():
            log_probs = []
            values = []
            entropies = []
            rewards = []
            boundaries = []
            terminations = []
            next_values_sequence = []
            rollout_steps = 0

            for _ in range(cfg.t_max):
                if stop_event.is_set():
                    break
                # Reserve BEFORE stepping. Do not let one worker consume another
                # worker's allowance or set the stop event at the last grant.
                with global_step.get_lock():
                    if stop_event.is_set():
                        break
                    if total_steps - global_step.value < num_envs:
                        budget_exhausted = True
                        break
                    global_step.value += num_envs
                    current_step = global_step.value
                    reserved_transitions += num_envs

                # A saved autograd input must not alias an env's reusable
                # observation buffer, which the next step may overwrite.
                state = torch.tensor(obs, dtype=torch.float32, device=device)
                action, log_prob, entropy, value = local_model.act(state)
                env_action = (
                    action.detach().cpu().numpy() if use_vec else int(action.item())
                )
                next_obs, reward, terminated, truncated, infos = env.step(env_action)
                # Record what was actually collected even if later processing fails.
                collected_transitions += num_envs
                worker_rollout_steps += 1
                rollout_steps += 1
                reward_array = np.asarray(reward, dtype=np.float64).reshape(num_envs)
                terminal_array = np.asarray(terminated, dtype=bool).reshape(num_envs)
                truncated_array = np.asarray(truncated, dtype=bool).reshape(num_envs)
                boundary_array = terminal_array | truncated_array
                timeout_array = truncated_array & ~terminal_array
                tensor_shape = (num_envs,) if use_vec else ()
                endpoint_values = torch.zeros(
                    tensor_shape, dtype=torch.float32, device=device
                )

                # Only timeout boundaries need per-step endpoint values. Ordinary
                # transitions use recursive returns; true terminals use zero.
                if use_vec and timeout_array.any():
                    indices = np.flatnonzero(timeout_array)
                    final_states = np.stack(
                        [
                            _final_observation(
                                infos, int(index), np.asarray(next_obs[index]).shape
                            )
                            for index in indices
                        ]
                    )
                    timeout_values = _values_for_observations(
                        local_model, final_states, device
                    )
                    endpoint_values[
                        torch.as_tensor(indices, dtype=torch.long, device=device)
                    ] = timeout_values
                elif not use_vec and timeout_array[0]:
                    endpoint_values = _values_for_observations(
                        local_model, next_obs, device
                    ).reshape(())

                log_probs.append(log_prob.reshape(tensor_shape))
                values.append(value.reshape(tensor_shape))
                entropies.append(entropy.reshape(tensor_shape))
                rewards.append(
                    torch.tensor(
                        reward_array, dtype=torch.float32, device=device
                    ).reshape(tensor_shape)
                )
                boundaries.append(
                    torch.tensor(
                        boundary_array, dtype=torch.bool, device=device
                    ).reshape(tensor_shape)
                )
                terminations.append(
                    torch.tensor(
                        terminal_array, dtype=torch.bool, device=device
                    ).reshape(tensor_shape)
                )
                next_values_sequence.append(endpoint_values)

                for index in range(num_envs):
                    if use_vec:

                        def info_value(key):
                            return _transition_info_value(
                                infos, key, index, bool(boundary_array[index])
                            )
                    else:

                        def info_value(key):
                            return (
                                infos.get(key) if isinstance(infos, Mapping) else None
                            )

                    native_reward = info_value("native_reward")
                    if native_reward is None:
                        # This fallback is only valid when rewards are unmodified,
                        # for example a plain test environment. Never undo shaping
                        # or guess native performance from scaled learning rewards.
                        if reward_shaping or reward_scale != 1.0:
                            raise RuntimeError(
                                "Transformed reward is missing native_reward episode info"
                            )
                        native_reward = reward_array[index]
                    native_reward = float(native_reward)
                    native_returns[index] += native_reward
                    learning_returns[index] += reward_array[index]
                    episode_lengths[index] += 1
                    if boundary_array[index]:
                        episode_return = info_value("native_episode_return")
                        episode_return = (
                            native_returns[index]
                            if episode_return is None
                            else float(episode_return)
                        )
                        episode_length = info_value("native_episode_length")
                        episode_length = (
                            episode_lengths[index]
                            if episode_length is None
                            else int(episode_length)
                        )
                        episodes += 1
                        result_queue.put(
                            {
                                "kind": "episode",
                                "worker_id": worker_id,
                                "env_index": index,
                                "episode_return": float(episode_return),
                                "native_episode_return": float(episode_return),
                                "learning_episode_return": float(
                                    learning_returns[index]
                                ),
                                "episode_length": int(episode_length),
                                "native_episode_length": int(episode_length),
                                "native_reward": native_reward,
                                "learning_reward": float(reward_array[index]),
                                "terminated": bool(terminal_array[index]),
                                "truncated": bool(truncated_array[index]),
                                "episode_limit_reached": bool(
                                    truncated_array[index]
                                    and effective_episode_limit is not None
                                    and episode_length >= effective_episode_limit
                                ),
                                "max_episode_steps": effective_episode_limit,
                                "global_step": current_step,
                                "collected_transitions": collected_transitions,
                            }
                        )
                        native_returns[index] = 0.0
                        learning_returns[index] = 0.0
                        episode_lengths[index] = 0

                obs = next_obs
                if not use_vec and boundary_array[0]:
                    # Capture a timeout endpoint BEFORE this reset. A true
                    # terminal never receives a value from the new episode.
                    obs, _ = env.reset()
                    break

            if rollout_steps == 0:
                break

            rewards_t = torch.stack(rewards)
            boundaries_t = torch.stack(boundaries)
            terminations_t = torch.stack(terminations)
            endpoint_values_t = torch.stack(next_values_sequence)
            last_value = endpoint_values_t[-1].clone()
            if use_vec:
                continuing = ~boundaries_t[-1]
                if bool(continuing.any()):
                    last_value[continuing] = _values_for_observations(
                        local_model, np.asarray(obs)[continuing.cpu().numpy()], device
                    )
            elif not bool(boundaries_t[-1]):
                last_value = _values_for_observations(local_model, obs, device).reshape(
                    ()
                )

            values_t = torch.stack(values)
            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)
            returns_t = compute_returns(
                rewards_t,
                boundaries_t,
                last_value,
                cfg.gamma,
                terminations=terminations_t,
                next_values_sequence=endpoint_values_t,
            )
            advantages = returns_t - values_t
            if use_vec:
                log_probs_t = log_probs_t.reshape(-1)
                values_t = values_t.reshape(-1)
                entropies_t = entropies_t.reshape(-1)
                returns_t = returns_t.reshape(-1)
                advantages = advantages.reshape(-1)

            fraction = max(1.0 - current_step / total_steps, 0.0)
            entropy_coef = (
                entropy_coef_end + (entropy_coef_start - entropy_coef_end) * fraction
                if anneal_entropy
                else None
            )
            total_loss, stats = shared_agent.compute_loss(
                advantages=advantages,
                log_probs=log_probs_t,
                values=values_t,
                returns=returns_t,
                entropies=entropies_t,
                entropy_coef=entropy_coef,
                normalize_advantages=normalize_advantages,
            )
            local_model.zero_grad()
            total_loss.backward()
            learning_rate = initial_lr * fraction if anneal_lr else None
            updates_attempted += 1
            # The agent owns the lock for optimizer LR/state changes and local
            # synchronization. A rejected update must not look like success.
            applied = shared_agent.apply_gradients(
                local_model, learning_rate=learning_rate, sync=True
            )
            if not applied:
                raise RuntimeError(
                    "A3C gradient update was rejected; no optimizer step completed"
                )
            optimizer_steps += 1
            result_queue.put(
                {
                    "kind": "update",
                    "worker_id": worker_id,
                    "stats": asdict(stats),
                    "global_step": current_step,
                    "rollout_steps": rollout_steps * num_envs,
                    "worker_rollout_steps": rollout_steps,
                    "collected_transitions": collected_transitions,
                    "optimizer_steps": optimizer_steps,
                    "update_applied": True,
                }
            )
            if budget_exhausted:
                break

    except BaseException as exc:
        failure = exc
        stop_event.set()
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    cleanup_failure = exc
                stop_event.set()

    summary = {
        "worker_id": worker_id,
        "global_step": current_step,
        "global_step_semantics": "reserved_transitions",
        "collected_transitions": collected_transitions,
        "transitions": collected_transitions,
        "reserved_transitions": reserved_transitions,
        "rollout_steps": collected_transitions,
        "worker_rollout_steps": worker_rollout_steps,
        "optimizer_steps": optimizer_steps,
        "updates_attempted": updates_attempted,
        "episodes": episodes,
    }
    if failure is not None:
        summary.update(
            {
                "kind": "worker_error",
                "exception_type": type(failure).__name__,
                "error": str(failure),
                "traceback": "".join(
                    traceback.format_exception(
                        type(failure), failure, failure.__traceback__
                    )
                ),
            }
        )
        if cleanup_failure is not None:
            summary["cleanup_error"] = "".join(
                traceback.format_exception(
                    type(cleanup_failure),
                    cleanup_failure,
                    cleanup_failure.__traceback__,
                )
            )
        try:
            result_queue.put(summary)
        finally:
            # A queue delivery alone is not reliable failure propagation: the
            # parent must also see a nonzero child exit if that delivery fails.
            raise failure.with_traceback(failure.__traceback__)
    summary.update(
        {
            "kind": "worker_done",
            "stopped": bool(stop_event.is_set()),
            "budget_exhausted": budget_exhausted,
        }
    )
    result_queue.put(summary)
