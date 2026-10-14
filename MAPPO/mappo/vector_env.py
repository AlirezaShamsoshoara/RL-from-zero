"""Spawn-safe parallel MultiWalker collection with one transition per worker step.

Workers own only environments. Policy/critic computation stays in the parent.
Every ``step`` fans out once to all workers and contributes ``num_envs`` joint
transitions, not ``num_envs * n_agents``. Native episode endings are reset inside
that same request: ``next_obs`` remains the pre-reset bootstrap endpoint and
``obs`` is the observation for the next interaction. No reset-only training
steps are inserted. Checkpoints do not resume these environments or their RNGs.

A custom ``env_factory(cfg, seed, render_mode=None)`` must be a picklable,
module-level callable. The application entrypoint must use a ``__main__`` guard
for the default spawn context. No environment or child is created on import.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import pickle
import threading
import time
import traceback
from typing import Optional


_START_LOCK = threading.Lock()
_WORKER_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}


class VectorWorkerError(RuntimeError):
    """An environment worker failed, exited, or exceeded an RPC deadline."""


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _timeout(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _seed(value, num_envs, n_agents):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("start_seed must be a nonnegative integer")
    if value + 10000 * (num_envs - 1) + n_agents - 1 >= 2**32:
        raise ValueError("worker seeds plus agent ranks must be below 2**32")
    return value


def _worker(connection, rank, cfg, seed, env_factory):
    """Import runtime dependencies only in an isolated, CPU-only worker."""
    os.environ.update(_WORKER_ENVIRONMENT)
    env = None
    operation = "init"
    try:
        import numpy as np
        from gymnasium import spaces
        from MAPPO import main
        from MAPPO.mappo.diagnostics import task_status

        # Also bound Torch's intra-op pool if the application imported it before
        # entering this function. Do not construct an agent or initialize CUDA.
        main.torch.set_num_threads(1)
        factory = main._make_env if env_factory is None else env_factory
        env = factory(cfg, seed, render_mode=None)
        obs_dim, act_dim, state_dim, n_agents, action_space = main.get_space_dims(
            env, return_action_space=True
        )
        order = list(env.possible_agents)
        if n_agents != cfg.n_agents:
            raise ValueError("configured agent count does not match environment")
        continuous = isinstance(action_space, spaces.Box)
        episode_rewards = np.zeros(n_agents, dtype=np.float64)
        episode_length = episode_agent_transitions = 0
        episode_terminated = episode_truncated = False
        initial_package_x = None
        environment_steps = 0

        def reset(reset_seed=None):
            nonlocal observations, episode_length, episode_agent_transitions
            nonlocal episode_terminated, episode_truncated, initial_package_x
            observations, _ = env.reset(seed=reset_seed)
            if not observations:
                raise RuntimeError("environment reset exposed no observed agents")
            if hasattr(env, "agents"):
                observed = set(observations)
                reported = set(env.agents)
                if observed != reported:
                    raise ValueError("reset observations and active agents disagree")
            array = main._observations(observations, order, obs_dim)
            episode_rewards.fill(0.0)
            episode_length = episode_agent_transitions = 0
            episode_terminated = episode_truncated = False
            initial_package_x = task_status(env).get("package_x")
            return array

        observations = {}
        initial_obs = reset(seed)
        connection.send(
            {
                "ok": True,
                "operation": "init",
                "rank": rank,
                "payload": {
                    "obs": initial_obs,
                    "possible_agents": order,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "state_dim": state_dim,
                    "n_agents": n_agents,
                    "action_type": "continuous" if continuous else "discrete",
                    "observation_space": env.observation_space(order[0]),
                    "action_space": action_space,
                    "seed": seed,
                },
            }
        )
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break  # Parent deliberately closed the transport.
            operation = request["operation"]
            if operation == "close":
                env.close()
                env = None
                connection.send({"ok": True, "operation": "close", "rank": rank})
                break
            if operation == "reset":
                array = reset(request.get("seed"))
                connection.send(
                    {
                        "ok": True,
                        "operation": operation,
                        "rank": rank,
                        "payload": {"obs": array},
                    }
                )
                continue
            if operation != "step":
                raise ValueError(f"unknown worker operation {operation!r}")
            if not observations:
                raise RuntimeError("cannot step an empty active-agent set")
            active = set(observations)
            alive = np.asarray([name in active for name in order], dtype=np.float32)
            actions = request["actions"]
            result = env.step(main._actions(actions, observations, order, continuous))
            next_observations, rewards, terminations, truncations, infos = result
            episode_length += 1
            environment_steps += 1
            episode_agent_transitions += len(active)
            capped = episode_length >= cfg.max_steps_per_episode
            endpoints, reward_array, terminal, dones, next_active, ended = (
                main._transition(
                    next_observations,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                    active,
                    order,
                    obs_dim,
                    capped,
                )
            )
            if hasattr(env, "agents"):
                reported = set(env.agents)
                if reported - set(order):
                    raise ValueError("environment reported unknown active agents")
                if not reported:
                    ended = True
                    dones[alive > 0.5] = 1.0
                elif set(next_active) != reported:
                    raise ValueError("next observations and active agents disagree")
            episode_rewards += reward_array
            episode_terminated |= any(
                bool(terminations.get(name, False)) for name in active
            )
            episode_truncated |= any(
                bool(truncations.get(name, False)) for name in active
            )
            episode = None
            if ended:
                # Capture diagnostics and returns before reset destroys the native
                # package endpoint. Diagnostics never enter observations or rewards.
                status = dict(task_status(env, initial_package_x=initial_package_x))
                package_x = status.get("package_x")
                displacement = (
                    float(package_x) - float(initial_package_x)
                    if package_x is not None and initial_package_x is not None
                    else None
                )
                if displacement is not None and not math.isfinite(displacement):
                    raise ValueError("package displacement must be finite")
                episode = {
                    **status,
                    "team_return": float(episode_rewards.mean()),
                    "per_agent_returns": episode_rewards.tolist(),
                    "length": episode_length,
                    "agent_transitions": episode_agent_transitions,
                    "agent_ids": order.copy(),
                    "initial_package_x": initial_package_x,
                    "package_displacement": displacement,
                    "termination_reason": status.get(
                        "termination_reason", status.get("reason", "unknown")
                    ),
                    "terminated": bool(episode_terminated),
                    "truncated": bool(episode_truncated),
                    "capped": bool(
                        capped and not (episode_terminated or episode_truncated)
                    ),
                }
                if not np.isfinite(episode_rewards).all():
                    raise ValueError("episode returns must be finite")
                current_obs = reset()
            else:
                observations = next_active
                current_obs = main._observations(observations, order, obs_dim)
            connection.send(
                {
                    "ok": True,
                    "operation": operation,
                    "rank": rank,
                    "payload": {
                        "obs": current_obs,
                        "next_obs": endpoints,
                        "rewards": reward_array,
                        "terminations": terminal,
                        "dones": dones,
                        "alive": alive,
                        "episode": episode,
                        "agent_transitions": len(active),
                        "environment_steps": environment_steps,
                    },
                }
            )
    except BaseException as exc:
        try:
            connection.send(
                {
                    "ok": False,
                    "operation": operation,
                    "rank": rank,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass  # Preserve the original failure already sent to the parent.
        connection.close()


class ParallelMultiwalker:
    """Synchronous fan-out/fan-in of spawn-safe native MultiWalker environments.

    ``obs_array`` has shape [E, N, O]. ``step(actions)`` accepts [E, N, D]
    continuous actions or [E, N] discrete indices and returns E dictionaries in
    rank order. All action rows are validated before any worker is stepped.
    Worker failures close the whole pool; malformed caller actions do not.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        start_seed: Optional[int] = None,
        env_factory=None,
        *,
        start_method: str = "spawn",
        startup_timeout: float = 180.0,
        step_timeout: float = 60.0,
    ):
        self._closed = False
        self._connections = []
        self._processes = []
        self._rpc_lock = threading.RLock()
        self.num_envs = _positive_integer(num_envs, "num_envs")
        self.n_agents = _positive_integer(cfg.n_agents, "n_agents")
        _positive_integer(cfg.max_steps_per_episode, "max_steps_per_episode")
        self.total_timesteps = _positive_integer(cfg.total_timesteps, "total_timesteps")
        if self.total_timesteps % self.num_envs:
            raise ValueError("total_timesteps must be divisible by num_envs")
        if cfg.n_walkers != self.n_agents:
            raise ValueError("n_walkers must match n_agents")
        self.start_seed = _seed(
            cfg.seed if start_seed is None else start_seed, self.num_envs, self.n_agents
        )
        self.seeds = [self.start_seed + 10000 * rank for rank in range(self.num_envs)]
        self.startup_timeout = _timeout(startup_timeout, "startup_timeout")
        self.step_timeout = _timeout(step_timeout, "step_timeout")
        if start_method not in {"spawn", "forkserver"}:
            raise ValueError("start_method must be spawn or forkserver")
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"{start_method} is unavailable on this platform")
        # Fail before starting any workers for unpicklable configs/factories.
        pickle.dumps((cfg, env_factory))
        self.start_method = start_method
        self.joint_timesteps = self.agent_transitions = 0
        context = mp.get_context(start_method)
        try:
            # Spawn imports the application before calling _worker, so the thread
            # and CUDA restrictions must already be present in its environment.
            # Restore the parent's environment immediately after process launch.
            with _START_LOCK:
                previous = {key: os.environ.get(key) for key in _WORKER_ENVIRONMENT}
                try:
                    os.environ.update(_WORKER_ENVIRONMENT)
                    for rank, seed in enumerate(self.seeds):
                        parent, child = context.Pipe()
                        process = context.Process(
                            target=_worker,
                            args=(child, rank, cfg, seed, env_factory),
                            name=f"mappo-env-{rank}",
                            daemon=True,
                        )
                        try:
                            process.start()
                        except BaseException:
                            parent.close()
                            raise
                        finally:
                            child.close()
                        self._connections.append(parent)
                        self._processes.append(process)
                finally:
                    for key, value in previous.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
            metadata = self._receive_many("init", self.startup_timeout)
            first = metadata[0]
            self.possible_agents = list(first["possible_agents"])
            self.obs_dim, self.act_dim, self.state_dim = (
                first["obs_dim"],
                first["act_dim"],
                first["state_dim"],
            )
            self.action_type = first["action_type"]
            self.single_observation_space = first["observation_space"]
            self.single_action_space = first["action_space"]
            for rank, item in enumerate(metadata):
                for key in (
                    "possible_agents",
                    "obs_dim",
                    "act_dim",
                    "state_dim",
                    "n_agents",
                    "action_type",
                    "observation_space",
                    "action_space",
                ):
                    if item[key] != first[key]:
                        raise VectorWorkerError(f"worker {rank} has mismatched {key}")
            self.metadata = {key: value for key, value in first.items() if key != "obs"}
            self._set_observations(metadata)
        except BaseException:
            self.close()
            raise

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("ParallelMultiwalker is closed")

    def _set_observations(self, results):
        import numpy as np

        array = np.stack([item["obs"] for item in results]).astype(
            np.float32, copy=False
        )
        if (
            array.shape != (self.num_envs, self.n_agents, self.obs_dim)
            or not np.isfinite(array).all()
        ):
            raise VectorWorkerError(
                "worker observations have invalid shape or non-finite values"
            )
        self.obs_array = array

    def _receive_many(self, operation, timeout):
        deadline = time.monotonic() + timeout
        pending = {
            connection: rank for rank, connection in enumerate(self._connections)
        }
        results = [None] * self.num_envs
        errors = []
        closing_requested = False
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                errors.append(
                    f"{operation} timed out waiting for workers {sorted(pending.values())}"
                )
                break
            ready = wait(list(pending), timeout=remaining)
            if not ready:
                continue
            for connection in ready:
                rank = pending.pop(connection)
                try:
                    message = connection.recv()
                    if not isinstance(message, dict) or message.get("rank") != rank:
                        raise ValueError("malformed worker response")
                    if message.get("operation") != operation:
                        raise ValueError("worker response operation mismatch")
                    if not message.get("ok"):
                        errors.append(
                            f"worker {rank} failed during {operation}: {message.get('error', 'unknown error')}\n"
                            f"{message.get('traceback', '')}"
                        )
                    else:
                        results[rank] = message["payload"]
                except (EOFError, OSError, ValueError, KeyError) as exc:
                    errors.append(
                        f"worker {rank} {operation} response failed: {type(exc).__name__}: {exc}; "
                        f"exitcode={self._processes[rank].exitcode}"
                    )
            if errors and not closing_requested:
                # Request sibling cleanup immediately. Their current request's
                # response still precedes the close acknowledgment on each pipe.
                for connection in self._connections:
                    try:
                        connection.send({"operation": "close"})
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                closing_requested = True
                # Drain outstanding small replies, but never wait a full step
                # timeout for a sibling stuck in native code after another fails.
                deadline = min(deadline, time.monotonic() + 5.0)
        if errors:
            raise VectorWorkerError("\n".join(errors))
        return results

    def _request_many(self, operation, payloads):
        self._ensure_open()
        try:
            for connection, payload in zip(self._connections, payloads):
                connection.send({"operation": operation, **payload})
            return self._receive_many(operation, self.step_timeout)
        except BaseException:
            self.close()
            raise

    def observation_space(self, agent_id=None):
        if agent_id is not None and agent_id not in self.possible_agents:
            raise KeyError(agent_id)
        return self.single_observation_space

    def action_space(self, agent_id=None):
        if agent_id is not None and agent_id not in self.possible_agents:
            raise KeyError(agent_id)
        return self.single_action_space

    def step(self, actions):
        import numpy as np

        with self._rpc_lock:
            self._ensure_open()
            if self.joint_timesteps + self.num_envs > self.total_timesteps:
                raise ValueError(
                    "step would exceed the configured joint timestep budget"
                )
            array = np.asarray(actions)
            if self.action_type == "continuous":
                expected = (self.num_envs, self.n_agents, self.act_dim)
                if array.shape != expected or not np.isfinite(array).all():
                    raise ValueError(
                        f"continuous actions must be finite with shape {expected}"
                    )
                array = array.astype(np.float32)
                if (
                    not np.isfinite(array).all()
                    or np.any(array < self.single_action_space.low)
                    or np.any(array > self.single_action_space.high)
                ):
                    raise ValueError(
                        "continuous actions are outside the action-space bounds"
                    )
            else:
                expected = (self.num_envs, self.n_agents)
                if array.shape != expected or not np.issubdtype(
                    array.dtype, np.integer
                ):
                    raise ValueError(
                        f"discrete actions must be integer indices with shape {expected}"
                    )
                if np.any(array < 0) or np.any(array >= self.act_dim):
                    raise ValueError(
                        "discrete actions are outside the action-space bounds"
                    )
                array = array.astype(np.int64)
            results = self._request_many(
                "step", [{"actions": row.copy()} for row in array]
            )
            try:
                self._set_observations(results)
                self.joint_timesteps += self.num_envs
                self.agent_transitions += sum(
                    item["agent_transitions"] for item in results
                )
            except BaseException:
                self.close()
                raise
            return results

    def reset(self, seed: Optional[int] = None):
        """Explicit resets are not transitions; normal episode resets are automatic."""
        with self._rpc_lock:
            self._ensure_open()
            if seed is not None:
                seed = _seed(seed, self.num_envs, self.n_agents)
            payloads = [
                {"seed": None if seed is None else seed + 10000 * rank}
                for rank in range(self.num_envs)
            ]
            results = self._request_many("reset", payloads)
            try:
                self._set_observations(results)
            except BaseException:
                self.close()
                raise
            return self.obs_array.copy()

    def close(self):
        """Close only this pool's children; graceful joins have a shared 5s bound."""
        with self._rpc_lock:
            if self._closed:
                return
            self._closed = True
            for connection in self._connections:
                try:
                    connection.send({"operation": "close"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
            deadline = time.monotonic() + 5.0
            for process in self._processes:
                try:
                    process.join(max(0.0, deadline - time.monotonic()))
                except (AssertionError, OSError):
                    pass
            for process in self._processes:
                if process.is_alive():
                    process.terminate()
            deadline = time.monotonic() + 1.0
            for process in self._processes:
                process.join(max(0.0, deadline - time.monotonic()))
            for process in self._processes:
                if process.is_alive():
                    process.kill()
            deadline = time.monotonic() + 1.0
            for process in self._processes:
                process.join(max(0.0, deadline - time.monotonic()))
            for connection in self._connections:
                connection.close()
            for process in self._processes:
                if not process.is_alive():
                    process.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass
