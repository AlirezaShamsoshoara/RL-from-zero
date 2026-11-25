from __future__ import annotations

import os
import queue
import time
from collections import deque
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from gymnasium import spaces
from tqdm import tqdm

from A3C.a3c.agent import A3CAgent
from A3C.a3c.config import Config
from A3C.a3c.logging_utils import setup_logger
from A3C.a3c.utils import load_checkpoint, save_checkpoint, set_seed
from A3C.a3c.worker import worker_process


def _log_mean(stats: list[Dict[str, float]]) -> Dict[str, float]:
    keys = set().union(*(item.keys() for item in stats)) if stats else set()
    return {
        key: float(np.mean([item[key] for item in stats if key in item]))
        for key in keys
    }


def _logger(cfg):
    return setup_logger(
        name="a3c",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )


def _agent(cfg, env, update_lock=None):
    if (
        not isinstance(env.observation_space, spaces.Box)
        or len(env.observation_space.shape) != 1
    ):
        raise ValueError("A3C supports only flat Box observations")
    if not isinstance(env.action_space, spaces.Discrete):
        raise ValueError("A3C supports only discrete actions")
    return A3CAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        learning_rate=cfg.learning_rate,
        entropy_coef=cfg.entropy_coef,
        value_loss_coef=cfg.value_loss_coef,
        max_grad_norm=cfg.max_grad_norm,
        device=cfg.device,
        update_lock=update_lock,
    )


def _cleanup_processes(processes, timeout):
    errors = []
    for proc in processes:
        try:
            try:
                proc.join(timeout=timeout)
            except (AssertionError, ValueError):
                continue
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=timeout)
            if proc.is_alive():
                raise RuntimeError("A3C worker could not be stopped")
            close = getattr(proc, "close", None)
            if close is not None:
                close()
        except Exception as error:
            errors.append(error)
            # A cleanup error must not abandon cleanup of sibling workers.
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=timeout)
            except Exception:
                pass
    if errors:
        raise RuntimeError("A3C worker cleanup failed") from errors[0]


def _join_success(processes, timeout):
    failures = []
    for rank, proc in enumerate(processes):
        try:
            proc.join(timeout=timeout)
            if proc.is_alive() or getattr(proc, "exitcode", None) != 0:
                failures.append(
                    f"worker {rank}: alive={proc.is_alive()}, exitcode={getattr(proc, 'exitcode', None)}"
                )
        except Exception as error:
            failures.append(f"worker {rank}: {error}")
    if failures:
        raise RuntimeError(
            "A3C completion requires clean worker exits: " + "; ".join(failures)
        )


def _save_agent(
    agent, path, step, score, cfg, *, completed=False, optimizer_steps=None
):
    # Python optimizer group dictionaries are process-local. The shared scalar
    # records the last successfully applied LR, under the same update lock.
    with agent._lock():
        learning_rate = float(agent.last_learning_rate.item())
        for group in agent.optimizer.param_groups:
            group["lr"] = learning_rate
        metadata = {
            "progress_semantics": "completed_transitions"
            if completed
            else "reserved_transitions",
            "training_completed": completed,
            "last_applied_learning_rate": learning_rate,
            "optimizer_steps": optimizer_steps,
        }
        save_checkpoint(
            path,
            agent.model,
            agent.optimizer,
            step,
            score,
            config=cfg.to_dict(),
            metadata=metadata,
            update_lock=agent.update_lock,
        )


def train(config: str = "A3C/configs/lunarlander.yaml", wandb_key: str = ""):
    cfg = Config.from_yaml(config).validate()
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
    env = result_queue = stop_event = pbar = None
    processes = []
    try:
        env = gym.make(cfg.env_id, **cfg.env_kwargs)
        ctx = mp.get_context("spawn")
        agent = _agent(cfg, env, update_lock=ctx.RLock())
        to_close, env = env, None
        to_close.close()
        global_step = ctx.Value("q", 0)
        result_queue, stop_event = ctx.Queue(), ctx.Event()
        for worker_id in range(cfg.num_workers):
            proc = ctx.Process(
                target=worker_process,
                args=(worker_id, cfg, agent, global_step, result_queue, stop_event),
                daemon=True,
            )
            processes.append(proc)
            proc.start()
        native_returns = deque(maxlen=max(200, cfg.best_window))
        learning_returns = deque(maxlen=max(200, cfg.best_window))
        episode_lengths = deque(maxlen=max(200, cfg.best_window))
        update_stats = []
        best_avg_return, best_path = None, None
        next_log = cfg.log_interval if cfg.log_interval else None
        next_checkpoint = cfg.checkpoint_interval if cfg.checkpoint_interval else None
        latest_step = previous_step = episodes = 0
        summaries = {}
        last_message = time.monotonic()
        pbar = tqdm(total=cfg.total_steps, desc="A3C transitions")
        while len(summaries) < cfg.num_workers:
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                failed = [
                    getattr(p, "exitcode", None)
                    for p in processes
                    if getattr(p, "exitcode", None) not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(f"A3C worker exited abnormally: {failed}")
                if not any(p.is_alive() for p in processes):
                    raise RuntimeError(
                        "A3C workers exited without all completion summaries"
                    )
                if time.monotonic() - last_message > cfg.worker_timeout:
                    raise TimeoutError("A3C workers stopped reporting progress")
                continue
            last_message = time.monotonic()
            if not isinstance(message, dict):
                raise RuntimeError("malformed A3C worker message")
            kind = message.get("kind")
            if kind in ("worker_error", "error"):
                raise RuntimeError(
                    f"A3C worker {message.get('worker_id')} failed: {message.get('error')}\n{message.get('traceback', '')}"
                )
            latest_step = max(latest_step, int(message.get("global_step", latest_step)))
            if latest_step > cfg.total_steps:
                raise RuntimeError("A3C worker exceeded transition budget")
            pbar.update(latest_step - previous_step)
            previous_step = latest_step
            if kind == "episode":
                native_returns.append(float(message["episode_return"]))
                learning_returns.append(
                    float(
                        message.get(
                            "learning_episode_return", message["episode_return"]
                        )
                    )
                )
                episode_lengths.append(int(message["episode_length"]))
                episodes += 1
                if cfg.save_best and len(native_returns) >= cfg.best_window:
                    mean = float(np.mean(list(native_returns)[-cfg.best_window :]))
                    if best_avg_return is None or mean > best_avg_return:
                        best_avg_return = mean
                        best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                        _save_agent(agent, best_path, latest_step, best_avg_return, cfg)
            elif kind == "update":
                update_stats.append(message["stats"])
            elif kind in ("worker_done", "done"):
                rank = int(message["worker_id"])
                if not 0 <= rank < cfg.num_workers or rank in summaries:
                    raise RuntimeError("invalid/duplicate worker completion")
                summaries[rank] = message
            else:
                raise RuntimeError(f"unknown worker message kind: {kind}")
            if next_log is not None and latest_step >= next_log:
                stats = _log_mean(update_stats)
                payload = {
                    "charts/avg_return": float(np.mean(native_returns))
                    if native_returns
                    else 0.0,
                    "charts/avg_learning_return": float(np.mean(learning_returns))
                    if learning_returns
                    else 0.0,
                    "charts/avg_length": float(np.mean(episode_lengths))
                    if episode_lengths
                    else 0.0,
                    "progress/global_step": latest_step,
                    "progress/reserved_transitions": latest_step,
                    "charts/learning_rate": float(agent.last_learning_rate.item()),
                }
                payload.update(
                    {
                        f"{'loss' if 'loss' in key else 'stats'}/{key}": value
                        for key, value in stats.items()
                    }
                )
                wandb.log(payload, step=latest_step)
                update_stats.clear()
                logger.info(
                    "step=%d native_return=%.2f learning_return=%.3f length=%.0f",
                    latest_step,
                    payload["charts/avg_return"],
                    payload["charts/avg_learning_return"],
                    payload["charts/avg_length"],
                )
                next_log = (latest_step // cfg.log_interval + 1) * cfg.log_interval
            if next_checkpoint is not None and latest_step >= next_checkpoint:
                _save_agent(
                    agent,
                    os.path.join(cfg.checkpoint_dir, f"checkpoint_{latest_step}.pt"),
                    latest_step,
                    best_avg_return,
                    cfg,
                )
                next_checkpoint = (
                    latest_step // cfg.checkpoint_interval + 1
                ) * cfg.checkpoint_interval
        _join_success(processes, cfg.shutdown_timeout)
        completed = sum(
            int(summary["collected_transitions"]) for summary in summaries.values()
        )
        optimizer_steps = sum(
            int(summary["optimizer_steps"]) for summary in summaries.values()
        )
        if completed != cfg.total_steps or global_step.value != cfg.total_steps:
            raise RuntimeError(
                f"incomplete A3C transition budget: {completed}/{cfg.total_steps}"
            )
        final_path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{completed}.pt")
        _save_agent(
            agent,
            final_path,
            completed,
            best_avg_return,
            cfg,
            completed=True,
            optimizer_steps=optimizer_steps,
        )
        wandb.log(
            {
                "progress/global_step": completed,
                "progress/completed_transitions": completed,
                "progress/optimizer_steps": optimizer_steps,
                "progress/episodes": episodes,
            },
            step=completed,
        )
        return {
            "global_step": completed,
            "optimizer_steps": optimizer_steps,
            "episodes": episodes,
            "best_avg_return": best_avg_return,
            "best_checkpoint": best_path,
        }
    finally:
        try:
            if stop_event is not None:
                stop_event.set()
            try:
                _cleanup_processes(processes, cfg.shutdown_timeout)
            finally:
                if env is not None:
                    env.close()
                if result_queue is not None:
                    result_queue.close()
                    result_queue.join_thread()
                if pbar is not None:
                    pbar.close()
        finally:
            run.finish()


def demo(
    config: str = "A3C/configs/lunarlander_greedy.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
    render_mode: Optional[str] = None,
    deterministic: bool = True,
    seed: Optional[int] = None,
):
    cfg = Config.from_yaml(config).validate()
    episodes = cfg.episodes if episodes is None else episodes
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    seed = cfg.seed if seed is None else seed
    if render_mode == "none":
        render_mode = None
    elif render_mode is None:
        render_mode = cfg.render_mode
    set_seed(seed)
    logger = _logger(cfg)
    env = gym.make(cfg.env_id, render_mode=render_mode, **cfg.env_kwargs)
    try:
        agent = _agent(cfg, env)
        load_checkpoint(model_path or cfg.inference_model_path, agent.model)
        agent.model.eval()
        returns, lengths, terminated_flags, truncated_flags = [], [], [], []
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total = 0.0
            terminated = truncated = False
            for step in range(cfg.max_steps_per_episode):
                with torch.no_grad():
                    action, _, _, _ = agent.model.act(
                        torch.as_tensor(obs, dtype=torch.float32),
                        deterministic=deterministic,
                    )
                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(step + 1)
            terminated_flags.append(bool(terminated))
            truncated_flags.append(bool(truncated))
            logger.info(
                "Episode %d: native return=%.2f length=%d", episode + 1, total, step + 1
            )
        return {
            "returns": returns,
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "lengths": lengths,
            "terminated": terminated_flags,
            "truncated": truncated_flags,
        }
    finally:
        env.close()


if __name__ == "__main__":
    import fire

    fire.Fire({"train": train, "demo": demo})
