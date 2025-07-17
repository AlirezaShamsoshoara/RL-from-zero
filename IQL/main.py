from __future__ import annotations
import os
from dataclasses import asdict
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm
import wandb
from gym import spaces

from IQL.iql.config import Config
from IQL.iql.utils import (
    set_seed,
    make_env,
    save_checkpoint,
    load_checkpoint,
    evaluate_policy,
)
from IQL.iql.agent import IQLAgent
from IQL.iql.dataset import build_dataset
from IQL.iql.logging_utils import setup_logger


def train(config: str = "IQL/configs/pendulum_random.yaml", wandb_key: str = ""):
    cfg = Config.from_yaml(config)
    if wandb_key:
        cfg.wandb_key = wandb_key

    logger = setup_logger(
        name="iql",
        level=cfg.log_level,
        to_console=cfg.log_to_console,
        to_file=cfg.log_to_file,
        log_file=cfg.log_file,
    )

    set_seed(cfg.seed)

    if getattr(cfg, "wandb_key", ""):
        import wandb as _wandb  # local import keeps optional dependency scoped

        _wandb.login(key=cfg.wandb_key)

    logger.info(f"Initializing wandb run={cfg.run_name}")
    run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.run_name,
        config=cfg.to_dict(),
    )

    eval_env = make_env(cfg.env_id, cfg.seed, render_mode=None, env_kwargs=cfg.env_kwargs)
    obs_space = eval_env.observation_space
    act_space = eval_env.action_space
    if not isinstance(obs_space, spaces.Box) or not isinstance(act_space, spaces.Box):
        raise TypeError("IQL currently supports continuous Box observation and action spaces only")
    if len(obs_space.shape) != 1 or len(act_space.shape) != 1:
        raise ValueError("IQL implementation expects flat observation and action spaces")

    agent = IQLAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        value_lr=cfg.value_lr,
        gamma=cfg.gamma,
        expectile=cfg.expectile,
        temperature=cfg.temperature,
        max_weight=cfg.max_weight,
        tau=cfg.tau,
        device=cfg.device,
    )

    dataset_env_id = cfg.dataset_env_id or cfg.env_id
    dataset, dataset_stats = build_dataset(
        source=cfg.dataset_source,
        env_id=dataset_env_id,
        seed=cfg.seed,
        device=torch.device(cfg.device),
        num_steps=cfg.dataset_steps,
        path=cfg.dataset_path,
        env_kwargs=cfg.dataset_env_kwargs,
        reward_scale=cfg.reward_scale,
        reward_shift=cfg.reward_shift,
        normalize_rewards=cfg.normalize_rewards,
        logger=logger,
    )
    wandb.log(
        {
            "dataset/size": dataset_stats.size,
            "dataset/reward_mean": dataset_stats.reward_mean,
            "dataset/reward_std": dataset_stats.reward_std,
            "dataset/terminal_fraction": dataset_stats.terminal_fraction,
        },
        step=0,
    )

    best_avg_return = -np.inf
    metric_buffer: List[Dict[str, float]] = []
    pbar = tqdm(range(1, cfg.total_updates + 1), desc="IQL Updates")

    for step in pbar:
        batch = dataset.sample(cfg.batch_size)
        stats = asdict(agent.update(batch))
        metric_buffer.append(stats)

        if step % cfg.log_interval == 0 or step == 1:
            mean_metrics = {
                key: float(np.mean([m[key] for m in metric_buffer]))
                for key in metric_buffer[0].keys()
            }
            metric_buffer.clear()
            log_payload = {
                "loss/critic": mean_metrics["critic_loss"],
                "loss/value": mean_metrics["value_loss"],
                "loss/actor": mean_metrics["actor_loss"],
                "stats/mean_advantage": mean_metrics["mean_advantage"],
                "stats/weight_mean": mean_metrics["weight_mean"],
                "stats/weight_max": mean_metrics["weight_max"],
                "progress/update": step,
            }
            wandb.log(log_payload, step=step)
            pbar.set_postfix(
                {
                    "actor": f"{log_payload['loss/actor']:.3f}",
                    "critic": f"{log_payload['loss/critic']:.3f}",
                }
            )

        if cfg.eval_interval > 0 and step % cfg.eval_interval == 0:
            returns = evaluate_policy(agent, eval_env, cfg.eval_episodes, cfg.seed + step)
            avg_return = float(np.mean(returns)) if returns else 0.0
            std_return = float(np.std(returns)) if returns else 0.0
            wandb.log(
                {
                    "eval/avg_return": avg_return,
                    "eval/std_return": std_return,
                    "eval/episodes": cfg.eval_episodes,
                },
                step=step,
            )
            logger.info(
                "Eval step=%d | avg_return=%.2f ± %.2f",
                step,
                avg_return,
                std_return,
            )
            if cfg.save_best and avg_return > best_avg_return:
                best_avg_return = avg_return
                best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                save_checkpoint(best_path, agent, step, best_avg_return)
                logger.info("New best avg return %.2f -> saved %s", best_avg_return, best_path)

        if cfg.checkpoint_interval > 0 and step % cfg.checkpoint_interval == 0:
            path = os.path.join(cfg.checkpoint_dir, f"checkpoint_{step}.pt")
            save_checkpoint(path, agent, step, best_avg_return)
            logger.info("Saved checkpoint: %s", path)

    eval_env.close()
    run.finish()
    logger.info("Training finished. Best avg return: %.2f", best_avg_return)


def demo(
    config: str = "IQL/configs/pendulum_random.yaml",
    model_path: Optional[str] = None,
    episodes: Optional[int] = None,
):
    cfg = Config.from_yaml(config)
    logger = setup_logger(
        name="iql",
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
        render_mode=cfg.render_mode or "human",
        env_kwargs=cfg.env_kwargs,
    )

    obs_space = env.observation_space
    act_space = env.action_space
    if not isinstance(obs_space, spaces.Box) or not isinstance(act_space, spaces.Box):
        raise TypeError("IQL demo requires continuous Box spaces")

    agent = IQLAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        value_lr=cfg.value_lr,
        gamma=cfg.gamma,
        expectile=cfg.expectile,
        temperature=cfg.temperature,
        max_weight=cfg.max_weight,
        tau=cfg.tau,
        device=cfg.device,
    )
    load_checkpoint(model_path, agent)
    agent.actor.eval()

    returns = evaluate_policy(agent, env, episodes, cfg.seed)
    for idx, ret in enumerate(returns, start=1):
        logger.info("Episode %d: return=%.2f", idx, ret)
    if returns:
        logger.info(
            "Average return over %d episodes: %.2f ± %.2f",
            episodes,
            float(np.mean(returns)),
            float(np.std(returns)),
        )
    env.close()


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "train": train,
            "demo": demo,
        }
    )
