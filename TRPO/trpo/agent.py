from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from .networks import GaussianPolicy, ValueNetwork


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: Optional[torch.Tensor] = None


def _flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([param.data.view(-1) for param in module.parameters()])


def _set_params(module: nn.Module, vector: torch.Tensor):
    pointer = 0
    for param in module.parameters():
        num = param.numel()
        param.data.copy_(vector[pointer : pointer + num].view_as(param))
        pointer += num


def _flat_grad(
    outputs: torch.Tensor,
    inputs: Iterable[torch.nn.Parameter],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        outputs,
        inputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )
    return torch.cat([g.contiguous().view(-1) for g in grads])


def _conjugate_gradient(
    Avp, b: torch.Tensor, iters: int, residual_tol: float = 1e-10
) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(iters):
        Avp_p = Avp(p)
        alpha = rdotr / (torch.dot(p, Avp_p) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x


class TRPOAgent:
    def __init__(
        self,
        obs_space: spaces.Box,
        act_space: spaces.Box,
        hidden_sizes: Iterable[int],
        activation: str,
        max_kl: float,
        cg_iters: int,
        cg_damping: float,
        line_search_coef: float,
        line_search_steps: int,
        vf_lr: float,
        vf_iters: int,
        entropy_coef: float,
        normalize_advantages: bool,
        device: str,
    ):
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("Observation space must be gym.spaces.Box")
        if not isinstance(act_space, spaces.Box):
            raise TypeError("Action space must be gym.spaces.Box")
        if len(obs_space.shape) != 1:
            raise ValueError("Only flat observation spaces are supported")
        if len(act_space.shape) != 1:
            raise ValueError("Only flat action spaces are supported")

        self.device = torch.device(device)
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = int(np.prod(act_space.shape))

        action_low = torch.as_tensor(act_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(act_space.high, dtype=torch.float32)

        self.policy = GaussianPolicy(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)
        self.value_fn = ValueNetwork(
            obs_dim=self.obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=vf_lr)

        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.line_search_coef = line_search_coef
        self.line_search_steps = line_search_steps
        self.vf_iters = vf_iters
        self.entropy_coef = entropy_coef
        self.normalize_advantages = normalize_advantages

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if obs.ndim == 1:
            obs = obs[None, :]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if deterministic:
            mean, _ = self.policy(obs_t)
            action = torch.tanh(mean) * self.policy.action_scale + self.policy.action_bias
            log_prob = self.policy.log_prob(obs_t, action)
        else:
            action, log_prob, _, _ = self.policy.sample(obs_t)
        value = self.value_fn(obs_t)
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_prob = self.policy.log_prob(obs, actions)
        mean, log_std = self.policy(obs)
        dist = torch.distributions.Normal(mean, torch.exp(log_std))
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def update(self, batch: Batch) -> Dict[str, float]:
        obs = batch.obs.to(self.device)
        actions = batch.actions.to(self.device)
        old_log_probs = batch.logprobs.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = batch.advantages.to(self.device)

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with torch.no_grad():
            old_mean, old_log_std = self.policy(obs)

        def loss_and_kl():
            log_prob = self.policy.log_prob(obs, actions)
            ratio = torch.exp(log_prob - old_log_probs)
            mean, log_std = self.policy(obs)
            dist = torch.distributions.Normal(mean, torch.exp(log_std))
            entropy = dist.entropy().sum(dim=-1).mean()
            loss_pi = -(ratio * advantages).mean() - self.entropy_coef * entropy
            kl = self.policy.kl_divergence(obs, old_mean, old_log_std).mean()
            return loss_pi, kl, entropy

        loss_pi, kl, entropy = loss_and_kl()
        loss_grad = _flat_grad(loss_pi, self.policy.parameters(), retain_graph=True)

        def fisher_vector_product(vec: torch.Tensor) -> torch.Tensor:
            kl_value = self.policy.kl_divergence(obs, old_mean, old_log_std).mean()
            grads = torch.autograd.grad(kl_value, self.policy.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([g.contiguous().view(-1) for g in grads])
            grad_vec = torch.dot(flat_grad_kl, vec)
            hvp = torch.autograd.grad(grad_vec, self.policy.parameters())
            flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp])
            return flat_hvp + self.cg_damping * vec

        step_direction = _conjugate_gradient(fisher_vector_product, -loss_grad, self.cg_iters)
        shs = torch.dot(step_direction, fisher_vector_product(step_direction))
        step_scale = torch.sqrt((2.0 * self.max_kl) / (shs + 1e-8))
        full_step = step_direction * step_scale
        prev_params = _flat_params(self.policy)

        def apply_step(step: torch.Tensor):
            new_params = prev_params + step
            _set_params(self.policy, new_params)

        loss_before = loss_pi.item()
        success, kl_val, loss_after, steps_taken = self._line_search(
            apply_step,
            prev_params,
            full_step,
            loss_before,
            obs,
            actions,
            advantages,
            old_log_probs,
            old_mean,
            old_log_std,
        )

        if not success:
            _set_params(self.policy, prev_params)

        value_loss = torch.tensor(0.0, device=self.device)
        for _ in range(self.vf_iters):
            value_pred = self.value_fn(obs)
            value_loss = 0.5 * (value_pred - returns).pow(2).mean()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        return {
            "loss/policy": float(loss_before),
            "loss/value": float(value_loss.item()),
            "stats/kl": float(kl_val),
            "stats/entropy": float(entropy.item()),
            "stats/line_search_steps": float(steps_taken),
            "stats/line_search_success": 1.0 if success else 0.0,
        }

    def _line_search(
        self,
        apply_step,
        prev_params: torch.Tensor,
        full_step: torch.Tensor,
        loss_before: float,
        obs: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_mean: torch.Tensor,
        old_log_std: torch.Tensor,
    ) -> Tuple[bool, float, float, int]:
        step_frac = 1.0
        for attempt in range(1, self.line_search_steps + 1):
            step = full_step * step_frac
            apply_step(step)
            with torch.no_grad():
                log_prob = self.policy.log_prob(obs, actions)
                ratio = torch.exp(log_prob - old_log_probs)
                mean, log_std = self.policy(obs)
                dist = torch.distributions.Normal(mean, torch.exp(log_std))
                entropy = dist.entropy().sum(dim=-1).mean()
                loss_pi = -(ratio * advantages).mean() - self.entropy_coef * entropy
                kl = self.policy.kl_divergence(obs, old_mean, old_log_std).mean()
            loss_val = float(loss_pi.item())
            kl_val = float(kl.item())
            if loss_val <= loss_before and kl_val <= self.max_kl:
                return True, kl_val, loss_val, attempt
            step_frac *= self.line_search_coef
        _set_params(self.policy, prev_params)
        return False, 0.0, loss_before, self.line_search_steps

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value_fn.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]):
        if "policy_state_dict" in state:
            self.policy.load_state_dict(state["policy_state_dict"])
        if "value_state_dict" in state:
            self.value_fn.load_state_dict(state["value_state_dict"])
        if "value_optimizer_state_dict" in state:
            self.value_optimizer.load_state_dict(state["value_optimizer_state_dict"])

    @staticmethod
    def compute_gae(
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_value: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T, N = rewards.shape
        advantages = np.zeros_like(rewards)
        lastgaelam = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            lastgaelam = delta + gamma * gae_lambda * next_non_terminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns
