from __future__ import annotations

import torch
from torch import nn


class RunningNorm(nn.Module):
    """Checkpointed input statistics; update only between on-policy rollouts."""

    def __init__(self, features: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("var", torch.ones(features))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))
        self.clip = clip

    @torch.no_grad()
    def update(self, samples):
        samples = torch.as_tensor(
            samples, dtype=self.mean.dtype, device=self.mean.device
        ).reshape(-1, len(self.mean))
        if not len(samples):
            return
        if not torch.isfinite(samples).all():
            raise ValueError("normalizer samples must be finite")
        n = len(samples)
        sample_mean = samples.mean(0)
        sample_var = samples.var(0, unbiased=False)
        delta = sample_mean - self.mean
        total = self.count + n
        mean = self.mean + delta * (n / total)
        m2 = (
            self.var * self.count
            + sample_var * n
            + delta.square() * self.count * n / total
        )
        self.mean.copy_(mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, samples):
        return ((samples - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(
            -self.clip, self.clip
        )
