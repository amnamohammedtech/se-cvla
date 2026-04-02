"""
models/policy/causal_policy.py

Causal Policy Module for SE-CVLA.

Takes the causal representation from DynamicSCMLearner and produces
trajectory predictions via a diffusion-based action head — matching
the Alpamayo 1.5 architecture but conditioned on causal structure.

Two paths:
  - "causal"   : trajectory conditioned on causal repr + causal graph
  - "standard" : ablation baseline (no causal conditioning)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion denoiser."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TrajectoryDenoiser(nn.Module):
    """
    Transformer-based denoiser for diffusion trajectory prediction.

    Denoises a noisy trajectory x_t conditioned on:
      - causal_repr: the causal state from SCM
      - fused_repr:  full multimodal context
      - timestep t:  diffusion step
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.horizon = cfg.policy.horizon
        self.action_dim = cfg.policy.action_dim
        d = cfg.policy.hidden_dim

        # Input projections
        self.traj_proj = nn.Linear(cfg.policy.action_dim, d)
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(d),
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )
        self.context_proj = nn.Linear(cfg.encoder.hidden_dim, d)

        # Graph conditioning: project flattened adjacency to hidden dim
        num_vars = cfg.scm.num_variables
        self.graph_proj = nn.Sequential(
            nn.Linear(num_vars * num_vars, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Transformer denoiser
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=8,
            dim_feedforward=d * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self._denoiser_module = nn.TransformerDecoder(layer, num_layers=cfg.policy.num_layers)

        # Output projection
        self.out_proj = nn.Linear(d, cfg.policy.action_dim)

    def forward(
        self,
        x_t: torch.Tensor,          # (B, horizon, action_dim) — noisy trajectory
        t: torch.Tensor,             # (B,) — diffusion timestep
        causal_repr: torch.Tensor,   # (B, hidden_dim) — from SCM
        context: torch.Tensor,       # (B, N_tokens, hidden_dim) — full context
        adj_soft: torch.Tensor,      # (d, d) — causal graph
        conditioning: str = "causal",
    ) -> torch.Tensor:
        """Returns denoised trajectory: (B, horizon, action_dim)."""
        B = x_t.shape[0]

        # Embed noisy trajectory
        traj_emb = self.traj_proj(x_t)           # (B, H, d)

        # Time embedding (broadcast to all timesteps)
        t_emb = self.time_emb(t).unsqueeze(1)     # (B, 1, d)
        traj_emb = traj_emb + t_emb

        # Build conditioning memory
        ctx = self.context_proj(context)           # (B, N, d)

        if conditioning == "causal":
            # Add causal repr as an extra conditioning token
            causal_tok = causal_repr.unsqueeze(1)  # (B, 1, hidden_dim)
            causal_tok = self.context_proj(causal_tok)  # (B, 1, d)

            # Add graph structure as a conditioning token
            graph_flat = adj_soft.flatten().unsqueeze(0).expand(B, -1)
            graph_tok = self.graph_proj(graph_flat).unsqueeze(1)  # (B, 1, d)

            memory = torch.cat([ctx, causal_tok, graph_tok], dim=1)
        else:
            memory = ctx  # standard (no-causal) ablation

        # Denoise
        out = self._denoiser_module(traj_emb, memory)
        return self.out_proj(out)                  # (B, H, action_dim)




class CausalPolicyModule(nn.Module):
    """
    Causal Policy Module.

    Wraps a DDPM-style diffusion process over trajectory predictions,
    conditioned on causal structure from the DynamicSCMLearner.

    At inference, runs `num_diffusion_steps` denoising iterations to
    produce a set of diverse trajectory samples.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.horizon = cfg.policy.horizon
        self.action_dim = cfg.policy.action_dim
        self.num_steps = cfg.policy.diffusion_steps
        self.cfg_scale = cfg.policy.cfg_scale
        self.conditioning = cfg.policy.conditioning

        self.denoiser = TrajectoryDenoiser(cfg)

        # DDPM noise schedule (cosine)
        betas = self._cosine_beta_schedule(self.num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_alphas_cumprod", alphas_cumprod.sqrt()
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()
        )

    def _cosine_beta_schedule(self, T: int, s: float = 0.008) -> torch.Tensor:
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return betas.clamp(0.0001, 0.9999).float()

    def forward_training(
        self,
        trajectory_gt: torch.Tensor,   # (B, H, 2)  ground-truth
        causal_repr: torch.Tensor,      # (B, D)
        context: torch.Tensor,          # (B, N, D)
        adj_soft: torch.Tensor,         # (d, d)
    ) -> torch.Tensor:
        """
        Training forward pass: add noise at a random timestep and predict
        the noise. Returns MSE loss on predicted noise.
        """
        B, H, _ = trajectory_gt.shape
        device = trajectory_gt.device

        # Random diffusion timesteps
        t = torch.randint(0, self.num_steps, (B,), device=device)

        # Add noise
        noise = torch.randn_like(trajectory_gt)
        sqrt_a = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        x_t = sqrt_a * trajectory_gt + sqrt_1a * noise

        # Predict noise
        noise_pred = self.denoiser(
            x_t, t.float(), causal_repr, context, adj_soft,
            conditioning=self.conditioning,
        )
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        causal_repr: torch.Tensor,
        context: torch.Tensor,
        adj_soft: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Inference: run DDPM reverse process to sample trajectories.

        Returns:
            trajectories: (B, num_samples, horizon, action_dim)
        """
        B = causal_repr.shape[0]
        device = causal_repr.device
        shape = (B * num_samples, self.horizon, self.action_dim)

        # Expand conditioning for multiple samples
        causal_repr_exp = causal_repr.repeat_interleave(num_samples, dim=0)
        context_exp = context.repeat_interleave(num_samples, dim=0)

        x = torch.randn(shape, device=device)

        for step in reversed(range(self.num_steps)):
            t = torch.full((B * num_samples,), step, device=device, dtype=torch.float)
            noise_pred = self.denoiser(
                x, t, causal_repr_exp, context_exp, adj_soft,
                conditioning=self.conditioning,
            )
            # DDPM reverse step
            beta_t = self.betas[step]
            alpha_t = 1.0 - beta_t
            alpha_bar_t = self.alphas_cumprod[step]
            x = (1.0 / alpha_t.sqrt()) * (
                x - (beta_t / (1 - alpha_bar_t).sqrt()) * noise_pred
            )
            if step > 0:
                x = x + beta_t.sqrt() * torch.randn_like(x)

        # Reshape: (B*num_samples, H, 2) → (B, num_samples, H, 2)
        return x.view(B, num_samples, self.horizon, self.action_dim)
