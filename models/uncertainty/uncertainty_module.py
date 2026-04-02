"""
models/uncertainty/uncertainty_module.py

Uncertainty Quantification Module for SE-CVLA.

Quantifies two types of uncertainty:
  - Epistemic  (model uncertainty) — reducible with more data
  - Aleatoric  (data uncertainty)  — irreducible, inherent to the task

Methods supported (configurable):
  - "deep_ensemble"  : ensemble of K models (gold standard)
  - "mc_dropout"     : Monte Carlo dropout at inference
  - "evidential"     : evidential deep learning (single forward pass)

Outputs are used for:
  - Risk-aware decision making (defer when epistemic uncertainty is high)
  - Calibration loss (L_uncertainty in the training objective)
  - OOD detection (Experiment 1)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class MCDropoutWrapper(nn.Module):
    """
    MC Dropout wrapper that keeps dropout active at inference.
    Wrap any nn.Module to get dropout-based uncertainty estimates.
    """

    def __init__(self, module: nn.Module, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.module = module
        self.dropout_rate = dropout_rate

    def forward(self, *args, **kwargs):
        # Force train mode for all dropout layers (MC Dropout)
        self._set_dropout_train(self.module)
        return self.module(*args, **kwargs)

    def _set_dropout_train(self, model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()


class EvidentialHead(nn.Module):
    """
    Evidential regression head (Normal-Inverse-Gamma prior).
    Single forward pass → (mu, v, alpha, beta) for epistemic + aleatoric.

    Reference: Amini et al., "Deep Evidential Regression", NeurIPS 2020.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim * 4)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns dict with mu, v, alpha, beta — all (B, out_dim).
        Epistemic uncertainty ∝ beta / (v * (alpha - 1))
        """
        out = self.head(x)                          # (B, out_dim * 4)
        mu, log_v, log_alpha, log_beta = out.chunk(4, dim=-1)
        v     = F.softplus(log_v)   + 1e-5
        alpha = F.softplus(log_alpha) + 1.0 + 1e-5  # α > 1
        beta  = F.softplus(log_beta)  + 1e-5
        return {"mu": mu, "v": v, "alpha": alpha, "beta": beta}

    @staticmethod
    def nig_loss(
        y: torch.Tensor,
        mu: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        lam: float = 1.0,
    ) -> torch.Tensor:
        """Normal-Inverse-Gamma negative log-likelihood + regulariser."""
        two_beta_lam = 2.0 * beta * (1 + v)
        nll = (
            0.5 * (two_beta_lam * math.pi / v).log()
            - alpha * two_beta_lam.log()
            + (alpha + 0.5) * (v * (y - mu) ** 2 + two_beta_lam).log()
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        reg = (y - mu).abs() * (2 * v + alpha)
        return (nll + lam * reg).mean()


import math


class UncertaintyModule(nn.Module):
    """
    Uncertainty Module wrapping trajectory predictions.

    In addition to the trajectory, this module outputs:
      - epistemic_uncertainty: (B,)  — model uncertainty per sample
      - aleatoric_uncertainty: (B,)  — data uncertainty per sample
      - is_ood:                (B,)  — binary OOD flag (epistem. > threshold)

    Args:
        cfg:        model config with uncertainty sub-config
        hidden_dim: dim of the causal repr fed into this module
    """

    def __init__(self, cfg: DictConfig, hidden_dim: int) -> None:
        super().__init__()
        self.cfg = cfg.uncertainty
        self.method = cfg.uncertainty.method
        self.risk_threshold = cfg.uncertainty.risk_threshold
        self.horizon = cfg.policy.horizon

        # Shared uncertainty head (post-trajectory features)
        traj_feat_dim = cfg.policy.horizon * cfg.policy.action_dim

        if self.method == "evidential":
            self.ev_head = EvidentialHead(
                in_dim=hidden_dim + traj_feat_dim,
                out_dim=cfg.policy.horizon * cfg.policy.action_dim,
            )
        else:
            # Epistemic / aleatoric heads for ensemble / MC dropout
            uncertainty_head_dim = hidden_dim + traj_feat_dim
            self.epistemic_head = nn.Sequential(
                nn.Linear(uncertainty_head_dim, 256),
                nn.GELU(),
                nn.Dropout(cfg.uncertainty.dropout_rate),
                nn.Linear(256, 1),
                nn.Softplus(),               # positive uncertainty
            )
            self.aleatoric_head = nn.Sequential(
                nn.Linear(uncertainty_head_dim, 256),
                nn.GELU(),
                nn.Linear(256, cfg.policy.horizon * cfg.policy.action_dim),
                nn.Softplus(),
            )

        # Calibration temperature scaling (learnable post-hoc calibration)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        causal_repr: torch.Tensor,       # (B, D)
        trajectory_pred: torch.Tensor,   # (B, H, 2)
        trajectory_gt: torch.Tensor | None = None,  # (B, H, 2)
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict:
            epistemic_unc: (B,)
            aleatoric_unc: (B,)
            is_ood:        (B,) bool
            uncertainty_loss: scalar (only meaningful if gt is given)
        """
        B = causal_repr.shape[0]
        device = causal_repr.device

        traj_flat = trajectory_pred.flatten(1)        # (B, H*2)
        features = torch.cat([causal_repr, traj_flat], dim=-1)  # (B, D + H*2)

        if self.method == "evidential":
            ev = self.ev_head(features)
            # Epistemic ~ beta / (v * (alpha - 1))
            epistemic = ev["beta"] / (ev["v"] * (ev["alpha"] - 1))
            epistemic = epistemic.mean(dim=-1) / self.temperature   # (B,)
            aleatoric  = (ev["beta"] / (ev["alpha"] - 1))
            aleatoric  = aleatoric.mean(dim=-1)                      # (B,)
        else:
            epistemic = self.epistemic_head(features).squeeze(-1)   # (B,)
            aleatoric_flat = self.aleatoric_head(features)           # (B, H*2)
            aleatoric = aleatoric_flat.mean(dim=-1)                  # (B,)

        is_ood = epistemic > self.risk_threshold

        # ── Uncertainty loss (calibration) ───────────────────────────────────
        uncertainty_loss = torch.tensor(0.0, device=device)
        if trajectory_gt is not None:
            # ECE-inspired calibration: uncertainty should correlate with error
            traj_error = (trajectory_pred - trajectory_gt).norm(dim=-1).mean(dim=-1)
            # L1 loss between predicted uncertainty and actual error
            calibration_loss = F.l1_loss(epistemic, traj_error.detach())
            # Risk-aware penalty: high uncertainty AND low error → penalise
            risk_penalty = (epistemic * (1.0 - traj_error.clamp(0, 1))).mean()
            uncertainty_loss = calibration_loss + 0.1 * risk_penalty

        return {
            "epistemic_uncertainty": epistemic,           # (B,)
            "aleatoric_uncertainty": aleatoric,           # (B,)
            "is_ood": is_ood,                             # (B,) bool
            "uncertainty_loss": uncertainty_loss,
        }

    @torch.no_grad()
    def mc_dropout_sample(
        self,
        causal_repr: torch.Tensor,
        trajectory_samples: torch.Tensor,   # (B, K, H, 2) — K policy samples
    ) -> dict[str, torch.Tensor]:
        """
        Compute MC Dropout uncertainty from multiple trajectory samples.
        This is an alternative to the forward() method for ensemble / MC dropout.

        Returns:
            epistemic:  (B,) — variance across samples
            aleatoric:  (B,) — mean per-sample variance
        """
        # Epistemic = variance of sample means
        traj_mean = trajectory_samples.mean(dim=1)               # (B, H, 2)
        epistemic = trajectory_samples.var(dim=1).mean(dim=(1, 2))  # (B,)

        # Aleatoric = mean of sample variances (estimated from trajectory spread)
        # Simplified: use spatial displacement variance
        aleatoric = (trajectory_samples - traj_mean.unsqueeze(1)).norm(
            dim=-1
        ).mean(dim=(1, 2))  # (B,)

        return {
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "is_ood": epistemic > self.risk_threshold,
        }
