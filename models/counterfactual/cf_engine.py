"""
models/counterfactual/cf_engine.py

Counterfactual Simulation Engine for SE-CVLA.

Implements the three-step counterfactual inference procedure:
  1. Abduction  — infer exogenous noise ε from observed data
  2. Intervention — fix parent variable(s) via do(X_i = v)
  3. Prediction  — propagate through modified SCM to get Y_{do(X_i=v)}

Used during:
  - Training: L_counterfactual loss (Experiment 3)
  - Inference: what-if trajectory planning
  - Evaluation: counterfactual accuracy benchmarking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualResult:
    """Result of a counterfactual query."""
    factual_traj: torch.Tensor         # (B, horizon, 2) — original prediction
    cf_trajs: torch.Tensor             # (B, num_cf_samples, horizon, 2)
    intervention_var: int              # which causal variable was intervened on
    intervention_value: torch.Tensor   # the value it was set to
    cf_causal_vars: torch.Tensor       # (B, d, var_dim) — post-intervention vars


class CounterfactualSimulationEngine(nn.Module):
    """
    Counterfactual Simulation Engine.

    Given a factual observation and its SCM state, this engine:
      1. Identifies candidate intervention variables (most causally influential)
      2. Applies a do-calculus intervention on the SCM
      3. Re-runs the causal policy to obtain counterfactual trajectories
      4. Computes counterfactual loss against any available ground truth

    Args:
        cfg:          full experiment config
        scm_learner:  DynamicSCMLearner instance (shared reference)
        policy:       CausalPolicyModule instance (shared reference)
    """

    def __init__(
        self,
        cfg: DictConfig,
        scm_learner: nn.Module,
        policy: nn.Module,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.scm = scm_learner
        self.policy = policy
        self.num_cf_samples = cfg.counterfactual.num_samples
        self.cf_horizon = cfg.counterfactual.cf_horizon

        # Intervention encoder: maps intervention type → variable delta
        self.intervention_encoder = nn.Embedding(
            len(cfg.counterfactual.intervention_types),
            cfg.scm.num_variables,
        )
        # Maps intervention embedding → delta values for causal variables
        self.intervention_mlp = nn.Sequential(
            nn.Linear(cfg.scm.num_variables, cfg.scm.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.scm.hidden_dim, cfg.scm.num_variables * 64),
        )

    def forward(
        self,
        pooled_repr: torch.Tensor,       # (B, D) — from MultimodalEncoder
        fused_repr: torch.Tensor,        # (B, N, D)
        scm_output: dict,                # output of DynamicSCMLearner.forward()
        intervention_type_idx: int = 0,  # index into intervention_types list
        trajectory_gt: torch.Tensor | None = None,
    ) -> tuple[CounterfactualResult, torch.Tensor]:
        """
        Perform counterfactual inference and compute CF loss.

        Returns:
            cf_result:  CounterfactualResult dataclass
            cf_loss:    scalar tensor (0 if no gt available)
        """
        B = pooled_repr.shape[0]
        device = pooled_repr.device
        d = self.cfg.scm.num_variables
        var_dim = 64

        # ── Step 1: Factual trajectory ──────────────────────────────────────
        factual_trajs = self.policy.sample(
            causal_repr=scm_output["causal_repr"],
            context=fused_repr,
            adj_soft=scm_output["adj_soft"],
            num_samples=1,
        ).squeeze(1)  # (B, H, 2)

        # ── Step 2: Abduction — identify exogenous noise residuals ──────────
        # Residuals: difference between observed vars and SEM predictions
        # (Simplified: use intervention embedding to generate delta)
        int_idx_t = torch.tensor(
            [intervention_type_idx], device=device
        ).expand(B)
        int_emb = self.intervention_encoder(int_idx_t)         # (B, d)
        delta = self.intervention_mlp(int_emb)                 # (B, d*var_dim)
        delta = delta.view(B, d, var_dim)

        # ── Step 3: Intervention — apply do(X_var = X_var + delta) ──────────
        # Select the variable with the largest expected effect (by delta norm)
        intervention_var = delta.norm(dim=-1).argmax(dim=-1).mode().values.item()

        # Create intervention dict for SCM forward pass
        intervened_var_values = (
            scm_output["causal_vars"][:, intervention_var] + delta[:, intervention_var]
        )
        interventions = {int(intervention_var): intervened_var_values}

        # ── Step 4: Re-run SCM with intervention ────────────────────────────
        cf_scm_output = self.scm(
            pooled_repr=pooled_repr,
            interventions=interventions,
        )

        # ── Step 5: Generate counterfactual trajectories ─────────────────────
        cf_trajs = self.policy.sample(
            causal_repr=cf_scm_output["causal_repr"],
            context=fused_repr,
            adj_soft=cf_scm_output["adj_soft"],
            num_samples=self.num_cf_samples,
        )  # (B, num_cf_samples, H, 2)

        # ── Counterfactual loss ──────────────────────────────────────────────
        cf_loss = torch.tensor(0.0, device=device)
        if trajectory_gt is not None:
            # Mean trajectory of CF samples should differ from factual
            # but remain physically plausible (supervised by GT when available)
            cf_mean = cf_trajs.mean(dim=1)               # (B, H, 2)

            # 1. Realism loss: CF traj should be close to GT (if GT is for CF)
            cf_loss = cf_loss + F.mse_loss(cf_mean[:, :min(self.cf_horizon, trajectory_gt.shape[1])], trajectory_gt[:, :min(self.cf_horizon, trajectory_gt.shape[1])])

            # 2. Diversity loss: CF samples should be diverse
            #    variance across samples should be > epsilon
            cf_var = cf_trajs.var(dim=1).mean()
            diversity_loss = F.relu(0.1 - cf_var)        # penalise collapse
            cf_loss = cf_loss + 0.1 * diversity_loss

            # 3. Causal consistency: CF should differ from factual
            h = min(self.cf_horizon, factual_trajs.shape[1])
            diff = (cf_mean[:, :h] - factual_trajs[:, :h]).norm(dim=-1).mean()
            consistency_loss = F.relu(0.05 - diff)       # penalise trivial identity CF
            cf_loss = cf_loss + 0.05 * consistency_loss

        result = CounterfactualResult(
            factual_traj=factual_trajs,
            cf_trajs=cf_trajs,
            intervention_var=int(intervention_var),
            intervention_value=intervened_var_values,
            cf_causal_vars=cf_scm_output["causal_vars"],
        )
        return result, cf_loss

    @torch.no_grad()
    def what_if(
        self,
        pooled_repr: torch.Tensor,
        fused_repr: torch.Tensor,
        variable_idx: int,
        value: torch.Tensor,
        num_samples: int = 8,
    ) -> torch.Tensor:
        """
        Interactive what-if query: fix variable `variable_idx` to `value`
        and return predicted trajectories.

        Returns:
            trajectories: (B, num_samples, horizon, 2)
        """
        interventions = {variable_idx: value}
        cf_scm = self.scm(pooled_repr=pooled_repr, interventions=interventions)
        _, adj_soft_cf = self.scm.graph_learner()
        return self.policy.sample(
            causal_repr=cf_scm["causal_repr"],
            context=fused_repr,
            adj_soft=adj_soft_cf,
            num_samples=num_samples,
        )
