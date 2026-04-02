"""
models/scm/dynamic_scm_learner.py

Dynamic Structural Causal Model (SCM) Learner — the core novel module of SE-CVLA.

Continuously learns and updates:
  1. A causal graph  G  (DAG adjacency matrix over latent variables)
  2. Structural Equations  f_i  for each variable: X_i = f_i(PA_i(X), ε_i)

Key design decisions:
  - Graph structure is learned via DAGMA-style continuous optimisation,
    with a differentiable acyclicity constraint.
  - Structural equations are parameterised as small MLPs (one per variable).
  - The graph and SEMs are updated every `scm_update_freq` gradient steps
    from the accumulated hidden representations.
  - Supports do-calculus interventions for counterfactual simulation.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Causal Variable Encoder: projects fused repr → latent causal variables
# ──────────────────────────────────────────────────────────────────────────────

class CausalVariableProjector(nn.Module):
    """Projects fused multimodal representation into causal variable space."""

    def __init__(self, hidden_dim: int, num_variables: int, var_dim: int = 64) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.var_dim = var_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_variables * var_dim),
        )
        self.norm = nn.LayerNorm(var_dim)

    def forward(self, pooled_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_repr: (B, hidden_dim)
        Returns:
            causal_vars: (B, num_variables, var_dim)
        """
        x = self.proj(pooled_repr)                   # (B, num_vars * var_dim)
        x = x.view(x.shape[0], self.num_variables, self.var_dim)
        return self.norm(x)


# ──────────────────────────────────────────────────────────────────────────────
# Graph Structure Learner (DAGMA-inspired)
# ──────────────────────────────────────────────────────────────────────────────

class GraphStructureLearner(nn.Module):
    """
    Learns the causal graph adjacency matrix W ∈ R^{d×d}.

    Uses DAGMA's matrix-exponential acyclicity characterisation:
        h(W) = tr(e^{W ⊙ W}) - d  ≥ 0,  with equality ↔ G is a DAG.

    W_{ij} > 0 means X_j → X_i (j is a parent of i).
    """

    def __init__(self, num_variables: int, max_parents: int, sparsity: float = 0.1) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.max_parents = max_parents
        self.sparsity = sparsity

        # Raw (unconstrained) weight matrix — masked to remove self-loops
        self.W_raw = nn.Parameter(
            torch.randn(num_variables, num_variables) * 0.01
        )
        # Register diagonal mask (no self-loops)
        diag_mask = 1.0 - torch.eye(num_variables)
        self.register_buffer("diag_mask", diag_mask)

    @property
    def W(self) -> torch.Tensor:
        """Soft adjacency matrix with self-loops zeroed out."""
        return self.W_raw * self.diag_mask

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            adj_soft:  (d, d)  — soft adjacency (sigmoid of W), used for training
            adj_hard:  (d, d)  — hard binary adjacency (top-k per variable), for inference
        """
        adj_soft = torch.sigmoid(self.W * 5.0)   # sharpen with temperature

        # Hard adjacency: keep at most max_parents strongest edges per node
        topk_vals, topk_idx = torch.topk(adj_soft, k=self.max_parents, dim=1)
        adj_hard = torch.zeros_like(adj_soft)
        adj_hard.scatter_(1, topk_idx, (topk_vals > 0.5).float())

        return adj_soft, adj_hard

    def acyclicity_loss(self) -> torch.Tensor:
        """
        DAGMA acyclicity penalty: h(W) = tr(e^{W⊙W}) - d.
        Should be driven to 0 during training.
        """
        d = self.num_variables
        W2 = self.W ** 2
        # Matrix exponential via torch.linalg.matrix_exp
        expm = torch.linalg.matrix_exp(W2)
        return expm.trace() - d

    def sparsity_loss(self) -> torch.Tensor:
        """L1 sparsity on the adjacency to encourage sparse graphs."""
        return self.W.abs().sum()


# ──────────────────────────────────────────────────────────────────────────────
# Structural Equation Models (one MLP per causal variable)
# ──────────────────────────────────────────────────────────────────────────────

class StructuralEquationModels(nn.Module):
    """
    Parameterises the structural equations X_i = f_i(PA_i(X), ε_i).
    One small MLP per causal variable, operating on parent variable embeddings.
    """

    def __init__(self, num_variables: int, var_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.var_dim = var_dim

        # Shared-weight MLP (could be per-variable; shared saves parameters)
        self.sems = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_variables * var_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, var_dim),
                nn.LayerNorm(var_dim),
            )
            for _ in range(num_variables)
        ])

    def forward(
        self,
        causal_vars: torch.Tensor,    # (B, d, var_dim)
        adj_soft: torch.Tensor,        # (d, d)
    ) -> torch.Tensor:
        """
        Compute updated variable values by passing each variable's parent
        information through its structural equation.

        Returns:
            updated_vars: (B, d, var_dim)
        """
        B, d, var_dim = causal_vars.shape
        outputs = []
        for i in range(d):
            # Parent mask for variable i: adj_soft[i] ∈ [0,1]^d
            parent_weights = adj_soft[i].unsqueeze(0).unsqueeze(-1)  # (1, d, 1)
            # Weighted parent inputs
            parent_input = (causal_vars * parent_weights)            # (B, d, var_dim)
            parent_input = parent_input.flatten(1)                    # (B, d*var_dim)
            out_i = self.sems[i](parent_input)                        # (B, var_dim)
            outputs.append(out_i)
        return torch.stack(outputs, dim=1)  # (B, d, var_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic SCM Learner (top-level module)
# ──────────────────────────────────────────────────────────────────────────────

class DynamicSCMLearner(nn.Module):
    """
    Main Dynamic SCM Learner module.

    Workflow per forward pass:
      1. Project fused repr → causal variables  (CausalVariableProjector)
      2. Learn/update causal graph              (GraphStructureLearner)
      3. Propagate through structural equations (StructuralEquationModels)
      4. Return updated causal state + graph for downstream modules

    The graph structure is updated online every `scm_update_freq` steps
    during self-evolving training (Stage 3).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_vars = cfg.scm.num_variables
        self.var_dim = 64
        self.update_freq = cfg.scm.update_frequency
        self._step = 0

        self.var_projector = CausalVariableProjector(
            hidden_dim=cfg.encoder.hidden_dim,
            num_variables=self.num_vars,
            var_dim=self.var_dim,
        )
        self.graph_learner = GraphStructureLearner(
            num_variables=self.num_vars,
            max_parents=cfg.scm.max_parents,
        )
        self.sems = StructuralEquationModels(
            num_variables=self.num_vars,
            var_dim=self.var_dim,
            hidden_dim=cfg.scm.hidden_dim,
        )

        # Output projection: causal state → policy hidden dim
        self.output_proj = nn.Linear(
            self.num_vars * self.var_dim,
            cfg.encoder.hidden_dim,
        )

    def forward(
        self,
        pooled_repr: torch.Tensor,        # (B, hidden_dim)
        interventions: dict | None = None, # {var_idx: value} for do-calculus
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pooled_repr:    Global context from MultimodalEncoder
            interventions:  Optional dict mapping variable index → fixed value tensor
                            for counterfactual / do-calculus queries.

        Returns dict with:
            causal_repr:    (B, hidden_dim)  — input for CausalPolicyModule
            causal_vars:    (B, d, var_dim)  — latent causal variables
            adj_soft:       (d, d)            — soft adjacency matrix
            adj_hard:       (d, d)            — hard DAG adjacency
            causal_loss:    scalar            — acyclicity + sparsity loss
        """
        # 1. Project to causal variable space
        causal_vars = self.var_projector(pooled_repr)          # (B, d, var_dim)

        # 2. Apply interventions (do-calculus: fix variable values)
        if interventions:
            causal_vars = causal_vars.clone()
            for var_idx, val in interventions.items():
                causal_vars[:, var_idx] = val

        # 3. Get causal graph
        adj_soft, adj_hard = self.graph_learner()

        # 4. Propagate through structural equations
        updated_vars = self.sems(causal_vars, adj_soft)         # (B, d, var_dim)

        # 5. Compute auxiliary losses for training
        acyclicity_loss = self.graph_learner.acyclicity_loss()
        sparsity_loss   = self.graph_learner.sparsity_loss()
        causal_loss = (
            self.cfg.scm.acyclicity_weight * acyclicity_loss
            + self.cfg.scm.get("sparsity_weight", 0.01) * sparsity_loss
        )

        # 6. Project to policy hidden dim
        causal_repr = self.output_proj(
            updated_vars.flatten(1)                              # (B, d*var_dim)
        )  # (B, hidden_dim)

        self._step += 1

        return {
            "causal_repr":  causal_repr,
            "causal_vars":  updated_vars,
            "adj_soft":     adj_soft,
            "adj_hard":     adj_hard,
            "causal_loss":  causal_loss,
        }

    @torch.no_grad()
    def get_causal_graph(self) -> torch.Tensor:
        """Return the current hard DAG adjacency matrix (for visualisation)."""
        _, adj_hard = self.graph_learner()
        return adj_hard.cpu()

    def scm_stability_score(
        self,
        adj_prev: torch.Tensor,
        adj_curr: torch.Tensor,
    ) -> float:
        """
        Compute SCM Stability Score: fraction of edges that remain stable.
        Used as an evaluation metric (Experiment 2).
        """
        agreement = (adj_prev == adj_curr).float()
        return agreement.mean().item()
