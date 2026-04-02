"""
training/losses/se_cvla_loss.py

Composite loss function for SE-CVLA.

Implements the full training objective from the PhD proposal:

    L = L_task  +  λ1·L_causal  +  λ2·L_counterfactual  +  λ3·L_uncertainty

Each component is described and unit-tested separately.
The SECVLALoss class is stateless: it takes pre-computed scalar losses
from each module and combines them with learned or fixed weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SECVLALoss(nn.Module):
    """
    Weighted combination of the four SE-CVLA loss components.

    Args:
        lambda_task:            weight for trajectory prediction loss (L_task)
        lambda_causal:          weight for causal consistency loss (L_causal)
        lambda_counterfactual:  weight for counterfactual accuracy loss (L_cf)
        lambda_uncertainty:     weight for calibration / risk loss (L_unc)
        learnable_weights:      if True, treat lambdas as learnable log-weights
                                (homoscedastic multi-task loss, Kendall et al.)
    """

    def __init__(
        self,
        lambda_task: float = 1.0,
        lambda_causal: float = 0.5,
        lambda_counterfactual: float = 0.3,
        lambda_uncertainty: float = 0.1,
        learnable_weights: bool = False,
    ) -> None:
        super().__init__()
        self.learnable = learnable_weights

        if learnable_weights:
            # Parameterise as log(σ²) — optimised by the main optimizer
            self.log_var_task    = nn.Parameter(torch.zeros(1))
            self.log_var_causal  = nn.Parameter(torch.zeros(1))
            self.log_var_cf      = nn.Parameter(torch.zeros(1))
            self.log_var_unc     = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("lw_task",   torch.tensor(lambda_task))
            self.register_buffer("lw_causal", torch.tensor(lambda_causal))
            self.register_buffer("lw_cf",     torch.tensor(lambda_counterfactual))
            self.register_buffer("lw_unc",    torch.tensor(lambda_uncertainty))

    def forward(
        self,
        task_loss:        torch.Tensor,
        causal_loss:      torch.Tensor,
        cf_loss:          torch.Tensor,
        uncertainty_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute weighted total loss.

        Returns:
            total_loss:  scalar tensor (backpropagatable)
            loss_dict:   dict of individual weighted losses for logging
        """
        if self.learnable:
            # Homoscedastic uncertainty weighting (Kendall & Gal, NeurIPS 2018)
            # L_i = L_i / (2σ_i²) + log σ_i
            total = (
                task_loss        / (2 * self.log_var_task.exp())    + self.log_var_task    * 0.5
                + causal_loss    / (2 * self.log_var_causal.exp())  + self.log_var_causal  * 0.5
                + cf_loss        / (2 * self.log_var_cf.exp())      + self.log_var_cf      * 0.5
                + uncertainty_loss / (2 * self.log_var_unc.exp())   + self.log_var_unc     * 0.5
            )
            w_task    = (1 / (2 * self.log_var_task.exp())).item()
            w_causal  = (1 / (2 * self.log_var_causal.exp())).item()
            w_cf      = (1 / (2 * self.log_var_cf.exp())).item()
            w_unc     = (1 / (2 * self.log_var_unc.exp())).item()
        else:
            total = (
                self.lw_task   * task_loss
                + self.lw_causal * causal_loss
                + self.lw_cf     * cf_loss
                + self.lw_unc    * uncertainty_loss
            )
            w_task, w_causal, w_cf, w_unc = (
                self.lw_task.item(), self.lw_causal.item(),
                self.lw_cf.item(),   self.lw_unc.item(),
            )

        loss_dict = {
            "task_loss":        task_loss.detach(),
            "causal_loss":      causal_loss.detach(),
            "cf_loss":          cf_loss.detach(),
            "uncertainty_loss": uncertainty_loss.detach(),
            "weighted_task":    (w_task   * task_loss).detach(),
            "weighted_causal":  (w_causal * causal_loss).detach(),
            "weighted_cf":      (w_cf     * cf_loss).detach(),
            "weighted_unc":     (w_unc    * uncertainty_loss).detach(),
        }
        return total, loss_dict


# ──────────────────────────────────────────────────────────────────────────────
# Individual task loss helpers (used in ablations / unit tests)
# ──────────────────────────────────────────────────────────────────────────────

def trajectory_huber_loss(
    pred: torch.Tensor,   # (B, H, 2)
    gt:   torch.Tensor,   # (B, H, 2)
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber (smooth-L1) loss on trajectory waypoints."""
    return F.huber_loss(pred, gt, delta=delta)


def causal_consistency_loss(
    adj_soft: torch.Tensor,             # (d, d) current soft adjacency
    adj_prev: torch.Tensor | None,      # (d, d) previous soft adjacency
    acyclicity_loss: torch.Tensor,      # scalar from GraphStructureLearner
    consistency_weight: float = 0.1,
) -> torch.Tensor:
    """
    Combined causal loss:
      - DAGMA acyclicity penalty (drives graph toward DAG)
      - Temporal consistency: discourage graph thrashing between steps
    """
    loss = acyclicity_loss
    if adj_prev is not None:
        # Penalise large changes in graph structure between consecutive batches
        temporal_drift = F.mse_loss(adj_soft, adj_prev.detach())
        loss = loss + consistency_weight * temporal_drift
    return loss


def counterfactual_error(
    cf_pred: torch.Tensor,   # (B, H, 2) — counterfactual prediction
    cf_gt:   torch.Tensor,   # (B, H, 2) — counterfactual ground truth
) -> torch.Tensor:
    """
    Counterfactual prediction error (Experiment 3 primary metric).
    Measures how accurately the model predicts the outcome of interventions.
    """
    return (cf_pred - cf_gt).norm(dim=-1).mean()


def expected_calibration_error(
    confidences: torch.Tensor,   # (B,)  predicted uncertainty (inverted)
    errors:      torch.Tensor,   # (B,)  actual ADE
    n_bins: int = 15,
) -> torch.Tensor:
    """
    Expected Calibration Error (ECE).
    Bins predictions by confidence and measures reliability.
    Lower = better calibrated.
    """
    device = confidences.device
    # Normalise to [0, 1]
    conf_norm  = 1.0 - confidences.clamp(0, 1)
    error_norm = errors / (errors.max() + 1e-8)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)
    ece = torch.tensor(0.0, device=device)
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (conf_norm >= lo) & (conf_norm < hi)
        if mask.sum() == 0:
            continue
        bin_conf  = conf_norm[mask].mean()
        bin_error = error_norm[mask].mean()
        bin_weight = mask.float().mean()
        ece += bin_weight * (bin_conf - bin_error).abs()
    return ece
