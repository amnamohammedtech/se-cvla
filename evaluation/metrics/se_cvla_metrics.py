"""
evaluation/metrics/se_cvla_metrics.py

Evaluation metrics for SE-CVLA.
Implements all metrics from the PhD proposal across four categories:

  Driving Metrics:
    - ADE   (Average Displacement Error)
    - FDE   (Final Displacement Error)
    - Collision Rate

  Causal Metrics:
    - CCS   (Causal Consistency Score)
    - Counterfactual Error
    - SCM Stability Score

  Uncertainty Metrics:
    - ECE   (Expected Calibration Error)
    - Risk-Aware Decision Score

  OOD Metrics:
    - ROC-AUC for ID vs OOD detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Results container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResults:
    # Driving
    ade: float = 0.0
    fde: float = 0.0
    collision_rate: float = 0.0
    route_completion: float = 0.0
    comfort_score: float = 0.0

    # Causal
    causal_consistency_score: float = 0.0
    counterfactual_error: float = 0.0
    scm_stability_score: float = 0.0

    # Uncertainty
    ece: float = 0.0
    risk_aware_score: float = 0.0

    # OOD
    roc_auc_ood: float = 0.0

    # Meta
    num_samples: int = 0
    experiment: str = ""

    def to_dict(self) -> dict[str, float]:
        return {
            "ADE": self.ade,
            "FDE": self.fde,
            "Collision Rate": self.collision_rate,
            "Route Completion": self.route_completion,
            "Comfort Score": self.comfort_score,
            "CCS": self.causal_consistency_score,
            "Counterfactual Error": self.counterfactual_error,
            "SCM Stability": self.scm_stability_score,
            "ECE": self.ece,
            "Risk-Aware Score": self.risk_aware_score,
            "ROC-AUC OOD": self.roc_auc_ood,
        }

    def pretty_print(self) -> str:
        lines = [f"=== SE-CVLA Evaluation Results [{self.experiment}] ==="]
        lines.append(f"  Samples evaluated : {self.num_samples}")
        lines.append("  ── Driving ──────────────────────────────")
        lines.append(f"  ADE               : {self.ade:.4f} m")
        lines.append(f"  FDE               : {self.fde:.4f} m")
        lines.append(f"  Collision Rate    : {self.collision_rate:.2%}")
        lines.append(f"  Route Completion  : {self.route_completion:.2%}")
        lines.append(f"  Comfort Score     : {self.comfort_score:.4f}")
        lines.append("  ── Causal ───────────────────────────────")
        lines.append(f"  CCS               : {self.causal_consistency_score:.4f}")
        lines.append(f"  CF Error          : {self.counterfactual_error:.4f} m")
        lines.append(f"  SCM Stability     : {self.scm_stability_score:.4f}")
        lines.append("  ── Uncertainty ──────────────────────────")
        lines.append(f"  ECE               : {self.ece:.4f}")
        lines.append(f"  Risk-Aware Score  : {self.risk_aware_score:.4f}")
        lines.append("  ── OOD ──────────────────────────────────")
        lines.append(f"  ROC-AUC OOD       : {self.roc_auc_ood:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Individual metric implementations
# ──────────────────────────────────────────────────────────────────────────────

def compute_ade(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Average Displacement Error.
    pred, gt: (B, H, 2)
    """
    return (pred - gt).norm(dim=-1).mean().item()


def compute_fde(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Final Displacement Error (error at last waypoint).
    pred, gt: (B, H, 2)
    """
    return (pred[:, -1] - gt[:, -1]).norm(dim=-1).mean().item()


def compute_min_ade(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    minADE — best of K trajectory samples.
    pred: (B, K, H, 2),  gt: (B, H, 2)
    """
    gt_exp = gt.unsqueeze(1)                            # (B, 1, H, 2)
    ade_k  = (pred - gt_exp).norm(dim=-1).mean(dim=-1)  # (B, K)
    return ade_k.min(dim=-1).values.mean().item()


def compute_collision_rate(
    trajectories: torch.Tensor,      # (B, H, 2) — predicted ego trajectory
    agent_positions: torch.Tensor,   # (B, H, A, 2) — agent positions over horizon
    collision_radius: float = 1.5,   # metres
) -> float:
    """
    Fraction of scenarios where the predicted trajectory collides with any agent.
    Collision defined as distance < collision_radius at any timestep.
    """
    # Expand ego traj: (B, H, 1, 2) vs agents: (B, H, A, 2)
    ego = trajectories.unsqueeze(2)
    dist = (ego - agent_positions).norm(dim=-1)          # (B, H, A)
    collision = (dist < collision_radius).any(dim=(1, 2)) # (B,)
    return collision.float().mean().item()


def compute_causal_consistency_score(
    adj_list: list[torch.Tensor],   # list of (d, d) adjacency matrices
) -> float:
    """
    Causal Consistency Score (CCS).
    Measures the fraction of causal edges that remain stable across a sequence
    of scenario evaluations. Range [0, 1] — higher is more consistent.

    adj_list: sequence of binary adjacency matrices from consecutive predictions.
    """
    if len(adj_list) < 2:
        return 1.0
    agreements = []
    for i in range(1, len(adj_list)):
        match = (adj_list[i - 1] == adj_list[i]).float().mean().item()
        agreements.append(match)
    return float(np.mean(agreements))


def compute_counterfactual_error(
    cf_pred: torch.Tensor,   # (B, H, 2) — model's CF prediction
    cf_gt:   torch.Tensor,   # (B, H, 2) — ground-truth CF trajectory
) -> float:
    """Counterfactual Error: ADE between CF prediction and GT."""
    return (cf_pred - cf_gt).norm(dim=-1).mean().item()


def compute_scm_stability(
    adj_prev: torch.Tensor,
    adj_curr: torch.Tensor,
) -> float:
    """
    SCM Stability Score: fraction of binary edge decisions that remain
    unchanged between two consecutive SCM graph snapshots.
    """
    return (adj_prev == adj_curr).float().mean().item()


def compute_ece(
    confidences: np.ndarray,   # (N,) predicted probability / confidence
    correct:     np.ndarray,   # (N,) binary correctness (1=within threshold)
    n_bins:      int = 15,
) -> float:
    """
    Expected Calibration Error.
    confidences: model's predicted probability of being within error threshold.
    correct: 1 if actual error ≤ threshold, else 0.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = correct[mask].mean()
        bin_size = mask.mean()
        ece += bin_size * abs(bin_conf - bin_acc)
    return float(ece)


def compute_risk_aware_score(
    trajectories: torch.Tensor,          # (B, H, 2)
    agent_positions: torch.Tensor,       # (B, H, A, 2)
    epistemic_unc: torch.Tensor,         # (B,)
    collision_radius: float = 1.5,
    defer_threshold: float = 0.8,
) -> float:
    """
    Risk-Aware Decision Score.
    Penalises the model for:
      - Taking actions when uncertainty is high (unc > threshold) AND collision occurs
      - Being overly conservative (deferring when situation is safe)

    Returns a score in [0, 1] where 1.0 is perfect risk calibration.
    """
    B = trajectories.shape[0]
    ego = trajectories.unsqueeze(2)
    dist = (ego - agent_positions).norm(dim=-1)         # (B, H, A)
    had_collision = (dist < collision_radius).any(dim=(1, 2)).float()  # (B,)

    deferred = (epistemic_unc > defer_threshold).float()               # (B,)

    # Good: deferred AND would have collided (avoided risk)
    good_defer  = (deferred * had_collision).mean().item()
    # Bad: did not defer AND collided (missed risk)
    bad_no_defer = ((1 - deferred) * had_collision).mean().item()
    # Bad: deferred unnecessarily (too conservative)
    bad_defer   = (deferred * (1 - had_collision)).mean().item()

    score = good_defer / (good_defer + bad_no_defer + bad_defer + 1e-8)
    return float(score)


def compute_ood_roc_auc(
    epistemic_unc_id:  np.ndarray,   # uncertainties for in-distribution samples
    epistemic_unc_ood: np.ndarray,   # uncertainties for OOD samples
) -> float:
    """
    ROC-AUC for ID vs OOD detection using epistemic uncertainty as the score.
    OOD samples should have higher epistemic uncertainty.
    AUC > 0.5 means the model can distinguish ID from OOD.
    """
    scores = np.concatenate([epistemic_unc_id, epistemic_unc_ood])
    labels = np.concatenate([
        np.zeros(len(epistemic_unc_id)),
        np.ones(len(epistemic_unc_ood)),
    ])
    return float(roc_auc_score(labels, scores))


# ──────────────────────────────────────────────────────────────────────────────
# Aggregator
# ──────────────────────────────────────────────────────────────────────────────

class MetricsAggregator:
    """
    Accumulates per-batch outputs and computes all final metrics at the end
    of an evaluation run.

    Usage::

        agg = MetricsAggregator()
        for batch in eval_loader:
            with torch.no_grad():
                out = model.predict(batch)
            agg.update(out, batch)
        results = agg.compute()
        print(results.pretty_print())
    """

    def __init__(self) -> None:
        self._preds:   list[torch.Tensor] = []
        self._gts:     list[torch.Tensor] = []
        self._ep_uncs: list[torch.Tensor] = []
        self._adjs:    list[torch.Tensor] = []
        self._is_ood:  list[torch.Tensor] = []

    def update(self, model_output: dict, batch) -> None:
        self._preds.append(model_output["best_trajectory"].cpu())
        self._gts.append(batch.trajectory_gt.cpu())
        self._ep_uncs.append(model_output["epistemic_uncertainty"].cpu())
        self._adjs.append(model_output["causal_graph"].cpu())
        self._is_ood.append(model_output["is_ood"].cpu())

    def compute(self, experiment: str = "") -> EvaluationResults:
        preds   = torch.cat(self._preds,   dim=0)   # (N, H, 2)
        gts     = torch.cat(self._gts,     dim=0)   # (N, H, 2)
        ep_uncs = torch.cat(self._ep_uncs, dim=0)   # (N,)

        ade = compute_ade(preds, gts)
        fde = compute_fde(preds, gts)
        ccs = compute_causal_consistency_score(self._adjs)

        # ECE: treat error < 2m as "correct"
        errors = (preds - gts).norm(dim=-1).mean(dim=-1).numpy()
        correct = (errors < 2.0).astype(float)
        conf = (1.0 - ep_uncs.numpy().clip(0, 1))
        ece = compute_ece(conf, correct)

        return EvaluationResults(
            ade=ade,
            fde=fde,
            collision_rate=0.0,       # requires agent positions; fill in sim eval
            causal_consistency_score=ccs,
            ece=ece,
            num_samples=len(preds),
            experiment=experiment,
        )
