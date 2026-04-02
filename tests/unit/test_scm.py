"""
tests/unit/test_scm.py
Unit tests for the Dynamic SCM Learner.
"""

import pytest
import torch
from omegaconf import OmegaConf

# ── Minimal config for unit tests ─────────────────────────────────────────────
MINIMAL_CFG = OmegaConf.create({
    "encoder": {"hidden_dim": 128, "num_heads": 4, "num_layers": 2,
                "dropout": 0.1, "vision_backbone": "mock",
                "language_backbone": "mock"},
    "scm": {
        "num_variables": 8, "hidden_dim": 64, "gnn_layers": 2,
        "gnn_heads": 4, "max_parents": 3, "acyclicity_weight": 1.0,
        "update_frequency": 100, "graph_prior": "sparse", "sem_type": "nonlinear",
        "sparsity_weight": 0.01,
    },
    "policy": {
        "action_dim": 2, "horizon": 16, "hidden_dim": 128,
        "num_layers": 2, "conditioning": "causal",
        "diffusion_steps": 10, "cfg_scale": 1.0,
    },
    "counterfactual": {
        "num_samples": 4, "cf_horizon": 8,
        "intervention_types": ["object_removal", "trajectory_perturbation"],
        "rollout_steps": 4,
    },
    "uncertainty": {
        "method": "evidential", "ensemble_size": 3,
        "mc_dropout_samples": 5, "dropout_rate": 0.1,
        "calibration_bins": 10, "risk_threshold": 0.8,
    },
})


@pytest.fixture
def scm():
    from models.scm.dynamic_scm_learner import DynamicSCMLearner
    return DynamicSCMLearner(MINIMAL_CFG)


def test_scm_forward_shape(scm):
    B = 4
    pooled = torch.randn(B, MINIMAL_CFG.encoder.hidden_dim)
    out = scm(pooled)

    assert out["causal_repr"].shape == (B, MINIMAL_CFG.encoder.hidden_dim)
    d = MINIMAL_CFG.scm.num_variables
    assert out["adj_soft"].shape == (d, d)
    assert out["adj_hard"].shape == (d, d)
    assert out["causal_vars"].shape == (B, d, 64)
    assert out["causal_loss"].ndim == 0   # scalar


def test_scm_acyclicity_loss_nonnegative(scm):
    loss = scm.graph_learner.acyclicity_loss()
    assert loss.item() >= 0.0


def test_scm_dag_property_after_training(scm):
    """After optimising acyclicity loss, the hard graph should be a DAG."""
    opt = torch.optim.Adam(scm.parameters(), lr=0.1)
    pooled = torch.randn(2, MINIMAL_CFG.encoder.hidden_dim)

    for _ in range(50):
        opt.zero_grad()
        out = scm(pooled)
        out["causal_loss"].backward()
        opt.step()

    adj_hard = scm.get_causal_graph()
    # Check no self-loops
    assert adj_hard.diagonal().sum().item() == 0.0


def test_scm_with_intervention(scm):
    B = 2
    d = MINIMAL_CFG.scm.num_variables
    pooled = torch.randn(B, MINIMAL_CFG.encoder.hidden_dim)

    # Apply intervention: fix variable 0 to zeros
    val = torch.zeros(B, 64)
    out_no_int = scm(pooled)
    out_with_int = scm(pooled, interventions={0: val})

    # Outputs should differ when intervention is non-trivial
    assert out_no_int["causal_repr"].shape == out_with_int["causal_repr"].shape


def test_scm_stability_score(scm):
    d = MINIMAL_CFG.scm.num_variables
    adj_a = torch.zeros(d, d)
    adj_b = torch.zeros(d, d)
    adj_b[0, 1] = 1.0   # one edge changed

    score = scm.scm_stability_score(adj_a, adj_b)
    expected = 1.0 - (1.0 / (d * d))
    assert abs(score - expected) < 1e-4
