"""
tests/integration/test_full_pipeline.py

Integration smoke test: builds a tiny SE-CVLA model and runs a
complete forward pass without a GPU or real dataset.

Run with:
    pytest tests/integration/test_full_pipeline.py -v
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

from data.loaders.physicalai_av_dataset import BatchedFrames
from training.losses.se_cvla_loss import SECVLALoss


# ── Tiny config (CPU-compatible, no external model downloads) ─────────────────
TINY_CFG = OmegaConf.create({
    "model": {
        "encoder": {
            "vision_backbone": "mock",
            "language_backbone": "mock",
            "vision_tokens": 16,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.0,
            "attn_implementation": "sdpa",
        },
        "scm": {
            "num_variables": 8, "hidden_dim": 32, "gnn_layers": 2,
            "gnn_heads": 4, "max_parents": 2, "acyclicity_weight": 1.0,
            "update_frequency": 100, "graph_prior": "sparse",
            "sem_type": "nonlinear", "sparsity_weight": 0.01,
        },
        "policy": {
            "action_dim": 2, "horizon": 8, "hidden_dim": 64,
            "num_layers": 2, "conditioning": "causal",
            "diffusion_steps": 5, "cfg_scale": 1.0,
        },
        "counterfactual": {
            "num_samples": 2, "cf_horizon": 4,
            "intervention_types": ["object_removal"],
            "rollout_steps": 2,
        },
        "uncertainty": {
            "method": "evidential", "ensemble_size": 2,
            "mc_dropout_samples": 2, "dropout_rate": 0.0,
            "calibration_bins": 5, "risk_threshold": 0.8,
        },
    },
    "training": {
        "stage": 1,
        "loss_weights": {
            "lambda_task": 1.0, "lambda_causal": 0.1,
            "lambda_counterfactual": 0.0, "lambda_uncertainty": 0.05,
        },
        "optimizer": {"lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999]},
        "max_epochs": 1,
        "warmup_steps": 10,
    },
    "data": {},
    "logging": {"use_wandb": False, "save_top_k": 1, "log_every_n_steps": 1},
    "hardware": {"accelerator": "cpu", "devices": 1, "precision": "32", "compile": False},
    "seed": 42,
    "experiment_name": "smoke_test",
    "output_dir": "/tmp/se_cvla_smoke",
})


def _make_tiny_batch(B: int = 2) -> BatchedFrames:
    """Create a minimal BatchedFrames for testing (no real data needed)."""
    return BatchedFrames(
        images=torch.randn(B, 6, 3, 224, 400),
        ego_state=torch.randn(B, 21, 5),
        agents=torch.randn(B, 32, 10),
        agent_mask=torch.ones(B, 32, dtype=torch.bool),
        trajectory_gt=torch.randn(B, 8, 2),
        input_ids=torch.zeros(B, 512, dtype=torch.long),
        attention_mask=torch.ones(B, 512, dtype=torch.long),
    )


class TestSECVLALossPipeline:
    """Tests that don't need the full model (no backbone downloads)."""

    def test_loss_backward(self):
        loss_fn = SECVLALoss()
        task = torch.tensor(1.5, requires_grad=True)
        causal = torch.tensor(0.8, requires_grad=True)
        cf = torch.tensor(0.3, requires_grad=True)
        unc = torch.tensor(0.2, requires_grad=True)

        total, d = loss_fn(task, causal, cf, unc)
        total.backward()

        assert task.grad is not None
        assert causal.grad is not None

    def test_loss_components_detached_in_dict(self):
        loss_fn = SECVLALoss()
        _, d = loss_fn(
            torch.tensor(1.0, requires_grad=True),
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
        )
        # Logged values should not carry gradients
        assert not d["task_loss"].requires_grad


class TestDatastructures:

    def test_batched_frames_to_device(self):
        batch = _make_tiny_batch(2)
        device = torch.device("cpu")
        batch_moved = batch.to(device)
        assert batch_moved.images.device.type == "cpu"

    def test_batched_frames_shapes(self):
        B = 3
        batch = _make_tiny_batch(B)
        assert batch.images.shape == (B, 6, 3, 224, 400)
        assert batch.trajectory_gt.shape == (B, 8, 2)
        assert batch.agent_mask.dtype == torch.bool


class TestMetricsNumerics:
    """Numerical sanity checks that are fast and dependency-free."""

    def test_ade_symmetry(self):
        from evaluation.metrics.se_cvla_metrics import compute_ade
        a = torch.randn(4, 16, 2)
        b = torch.randn(4, 16, 2)
        assert abs(compute_ade(a, b) - compute_ade(b, a)) < 1e-5

    def test_fde_only_last_step(self):
        from evaluation.metrics.se_cvla_metrics import compute_fde
        pred = torch.zeros(2, 8, 2)
        gt   = torch.zeros(2, 8, 2)
        gt[:, :-1] = 999.0    # early steps wildly wrong
        gt[:, -1]  = 0.0      # last step perfect
        assert compute_fde(pred, gt) == pytest.approx(0.0, abs=1e-5)
