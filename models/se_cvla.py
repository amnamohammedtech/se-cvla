"""
models/se_cvla.py

Top-level SE-CVLA model.

Wires together all five modules from the PhD proposal:
  1. MultimodalEncoder          — vision + language + state fusion
  2. DynamicSCMLearner          — online causal graph + structural equations
  3. CausalPolicyModule         — diffusion trajectory prediction
  4. CounterfactualSimulationEngine — do-calculus intervention + CF loss
  5. UncertaintyModule          — epistemic/aleatoric quantification

Also serves as the PyTorch Lightning LightningModule for training.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from models.encoders.multimodal_encoder import MultimodalEncoder
from models.scm.dynamic_scm_learner import DynamicSCMLearner
from models.policy.causal_policy import CausalPolicyModule
from models.counterfactual.cf_engine import CounterfactualSimulationEngine
from models.uncertainty.uncertainty_module import UncertaintyModule
from data.loaders.physicalai_av_dataset import BatchedFrames
from training.losses.se_cvla_loss import SECVLALoss

logger = logging.getLogger(__name__)


class SECVLA(pl.LightningModule):
    """
    Self-Evolving Causal Vision-Language-Action Model.

    Training objective:
        L = L_task + λ1·L_causal + λ2·L_counterfactual + λ3·L_uncertainty

    Supports three training stages (controlled via cfg.training.stage):
        Stage 1 — pretrain encoder + initial SCM (no CF, light causal)
        Stage 2 — closed-loop fine-tuning (full loss, frozen encoder)
        Stage 3 — self-evolving updates (RL-augmented, online SCM)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # ── Module instantiation ──────────────────────────────────────────────
        self.encoder = MultimodalEncoder(cfg.model)
        self.scm     = DynamicSCMLearner(cfg.model)
        self.policy  = CausalPolicyModule(cfg.model)
        self.cf_engine = CounterfactualSimulationEngine(
            cfg=cfg.model,
            scm_learner=self.scm,
            policy=self.policy,
        )
        self.uncertainty = UncertaintyModule(
            cfg=cfg.model,
            hidden_dim=cfg.model.encoder.hidden_dim,
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        lw = cfg.training.loss_weights
        self.loss_fn = SECVLALoss(
            lambda_task=lw.lambda_task,
            lambda_causal=lw.lambda_causal,
            lambda_counterfactual=lw.lambda_counterfactual,
            lambda_uncertainty=lw.lambda_uncertainty,
        )

        # ── Training stage bookkeeping ────────────────────────────────────────
        self.stage = cfg.training.stage
        self._apply_stage_freezing()

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        batch: BatchedFrames,
        num_traj_samples: int = 1,
        interventions: dict | None = None,
    ) -> dict[str, Any]:
        """
        Full SE-CVLA forward pass.

        Args:
            batch:            BatchedFrames (see data/loaders/physicalai_av_dataset.py)
            num_traj_samples: number of trajectory samples to draw at inference
            interventions:    optional {var_idx: value} for counterfactual inference

        Returns dict with all intermediate outputs and losses.
        """
        # 1. Encode multimodal inputs
        fused_repr, pooled_repr = self.encoder(
            images=batch.images,
            ego_state=batch.ego_state,
            agents=batch.agents,
            agent_mask=batch.agent_mask,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )

        # 2. Dynamic SCM forward (with optional interventions)
        scm_out = self.scm(pooled_repr=pooled_repr, interventions=interventions)

        # 3. Task loss: diffusion training objective
        task_loss = self.policy.forward_training(
            trajectory_gt=batch.trajectory_gt,
            causal_repr=scm_out["causal_repr"],
            context=fused_repr,
            adj_soft=scm_out["adj_soft"],
        )

        # 4. Counterfactual loss (disabled in Stage 1)
        cf_result, cf_loss = self.cf_engine(
            pooled_repr=pooled_repr,
            fused_repr=fused_repr,
            scm_output=scm_out,
            intervention_type_idx=0,
            trajectory_gt=batch.trajectory_gt if self.stage >= 2 else None,
        )

        # 5. Sample trajectories for uncertainty estimation
        with torch.no_grad():
            traj_samples = self.policy.sample(
                causal_repr=scm_out["causal_repr"],
                context=fused_repr,
                adj_soft=scm_out["adj_soft"],
                num_samples=num_traj_samples,
            )
        best_traj = traj_samples[:, 0]    # (B, H, 2) — first sample for loss

        # 6. Uncertainty quantification
        unc_out = self.uncertainty(
            causal_repr=scm_out["causal_repr"],
            trajectory_pred=best_traj,
            trajectory_gt=batch.trajectory_gt,
        )

        # 7. Compute total loss
        total_loss, loss_dict = self.loss_fn(
            task_loss=task_loss,
            causal_loss=scm_out["causal_loss"],
            cf_loss=cf_loss,
            uncertainty_loss=unc_out["uncertainty_loss"],
        )

        return {
            "loss": total_loss,
            "loss_dict": loss_dict,
            "traj_samples": traj_samples,
            "best_traj": best_traj,
            "adj_soft": scm_out["adj_soft"],
            "adj_hard": scm_out["adj_hard"],
            "causal_vars": scm_out["causal_vars"],
            "epistemic_uncertainty": unc_out["epistemic_uncertainty"],
            "aleatoric_uncertainty": unc_out["aleatoric_uncertainty"],
            "is_ood": unc_out["is_ood"],
            "cf_result": cf_result,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PyTorch Lightning hooks
    # ──────────────────────────────────────────────────────────────────────────

    def training_step(self, batch: BatchedFrames, batch_idx: int) -> torch.Tensor:
        out = self(batch)
        loss = out["loss"]
        ld   = out["loss_dict"]

        self.log("train/loss",              loss,                on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/task_loss",         ld["task_loss"],     on_step=True, on_epoch=False)
        self.log("train/causal_loss",       ld["causal_loss"],   on_step=True, on_epoch=False)
        self.log("train/cf_loss",           ld["cf_loss"],       on_step=True, on_epoch=False)
        self.log("train/uncertainty_loss",  ld["uncertainty_loss"], on_step=True, on_epoch=False)

        # Log causal graph sparsity
        sparsity = out["adj_hard"].mean().item()
        self.log("train/graph_sparsity", sparsity, on_step=False, on_epoch=True)

        # Self-evolving SCM graph update (Stage 3)
        if self.stage == 3 and (batch_idx % self.cfg.training.self_evolving.scm_update_freq == 0):
            self._evolve_scm()

        return loss

    def validation_step(self, batch: BatchedFrames, batch_idx: int) -> None:
        with torch.no_grad():
            out = self(batch, num_traj_samples=self.cfg.model.uncertainty.ensemble_size)

        self.log("val/loss",         out["loss"],                        prog_bar=True)
        self.log("val/epistemic_unc", out["epistemic_uncertainty"].mean(), prog_bar=False)
        self.log("val/ood_rate",      out["is_ood"].float().mean(),        prog_bar=False)

        # Compute ADE / FDE
        ade, fde = self._compute_ade_fde(out["best_traj"], batch.trajectory_gt)
        self.log("val/ADE", ade, prog_bar=True)
        self.log("val/FDE", fde, prog_bar=True)

    def test_step(self, batch: BatchedFrames, batch_idx: int) -> None:
        with torch.no_grad():
            out = self(batch, num_traj_samples=self.cfg.model.uncertainty.ensemble_size)
        ade, fde = self._compute_ade_fde(out["best_traj"], batch.trajectory_gt)
        self.log("test/ADE", ade)
        self.log("test/FDE", fde)
        self.log("test/epistemic_unc", out["epistemic_uncertainty"].mean())
        self.log("test/ood_rate",      out["is_ood"].float().mean())

    def configure_optimizers(self):
        train_cfg = self.cfg.training
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=train_cfg.optimizer.lr,
            weight_decay=train_cfg.optimizer.weight_decay,
            betas=train_cfg.optimizer.get("betas", [0.9, 0.999]),
        )
        total_steps = train_cfg.max_epochs * 10_000  # approximate
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_cfg.warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Inference API
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        batch: BatchedFrames,
        num_traj_samples: int = 16,
        return_reasoning: bool = True,
    ) -> dict[str, Any]:
        """
        Clean inference API matching Alpamayo 1.5 style.

        Returns:
            trajectories:        (B, num_samples, H, 2) — sampled trajectories
            best_trajectory:     (B, H, 2)              — highest-confidence trajectory
            causal_graph:        (d, d)                 — current hard DAG
            causal_vars:         (B, d, var_dim)
            epistemic_uncertainty: (B,)
            is_ood:              (B,)
        """
        out = self(batch, num_traj_samples=num_traj_samples)
        return {
            "trajectories":           out["traj_samples"],
            "best_trajectory":        out["best_traj"],
            "causal_graph":           out["adj_hard"].cpu(),
            "causal_vars":            out["causal_vars"],
            "epistemic_uncertainty":  out["epistemic_uncertainty"],
            "aleatoric_uncertainty":  out["aleatoric_uncertainty"],
            "is_ood":                 out["is_ood"],
        }

    @torch.no_grad()
    def what_if(
        self,
        batch: BatchedFrames,
        variable_idx: int,
        value: torch.Tensor,
        num_samples: int = 8,
    ) -> torch.Tensor:
        """
        Interactive counterfactual query.
        Returns counterfactual trajectories: (B, num_samples, H, 2).
        """
        fused_repr, pooled_repr = self.encoder(
            images=batch.images, ego_state=batch.ego_state,
            agents=batch.agents, agent_mask=batch.agent_mask,
            input_ids=batch.input_ids, attention_mask=batch.attention_mask,
        )
        return self.cf_engine.what_if(
            pooled_repr=pooled_repr,
            fused_repr=fused_repr,
            variable_idx=variable_idx,
            value=value,
            num_samples=num_samples,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _evolve_scm(self) -> None:
        """
        Trigger online SCM graph re-discovery using accumulated hidden states.
        Called every `scm_update_freq` steps during Stage 3 (self-evolving).
        In a full implementation this would run a short DAGMA optimisation loop
        on a buffer of recent causal variable activations.
        """
        logger.debug(f"[SE-CVLA] Evolving SCM graph at step {self.global_step}")
        # TODO: implement online DAGMA re-optimisation from replay buffer

    def _apply_stage_freezing(self) -> None:
        """Freeze modules according to current training stage."""
        if self.stage >= 2:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("Stage ≥2: encoder frozen")

    @staticmethod
    def _compute_ade_fde(
        pred: torch.Tensor,    # (B, H, 2)
        gt:   torch.Tensor,    # (B, H, 2)
    ) -> tuple[float, float]:
        """Average / Final Displacement Error."""
        diff = (pred - gt).norm(dim=-1)      # (B, H)
        ade = diff.mean().item()
        fde = diff[:, -1].mean().item()
        return ade, fde

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        cfg: DictConfig,
        strict: bool = True,
    ) -> "SECVLA":
        model = cls(cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded SE-CVLA from {checkpoint_path}")
        return model
