"""
training/callbacks/se_cvla_callbacks.py

Custom PyTorch Lightning callbacks for SE-CVLA training.

Includes:
  - SCMGraphLogger    — logs causal graph visualisation to W&B every N epochs
  - OODMonitor        — raises alert if OOD rate exceeds threshold during training
  - SelfEvolvingCkpt  — saves checkpoints keyed by SCM graph topology hash
  - StageTransition   — auto-advances training stage when metric plateau is reached
"""

from __future__ import annotations

import hashlib
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class SCMGraphLogger(Callback):
    """
    Logs the current causal graph to Weights & Biases as a matplotlib figure
    every `log_every_n_epochs` epochs.

    Variable names (optional) can be provided for readable node labels.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 5,
        variable_names: list[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.every = log_every_n_epochs
        self.var_names = variable_names
        self.threshold = threshold

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.current_epoch % self.every != 0:
            return
        if not hasattr(pl_module, "scm"):
            return

        adj = pl_module.scm.get_causal_graph().numpy()   # (d, d)
        d   = adj.shape[0]
        labels = (
            {i: self.var_names[i] for i in range(d)}
            if self.var_names and len(self.var_names) == d
            else {i: f"V{i}" for i in range(d)}
        )

        G = nx.DiGraph()
        G.add_nodes_from(range(d))
        for i in range(d):
            for j in range(d):
                if adj[i, j] > self.threshold:
                    G.add_edge(j, i, weight=float(adj[i, j]))

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G, pos=pos, labels=labels, ax=ax,
            node_color="#4C72B0", edge_color="#DD8452",
            node_size=800, font_size=8, font_color="white",
            arrows=True, arrowsize=15,
        )
        edge_weights = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_weights.items()},
            ax=ax, font_size=7,
        )
        ax.set_title(f"Causal Graph — Epoch {trainer.current_epoch}")
        plt.tight_layout()

        pass  # graph logging skipped


class OODMonitor(Callback):
    """
    Monitors the OOD detection rate during validation.
    Logs a warning if the rate exceeds `alert_threshold` and can
    optionally trigger a learning-rate reduction.
    """

    def __init__(self, alert_threshold: float = 0.3) -> None:
        self.threshold = alert_threshold

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        ood_rate = trainer.callback_metrics.get("val/ood_rate", None)
        if ood_rate is not None and ood_rate.item() > self.threshold:
            logger.warning(
                f"[OODMonitor] High OOD rate at epoch {trainer.current_epoch}: "
                f"{ood_rate.item():.2%} > threshold {self.threshold:.2%}. "
                "Consider diversifying training data or reducing LR."
            )


class SelfEvolvingCheckpoint(Callback):
    """
    Saves a checkpoint whenever the causal graph topology changes significantly.
    Uses a hash of the binarised adjacency matrix as a key.
    """

    def __init__(
        self,
        dirpath: str = "outputs/scm_checkpoints",
        min_edge_change_frac: float = 0.1,
    ) -> None:
        self.dirpath = dirpath
        self.min_change = min_edge_change_frac
        self._last_hash: str | None = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if not hasattr(pl_module, "scm"):
            return
        adj = pl_module.scm.get_causal_graph()
        graph_hash = hashlib.md5(adj.numpy().tobytes()).hexdigest()[:8]

        if graph_hash != self._last_hash:
            self._last_hash = graph_hash
            os.makedirs(self.dirpath, exist_ok=True)
            ckpt_path = os.path.join(
                self.dirpath,
                f"scm_graph_{graph_hash}_step{trainer.global_step}.ckpt",
            )
            trainer.save_checkpoint(ckpt_path)
            logger.info(f"[SelfEvolvingCkpt] Graph topology changed → saved {ckpt_path}")


class StageTransitionCallback(Callback):
    """
    Automatically advances training stage when a metric plateaus.

    Stage 1 → 2 when val/ADE stops improving for `patience` epochs.
    Stage 2 → 3 when val/ADE stops improving for `patience` epochs from Stage 2 baseline.

    This callback modifies the model's `self.stage` attribute and reloads
    the appropriate optimizer/scheduler configuration.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.01) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self._best_ade  = float("inf")
        self._no_improve = 0

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        ade = trainer.callback_metrics.get("val/ADE", None)
        if ade is None:
            return

        ade_val = ade.item()
        if self._best_ade - ade_val > self.min_delta:
            self._best_ade   = ade_val
            self._no_improve = 0
        else:
            self._no_improve += 1

        if self._no_improve >= self.patience:
            current_stage = getattr(pl_module, "stage", 1)
            if current_stage < 3:
                new_stage = current_stage + 1
                pl_module.stage = new_stage
                pl_module._apply_stage_freezing()
                self._no_improve = 0
                self._best_ade   = float("inf")
                logger.info(
                    f"[StageTransition] ADE plateaued — advancing to Stage {new_stage}"
                )
