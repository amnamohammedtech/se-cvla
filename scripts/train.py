"""
scripts/train.py

Main training entry point for SE-CVLA.

Usage:
    # Stage 1: pretrain
    python scripts/train.py experiment=stage1_pretrain

    # Stage 2: closed-loop fine-tuning
    python scripts/train.py experiment=stage2_closedloop

    # Stage 3: self-evolving
    python scripts/train.py experiment=stage3_selfevolving

    # Override any config value:
    python scripts/train.py training.batch_size=4 model.scm.num_variables=64

Powered by Hydra for config composition and PyTorch Lightning for training.
"""

from __future__ import annotations

import logging
import os

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from data.loaders.physicalai_av_dataset import build_dataloader
from models.se_cvla import SECVLA
from training.callbacks.se_cvla_callbacks import (
    OODMonitor,
    SCMGraphLogger,
    SelfEvolvingCheckpoint,
    StageTransitionCallback,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Reproducibility ───────────────────────────────────────────────────────
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.encoder.language_backbone,
        trust_remote_code=True,
    )

    # ── Data loaders ─────────────────────────────────────────────────────────
    train_loader = build_dataloader(cfg.data, split="train", tokenizer=tokenizer)
    val_loader   = build_dataloader(cfg.data, split="val",   tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────────
    if hasattr(cfg.training, "resume_from") and cfg.training.resume_from:
        model = SECVLA.from_pretrained(cfg.training.resume_from, cfg)
        logger.info(f"Resumed from {cfg.training.resume_from}")
    else:
        model = SECVLA(cfg)

    if cfg.hardware.compile:
        model = torch.compile(model)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.output_dir, "checkpoints"),
            filename="secvla-{epoch:02d}-{val/ADE:.4f}",
            monitor="val/ADE",
            mode="min",
            save_top_k=cfg.logging.save_top_k,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
        SCMGraphLogger(log_every_n_epochs=5),
        OODMonitor(alert_threshold=0.3),
        StageTransitionCallback(patience=3),
    ]
    if cfg.training.stage == 3:
        callbacks.append(
            SelfEvolvingCheckpoint(
                dirpath=os.path.join(cfg.output_dir, "scm_ckpts"),
            )
        )

    # ── Logger ────────────────────────────────────────────────────────────────
    pl_logger = None
    if cfg.logging.use_wandb:
        pl_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
        gradient_clip_val=cfg.training.gradient_clip,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=pl_logger,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        deterministic=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Training complete. Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
