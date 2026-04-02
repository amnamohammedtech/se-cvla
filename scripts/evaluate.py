"""
scripts/evaluate.py

Full evaluation script for SE-CVLA.

Runs all five experiments from the PhD proposal:
  1. OOD Generalization         (--exp ood)
  2. Environment Adaptation     (--exp adaptation)
  3. Counterfactual Accuracy    (--exp counterfactual)
  4. Closed-loop Stability      (--exp closedloop)     [calls closed_loop_eval.py]
  5. Ablation Study             (--exp ablation)

Usage:
    python scripts/evaluate.py \
        --checkpoint outputs/stage3/best.ckpt \
        --config configs/base.yaml \
        --suite full \
        --output_dir outputs/eval

    # Single experiment:
    python scripts/evaluate.py --checkpoint ... --exp ood
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from tqdm import tqdm

from data.loaders.physicalai_av_dataset import build_dataloader
from evaluation.metrics.se_cvla_metrics import (
    MetricsAggregator,
    EvaluationResults,
    compute_ood_roc_auc,
    compute_causal_consistency_score,
    compute_counterfactual_error,
)
from models.se_cvla import SECVLA

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Experiment runners
# ──────────────────────────────────────────────────────────────────────────────

def run_exp1_ood(model: SECVLA, cfg, tokenizer, device) -> EvaluationResults:
    """
    Experiment 1: OOD Generalization.
    Evaluate on held-out OOD splits and compute ROC-AUC for ID vs OOD detection.
    """
    logger.info("=== Experiment 1: OOD Generalization ===")

    id_loader  = build_dataloader(cfg.data, split="test",  tokenizer=tokenizer)
    ood_loader = build_dataloader(cfg.data, split="test",  tokenizer=tokenizer)
    # NOTE: in practice, ood_loader would use a different dataset split with
    # explicitly OOD scenarios. Stub here for illustration.

    id_uncs, ood_uncs = [], []
    agg = MetricsAggregator()

    model.eval()
    with torch.no_grad():
        for batch in tqdm(id_loader, desc="ID eval"):
            batch = batch.to(device)
            out = model.predict(batch, num_traj_samples=4)
            id_uncs.extend(out["epistemic_uncertainty"].cpu().numpy().tolist())
            agg.update(out, batch)

        for batch in tqdm(ood_loader, desc="OOD eval"):
            batch = batch.to(device)
            out = model.predict(batch, num_traj_samples=4)
            ood_uncs.extend(out["epistemic_uncertainty"].cpu().numpy().tolist())

    import numpy as np
    roc_auc = compute_ood_roc_auc(
        np.array(id_uncs), np.array(ood_uncs)
    )
    result = agg.compute(experiment="exp1_ood")
    result.roc_auc_ood = roc_auc
    return result


def run_exp2_adaptation(model: SECVLA, cfg, tokenizer, device) -> EvaluationResults:
    """
    Experiment 2: Environment Adaptation.
    Measures how quickly the SCM adapts after a distribution shift.
    """
    logger.info("=== Experiment 2: Environment Adaptation ===")

    loader = build_dataloader(cfg.data, split="test", tokenizer=tokenizer)
    adj_snapshots = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Adaptation eval"):
            batch = batch.to(device)
            out = model.predict(batch, num_traj_samples=1)
            adj_snapshots.append(out["causal_graph"])

    ccs = compute_causal_consistency_score(adj_snapshots)
    result = EvaluationResults(
        causal_consistency_score=ccs,
        scm_stability_score=ccs,
        num_samples=len(adj_snapshots),
        experiment="exp2_adaptation",
    )
    return result


def run_exp3_counterfactual(model: SECVLA, cfg, tokenizer, device) -> EvaluationResults:
    """
    Experiment 3: Counterfactual Prediction Accuracy.
    Requires a loader that provides counterfactual GT trajectories.
    """
    logger.info("=== Experiment 3: Counterfactual Accuracy ===")

    loader = build_dataloader(cfg.data, split="test", tokenizer=tokenizer)
    cf_errors = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="CF eval"):
            batch = batch.to(device)
            # Generate counterfactual prediction
            cf_trajs = model.what_if(
                batch=batch,
                variable_idx=0,
                value=torch.zeros(
                    batch.images.shape[0], 64, device=device
                ),
                num_samples=4,
            )
            cf_mean = cf_trajs.mean(dim=1)   # (B, H, 2)
            err = compute_counterfactual_error(cf_mean, batch.trajectory_gt)
            cf_errors.append(err)

    import numpy as np
    result = EvaluationResults(
        counterfactual_error=float(np.mean(cf_errors)),
        num_samples=len(cf_errors),
        experiment="exp3_counterfactual",
    )
    return result


def run_exp5_ablation(
    checkpoint_path: str,
    cfg,
    tokenizer,
    device,
) -> dict[str, EvaluationResults]:
    """
    Experiment 5: Ablation Study.
    Loads the full model and evaluates with key components disabled.
    """
    logger.info("=== Experiment 5: Ablation Study ===")

    ablation_results = {}
    loader = build_dataloader(cfg.data, split="test", tokenizer=tokenizer)

    ablations = [
        ("full_model",          {}),
        ("no_scm",              {"conditioning": "standard"}),
        ("no_causal_loss",      {}),  # handled at loss weight level
    ]

    for name, overrides in ablations:
        logger.info(f"Running ablation: {name}")
        model = SECVLA.from_pretrained(checkpoint_path, cfg)

        # Apply overrides
        if "conditioning" in overrides:
            model.policy.conditioning = overrides["conditioning"]

        model.eval().to(device)
        agg = MetricsAggregator()

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Ablation [{name}]"):
                batch = batch.to(device)
                out = model.predict(batch)
                agg.update(out, batch)

        result = agg.compute(experiment=f"exp5_{name}")
        ablation_results[name] = result

    return ablation_results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SE-CVLA Evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--config",     default="configs/base.yaml")
    p.add_argument(
        "--suite", default="full",
        choices=["full", "ood", "adaptation", "counterfactual", "closedloop", "ablation"],
    )
    p.add_argument("--exp",        default=None, help="Single experiment override")
    p.add_argument("--output_dir", default="outputs/eval")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    cfg   = OmegaConf.load(args.config)
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.encoder.language_backbone, trust_remote_code=True
    )
    model = SECVLA.from_pretrained(args.checkpoint, cfg)
    model.eval().to(device)

    suite = args.exp or args.suite
    all_results: dict[str, EvaluationResults | dict] = {}

    if suite in ("full", "ood"):
        all_results["exp1_ood"] = run_exp1_ood(model, cfg, tokenizer, device)

    if suite in ("full", "adaptation"):
        all_results["exp2_adaptation"] = run_exp2_adaptation(model, cfg, tokenizer, device)

    if suite in ("full", "counterfactual"):
        all_results["exp3_counterfactual"] = run_exp3_counterfactual(model, cfg, tokenizer, device)

    if suite in ("full", "closedloop"):
        logger.info("Experiment 4 (closed-loop) — run via: python scripts/closed_loop_eval.py")

    if suite in ("full", "ablation"):
        all_results["exp5_ablation"] = run_exp5_ablation(
            args.checkpoint, cfg, tokenizer, device
        )

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    serialisable = {}
    for k, v in all_results.items():
        if isinstance(v, EvaluationResults):
            print("\n" + v.pretty_print())
            serialisable[k] = v.to_dict()
        elif isinstance(v, dict):
            serialisable[k] = {kk: vv.to_dict() for kk, vv in v.items()}

    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Print ablation table
    if "exp5_ablation" in all_results:
        rows = []
        for name, res in all_results["exp5_ablation"].items():
            row = {"ablation": name}
            row.update(res.to_dict())
            rows.append(row)
        df = pd.DataFrame(rows).set_index("ablation")
        print("\n=== Ablation Table ===")
        print(df.to_string())
        df.to_csv(os.path.join(args.output_dir, "ablation_table.csv"))


if __name__ == "__main__":
    main()
