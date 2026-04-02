"""
scripts/closed_loop_eval.py

Experiment 4: Closed-loop simulation evaluation for SE-CVLA.

Runs the full agent-in-the-loop evaluation in PhysicalAI-AV (or nuPlan/CARLA)
and produces collision rate, route completion, comfort score, and causal
stability metrics across a large set of scenarios.

Usage:
    python scripts/closed_loop_eval.py \
        --checkpoint outputs/stage3/best.ckpt \
        --sim physicalai \
        --episodes 500 \
        --output_dir outputs/closed_loop

    # Compare against baselines:
    python scripts/closed_loop_eval.py \
        --checkpoint outputs/stage3/best.ckpt \
        --baselines alpamayo_1.5 \
        --episodes 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from models.se_cvla import SECVLA
from simulation.interfaces.closed_loop_interface import ClosedLoopEvaluator, EpisodeResult
from simulation.wrappers.physicalai_wrapper import PhysicalAIAVWrapper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_sim(sim_name: str, **kwargs):
    """Factory for simulation backends."""
    if sim_name == "physicalai":
        return PhysicalAIAVWrapper(**kwargs)
    elif sim_name == "nuplan":
        from simulation.wrappers.nuplan_wrapper import NuPlanWrapper
        return NuPlanWrapper(**kwargs)
    elif sim_name == "carla":
        from simulation.wrappers.carla_wrapper import CARLAWrapper
        return CARLAWrapper(**kwargs)
    else:
        raise ValueError(f"Unknown simulator: {sim_name!r}")


def aggregate_episode_results(results: list[EpisodeResult]) -> dict:
    n = len(results)
    return {
        "num_episodes":      n,
        "collision_rate":    sum(r.collision for r in results) / n,
        "avg_route_compl":   sum(r.route_completion for r in results) / n,
        "avg_comfort":       sum(r.comfort_score for r in results) / n,
        "avg_epistemic_unc": sum(r.avg_epistemic_unc for r in results) / n,
        "avg_steps":         sum(r.total_steps for r in results) / n,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SE-CVLA Closed-Loop Evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/base.yaml")
    p.add_argument("--sim",        default="physicalai",
                   choices=["physicalai", "nuplan", "carla"])
    p.add_argument("--episodes",   type=int, default=100)
    p.add_argument("--max_steps",  type=int, default=160)
    p.add_argument("--output_dir", default="outputs/closed_loop")
    p.add_argument("--baselines",  nargs="*", default=[],
                   help="Names of baseline checkpoints to compare")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--render",     action="store_true")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    cfg    = OmegaConf.load(args.config)
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.encoder.language_backbone, trust_remote_code=True
    )
    model = SECVLA.from_pretrained(args.checkpoint, cfg)
    model.eval().to(device)

    sim = build_sim(args.sim, split="val")

    evaluator = ClosedLoopEvaluator(
        model=model,
        env=sim,
        tokenizer=tokenizer,
        device=device,
        max_steps=args.max_steps,
        num_episodes=args.episodes,
        render=args.render,
    )

    logger.info(f"Running closed-loop eval: {args.episodes} episodes on {args.sim}")
    results = evaluator.run()
    summary = aggregate_episode_results(results)

    # ── Save results ──────────────────────────────────────────────────────────
    all_summaries = {"SE-CVLA": summary}

    # Optionally evaluate baselines for comparison table
    for baseline_ckpt in args.baselines:
        baseline_name = os.path.basename(baseline_ckpt).replace(".ckpt", "")
        logger.info(f"Evaluating baseline: {baseline_name}")
        try:
            bl_model = SECVLA.from_pretrained(baseline_ckpt, cfg)
            bl_model.eval().to(device)
            bl_eval = ClosedLoopEvaluator(
                model=bl_model, env=sim, tokenizer=tokenizer,
                device=device, max_steps=args.max_steps,
                num_episodes=args.episodes,
            )
            bl_results = bl_eval.run()
            all_summaries[baseline_name] = aggregate_episode_results(bl_results)
        except Exception as e:
            logger.warning(f"Failed to evaluate baseline {baseline_name}: {e}")

    # Print comparison table
    df = pd.DataFrame(all_summaries).T
    print("\n=== Closed-Loop Evaluation Results ===")
    print(df.to_string(float_format="{:.4f}".format))

    # Save
    results_path = os.path.join(args.output_dir, "closed_loop_results.json")
    with open(results_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    df.to_csv(os.path.join(args.output_dir, "closed_loop_table.csv"))
    logger.info(f"Results saved to {args.output_dir}")

    sim.close()


if __name__ == "__main__":
    main()
