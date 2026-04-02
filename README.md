# SE-CVLA: Self-Evolving Causal Vision-Language-Action Model

**Research for causal autonomous driving on PhysicalAI-AV dataset**

> *"Autonomous systems that continuously learn and refine their own Structural Causal Models (SCMs) in closed-loop interaction will significantly improve robustness, safety, and interpretability."*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SE-CVLA Framework                           │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Multimodal  │    │  Dynamic SCM │    │   Causal Policy      │  │
│  │  Encoder     │───▶│  Learner     │───▶│   Module             │  │
│  │ (vision+lang │    │ (graph + SEM)│    │ (action generation)  │  │
│  │  + state)    │    └──────────────┘    └──────────────────────┘  │
│  └──────────────┘           │                       │              │
│                             ▼                       ▼              │
│                   ┌──────────────────┐   ┌─────────────────────┐  │
│                   │ Counterfactual   │   │  Uncertainty Module  │  │
│                   │ Simulation Engine│   │  (epistemic+aleatoric│  │
│                   └──────────────────┘   └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
se-cvla/
├── configs/                    # Hydra YAML experiment configs
│   ├── base.yaml
│   ├── model/
│   ├── training/
│   └── experiment/
├── data/
│   ├── loaders/                # PhysicalAI-AV dataset loaders
│   └── augmentation/           # Causal data augmentation
├── models/
│   ├── encoders/               # Multimodal encoder (vision+lang+state)
│   ├── scm/                    # Dynamic Structural Causal Model
│   ├── policy/                 # Causal policy module
│   ├── counterfactual/         # Counterfactual simulation engine
│   └── uncertainty/            # Uncertainty quantification
├── training/
│   ├── losses/                 # Task + causal + CF + uncertainty losses
│   └── callbacks/              # Training callbacks and checkpointing
├── evaluation/
│   ├── metrics/                # ADE/FDE, CCS, ECE, OOD metrics
│   └── visualizers/            # Causal graph + trajectory visualization
├── simulation/
│   ├── interfaces/             # Abstract closed-loop sim interface
│   └── wrappers/               # nuPlan / CARLA / PhysAI wrappers
├── scripts/
│   ├── train.py                # Main training entry point
│   ├── evaluate.py             # Evaluation entry point
│   ├── run_experiment.py       # Full experiment runner
│   └── closed_loop_eval.py     # Closed-loop simulation evaluation
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_scm_visualization.ipynb
│   ├── 03_counterfactual_analysis.ipynb
│   └── 04_ood_evaluation.ipynb
└── tests/
    ├── unit/
    └── integration/
```

## Quickstart

### 1. Environment Setup

```bash
# Create environment
uv venv se_cvla_venv
source se_cvla_venv/bin/activate
uv sync --active

# Or with conda
conda env create -f environment.yaml
conda activate se-cvla
```

### 2. Dataset Access

Request access to the gated HuggingFace dataset, then authenticate:

```bash
huggingface-cli login
```

### 3. Training (3-stage pipeline)

```bash
# Stage 1: Pretrain VLA backbone + initial SCM
python scripts/train.py experiment=stage1_pretrain

# Stage 2: Closed-loop simulation fine-tuning
python scripts/train.py experiment=stage2_closedloop

# Stage 3: Self-evolving updates
python scripts/train.py experiment=stage3_selfevolving
```

### 4. Evaluation

```bash
# Full evaluation suite (all 5 experiments from the proposal)
python scripts/evaluate.py --checkpoint outputs/stage3/best.ckpt --suite full

# Closed-loop simulation
python scripts/closed_loop_eval.py --sim physicalai --episodes 100
```

## Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{causal}} + \lambda_2 \mathcal{L}_{\text{counterfactual}} + \lambda_3 \mathcal{L}_{\text{uncertainty}}$$

| Component | Description |
|-----------|-------------|
| `L_task` | Trajectory prediction (ADE/FDE) |
| `L_causal` | SCM consistency + acyclicity regularization |
| `L_counterfactual` | Counterfactual prediction accuracy |
| `L_uncertainty` | Calibration (ECE) + risk-aware penalty |

## Evaluation Experiments

| # | Experiment | Key Metric |
|---|-----------|------------|
| 1 | OOD Generalization | ROC-AUC ID vs OOD |
| 2 | Environment Adaptation | SCM Stability Score |
| 3 | Counterfactual Accuracy | Counterfactual Error |
| 4 | Closed-loop Stability | Collision Rate |
| 5 | Ablation Study | All metrics |

## Citation

```bibtex
@phdthesis{secvla2025,
  title     = {Self-Evolving Causal Vision-Language-Action Models for Autonomous Driving},
  author    = {Amna},
  year      = {2026},
  school    = {MBZUAI},
}
```

## License

Apache License 2.0 — model weights non-commercial (see `MODEL_LICENSE`).
