"""
data/augmentation/causal_augmentation.py

Causal data augmentation for SE-CVLA training.
Implements the three augmentation strategies from the PhD proposal:
  1. Interventions  (object removal, trajectory changes)
  2. Sensor noise injection
  3. Counterfactual scenario generation
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CausalAugmentor:
    """
    Applies a configurable list of causal augmentations to a DrivingFrame.

    Each augmentation config entry looks like:
        {"type": "sensor_noise", "prob": 0.15, "noise_level": 0.1}

    Args:
        aug_cfgs: list of augmentation config dicts (from Hydra yaml)
    """

    REGISTRY: dict[str, type["BaseAugmentation"]] = {}

    def __init__(self, aug_cfgs: list[dict]) -> None:
        self.augmentations: list[BaseAugmentation] = []
        for cfg in aug_cfgs:
            aug_type = cfg["type"]
            if aug_type not in self.REGISTRY:
                logger.warning(f"Unknown augmentation type: {aug_type!r}, skipping")
                continue
            self.augmentations.append(self.REGISTRY[aug_type](cfg))

    def __call__(self, frame: Any) -> Any:
        """Apply all augmentations in sequence (each with its own probability)."""
        for aug in self.augmentations:
            if random.random() < aug.prob:
                frame = aug(frame)
        return frame

    @classmethod
    def register(cls, name: str):
        def decorator(klass):
            cls.REGISTRY[name] = klass
            return klass
        return decorator


class BaseAugmentation:
    def __init__(self, cfg: dict) -> None:
        self.prob = cfg.get("prob", 0.5)
        self.cfg = cfg

    def __call__(self, frame: Any) -> Any:
        raise NotImplementedError


@CausalAugmentor.register("object_removal")
class ObjectRemovalAugmentation(BaseAugmentation):
    """
    Causal intervention: remove a random agent from the scene.
    This simulates the do(X=0) intervention on an agent variable,
    useful for training counterfactual reasoning.
    """

    def __call__(self, frame: Any) -> Any:
        if frame.agent_mask.sum() == 0:
            return frame  # no agents to remove

        valid_indices = frame.agent_mask.nonzero(as_tuple=True)[0]
        idx = valid_indices[torch.randint(len(valid_indices), (1,))].item()

        # Zero out agent features and mark as invalid
        frame.agents[idx] = 0.0
        frame.agent_mask[idx] = False

        # Append intervention label to language prompt
        frame.intervention_label = f"object_removal:agent_{idx}"
        return frame


@CausalAugmentor.register("trajectory_perturbation")
class TrajectoryPerturbationAugmentation(BaseAugmentation):
    """
    Perturb the ground-truth trajectory with Gaussian noise.
    Simulates counterfactual trajectories for contrastive training.

    Args (from cfg):
        noise_std: standard deviation of positional noise (meters)
    """

    def __call__(self, frame: Any) -> Any:
        noise_std = self.cfg.get("noise_std", 0.5)
        noise = torch.randn_like(frame.trajectory_gt) * noise_std
        frame.trajectory_gt = frame.trajectory_gt + noise
        frame.intervention_label = f"trajectory_perturbation:std={noise_std}"
        return frame


@CausalAugmentor.register("sensor_noise")
class SensorNoiseAugmentation(BaseAugmentation):
    """
    Inject additive Gaussian noise into camera images and ego-state features.
    Simulates sensor degradation and tests epistemic uncertainty estimation.

    Args (from cfg):
        noise_level: noise amplitude as fraction of signal range
    """

    def __call__(self, frame: Any) -> Any:
        noise_level = self.cfg.get("noise_level", 0.1)

        # Image noise (additive Gaussian, clipped to valid range)
        img_noise = torch.randn_like(frame.images) * noise_level
        frame.images = (frame.images + img_noise).clamp(-3.0, 3.0)

        # State noise
        state_noise = torch.randn_like(frame.ego_state) * noise_level * 0.5
        frame.ego_state = frame.ego_state + state_noise

        return frame


@CausalAugmentor.register("counterfactual_scenario")
class CounterfactualScenarioAugmentation(BaseAugmentation):
    """
    Generate counterfactual scenarios by combining multiple causal interventions.
    This produces paired (factual, counterfactual) samples that supervise
    the CounterfactualSimulationEngine.

    Randomly applies 1-3 of the available primitive interventions.
    """

    PRIMITIVES = ["object_removal", "trajectory_perturbation", "sensor_noise"]

    def __call__(self, frame: Any) -> Any:
        n_interventions = random.randint(1, 3)
        chosen = random.sample(self.PRIMITIVES, n_interventions)

        labels = []
        for prim in chosen:
            aug_cls = CausalAugmentor.REGISTRY.get(prim)
            if aug_cls is not None:
                aug = aug_cls({"prob": 1.0})  # always apply in CF mode
                frame = aug(frame)
                labels.append(prim)

        frame.intervention_label = "counterfactual:" + "+".join(labels)
        return frame


@CausalAugmentor.register("weather_change")
class WeatherChangeAugmentation(BaseAugmentation):
    """
    Simulate weather changes through image-space transformations.
    Approximates rain/fog effects without requiring a full renderer.

    Args (from cfg):
        weather: "rain" | "fog" | "night" | "random"
    """

    def __call__(self, frame: Any) -> Any:
        weather = self.cfg.get("weather", "random")
        if weather == "random":
            weather = random.choice(["rain", "fog", "night"])

        if weather == "fog":
            frame.images = self._apply_fog(frame.images)
        elif weather == "rain":
            frame.images = self._apply_rain(frame.images)
        elif weather == "night":
            frame.images = self._apply_night(frame.images)

        frame.intervention_label = f"weather:{weather}"
        return frame

    @staticmethod
    def _apply_fog(images: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """Blend images toward a white/grey fog layer."""
        fog_layer = torch.ones_like(images) * 0.8
        return alpha * fog_layer + (1 - alpha) * images

    @staticmethod
    def _apply_rain(images: torch.Tensor) -> torch.Tensor:
        """Add vertical streak artifacts to simulate rain."""
        B, C, H, W = images.shape
        streaks = torch.zeros_like(images)
        num_streaks = random.randint(20, 80)
        for _ in range(num_streaks):
            x = random.randint(0, W - 1)
            length = random.randint(H // 8, H // 3)
            y_start = random.randint(0, H - length)
            streaks[:, :, y_start:y_start + length, x] = 0.6
        return (images + streaks * 0.3).clamp(-3.0, 3.0)

    @staticmethod
    def _apply_night(images: torch.Tensor) -> torch.Tensor:
        """Darken images and reduce colour saturation to simulate night."""
        return images * 0.25
