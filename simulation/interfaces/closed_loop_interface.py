"""
simulation/interfaces/closed_loop_interface.py

Abstract closed-loop simulation interface for SE-CVLA.

Defines the contract between SE-CVLA and any driving simulator
(PhysicalAI-AV, nuPlan, CARLA). Concrete wrappers live in
simulation/wrappers/.

The interface follows the standard gymnasium Env API extended with
causal-specific hooks:
  - step_with_causal_feedback()  — returns causal state alongside obs
  - intervene()                  — apply do-calculus interventions in sim
  - get_counterfactual_obs()     — obtain CF observation from sim

Implements the closed-loop evaluation loop:
    obs = env.reset()
    for t in range(horizon):
        action = model.predict(obs)
        obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimObservation:
    """Observation returned at each simulation step."""
    images: np.ndarray              # (num_cameras, H, W, C)  uint8
    ego_state: np.ndarray           # (state_dim,)
    agents: np.ndarray              # (max_agents, feat_dim)
    agent_mask: np.ndarray          # (max_agents,) bool
    timestamp: float = 0.0
    scenario_id: str = ""

    # Optional causal ground truth from simulator
    causal_graph_gt: np.ndarray | None = None   # (d, d) if simulator provides it


@dataclass
class SimInfo:
    """Supplementary info returned from step()."""
    collision: bool = False
    off_route: bool = False
    time_limit_reached: bool = False
    comfort_violation: bool = False
    scenario_type: str = "unknown"
    causal_events: list[str] = field(default_factory=list)  # e.g. ["jaywalker_appeared"]


@dataclass
class EpisodeResult:
    """Final result of a single closed-loop simulation episode."""
    scenario_id: str
    total_steps: int
    collision: bool
    route_completion: float         # fraction [0, 1]
    comfort_score: float
    avg_epistemic_unc: float
    trajectory: np.ndarray          # (T, 2)
    adj_snapshots: list[np.ndarray] # causal graph at each step
    rewards: list[float]


# ──────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ──────────────────────────────────────────────────────────────────────────────

class ClosedLoopSimInterface(abc.ABC):
    """
    Abstract base class for all SE-CVLA simulation backends.

    Subclass this and implement the six abstract methods to plug in
    any simulator.  See simulation/wrappers/ for concrete examples.
    """

    @abc.abstractmethod
    def reset(self, scenario_id: str | None = None) -> SimObservation:
        """Reset the simulation and return the initial observation."""

    @abc.abstractmethod
    def step(
        self,
        ego_trajectory: np.ndarray,   # (H, 2) predicted waypoints
        dt: float = 0.1,
    ) -> tuple[SimObservation, float, bool, SimInfo]:
        """
        Advance the simulation by one timestep.

        Returns:
            obs:    next observation
            reward: scalar reward
            done:   whether episode has ended
            info:   supplementary info
        """

    @abc.abstractmethod
    def get_scenario_ids(self) -> list[str]:
        """Return list of available scenario IDs for evaluation."""

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up simulator resources."""

    # Optional but recommended
    def intervene(self, variable_idx: int, value: Any) -> SimObservation:
        """
        Apply a do-calculus intervention in the simulation.
        Not all backends support this — default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support causal interventions."
        )

    def get_counterfactual_obs(
        self,
        intervention: dict[str, Any],
    ) -> SimObservation:
        """
        Return a counterfactual observation given an intervention description.
        Default: not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support CF observations."
        )

    def render(self) -> np.ndarray | None:
        """Optionally return an RGB frame for visualisation."""
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Closed-loop evaluation runner
# ──────────────────────────────────────────────────────────────────────────────

class ClosedLoopEvaluator:
    """
    Runs SE-CVLA in a closed-loop simulation and collects episode results.

    Args:
        model:          SE-CVLA model instance (in eval mode)
        env:            ClosedLoopSimInterface implementation
        tokenizer:      HuggingFace tokenizer for language inputs
        device:         torch device
        max_steps:      maximum steps per episode
        num_episodes:   number of episodes to evaluate
        render:         whether to render and collect frames
    """

    def __init__(
        self,
        model,
        env: ClosedLoopSimInterface,
        tokenizer,
        device: torch.device,
        max_steps: int = 160,
        num_episodes: int = 100,
        render: bool = False,
    ) -> None:
        self.model = model.eval()
        self.env = env
        self.tokenizer = tokenizer
        self.device = device
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.render = render

    def run(self) -> list[EpisodeResult]:
        """Run all episodes and return results."""
        scenario_ids = self.env.get_scenario_ids()
        results: list[EpisodeResult] = []

        for ep_idx in range(self.num_episodes):
            scenario_id = scenario_ids[ep_idx % len(scenario_ids)]
            result = self._run_episode(scenario_id)
            results.append(result)
            logger.info(
                f"Episode {ep_idx+1}/{self.num_episodes} | {scenario_id} | "
                f"Collision: {result.collision} | "
                f"Route: {result.route_completion:.1%} | "
                f"UNC: {result.avg_epistemic_unc:.4f}"
            )

        self._log_summary(results)
        return results

    def _run_episode(self, scenario_id: str) -> EpisodeResult:
        obs = self.env.reset(scenario_id=scenario_id)
        trajectory_log:    list[np.ndarray]  = []
        adj_log:           list[np.ndarray]  = []
        ep_uncs:           list[float]       = []
        rewards:           list[float]       = []
        collision = False
        route_completion = 0.0

        for step_idx in range(self.max_steps):
            batch = self._obs_to_batch(obs)
            with torch.no_grad():
                out = self.model.predict(batch, num_traj_samples=4)

            # Use best trajectory (lowest uncertainty sample)
            best_traj = out["best_trajectory"][0].cpu().numpy()  # (H, 2)
            ep_uncs.append(out["epistemic_uncertainty"][0].item())
            adj_log.append(out["causal_graph"].numpy())
            trajectory_log.append(best_traj[0])  # record first waypoint

            next_obs, reward, done, info = self.env.step(best_traj)
            rewards.append(reward)

            if info.collision:
                collision = True
            route_completion = getattr(info, "route_completion", step_idx / self.max_steps)

            obs = next_obs
            if done:
                break

        comfort = self._compute_comfort(np.array(trajectory_log))
        return EpisodeResult(
            scenario_id=scenario_id,
            total_steps=len(trajectory_log),
            collision=collision,
            route_completion=route_completion,
            comfort_score=comfort,
            avg_epistemic_unc=float(np.mean(ep_uncs)),
            trajectory=np.array(trajectory_log),
            adj_snapshots=adj_log,
            rewards=rewards,
        )

    def _obs_to_batch(self, obs: SimObservation):
        """Convert a SimObservation to a BatchedFrames for model input."""
        from data.loaders.physicalai_av_dataset import BatchedFrames
        import torchvision.transforms.functional as TF
        from PIL import Image

        # Images
        imgs = []
        for cam_img in obs.images:  # (num_cameras, H, W, C)
            pil = Image.fromarray(cam_img)
            t = TF.to_tensor(TF.resize(pil, [224, 400]))
            t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            imgs.append(t)
        images = torch.stack(imgs).unsqueeze(0).to(self.device)  # (1, C_n, C, H, W)

        # Ego state
        ego = torch.tensor(obs.ego_state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Agents
        agents = torch.tensor(obs.agents, dtype=torch.float32).unsqueeze(0).to(self.device)
        agent_mask = torch.tensor(obs.agent_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        # Dummy language tokens (inference: no ground-truth CoC available)
        dummy_ids   = torch.zeros(1, 512, dtype=torch.long, device=self.device)
        dummy_mask  = torch.zeros(1, 512, dtype=torch.long, device=self.device)
        dummy_traj  = torch.zeros(1, 64, 2, device=self.device)

        return BatchedFrames(
            images=images, ego_state=ego, agents=agents, agent_mask=agent_mask,
            trajectory_gt=dummy_traj, input_ids=dummy_ids, attention_mask=dummy_mask,
        )

    @staticmethod
    def _compute_comfort(traj: np.ndarray) -> float:
        """Jerk-based comfort score. Lower jerk → higher score."""
        if len(traj) < 3:
            return 1.0
        vel   = np.diff(traj, axis=0)
        accel = np.diff(vel,  axis=0)
        jerk  = np.diff(accel, axis=0)
        jerk_norm = np.linalg.norm(jerk, axis=-1).mean()
        return float(1.0 / (1.0 + jerk_norm))

    @staticmethod
    def _log_summary(results: list[EpisodeResult]) -> None:
        n = len(results)
        col_rate = sum(r.collision for r in results) / n
        avg_rc   = sum(r.route_completion for r in results) / n
        avg_unc  = sum(r.avg_epistemic_unc for r in results) / n
        logger.info(
            f"\n[ClosedLoopEval Summary]\n"
            f"  Episodes      : {n}\n"
            f"  Collision Rate: {col_rate:.2%}\n"
            f"  Route Compl.  : {avg_rc:.2%}\n"
            f"  Avg Ep. Unc.  : {avg_unc:.4f}"
        )
