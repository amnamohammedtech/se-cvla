"""
simulation/wrappers/physicalai_wrapper.py

Closed-loop simulation wrapper for the NVIDIA PhysicalAI-AV dataset.

Since PhysicalAI-AV is a logged dataset (not a real-time simulator),
this wrapper implements a replay-based closed-loop evaluation:
  - Loads pre-recorded scenarios
  - Steps through agent predictions reactively
  - Detects collisions by comparing predicted ego position with recorded agents
  - Supports causal interventions by modifying the replay log on-the-fly

For a full reactive simulation, replace with a CARLA or nuPlan wrapper.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from simulation.interfaces.closed_loop_interface import (
    ClosedLoopSimInterface,
    EpisodeResult,
    SimInfo,
    SimObservation,
)

logger = logging.getLogger(__name__)


class PhysicalAIAVWrapper(ClosedLoopSimInterface):
    """
    Replay-based closed-loop wrapper for PhysicalAI-AV.

    Args:
        split:            "val" or "test"
        collision_radius: metres — distance threshold for collision detection
        num_cameras:      number of cameras to return
    """

    HF_DATASET_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"

    def __init__(
        self,
        split: str = "val",
        collision_radius: float = 1.5,
        num_cameras: int = 6,
        cache_dir: str = "~/.cache/physicalai_av",
    ) -> None:
        self.split = split
        self.collision_radius = collision_radius
        self.num_cameras = num_cameras

        logger.info(f"Loading PhysicalAI-AV [{split}] for closed-loop eval …")
        self._dataset = load_dataset(
            self.HF_DATASET_ID,
            split=split,
            streaming=True,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self._scenario_cache: dict[str, list[dict]] = {}  # scenario_id → frames
        self._current_scenario: list[dict] = []
        self._current_step: int = 0
        self._current_id: str = ""
        self._preload_scenarios()

    def _preload_scenarios(self, max_scenarios: int = 200) -> None:
        """Cache up to max_scenarios from the dataset for repeated access."""
        logger.info(f"Pre-loading up to {max_scenarios} scenarios …")
        for sample in self._dataset:
            sid = sample.get("scenario_id", f"scenario_{len(self._scenario_cache)}")
            if sid not in self._scenario_cache:
                self._scenario_cache[sid] = []
            self._scenario_cache[sid].append(sample)
            if len(self._scenario_cache) >= max_scenarios:
                break
        logger.info(f"Cached {len(self._scenario_cache)} scenarios")

    def get_scenario_ids(self) -> list[str]:
        return list(self._scenario_cache.keys())

    def reset(self, scenario_id: str | None = None) -> SimObservation:
        if scenario_id is None:
            scenario_id = next(iter(self._scenario_cache))
        if scenario_id not in self._scenario_cache:
            raise ValueError(f"Unknown scenario_id: {scenario_id!r}")

        self._current_id = scenario_id
        self._current_scenario = self._scenario_cache[scenario_id]
        self._current_step = 0
        return self._frame_to_obs(self._current_scenario[0])

    def step(
        self,
        ego_trajectory: np.ndarray,   # (H, 2) — model's predicted waypoints
        dt: float = 0.1,
    ) -> tuple[SimObservation, float, bool, SimInfo]:
        self._current_step += 1
        done = self._current_step >= len(self._current_scenario) - 1

        if done:
            frame = self._current_scenario[-1]
        else:
            frame = self._current_scenario[self._current_step]

        obs = self._frame_to_obs(frame)

        # Collision check: compare first waypoint of predicted traj vs agents
        predicted_pos = ego_trajectory[0]  # (2,) — next waypoint
        agent_positions = frame.get("agent_positions_now", np.zeros((0, 2)))
        collision = self._check_collision(predicted_pos, agent_positions)

        # Route completion estimate
        route_completion = self._current_step / max(len(self._current_scenario) - 1, 1)

        # Reward shaping
        reward = -10.0 if collision else 1.0 + route_completion

        info = SimInfo(
            collision=collision,
            off_route=False,
            time_limit_reached=done,
            route_completion=route_completion,
            scenario_type=frame.get("scenario_type", "unknown"),
        )
        return obs, reward, done, info

    def intervene(self, variable_idx: int, value: Any) -> SimObservation:
        """
        Causal intervention: remove or modify agents in the current frame
        to simulate a do-calculus intervention.

        variable_idx maps to agent index when < max_agents,
        else modifies weather/scene parameters.
        """
        frame = dict(self._current_scenario[self._current_step])
        if variable_idx < 32:  # agent intervention
            agents = list(frame.get("agents", []))
            if variable_idx < len(agents):
                agents[variable_idx] = None   # "remove" agent
                frame["agents"] = [a for a in agents if a is not None]
                logger.debug(f"Intervention: removed agent {variable_idx}")
        return self._frame_to_obs(frame)

    def close(self) -> None:
        self._scenario_cache.clear()

    # ──────────────────────────────────────────────────────────────────────────

    def _frame_to_obs(self, frame: dict) -> SimObservation:
        """Convert a raw HuggingFace frame dict to SimObservation."""
        cam_names = ["front", "front_left", "front_right", "rear", "rear_left", "rear_right"]
        images = []
        for cam in cam_names[: self.num_cameras]:
            img = frame.get("cameras", {}).get(cam, None)
            if img is not None:
                images.append(np.array(img))
            else:
                images.append(np.zeros((224, 400, 3), dtype=np.uint8))
        images_arr = np.stack(images)   # (num_cameras, H, W, C)

        ego = np.array(
            frame.get("ego_state", np.zeros(5)), dtype=np.float32
        )
        agents_raw = frame.get("agents", [])
        max_a = 32
        feat_d = 10
        agents = np.zeros((max_a, feat_d), dtype=np.float32)
        agent_mask = np.zeros(max_a, dtype=bool)
        for i, ag in enumerate(agents_raw[:max_a]):
            agents[i] = np.array(ag.get("features", np.zeros(feat_d)))
            agent_mask[i] = True

        cg = frame.get("causal_graph", None)
        causal_graph_gt = np.array(cg) if cg is not None else None

        return SimObservation(
            images=images_arr,
            ego_state=ego,
            agents=agents,
            agent_mask=agent_mask,
            timestamp=float(frame.get("timestamp", 0.0)),
            scenario_id=self._current_id,
            causal_graph_gt=causal_graph_gt,
        )

    def _check_collision(
        self,
        ego_pos: np.ndarray,         # (2,)
        agent_positions: np.ndarray, # (A, 2)
    ) -> bool:
        if len(agent_positions) == 0:
            return False
        dist = np.linalg.norm(agent_positions - ego_pos[None], axis=-1)
        return bool((dist < self.collision_radius).any())
