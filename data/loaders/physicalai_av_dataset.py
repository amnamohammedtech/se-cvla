from __future__ import annotations
import logging, os
from dataclasses import dataclass
from typing import Iterator
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from PIL import Image
from data.augmentation.causal_augmentation import CausalAugmentor

logger = logging.getLogger(__name__)

@dataclass
class DrivingFrame:
    images: torch.Tensor
    ego_state: torch.Tensor
    agents: torch.Tensor
    agent_mask: torch.Tensor
    trajectory_gt: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    scenario_id: str = ""
    timestamp: float = 0.0
    split: str = "train"
    causal_graph_gt: torch.Tensor | None = None
    intervention_label: str | None = None

@dataclass
class BatchedFrames:
    images: torch.Tensor
    ego_state: torch.Tensor
    agents: torch.Tensor
    agent_mask: torch.Tensor
    trajectory_gt: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    causal_graph_gt: torch.Tensor | None = None

    def to(self, device) -> "BatchedFrames":
        return BatchedFrames(
            images=self.images.to(device),
            ego_state=self.ego_state.to(device),
            agents=self.agents.to(device),
            agent_mask=self.agent_mask.to(device),
            trajectory_gt=self.trajectory_gt.to(device),
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            causal_graph_gt=(self.causal_graph_gt.to(device) if self.causal_graph_gt is not None else None),
        )

class PhysicalAIAVDataset(IterableDataset):
    def __init__(self, cfg: DictConfig, split: str = "train", tokenizer=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.tokenizer = tokenizer
        self.augmentor = CausalAugmentor(cfg.augmentation.get(split, []))
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        token = open(token_path).read().strip() if os.path.exists(token_path) else None
        logger.info(f"Initialising PhysicalAIAVDatasetInterface [{split}] ...")
        from physical_ai_av import PhysicalAIAVDatasetInterface
        self._interface = PhysicalAIAVDatasetInterface(
            token=token,
            cache_dir=os.path.expanduser(cfg.dataset.cache_dir),
            confirm_download_threshold_gb=float("inf"),
        )
        self._interface.download_metadata()
        idx = self._interface.clip_index
        self._clip_ids = list(idx[idx["split"] == split].index)
        logger.info(f"Split [{split}]: {len(self._clip_ids)} clips")
        self.img_transform = transforms.Compose([
            transforms.Resize(tuple(cfg.cameras.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.cameras.normalize_mean, std=cfg.cameras.normalize_std),
        ])

    def __iter__(self) -> Iterator[DrivingFrame]:
        for clip_id in self._clip_ids:
            try:
                frame = self._load_clip(clip_id)
                frame = self.augmentor(frame)
                yield frame
            except Exception as e:
                logger.warning(f"Skipping clip {clip_id}: {e}")
                continue

    def _load_clip(self, clip_id: str) -> DrivingFrame:
        cfg = self.cfg
        cam_names = cfg.cameras.names[:cfg.cameras.num_cameras]
        images_list = []
        for cam in cam_names:
            try:
                frames = self._interface.get_clip_camera_frames(clip_id, cam)
                mid = len(frames) // 2
                img = Image.fromarray(frames[mid]).convert("RGB")
                images_list.append(self.img_transform(img))
            except Exception:
                images_list.append(torch.zeros(3, *cfg.cameras.image_size))
        images = torch.stack(images_list)
        try:
            ego_raw = self._interface.get_clip_egomotion(clip_id)
            hist = cfg.ego_state.history_len
            feat = len(cfg.ego_state.features)
            ego_arr = np.zeros((hist, feat), dtype=np.float32)
            n_steps = min(hist, len(ego_raw))
            ego_arr[:n_steps] = np.array(ego_raw[:n_steps])[:, :feat]
            ego_state = torch.tensor(ego_arr)
        except Exception:
            ego_state = torch.zeros(cfg.ego_state.history_len, len(cfg.ego_state.features))
        max_a, feat_d = cfg.agents.max_agents, cfg.agents.feature_dim
        agents = torch.zeros(max_a, feat_d)
        agent_mask = torch.zeros(max_a, dtype=torch.bool)
        traj = torch.zeros(cfg.trajectory.future_len, 2)
        seq_len = cfg.language.max_tokens
        if self.tokenizer is not None:
            tokens = self.tokenizer("The ego vehicle continues on its current path.",
                max_length=seq_len, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(seq_len, dtype=torch.long)
            attention_mask = torch.zeros(seq_len, dtype=torch.long)
        return DrivingFrame(images=images, ego_state=ego_state, agents=agents,
            agent_mask=agent_mask, trajectory_gt=traj, input_ids=input_ids,
            attention_mask=attention_mask, scenario_id=clip_id, split=self.split)

def collate_driving_frames(frames: list[DrivingFrame]) -> BatchedFrames:
    has_cg = frames[0].causal_graph_gt is not None
    return BatchedFrames(
        images=torch.stack([f.images for f in frames]),
        ego_state=torch.stack([f.ego_state for f in frames]),
        agents=torch.stack([f.agents for f in frames]),
        agent_mask=torch.stack([f.agent_mask for f in frames]),
        trajectory_gt=torch.stack([f.trajectory_gt for f in frames]),
        input_ids=torch.stack([f.input_ids for f in frames]),
        attention_mask=torch.stack([f.attention_mask for f in frames]),
        causal_graph_gt=(torch.stack([f.causal_graph_gt for f in frames]) if has_cg else None),
    )

def build_dataloader(cfg: DictConfig, split: str, tokenizer=None) -> DataLoader:
    dataset = PhysicalAIAVDataset(cfg, split=split, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=cfg.dataloader.get("batch_size", 4),
        num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor, collate_fn=collate_driving_frames)
