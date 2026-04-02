"""
models/encoders/multimodal_encoder.py
Clean multimodal encoder for SE-CVLA using CLIP vision + Qwen2.5 language.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig
from transformers import AutoModel


class VisionEncoder(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.num_cameras = 6
        self.backbone = AutoModel.from_pretrained(cfg.encoder.vision_backbone)
        # CLIP vision model hidden size
        vis_cfg = getattr(self.backbone.config, 'vision_config', self.backbone.config)
        backbone_dim = getattr(vis_cfg, 'hidden_size', 768)
        self.proj = nn.Linear(backbone_dim, cfg.encoder.hidden_dim)
        self.norm = nn.LayerNorm(cfg.encoder.hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N_cam, C, H, W = images.shape
        flat = rearrange(images, 'b n c h w -> (b n) c h w')
        # CLIP vision model forward
        vis_out = self.backbone.vision_model(pixel_values=flat, interpolate_pos_encoding=True)
        features = vis_out.last_hidden_state          # (B*N, patches, dim)
        features = self.norm(self.proj(features))
        return rearrange(features, '(b n) p d -> b (n p) d', b=B, n=N_cam)


class StateEncoder(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        ego_input_dim = 21 * 5
        self.ego_mlp = nn.Sequential(
            nn.Linear(ego_input_dim, cfg.encoder.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.encoder.hidden_dim, cfg.encoder.hidden_dim),
            nn.LayerNorm(cfg.encoder.hidden_dim),
        )
        self.agent_proj = nn.Linear(10, cfg.encoder.hidden_dim)
        self.agent_queries = nn.Parameter(torch.randn(1, 8, cfg.encoder.hidden_dim) * 0.02)
        self.agent_cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.encoder.hidden_dim,
            num_heads=cfg.encoder.num_heads,
            batch_first=True,
            dropout=cfg.encoder.dropout,
        )

    def forward(self, ego_state, agents, agent_mask):
        B = ego_state.shape[0]
        ego_token = self.ego_mlp(ego_state.flatten(1)).unsqueeze(1)
        agent_feats = self.agent_proj(agents)
        queries = self.agent_queries.expand(B, -1, -1)
        agent_tokens, _ = self.agent_cross_attn(
            query=queries, key=agent_feats, value=agent_feats,
            key_padding_mask=~agent_mask,
        )
        return ego_token, agent_tokens


class MultimodalEncoder(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(cfg)
        self.state_encoder = StateEncoder(cfg)

        # Language backbone (Qwen2.5 text model)
        self.language_backbone = AutoModel.from_pretrained(cfg.encoder.language_backbone)
        lang_cfg = getattr(self.language_backbone.config, 'text_config', self.language_backbone.config)
        lang_dim = getattr(lang_cfg, 'hidden_size', 1536)
        self.lang_proj = nn.Linear(lang_dim, cfg.encoder.hidden_dim)
        self.lang_norm = nn.LayerNorm(cfg.encoder.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.encoder.hidden_dim, nhead=cfg.encoder.num_heads,
            dim_feedforward=cfg.encoder.hidden_dim * 4,
            dropout=cfg.encoder.dropout, batch_first=True, norm_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder.num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.encoder.hidden_dim) * 0.02)

    def forward(self, images, ego_state, agents, agent_mask, input_ids, attention_mask):
        B = images.shape[0]
        vision_tokens = self.vision_encoder(images)
        ego_token, agent_tokens = self.state_encoder(ego_state, agents, agent_mask)
        lang_out = self.language_backbone(input_ids=input_ids, attention_mask=attention_mask)
        lang_tokens = self.lang_norm(self.lang_proj(lang_out.last_hidden_state))
        cls = self.cls_token.expand(B, -1, -1)
        all_tokens = torch.cat([cls, vision_tokens, lang_tokens, ego_token, agent_tokens], dim=1)
        fused_repr = self.fusion_transformer(all_tokens)
        pooled_repr = fused_repr[:, 0]
        return fused_repr, pooled_repr
