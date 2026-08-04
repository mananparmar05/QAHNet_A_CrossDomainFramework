
"""
QAHNet: Quality-Aware Hybrid Network.

Main architecture combining:
  C1: FiLM-conditioned Quality Token
  C2: Multi-task Quality Prediction
  C3: Multimodal Clinical Metadata Fusion

Architecture:
  CNN Backbone (EfficientNet-B0) -> Patch Tokenizer -> 
  [CLS] + [QTok] + [Meta_1..11] + [Patch_1..49] ->
  FiLM-modulated Transformer Encoder (2 layers, 4 heads) ->
  Multi-task Heads (infection, severity, quality)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from .film_module import FiLMQualityToken
from .metadata_encoder import MetadataEncoder


class FiLMMultiHeadAttention(nn.Module):
    """
    Multi-head attention with FiLM modulation.
    
    Standard attention: Attn = softmax(QK^T / sqrt(d))
    FiLM attention:     Attn = gamma * softmax(QK^T / sqrt(d)) + beta
    """

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, gamma=None, beta=None):
        """
        Args:
            x: [B, N, D] - token sequence
            gamma: [B, H, 1, 1] - FiLM scale (optional)
            beta:  [B, H, 1, 1] - FiLM shift (optional)
        """
        B, N, D = x.shape
        H = self.num_heads

        q = self.q_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        k = self.k_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)

        # FiLM modulation: gamma * attn + beta
        if gamma is not None and beta is not None:
            attn = gamma * attn + beta

        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class FiLMTransformerBlock(nn.Module):
    """Transformer block with FiLM-modulated attention."""

    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = FiLMMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, gamma=None, beta=None):
        x = x + self.attn(self.norm1(x), gamma, beta)
        x = x + self.mlp(self.norm2(x))
        return x


class QAHNet(nn.Module):
    """
    Quality-Aware Hybrid Network.

    Args:
        num_classes_infection: int (default 1 for binary)
        num_classes_severity: int (default 3)
        embed_dim: transformer embedding dimension
        num_heads: number of attention heads
        num_layers: number of transformer layers
        dropout: dropout rate
        use_quality_token: whether to use FiLM QTok (C1)
        use_metadata: whether to use metadata tokens (C3)
        use_quality_head: whether to predict quality (C2)
        use_film: whether to use FiLM modulation (vs passive QTok)
    """

    def __init__(
        self,
        num_classes_infection=1,
        num_classes_severity=3,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        use_quality_token=True,
        use_metadata=True,
        use_quality_head=True,
        use_film=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_quality_token = use_quality_token
        self.use_metadata = use_metadata
        self.use_quality_head = use_quality_head
        self.use_film = use_film

        # ── Stage 1: CNN Backbone ──
        efficientnet = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn_features = efficientnet.features  # Output: [B, 1280, 7, 7]
        self.cnn_out_dim = 1280

        # ── Stage 2: Patch Tokenizer ──
        # Project CNN features to embed_dim
        self.patch_proj = nn.Sequential(
            nn.Linear(self.cnn_out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        # Positional embeddings for 49 spatial tokens
        self.spatial_pos = nn.Parameter(torch.randn(1, 49, embed_dim) * 0.02)

        # ── CLS Token ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # ── Stage 3: Quality Token (C1) ──
        if use_quality_token:
            self.quality_module = FiLMQualityToken(embed_dim, num_heads)

        # ── Stage 4: Metadata Encoder (C3) ──
        if use_metadata:
            self.metadata_encoder = MetadataEncoder(embed_dim)

        # ── Stage 5: Transformer Encoder ──
        self.transformer_blocks = nn.ModuleList([
            FiLMTransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # ── Stage 6: Multi-Task Heads ──
        self.infection_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes_infection),
        )

        self.severity_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes_severity),
        )

        if use_quality_head:
            self.quality_head = nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),  # output quality in [0, 1]
            )

    def freeze_cnn(self):
        """Freeze CNN backbone for initial training epochs."""
        for param in self.cnn_features.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        """Unfreeze CNN backbone for fine-tuning."""
        for param in self.cnn_features.parameters():
            param.requires_grad = True

    def forward(self, images, quality_score=None, metadata=None):
        """
        Args:
            images: [B, 3, 224, 224]
            quality_score: [B] or [B, 1] float in [0, 1] (optional)
            metadata: dict of tensors (optional)

        Returns:
            dict with keys:
                'infection': [B, 1] logits
                'severity':  [B, 3] logits
                'quality':   [B, 1] predicted quality (if use_quality_head)
        """
        B = images.shape[0]

        # ── CNN Feature Extraction ──
        feat_maps = self.cnn_features(images)  # [B, 1280, 7, 7]

        # ── Patch Tokenization ──
        B_feat, C_feat, H_feat, W_feat = feat_maps.shape
        N_spatial = H_feat * W_feat
        patches = feat_maps.flatten(2).transpose(1, 2)  # [B, N_spatial, 1280]
        patch_tokens = self.patch_proj(patches)  # [B, N_spatial, D]

        # Adapt positional embeddings to actual spatial size
        if N_spatial != self.spatial_pos.shape[1]:
            pos = self.spatial_pos.transpose(1, 2)  # [1, D, 49]
            pos = F.interpolate(pos, size=N_spatial, mode='linear', align_corners=False)
            patch_tokens = patch_tokens + pos.transpose(1, 2)  # [1, N_spatial, D]
        else:
            patch_tokens = patch_tokens + self.spatial_pos

        # ── Build token sequence ──
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        tokens = [cls_token]

        # Quality token
        gamma, beta = None, None
        if self.use_quality_token and quality_score is not None:
            q_token, gamma, beta = self.quality_module(quality_score)
            tokens.append(q_token)
            if not self.use_film:
                gamma, beta = None, None  # disable FiLM, keep passive token

        # Metadata tokens
        if self.use_metadata and metadata is not None:
            meta_tokens = self.metadata_encoder(metadata)  # [B, 11, D]
            tokens.append(meta_tokens)

        # Patch tokens
        tokens.append(patch_tokens)

        # Concatenate all tokens
        x = torch.cat(tokens, dim=1)  # [B, N_total, D]

        # ── Transformer Encoder with FiLM ──
        for block in self.transformer_blocks:
            x = block(x, gamma, beta)

        x = self.final_norm(x)

        # ── Extract CLS token ──
        cls_out = x[:, 0]  # [B, D]

        # ── Multi-task heads ──
        outputs = {
            'infection': self.infection_head(cls_out),
            'severity': self.severity_head(cls_out),
        }

        if self.use_quality_head:
            outputs['quality'] = self.quality_head(cls_out)

        return outputs

    def get_param_groups(self, lr_cnn=1e-5, lr_transformer=1e-4, lr_heads=3e-4):
        """Get parameter groups with differential learning rates."""
        cnn_params = list(self.cnn_features.parameters())
        head_params = (
            list(self.infection_head.parameters()) +
            list(self.severity_head.parameters())
        )
        if self.use_quality_head:
            head_params += list(self.quality_head.parameters())

        # Everything else = transformer + tokenizer + quality module + metadata encoder
        other_ids = set(id(p) for p in cnn_params + head_params)
        transformer_params = [p for p in self.parameters() if id(p) not in other_ids]

        return [
            {'params': cnn_params, 'lr': lr_cnn},
            {'params': transformer_params, 'lr': lr_transformer},
            {'params': head_params, 'lr': lr_heads},
        ]


def build_model(variant='F', device='cpu'):
    """
    Build model variant for ablation study.

    Variants:
        A: CNN-only (EfficientNet-B0, no transformer)
        B: CNN + Transformer
        C: B + Passive QTok (no FiLM)
        D: B + FiLM QTok
        E: D + Metadata
        F: Full QAHNet (D + Metadata + Quality Head)
    """
    configs = {
        'A': dict(use_quality_token=False, use_metadata=False, use_quality_head=False, use_film=False),
        'B': dict(use_quality_token=False, use_metadata=False, use_quality_head=False, use_film=False),
        'C': dict(use_quality_token=True,  use_metadata=False, use_quality_head=False, use_film=False),
        'D': dict(use_quality_token=True,  use_metadata=False, use_quality_head=False, use_film=True),
        'E': dict(use_quality_token=True,  use_metadata=True,  use_quality_head=False, use_film=True),
        'F': dict(use_quality_token=True,  use_metadata=True,  use_quality_head=True,  use_film=True),
    }

    if variant == 'A':
        # CNN-only: use pooled features directly, no transformer
        return CNNOnly(device=device)

    config = configs[variant]
    model = QAHNet(**config)
    return model.to(device)


class CNNOnly(nn.Module):
    """
    Ablation Model A: CNN-only baseline (EfficientNet-B0).
    No transformer, no quality token, no metadata.
    """

    def __init__(self, embed_dim=256, dropout=0.3, device='cpu'):
        super().__init__()
        efficientnet = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        self.features = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, embed_dim)

        self.infection_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

        self.use_quality_head = False
        self.to(device)

    def freeze_cnn(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_cnn(self):
        for p in self.features.parameters():
            p.requires_grad = True

    def forward(self, images, quality_score=None, metadata=None):
        feat = self.features(images)
        feat = self.pool(feat).flatten(1)
        feat = self.proj(feat)
        return {
            'infection': self.infection_head(feat),
            'severity': self.severity_head(feat),
        }

    def get_param_groups(self, lr_cnn=1e-5, lr_transformer=1e-4, lr_heads=3e-4):
        cnn_params = list(self.features.parameters())
        head_params = list(self.infection_head.parameters()) + list(self.severity_head.parameters())
        other_ids = set(id(p) for p in cnn_params + head_params)
        other_params = [p for p in self.parameters() if id(p) not in other_ids]
        return [
            {'params': cnn_params, 'lr': lr_cnn},
            {'params': other_params, 'lr': lr_transformer},
            {'params': head_params, 'lr': lr_heads},
        ]

