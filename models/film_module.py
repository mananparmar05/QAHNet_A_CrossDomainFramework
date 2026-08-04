"""
FiLM (Feature-wise Linear Modulation) Quality Token Module.

Novel contribution C1: Uses FiLM conditioning to generate scale (gamma) and
shift (beta) parameters that directly modulate transformer attention weights
based on image quality.
"""
import torch
import torch.nn as nn


class FiLMQualityToken(nn.Module):
    """
    Generates a quality token and FiLM parameters (gamma, beta) from
    an image quality score.

    Quality score is a scalar in [0, 1]:
        1.0 = clean, high-quality image
        0.0 = severely degraded image
    """

    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Learnable base quality token
        self.quality_base = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # MLP: quality score -> quality embedding
        self.quality_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

        # FiLM generators: quality embedding -> gamma, beta per head
        self.film_gamma = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_heads),
            nn.Sigmoid(),  # gamma in [0, 1]
        )

        self.film_beta = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_heads),
            nn.Tanh(),  # beta in [-1, 1]
        )

    def forward(self, quality_score):
        """
        Args:
            quality_score: [B, 1] or [B]
        Returns:
            quality_token: [B, 1, embed_dim]
            gamma: [B, num_heads, 1, 1]
            beta:  [B, num_heads, 1, 1]
        """
        if quality_score.dim() == 1:
            quality_score = quality_score.unsqueeze(-1)

        quality_embed = self.quality_encoder(quality_score)  # [B, embed_dim]

        # Token = learnable base + quality-conditioned embedding
        quality_token = self.quality_base + quality_embed.unsqueeze(1)  # [B, 1, D]

        # FiLM parameters
        gamma = self.film_gamma(quality_embed).unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        beta = self.film_beta(quality_embed).unsqueeze(-1).unsqueeze(-1)

        return quality_token, gamma, beta
