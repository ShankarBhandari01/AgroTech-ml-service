import numpy as np
import math
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn


class PyTorchCrossAttentionFusion(nn.Module):
    """
    PyTorch Deep Learning Cross-Attention Fusion Layer.
    Fuses Multi-Spectral Vision Transformer Satellite Embeddings (Query) with
    Probabilistic Temporal Weather Sequences (Key & Value).
    """
    def __init__(self, spatial_dim: int = 4, temporal_dim: int = 4, embed_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, embed_dim)
        self.temporal_proj = nn.Linear(temporal_dim, embed_dim)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, spatial_feats: torch.Tensor, temporal_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - spatial_feats: [Batch, Spatial_Dim]
        - temporal_seq: [Batch, Time_Steps, Temporal_Dim]
        """
        if spatial_feats.dim() == 1:
            spatial_feats = spatial_feats.unsqueeze(0)
        if temporal_seq.dim() == 2:
            temporal_seq = temporal_seq.unsqueeze(0)

        # Projections
        Q = self.spatial_proj(spatial_feats).unsqueeze(1) # [Batch, 1, Embed_Dim]
        K = self.temporal_proj(temporal_seq)             # [Batch, Time_Steps, Embed_Dim]
        V = K                                            # [Batch, Time_Steps, Embed_Dim]

        # Multi-head Cross Attention
        attn_output, attn_weights = self.multihead_attn(query=Q, key=K, value=V) # attn_output: [Batch, 1, Embed_Dim]
        attn_output = attn_output.squeeze(1) # [Batch, Embed_Dim]

        # Residual + Norm
        fused = self.norm(Q.squeeze(1) + self.fc(attn_output))
        
        return fused, attn_weights.squeeze(1)


class CrossAttentionFusionLayer:
    """
    NumPy / Scikit-Learn Fast Microclimate Cross-Attention Fusion Layer.
    Calculates sub-millisecond spatiotemporal cross-attention matrices on CPU.
    """
    def __init__(self, spatial_dim: int = 4, temporal_dim: int = 4, embed_dim: int = 32, num_heads: int = 4):
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        np.random.seed(42)
        self.W_q = np.random.randn(spatial_dim, embed_dim) / np.sqrt(spatial_dim)
        self.W_k = np.random.randn(temporal_dim, embed_dim) / np.sqrt(temporal_dim)
        self.W_v = np.random.randn(temporal_dim, embed_dim) / np.sqrt(temporal_dim)
        self.W_o = np.random.randn(embed_dim, embed_dim) / np.sqrt(embed_dim)

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def forward(self, spatial_feats: np.ndarray, temporal_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if spatial_feats.ndim == 1:
            spatial_feats = np.expand_dims(spatial_feats, axis=0)
        if temporal_seq.ndim == 2:
            temporal_seq = np.expand_dims(temporal_seq, axis=0)

        Q = np.dot(spatial_feats, self.W_q) # [Batch, Embed_Dim]
        K = np.matmul(temporal_seq, self.W_k) # [Batch, Time_Steps, Embed_Dim]
        V = np.matmul(temporal_seq, self.W_v) # [Batch, Time_Steps, Embed_Dim]

        Q_expanded = np.expand_dims(Q, axis=1) # [Batch, 1, Embed_Dim]
        scores = np.sum(Q_expanded * K, axis=-1) / math.sqrt(self.embed_dim)

        attn_weights = self._softmax(scores, axis=-1) # [Batch, Time_Steps]

        attn_weights_expanded = np.expand_dims(attn_weights, axis=-1)
        context = np.sum(attn_weights_expanded * V, axis=1) # [Batch, Embed_Dim]

        fused = np.dot(context, self.W_o) # [Batch, Embed_Dim]

        return fused, attn_weights
