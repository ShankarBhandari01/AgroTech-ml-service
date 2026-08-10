"""Spatiotemporal cross-attention fusion (NumPy, CPU).

NOTE: the projection matrices are fixed random draws, never trained, so `fusion_score` and
`peak_incubation_hour` are deterministic functions of the inputs but carry no learned signal.
They are diagnostics, not features — see docs/model-design.md. The PyTorch variant that used to
live here was never instantiated and is deleted along with the torch dependency.
"""

import math
from typing import Tuple

import numpy as np


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
