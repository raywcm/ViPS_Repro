"""Geometry residual adapter: F_native + alpha * delta, distilled to VGGT."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFrameAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):  # [B,F,P,C]
        B, F, P, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.permute(0, 2, 1, 3).reshape(B * P, F, C)
        k = k.permute(0, 2, 1, 3).reshape(B * P, F, C)
        v = v.permute(0, 2, 1, 3).reshape(B * P, F, C)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        out = attn @ v
        out = out.reshape(B, P, F, C).permute(0, 2, 1, 3)
        return self.out(out) + x


def _project(mlp, hidden, spatial_out):
    """concat 4 layers -> MLP -> pool to spatial_out x spatial_out -> [B,F,P,C]"""
    layers = [hidden[str(i)] for i in range(len(hidden))]
    x = torch.cat([h for h in layers], dim=-1)  # [B,F,729,C*n]
    B, nf, P, C = x.shape
    x = mlp(x)  # [B,F,729,out]
    s = int(P ** 0.5)
    x = x.reshape(B * nf, s, s, -1).permute(0, 3, 1, 2)
    x = F.adaptive_avg_pool2d(x, (spatial_out, spatial_out))
    x = x.permute(0, 2, 3, 1).reshape(B, nf, spatial_out, spatial_out, -1)
    return x


class GeometryResidualAdapter(nn.Module):
    def __init__(self, in_dim=1152, n_layers=4, out_dim=2048, use_cross_frame=True,
                 spatial_out=14, alpha=1.0, train_native=False, seed=0):
        super().__init__()
        self.in_dim = in_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.use_cross_frame = use_cross_frame
        self.spatial_out = spatial_out
        self.alpha = alpha
        self.train_native = train_native
        # frozen native projector
        g = torch.Generator().manual_seed(seed)
        self.native = nn.Sequential(
            nn.Linear(in_dim * n_layers, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        for m in self.native:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02, generator=g)
                nn.init.zeros_(m.bias)
        # residual branch
        self.fuse = nn.Sequential(
            nn.Linear(in_dim * n_layers, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.cross = CrossFrameAttention(out_dim) if use_cross_frame else None

    def forward(self, hidden):
        if self.train_native:
            f_native = _project(self.native, hidden, self.spatial_out)
        else:
            with torch.no_grad():
                f_native = _project(self.native, hidden, self.spatial_out)
        def _flat(t):
            B, nf = t.shape[:2]
            return t.reshape(B, nf, self.spatial_out * self.spatial_out, self.out_dim)
        delta = _project(self.fuse, hidden, self.spatial_out)
        if self.cross is not None:
            delta = self.cross(_flat(delta)).reshape(delta.shape)
        out = f_native + self.alpha * delta
        return out, f_native, delta
