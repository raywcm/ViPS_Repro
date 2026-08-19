"""ViPS: efficient prior proxy and dynamic prior fusion."""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn


VIPS_PRIOR_DIMS = OrderedDict((
    ("vggt", 2048), ("da3", 1024), ("radio", 1024),
    ("trace", 1024), ("wan", 1536),
))


class PriorProxyMLP(nn.Module):
    """Paper Stage-1 token-wise MLP: D_base -> 4096 -> D_prior."""
    def __init__(self, in_dim=2048, hidden_dim=4096, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class EfficientPriorProxy(nn.Module):
    """Five independent prior proxies sharing only the VGGT input feature."""
    def __init__(self, base_dim=2048, hidden_dim=4096, prior_dims: Mapping[str, int] | None = None):
        super().__init__()
        self.base_dim = int(base_dim)
        self.prior_dims = OrderedDict(prior_dims or VIPS_PRIOR_DIMS)
        self.proxies = nn.ModuleDict({
            name: PriorProxyMLP(self.base_dim, hidden_dim, dim)
            for name, dim in self.prior_dims.items()
        })

    def forward(self, f_base, experts=None):
        if f_base.shape[-1] != self.base_dim:
            raise ValueError(f"ViPS F_base expected last dim {self.base_dim}, got {tuple(f_base.shape)}")
        names = tuple(experts or self.proxies.keys())
        unknown = tuple(name for name in names if name not in self.proxies)
        if unknown:
            raise ValueError(f"ViPS Proxy has no experts {unknown}; available={tuple(self.proxies)}")
        return {name: self.proxies[name](f_base) for name in names}


class DynamicPriorFusion(nn.Module):
    """Five-expert fusion with per-layer 1x1 ZeroConv and an instruction gate."""
    def __init__(self, hidden_size, layers=(3, 8, 13, 18, 23), prior_dims=None, share_gate=True, fusion_mode="gate",
                 include_null=False, gate_cond="last", gate_bias=None, gate_temp=1.0, hybrid_beta_per_dataset=False,
                 layer_gate=False, layer_gate_per_dataset=False):
        super().__init__()
        self.layers = tuple(int(x) for x in layers)
        self.prior_dims = OrderedDict(prior_dims or VIPS_PRIOR_DIMS)
        self.names = tuple(self.prior_dims)
        self.share_gate = bool(share_gate)
        self.fusion_mode = str(fusion_mode)
        self.include_null = bool(include_null)
        self.gate_cond = str(gate_cond)
        self.gate_bias = str(gate_bias) if gate_bias else None
        self.gate_temp = float(gate_temp)
        self.hybrid_beta_per_dataset = bool(hybrid_beta_per_dataset)
        self.layer_gate = bool(layer_gate)
        self.layer_gate_per_dataset = bool(layer_gate_per_dataset)
        if self.gate_cond not in ("last", "mean", "cat"):
            raise ValueError(f"gate_cond must be last/mean/cat, got {self.gate_cond!r}")
        self.zero_convs = nn.ModuleDict({
            name: nn.ModuleDict({str(layer): nn.Conv2d(dim, hidden_size, 1, bias=True) for layer in self.layers})
            for name, dim in self.prior_dims.items()
        })
        n_out = len(self.names) + (1 if self.include_null else 0)

        def gate():
            in_dim = hidden_size if self.gate_cond != "cat" else hidden_size * 2
            return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_size // 2), nn.GELU(), nn.Linear(hidden_size // 2, n_out))

        self.weight_mlp = gate() if self.share_gate else None
        self.weight_mlps = nn.ModuleDict() if self.share_gate else nn.ModuleDict({str(layer): gate() for layer in self.layers})
        if self.fusion_mode == "hybrid":
            if self.hybrid_beta_per_dataset:
                # Per (layer, dataset) beta.
                self.hybrid_beta = nn.ParameterDict({str(layer): nn.Parameter(torch.zeros(5)) for layer in self.layers})
            else:
                # Per-layer beta between gate and mean fusion.
                self.hybrid_beta = nn.ParameterDict({str(layer): nn.Parameter(torch.zeros(1)) for layer in self.layers})
        if self.layer_gate:
            # Layer-level gate scale.
            if self.layer_gate_per_dataset:
                self.layer_scales = nn.ParameterDict({str(layer): nn.Parameter(torch.zeros(5)) for layer in self.layers})
            else:
                self.layer_scales = nn.ParameterDict({str(layer): nn.Parameter(torch.zeros(1)) for layer in self.layers})
        self.reset_parameters()

    def reset_parameters(self):
        for expert in self.zero_convs.values():
            for conv in expert.values():
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)
        gates = [self.weight_mlp] if self.share_gate else self.weight_mlps.values()
        for gate in gates:
            nn.init.ones_(gate[0].weight); nn.init.zeros_(gate[0].bias)
            nn.init.xavier_uniform_(gate[1].weight); nn.init.zeros_(gate[1].bias)
            # Equal logits at step zero; ZeroConv keeps the residual exactly zero.
            nn.init.zeros_(gate[-1].weight); nn.init.zeros_(gate[-1].bias)
            if self.gate_bias is not None:
                names = list(self.names) + (["__null__"] if self.include_null else [])
                if self.gate_bias in names:
                    with torch.no_grad():
                        gate[-1].bias[names.index(self.gate_bias)] = 1.5
        if getattr(self, 'layer_gate', False) and hasattr(self, 'layer_scales'):
            with torch.no_grad():
                for p in self.layer_scales.values():
                    p.fill_(2.0)

    def forward(self, layer_idx, priors, query_states, query_mean=None, return_stats=False, null_mask=None, dataset_ids=None):
        key = str(layer_idx)
        if layer_idx not in self.layers:
            raise ValueError(f"layer {layer_idx} is not a ViPS injection layer")
        missing = set(self.names).difference(priors)
        if missing:
            raise ValueError(f"ViPS missing priors: {sorted(missing)}")
        if self.fusion_mode == "mean":
            # Uniform weights over projected priors.
            weights = torch.full((query_states.shape[0], len(self.names)), 1.0 / len(self.names), dtype=query_states.dtype, device=query_states.device)
        else:
            gate = self.weight_mlp if self.share_gate else self.weight_mlps[key]
            if self.gate_cond == "mean":
                if query_mean is None:
                    raise ValueError("gate_cond='mean' requires query_mean (mean-pooled instruction)")
                g_in = query_mean
            elif self.gate_cond == "cat":
                if query_mean is None:
                    raise ValueError("gate_cond='cat' requires query_mean (mean-pooled instruction)")
                g_in = torch.cat([query_states, query_mean], dim=-1)
            else:
                g_in = query_states
            logits = gate(g_in)
            if self.gate_temp != 1.0:
                logits = logits / self.gate_temp
            if self.include_null and null_mask is not None:
                # Mask the null logit for datasets without a null option.
                mask_val = torch.full_like(logits[:, -1], -1e9)
                logits[:, -1] = torch.where(null_mask.to(device=logits.device), logits[:, -1], mask_val)
            weights = torch.softmax(logits, dim=-1)  # [B, K(+1)]
        projected = []
        for name in self.names:
            x = priors[name]
            if x.ndim != 5 or x.shape[-1] != self.prior_dims[name]:
                raise ValueError(f"{name} must be [B,F,H,W,{self.prior_dims[name]}], got {tuple(x.shape)}")
            # Normalize per token to control heterogeneous teacher scale.
            x = F.normalize(x.float(), dim=-1).to(x.dtype)
            b, frames, height, width, channels = x.shape
            z = self.zero_convs[name][key](x.reshape(b * frames, height, width, channels).permute(0, 3, 1, 2))
            projected.append(z.permute(0, 2, 3, 1).reshape(b, frames, height, width, -1))
        stacked = torch.stack(projected, dim=1)  # B,K,F,H,W,C
        # Null weight scales the total residual.
        expert_weights = weights[:, : len(self.names)]
        if self.fusion_mode == "hybrid":
            mean_weights = torch.full_like(expert_weights, 1.0 / len(self.names))
            weighted = stacked * expert_weights[:, :, None, None, None, None].to(stacked.dtype)
            fused_gate = weighted.sum(1)
            fused_mean = (stacked * mean_weights[:, :, None, None, None, None]).sum(1)
            beta = torch.sigmoid(self.hybrid_beta[key]).to(fused_gate.dtype)
            if self.hybrid_beta_per_dataset:
                if dataset_ids is None:
                    beta = beta.mean().view(1, 1, 1, 1, 1)
                else:
                    beta = beta[dataset_ids].view(-1, 1, 1, 1, 1)
            fused = beta * fused_gate + (1 - beta) * fused_mean
        else:
            weighted = stacked * expert_weights[:, :, None, None, None, None].to(stacked.dtype)
            fused = weighted.sum(1)
        if self.layer_gate:
            scale = torch.tanh(self.layer_scales[key]).to(fused.dtype)
            if self.layer_gate_per_dataset:
                if dataset_ids is None:
                    scale = scale.mean().view(1, 1, 1, 1, 1)
                else:
                    scale = scale[dataset_ids].view(-1, 1, 1, 1, 1)
            fused = fused * scale
        if not return_stats:
            return fused, weights
        # Detached scalar diagnostics.
        projected_norm = torch.stack([
            value.float().norm(dim=-1).mean(dim=(1, 2, 3)) for value in projected
        ], dim=-1)
        effective_norm = torch.stack([
            weighted[:, index].float().norm(dim=-1).mean(dim=(1, 2, 3))
            for index in range(len(self.names))
        ], dim=-1)
        stats = {
            "weights": weights.detach(),
            "projected_norm": projected_norm.detach(),
            "effective_norm": effective_norm.detach(),
            "effective_ratio": (effective_norm / effective_norm.sum(-1, keepdim=True).clamp_min(1e-12)).detach(),
            "gate_entropy": (-(weights.float() * weights.float().clamp_min(1e-12)).log().sum(-1)).detach(),
            "fused_norm": fused.float().norm(dim=-1).mean(dim=(1, 2, 3)).detach(),
        }
        if self.include_null:
            stats["null_weight"] = weights[:, -1].detach()
        return fused, weights, stats
