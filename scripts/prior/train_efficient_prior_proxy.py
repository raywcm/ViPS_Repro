#!/usr/bin/env python3
"""ViPS Stage 1：联合训练 VGGT -> {VGGT, DA3, RADIO, Trace, Wan} Proxy。

五个 ScanNet-series 任务会重复引用同一组四帧。本脚本以四帧路径的 SHA1
为主键去重，严格配对五份离线 cache。论文主配置优化五路 MSE/L2 之和；
cosine 只作诊断指标，除非显式设置 ``--cosine-weight`` 做消融。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(os.environ.get("VIPS_WS", str(PROJECT_ROOT.parents[1])))
sys.path.insert(0, str(PROJECT_ROOT))

from llava.model.vips import EfficientPriorProxy  # noqa: E402


TEACHER_KEYS = OrderedDict(
    (("vggt", "f_base"), ("da3", "da3_feature"), ("radio", "radio_feature"),
     ("trace", "trace_feature"), ("wan", "wan_feature"))
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = WORKSPACE_ROOT
    parser.add_argument(
        "--selection-manifest",
        default=str(root / "prior/scannet_series_5tasks_10pct_4frames_len2048_seed0_manifest.jsonl"),
    )
    parser.add_argument(
        "--vggt-dir", default=str(root / "prior/vggt_14x18_scannet_series_10pct_unique563")
    )
    parser.add_argument(
        "--da3-dir",
        default=(
            str(root / "prior/da3_large_1_1_layer23_grid14x18_scannet_series_5tasks_10pct_4frames_len2048_seed0_full22310")
        ),
    )
    parser.add_argument(
        "--radio-dir",
        default=str(root / "prior/radio_v2_5_l_final_spatial_grid14x18_scannet_series_10pct_unique563"),
    )
    parser.add_argument(
        "--trace-dir",
        default=str(root / "prior/trace_anything_final_decoder_grid14x18_scannet_series_10pct_unique563"),
    )
    parser.add_argument(
        "--wan-dir",
        default=str(root / "prior/wan2.1_t2v1.3b_block20_t300_grid14x18_scannet_series_10pct"),
    )
    parser.add_argument(
        "--output-dir", default=str(root / "outputs/vips_efficient_proxy_5experts_scannet_series_10pct")
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--cosine-weight", type=float, default=0.0,
        help="默认0（论文L2）；非0仅用于和旧Proxy recipe对比。",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=0, help="0=全部；正数=smoke。")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--frames", type=int, default=4,
        help="每组输入帧数；32帧 prior 训练时设为32。",
    )
    return parser.parse_args()


def frame_group_id(paths: tuple[str, ...]) -> str:
    payload = json.dumps(paths, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:20]


class EfficientPriorDataset(Dataset):
    """按唯一四帧组配对 F_base 与五路 teacher。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.frames = args.frames
        self.vggt_dir = Path(args.vggt_dir)
        self.teacher_dirs = {
            "da3": Path(args.da3_dir),
            "radio": Path(args.radio_dir),
            "trace": Path(args.trace_dir),
            "wan": Path(args.wan_dir),
        }
        groups: dict[tuple[str, ...], dict] = {}
        with Path(args.selection_manifest).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                paths = tuple(record["image_paths"])
                if len(paths) != args.frames:
                    raise ValueError(
                        f"样本 {record['sample_id']} 帧数={len(paths)}，期望={args.frames}"
                    )
                groups.setdefault(paths, record)
        self.items = []
        missing = []
        for paths, representative in sorted(groups.items()):
            identifier = frame_group_id(paths)
            files = {
                "vggt": self.vggt_dir / f"{identifier}.pt",
                # 新版 DA3 可按帧组去重保存；旧4帧缓存仍兼容 sample_id 命名。
                "da3": self.teacher_dirs["da3"] / f"{identifier}.pt",
                "radio": self.teacher_dirs["radio"] / f"{identifier}.pt",
                "trace": self.teacher_dirs["trace"] / f"{identifier}.pt",
                "wan": self.teacher_dirs["wan"] / f"{identifier}.pt",
            }
            da3_group_path = files["da3"]
            if not da3_group_path.is_file():
                files["da3"] = self.teacher_dirs["da3"] / f"{representative['sample_id']}.pt"
            absent = [f"{name}:{path}" for name, path in files.items() if not path.is_file()]
            if absent:
                missing.extend(absent)
            self.items.append((identifier, paths, files))
        if missing:
            raise FileNotFoundError(
                f"五路 prior 配对不完整：缺少 {len(missing)} 个文件，首个={missing[0]}"
            )
        if not self.items:
            raise ValueError("没有可用于 Proxy 对齐的 prior 配对。")
        # teacher 通道数由实际缓存推断：旧 DA3 layer23 是1024，新 DualDPT
        # depth output_conv1 是128。其余专家仍沿用原有维度。这样同一训练器
        # 可复用，但 checkpoint 会显式保存本次真实 prior_dims。
        _, _, first_files = self.items[0]
        self.prior_dims = OrderedDict((("vggt", 2048),))
        for name, key in TEACHER_KEYS.items():
            if name == "vggt":
                continue
            value = torch.load(first_files[name], map_location="cpu", weights_only=True).get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape[:3]) != (self.frames, 14, 18):
                raise ValueError(
                    f"{first_files[name]}:{key} 形状异常，期望前缀="
                    f"({self.frames}, 14, 18)，实际={getattr(value, 'shape', None)}"
                )
            self.prior_dims[name] = int(value.shape[-1])

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _load(path: Path, key: str, expected_shape: tuple[int, ...]) -> torch.Tensor:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if key not in obj:
            raise KeyError(f"{path} 缺少 key={key!r}。")
        value = obj[key]
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{path.name}:{key} 形状异常，期望={expected_shape}，实际={tuple(value.shape)}"
            )
        if not torch.isfinite(value.float()).all():
            raise FloatingPointError(f"{path}:{key} 含 NaN/Inf。")
        return value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        identifier, _, files = self.items[index]
        f_base = self._load(files["vggt"], "f_base", (self.frames, 14, 18, 2048))
        result: dict[str, torch.Tensor | str] = {
            "group_id": identifier,
            "f_base": f_base,
            # 论文 Eq.4 对 K=5 个 Proxy 都有 teacher。当 base model
            # 为 VGGT 时，VGGT Proxy 的目标就是同一份 E_base(V)=F_base。
            "vggt": f_base,
        }
        for name, key in TEACHER_KEYS.items():
            if name == "vggt":
                continue
            result[name] = self._load(
                files[name], key, (self.frames, 14, 18, self.prior_dims[name])
            )
        return result


def proxy_metrics(
    predictions: dict[str, torch.Tensor],
    teachers: dict[str, torch.Tensor],
    cosine_weight: float,
    prior_dims,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metrics = {}
    losses = []
    for name in prior_dims:
        prediction = predictions[name].float()
        teacher = teachers[name].float()
        mse = F.mse_loss(prediction, teacher)
        cosine = F.cosine_similarity(prediction, teacher, dim=-1).mean()
        branch_loss = mse + cosine_weight * (1.0 - cosine)
        metrics[f"{name}_mse"] = mse
        metrics[f"{name}_cosine"] = cosine
        # ViPS 表 5 使用 cosine distance（越低越好），而旧实验日志使用
        # cosine similarity（越高越好）。同时记录两者，避免把 0.252
        # distance 与 0.87 similarity 直接比较。
        metrics[f"{name}_cosine_distance"] = 1.0 - cosine
        losses.append(branch_loss)
    total = torch.stack(losses).sum()
    metrics["loss"] = total
    metrics["mean_mse"] = torch.stack(
        [metrics[f"{name}_mse"] for name in prior_dims]
    ).mean()
    metrics["mean_cosine"] = torch.stack(
        [metrics[f"{name}_cosine"] for name in prior_dims]
    ).mean()
    metrics["mean_cosine_distance"] = 1.0 - metrics["mean_cosine"]
    return total, metrics


def move_batch(batch: dict, device: torch.device, prior_dims):
    f_base = batch["f_base"].to(device, non_blocking=True)
    teachers = {
        name: batch[name].to(device, non_blocking=True) for name in prior_dims
    }
    return f_base, teachers


@torch.no_grad()
def evaluate(model, loader, device, cosine_weight, prior_dims):
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="Validation", leave=False):
        f_base, teachers = move_batch(batch, device, prior_dims)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, metrics = proxy_metrics(model(f_base), teachers, cosine_weight, prior_dims)
        batch_size = f_base.shape[0]
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value.item() * batch_size
        count += batch_size
    return {name: value / count for name, value in totals.items()}


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_proxy(path: Path, model, args, epoch, metrics, prior_dims) -> None:
    torch.save(
        {
            "model": cpu_state_dict(model),
            "epoch": epoch,
            "val": metrics,
            "base_dim": 2048,
            "hidden_dim": 4096,
            "prior_dims": dict(prior_dims),
            "alignment_objective": (
                "sum_5proxy_mse" if args.cosine_weight == 0 else "sum_5proxy_mse_plus_cosine_ablation"
            ),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Efficient Prior Proxy 训练需要 CUDA。")
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 (0,1) 内。")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = EfficientPriorDataset(args)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    if args.max_groups:
        indices = indices[: args.max_groups]
    if len(indices) < 2:
        raise ValueError("至少需要两个唯一四帧组。")
    val_size = max(1, round(len(indices) * args.val_ratio))
    train_set = Subset(dataset, indices[val_size:])
    val_set = Subset(dataset, indices[:val_size])
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    device = torch.device("cuda")
    model = EfficientPriorProxy(prior_dims=dataset.prior_dims).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "unique_groups": len(dataset),
        "train_groups": len(train_set),
        "val_groups": len(val_set),
        "prior_dims": dict(dataset.prior_dims),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = output_dir / "metrics.jsonl"
    print(
        f"唯一四帧组={len(dataset)}，训练={len(train_set)}，验证={len(val_set)}，"
        f"联合Proxy参数={config['parameters']:,}，objective="
        f"{'sum_5proxy(MSE)' if args.cosine_weight == 0 else 'sum_5proxy(MSE+cosine)'}"
    )

    best_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in progress:
            f_base, teachers = move_batch(batch, device, dataset.prior_dims)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, metrics = proxy_metrics(
                    model(f_base), teachers, args.cosine_weight, dataset.prior_dims
                )
            loss.backward()
            optimizer.step()
            batch_size = f_base.shape[0]
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value.item() * batch_size
            count += batch_size
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                cosine=f"{metrics['mean_cosine'].item():.4f}",
            )
        train_metrics = {name: value / count for name, value in totals.items()}
        val_metrics = evaluate(
            model, val_loader, device=device, cosine_weight=args.cosine_weight,
            prior_dims=dataset.prior_dims
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        if val_metrics["mean_mse"] < best_mse:
            best_mse = val_metrics["mean_mse"]
            save_proxy(output_dir / "best.pt", model, args, epoch, val_metrics, dataset.prior_dims)
        if args.save_every and epoch % args.save_every == 0:
            save_proxy(
                output_dir / f"epoch-{epoch}.pt", model, args, epoch, val_metrics, dataset.prior_dims
            )
    save_proxy(output_dir / "last.pt", model, args, args.epochs, val_metrics, dataset.prior_dims)
    print(f"训练完成：best mean MSE={best_mse:.6f}，{output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
