#!/usr/bin/env python3
"""Extract one normalized intermediate RADIO spatial feature as a ViPS teacher.

RADIO's ``forward_intermediates`` uses zero-based block indices.  For example,
``--radio-block 12`` selects the output immediately after block 12.  Only the
selected spatial feature is saved; summary tokens and other intermediate
layers are deliberately discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(os.environ.get("VIPS_WS", str(PROJECT_DIR.parents[1])))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio-code-dir", default=str(WORKSPACE_ROOT / "code/RADIO"))
    parser.add_argument(
        "--radio-checkpoint",
        default=str(WORKSPACE_ROOT / "models/RADIO-v2.5-L/radio-v2.5-l_half.pth.tar"),
    )
    parser.add_argument(
        "--selection-manifest",
        default=str(WORKSPACE_ROOT / "prior/scannet_series_5tasks_100pct_4frames_len2048_seed0_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "prior/radio_v2_5_l_block12_spatial_grid14x18_scannet_series_100pct_4frames_unique"),
    )
    parser.add_argument("--radio-block", type=int, default=12)
    parser.add_argument("--input-height", type=int, default=224)
    parser.add_argument("--input-width", type=int, default=288)
    parser.add_argument("--target-height", type=int, default=14)
    parser.add_argument("--target-width", type=int, default=18)
    parser.add_argument("--expected-dim", type=int, default=1024)
    parser.add_argument("--expected-frames", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="0=全部唯一帧组。")
    return parser.parse_args()


def frame_group_id(paths: tuple[str, ...]) -> str:
    payload = json.dumps(paths, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:20]


def unique_records(path: str) -> list[dict]:
    groups = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                paths = tuple(record["image_paths"])
                groups.setdefault(paths, record)
    return sorted(groups.values(), key=lambda record: tuple(record["image_paths"]))


def load_frames(paths: list[str], size: tuple[int, int], device: torch.device):
    frames = []
    for path in paths:
        with Image.open(path) as image:
            tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False
        ).squeeze(0)
        frames.append(tensor)
    return torch.stack(frames).to(device, non_blocking=True)


def main():
    args = parse_args()
    if args.radio_block < 0:
        raise ValueError("--radio-block 必须为非负的零基 block 索引。")
    if not torch.cuda.is_available():
        raise RuntimeError("RADIO teacher 提取要求 CUDA。")
    code_dir, checkpoint = Path(args.radio_code_dir), Path(args.radio_checkpoint)
    if not (code_dir / "hubconf.py").is_file():
        raise FileNotFoundError(f"缺少 RADIO 官方代码：{code_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"缺少 RADIOv2.5-L 权重：{checkpoint}")
    if args.input_height % 16 or args.input_width % 16:
        raise ValueError("RADIOv2.5-L 输入高宽必须是16的倍数。")

    records = unique_records(args.selection_manifest)
    if args.limit:
        records = records[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    model = torch.hub.load(
        str(code_dir), "radio_model", source="local", version=str(checkpoint), progress=False
    ).cuda().eval()
    device = torch.device("cuda")
    completed = []
    for record in tqdm(records, desc=f"Collecting RADIO block {args.radio_block} features"):
        paths = tuple(record["image_paths"])
        identifier = frame_group_id(paths)
        output_path = output_dir / f"{identifier}.pt"
        if not output_path.is_file():
            images = load_frames(list(paths), (args.input_height, args.input_width), device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.forward_intermediates(
                    images,
                    indices=[args.radio_block],
                    output_fmt="NCHW",
                    norm=True,
                    stop_early=True,
                    aggregation="sparse",
                    intermediates_only=True,
                )
            if len(outputs) != 1:
                raise RuntimeError(
                    f"RADIO block 输出数量异常：type={type(outputs)}，len={len(outputs)}"
                )
            # radio_model.forward_intermediates 在不请求 prefix token 时，
            # 官方实现返回 Tensor 列表；某些版本/配置则返回 RadioOutput。
            # 两者都表示同一个 spatial feature。
            item = outputs[0]
            feature = item.features if hasattr(item, "features") else item
            if not torch.is_tensor(feature):
                raise RuntimeError(f"RADIO block 输出元素异常：{type(item)}")
            if feature.ndim != 4:
                raise ValueError(f"RADIO block 特征维度异常：{tuple(feature.shape)}")
            aligned = F.interpolate(
                feature.float(),
                size=(args.target_height, args.target_width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).contiguous().to(torch.bfloat16).cpu()
            expected = (args.expected_frames, args.target_height, args.target_width, args.expected_dim)
            if tuple(aligned.shape) != expected:
                raise ValueError(f"RADIO block teacher形状={tuple(aligned.shape)}，期望={expected}")
            if not torch.isfinite(aligned.float()).all():
                raise FloatingPointError(f"RADIO block teacher含NaN/Inf：{identifier}")
            temporary = output_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "radio_feature": aligned,
                    "format": "RADIOv2.5-L normalized intermediate spatial_features",
                    "stage": "backbone_block_spatial_features",
                    "radio_block": args.radio_block,
                    "zero_based_block_index": True,
                    "summary_saved": False,
                    "input_size": [args.input_height, args.input_width],
                    "target_grid": [args.target_height, args.target_width],
                    "dtype": "bfloat16",
                    "image_paths": list(paths),
                },
                temporary,
            )
            os.replace(temporary, output_path)
        completed.append(
            {
                "group_id": identifier,
                "feature_file": output_path.name,
                "image_paths": list(paths),
                "shape": [args.expected_frames, args.target_height, args.target_width, args.expected_dim],
                "dtype": "bfloat16",
                "radio_block": args.radio_block,
            }
        )

    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        for item in completed:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(f"RADIO block {args.radio_block} teacher采集完成：groups={len(completed)}，目录={output_dir}")


if __name__ == "__main__":
    main()
