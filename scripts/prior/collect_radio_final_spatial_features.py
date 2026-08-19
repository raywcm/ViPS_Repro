#!/usr/bin/env python3
"""提取 RADIOv2.5-L 最终 spatial_features，作为 ViPS RADIO teacher。

只使用普通 ``forward`` 返回的第二项，不保存 summary，也不调用
``forward_intermediates``。五任务样本按四帧路径去重，输出统一为
``[4,14,18,1024]`` BF16。
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
        default=str(WORKSPACE_ROOT / "prior/scannet_series_5tasks_10pct_4frames_len2048_seed0_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "prior/radio_v2_5_l_final_spatial_grid14x18_scannet_series_10pct_unique563"),
    )
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


def spatial_to_nchw(spatial: torch.Tensor, input_hw: tuple[int, int], patch_size: int):
    if spatial.ndim == 4:
        return spatial
    if spatial.ndim != 3:
        raise ValueError(f"RADIO spatial_features 维度异常：{tuple(spatial.shape)}")
    height, width = input_hw[0] // patch_size, input_hw[1] // patch_size
    if spatial.shape[1] != height * width:
        raise ValueError(
            f"RADIO token数异常：{spatial.shape[1]}，期望 {height}×{width}={height*width}"
        )
    return spatial.transpose(1, 2).reshape(spatial.shape[0], spatial.shape[2], height, width)


def main():
    args = parse_args()
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
        str(code_dir),
        "radio_model",
        source="local",
        version=str(checkpoint),
        progress=False,
    ).cuda().eval()
    patch_size = int(getattr(model, "patch_size", 16) or 16)
    device = torch.device("cuda")

    # 断点续跑时重建 manifest，避免 append 造成重复行。
    completed = []
    for record in tqdm(records, desc="Collecting RADIO final spatial features"):
        paths = tuple(record["image_paths"])
        identifier = frame_group_id(paths)
        output_path = output_dir / f"{identifier}.pt"
        if not output_path.is_file():
            images = load_frames(list(paths), (args.input_height, args.input_width), device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                # RADIO正常forward的第二项就是最终归一化 spatial patch tokens。
                _summary, spatial_features = model(images, feature_fmt="NCHW")
            feature_nchw = spatial_to_nchw(
                spatial_features, (args.input_height, args.input_width), patch_size
            ).float()
            aligned = F.interpolate(
                feature_nchw,
                size=(args.target_height, args.target_width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).contiguous().to(torch.bfloat16).cpu()
            expected = (args.expected_frames, args.target_height, args.target_width, args.expected_dim)
            if tuple(aligned.shape) != expected:
                raise ValueError(f"RADIO teacher形状={tuple(aligned.shape)}，期望={expected}")
            if not torch.isfinite(aligned.float()).all():
                raise FloatingPointError(f"RADIO teacher含NaN/Inf：{identifier}")
            temporary = output_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "radio_feature": aligned,
                    "format": "RADIOv2.5-L final normalized spatial_features",
                    "stage": "final_backbone_spatial_features",
                    "summary_saved": False,
                    "intermediate_layers_saved": False,
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
            }
        )

    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        for record in completed:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(f"RADIO teacher采集完成：groups={len(completed)}，目录={output_dir}")


if __name__ == "__main__":
    main()
