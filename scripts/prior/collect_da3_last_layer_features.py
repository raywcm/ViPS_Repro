#!/usr/bin/env python3
"""提取指定网格的 DA3 BF16 教师特征。

输入样本读取训练选择 manifest，因而与指定训练运行使用完全相同的四帧、
抽样和长度过滤结果。manifest 可以是 VGGT F_base manifest，也可以是仅包含
样本/图像路径的轻量 manifest。

默认 ``backbone_layer`` 保留旧实验：保存 Prediction.aux 中指定 DINOv2
block 的空间特征。新推荐的 ``depth_output_conv1`` 会捕获 DualDPT 主 depth
branch 的 ``scratch.output_conv1`` 输出（256 -> 128），即完成多尺度 DPT 融合
但尚未进入 ``output_conv2`` 预测头的 dense depth prior。两种模式都只保存一
个目标网格特征，不保存深度图、置信度图或其他中间层。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(os.environ.get("VIPS_WS", str(PROJECT_DIR.parents[1])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--da3-code-dir", default=str(WORKSPACE_ROOT / "code/Depth-Anything-3")
    )
    parser.add_argument(
        "--model-path", default=str(WORKSPACE_ROOT / "models/DA3-LARGE-1.1")
    )
    parser.add_argument(
        "--input-manifest", "--vggt-manifest",
        dest="input_manifest",
        default=(
            str(WORKSPACE_ROOT / "prior/vggt_f_base_scannet_det_5pct_4frames_len2048/manifest.jsonl")
        ),
    )
    parser.add_argument("--media-root", default=str(WORKSPACE_ROOT / "data/media"))
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "prior"))
    parser.add_argument("--limit", type=int, default=0, help="0=全部记录。")
    parser.add_argument(
        "--deduplicate-by-frame-group",
        action="store_true",
        help="按完全相同的帧组仅提取一次，并以 group_id 命名特征文件。",
    )
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=23,
        help="DA3-Large 最后一个 DINOv2 backbone block 的索引。",
    )
    parser.add_argument(
        "--feature-source",
        choices=("backbone_layer", "depth_output_conv1"),
        default="backbone_layer",
        help=(
            "教师特征位置：backbone_layer 为旧的 DINOv2 block；"
            "depth_output_conv1 为 DualDPT 主 depth branch 的 output_conv1 后特征。"
        ),
    )
    parser.add_argument("--target-height", type=int, default=14)
    parser.add_argument("--target-width", type=int, default=14)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--expected-frames", type=int, default=4)
    parser.add_argument(
        "--dataset-tag",
        default="scannet_det_5pct_4frames_len2048",
        help="写入目录名中的数据选择标识，避免不同训练集的 prior 混用。",
    )
    return parser.parse_args()


def frame_group_id(paths: list[str]) -> str:
    payload = json.dumps(paths, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:20]


def read_manifest(path: Path, limit: int, deduplicate: bool) -> list[dict]:
    records = []
    seen = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if deduplicate:
                group = tuple(record["image_paths"])
                if group in seen:
                    continue
                seen.add(group)
                record["group_id"] = frame_group_id(record["image_paths"])
            records.append(record)
            if limit and len(records) >= limit:
                break
    if limit and len(records) != limit:
        raise RuntimeError(f"VGGT manifest 仅有 {len(records)} 条，少于要求的 {limit} 条。")
    return records


def downsample_feature(
    feature: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    channels_first: bool = False,
) -> torch.Tensor:
    """将 DA3 aux 特征双线性下采样为 [T,H,W,C]。

    模型 forward 的原始 aux 是 [B,T,H,W,C]；官方 OutputProcessor 会在
    单样本 inference 后移除 batch 维，因此 Prediction.aux 实际为
    [T,H,W,C]。同时兼容这两种公开接口形状。
    """
    # OutputProcessor 会将 Prediction.aux 移到 CPU 并转换为 numpy.ndarray；
    # Depth output_conv1 hook 则直接得到 [T,C,H,W] Tensor。``as_tensor`` 对
    # Tensor 不复制，对 ndarray 也共享内存。
    feature = torch.as_tensor(feature)
    if feature.ndim == 5 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 4:
        raise RuntimeError(f"DA3 aux 特征形状异常：{tuple(feature.shape)}")
    # backbone aux 是 [T,H,W,C]；DPT hook 则严格为 [T,C,H,W]。不要根据
    # 尺寸猜测布局：DA3-Large 的 output_conv1 中 C=128 小于某些输入宽高，
    # 启发式会把它误当 NHWC，产生错误的288维缓存。
    if channels_first:
        feature_nchw = feature.float()
    else:
        feature_nchw = feature.permute(0, 3, 1, 2).float()
    feature_14 = F.interpolate(
        feature_nchw,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return feature_14.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)


def normalize_frame_paths(paths: list[str], expected_frames: int) -> list[str]:
    """统一输出精确帧数，兼容少数 ScanNet 场景只有 12 个可用源帧。"""
    if expected_frames <= 0:
        raise ValueError(f"expected_frames 必须为正数，实际={expected_frames}")
    if not paths:
        raise FileNotFoundError("样本没有可用 RGB 帧。")
    # 帧数不足时会重复相邻/末帧；帧数过多时等距重采样。两种情况均保留
    # 时间顺序，保证 4/32 帧 teacher 输入与对应 manifest 一致。
    indices = torch.linspace(0, len(paths) - 1, expected_frames).round().long().tolist()
    return [paths[index] for index in indices]


def register_depth_output_conv1_hook(model: torch.nn.Module) -> tuple[list[torch.Tensor], object]:
    """捕获 DualDPT 主 depth head 的 output_conv1 输出。"""
    try:
        module = model.model.head.scratch.output_conv1
    except AttributeError as error:
        raise RuntimeError(
            "当前 DA3 checkpoint 未暴露 DualDPT depth output_conv1；"
            "请确认使用 DA3-LARGE-1.1 / DualDPT 配置。"
        ) from error
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        captured.append(output.detach())

    return captured, module.register_forward_hook(capture)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，DA3 教师特征提取必须在 GPU 环境运行。")

    da3_code_dir = Path(args.da3_code_dir)
    sys.path.insert(0, str(da3_code_dir / "src"))
    from depth_anything_3.api import DepthAnything3

    model_path = Path(args.model_path)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"未找到 DA3-Large 权重：{model_path}。请先下载官方 "
            "depth-anything/DA3-LARGE-1.1 到该目录。"
        )

    records = read_manifest(
        Path(args.input_manifest), args.limit, args.deduplicate_by_frame_group
    )
    run_name = "all" if args.limit == 0 else f"limit{args.limit}"
    source_tag = (
        f"layer{args.feature_layer}"
        if args.feature_source == "backbone_layer"
        else "depth_output_conv1_v2"
    )
    output_dir = (
        Path(args.output_dir)
        / f"da3_large_1_1_{source_tag}_grid{args.target_height}x{args.target_width}_"
        f"{args.dataset_tag}_{run_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    model = DepthAnything3.from_pretrained(str(model_path)).to("cuda").eval()
    media_root = Path(args.media_root)
    key = f"feat_layer_{args.feature_layer}"
    captured_depth_feature: list[torch.Tensor] | None = None
    depth_hook = None
    if args.feature_source == "depth_output_conv1":
        captured_depth_feature, depth_hook = register_depth_output_conv1_hook(model)

    try:
        with manifest_path.open("a", encoding="utf-8") as manifest:
            for record in tqdm(records, desc=f"Collecting DA3 {source_tag} features"):
                identifier = record.get("group_id", record["sample_id"])
                output_path = output_dir / f"{identifier}.pt"
                if output_path.exists():
                    continue

                source_paths = []
                for image_path in record["image_paths"]:
                    path = Path(image_path)
                    # 新 manifest 保存绝对路径；旧 VGGT manifest 保存相对
                    # media_root 的路径。两种格式都兼容，且不重复添加前缀。
                    if not path.is_absolute():
                        path = media_root / path
                    source_paths.append(str(path))
                image_paths = normalize_frame_paths(source_paths, args.expected_frames)
                if not all(Path(path).is_file() for path in image_paths):
                    raise FileNotFoundError(
                        f"样本 {record['sample_id']} 的 RGB 帧不完整：{image_paths}"
                    )

                if captured_depth_feature is not None:
                    captured_depth_feature.clear()
                prediction = model.inference(
                    image=image_paths,
                    process_res=args.process_res,
                    process_res_method="upper_bound_resize",
                    # 保持输入帧顺序，与 VGGT F_base 和训练数据严格对应。
                    ref_view_strategy="first",
                    export_feat_layers=(
                        [args.feature_layer]
                        if args.feature_source == "backbone_layer"
                        else []
                    ),
                )
                if args.feature_source == "backbone_layer":
                    if prediction.aux is None or key not in prediction.aux:
                        raise RuntimeError(
                            f"DA3 未返回 {key}；aux keys={list((prediction.aux or {}).keys())}"
                        )
                    raw_feature = prediction.aux[key]
                else:
                    if captured_depth_feature is None or len(captured_depth_feature) != 1:
                        raise RuntimeError(
                            "DA3 depth output_conv1 hook 未捕获唯一特征，"
                            f"实际捕获={0 if captured_depth_feature is None else len(captured_depth_feature)}。"
                        )
                    raw_feature = captured_depth_feature.pop()
                feature = downsample_feature(
                    raw_feature,
                    (args.target_height, args.target_width),
                    channels_first=args.feature_source == "depth_output_conv1",
                ).cpu()
                if feature.shape[0] != args.expected_frames:
                    raise RuntimeError(
                        f"帧数不一致：期望 {args.expected_frames}，得到 {tuple(feature.shape)}"
                    )

                temporary_path = output_path.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "da3_feature": feature,
                        "format": (
                            "DA3 DualDPT depth output_conv1 spatial feature"
                            if args.feature_source == "depth_output_conv1"
                            else "DA3 DINOv2 backbone block spatial feature"
                        ),
                        "stage": (
                            "after_depth_output_conv1_before_output_conv2"
                            if args.feature_source == "depth_output_conv1"
                            else "before_depth_head"
                        ),
                        "model": "DA3-LARGE-1.1",
                        "feature_source": args.feature_source,
                        "feature_layer": args.feature_layer if args.feature_source == "backbone_layer" else None,
                        "target_grid": [args.target_height, args.target_width],
                        "dtype": "bfloat16",
                        "sample_index": record["sample_index"],
                        "source_vggt_feature": record.get("feature_file"),
                        "image_paths": image_paths,
                        "original_image_paths": record["image_paths"],
                    },
                    temporary_path,
                )
                os.replace(temporary_path, output_path)
                manifest.write(
                    json.dumps(
                        {
                            "sample_index": record["sample_index"],
                            "sample_id": record["sample_id"],
                            "group_id": record.get("group_id"),
                            "feature_file": output_path.name,
                            "shape": list(feature.shape),
                            "dtype": "bfloat16",
                            "feature_source": args.feature_source,
                            "target_grid": [args.target_height, args.target_width],
                            "source_vggt_feature": record.get("feature_file"),
                        }
                    )
                    + "\n"
                )
                manifest.flush()
    finally:
        if depth_hook is not None:
            depth_hook.remove()

    print(f"DA3 教师特征采集完成：{output_dir}")


if __name__ == "__main__":
    main()
