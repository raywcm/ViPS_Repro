#!/usr/bin/env python3
"""提取 TraceAnything teacher 特征，作为 ViPS Trace prior。

严格复用官方图像预处理、时间编码、encoder 和 decoder；支持导出 encoder final、
decoder block12/block18/final，以及 Track PixelHead 的 refinenet3/refinenet1。
后两者执行官方 Track PixelHead 的多尺度 DPT，并在 trajectory regression head
之前捕获任务特化的 dense feature；refinenet1 的输出是 ``p1``。

五任务样本按帧组去重，输出统一到 ``[F,14,18,C]`` BF16。
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
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(os.environ.get("VIPS_WS", str(PROJECT_DIR.parents[1])))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-code-dir", default=str(WORKSPACE_ROOT / "code/TraceAnything"))
    parser.add_argument(
        "--trace-checkpoint",
        default=str(WORKSPACE_ROOT / "models/TraceAnything/trace_anything.pt"),
    )
    parser.add_argument(
        "--trace-config",
        default=str(WORKSPACE_ROOT / "code/TraceAnything/configs/eval.yaml"),
    )
    parser.add_argument(
        "--selection-manifest",
        default=str(WORKSPACE_ROOT / "prior/scannet_series_5tasks_10pct_4frames_len2048_seed0_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "prior/trace_anything_final_decoder_grid14x18_scannet_series_10pct_unique563"),
    )
    parser.add_argument("--long-side", type=int, default=512)
    parser.add_argument("--encoder-chunk-size", type=int, default=1)
    parser.add_argument("--target-height", type=int, default=14)
    parser.add_argument("--target-width", type=int, default=18)
    parser.add_argument(
        "--feature-source",
        choices=(
            "encoder_final",
            "encoder_block5",
            "encoder_block11",
            "encoder_block17",
            "encoder_block23",
            "decoder_block5",
            "decoder_block11",
            "decoder_block17",
            "decoder_block23",
            "decoder_block12",
            "decoder_block18",
            "final_decoder",
            "track_dpt_refinenet3",
            "track_dpt_refinenet2",
            "track_dpt_refinenet1",
        ),
        default="final_decoder",
        help=(
            "encoder_final=T0；encoder_block{N}=enc中间层；"
            "decoder_block{N}=fusion中间层；decoder_block12=T1；"
            "decoder_block18=T2；final_decoder=T3（=fusion_postnorm）；"
            "track_dpt_refinenet3=T4；track_dpt_refinenet2=dpt_mid；"
            "track_dpt_refinenet1=T5（=dpt_late）。"
        ),
    )
    parser.add_argument(
        "--expected-dim", type=int, default=None,
        help="默认按 feature-source 推断：encoder/decoder=1024，track DPT=256。",
    )
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


def to_plain_dict(value):
    return OmegaConf.to_container(value, resolve=True) if not isinstance(value, dict) else value


def get_state_dict(checkpoint):
    if (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("state_dict"), dict)
    ):
        return checkpoint["state_dict"]
    return checkpoint


def build_feature_model(
    code_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    feature_source: str,
):
    sys.path.insert(0, str(code_dir))
    from trace_anything.trace_anything import TraceAnything

    config = OmegaConf.load(config_path)
    net_config = config.get("model", {}).get("net", None) or config.get("net", None)
    if net_config is None:
        raise KeyError("Trace配置缺少 model.net 或 net。")
    model = TraceAnything(
        encoder_args=to_plain_dict(net_config["encoder_args"]),
        decoder_args=to_plain_dict(net_config["decoder_args"]),
        head_args=to_plain_dict(net_config["head_args"]),
        targeting_mechanism=net_config.get("targeting_mechanism", "bspline_conf"),
        poly_degree=net_config.get("poly_degree", 10),
        # 不构造额外 local head；time/track head 在加载后也会删除。
        whether_local=False,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    state = get_state_dict(checkpoint)
    if state and all(key.startswith("net.") for key in state):
        state = {key[4:]: value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    # 官方 checkpoint 可包含未构造的 local head；encoder/decoder 缺键则不能继续。
    # 新 p1 模式还必须确保 Track PixelHead 的 DPT 权重已加载，避免静默使用
    # 随机初始化的 refinenet1。
    required_prefixes = ("encoder.", "decoder.")
    if feature_source.startswith("track_dpt_"):
        # 官方 checkpoint 以 ``ds_head_track`` 命名；``track_head`` 只是同一
        # 模块的运行时别名，不能把别名键缺失误判为未加载。
        required_prefixes += ("ds_head_track.",)
    critical_missing = [
        key for key in incompatible.missing_keys
        if key.startswith(required_prefixes)
    ]
    if critical_missing:
        raise RuntimeError(f"Trace encoder/decoder 权重缺失：{critical_missing[:5]}")
    del checkpoint, state

    # final decoder 模式不需要任何预测 head。p1 模式则保留官方 Track
    # PixelHead；其 DPT 使用 [0,12,18,24] 多阶段 feature，hook 位于
    # refinenet1 输出、trajectory regression head 之前。
    keep_track_head = feature_source.startswith("track_dpt_")
    removable = ["time_head", "local_head", "ds_head_time", "ds_head_local"]
    if not keep_track_head:
        removable.extend(("track_head", "ds_head_track"))
    for name in removable:
        if hasattr(model, name):
            delattr(model, name)
    return model


def normalize_frame_paths(paths: list[str], expected_frames: int) -> list[str]:
    """以时间顺序重采样/补齐，兼容原始场景仅有12个可用 RGB 帧。"""
    if not paths:
        raise FileNotFoundError("Trace sample 没有可用 RGB 帧。")
    indices = torch.linspace(0, len(paths) - 1, expected_frames).round().long().tolist()
    return [paths[index] for index in indices]


def preprocess_views(paths: list[str], long_side: int, device: torch.device):
    tensors = []
    target_hw = None
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            if height > width:
                image = image.transpose(Image.Transpose.ROTATE_90)
                width, height = image.size
            if width >= height:
                resized = (long_side, int(height * long_side / width))
            else:
                resized = (int(width * long_side / height), long_side)
            image = image.resize(resized, Image.Resampling.BILINEAR)
            if target_hw is None:
                target_hw = (image.height - image.height % 16, image.width - image.width % 16)
            target_height, target_width = target_hw
            image = image.crop((0, 0, target_width, target_height))
            tensor = pil_to_tensor(image).float().div_(255.0)
            tensor = tensor.sub_(0.5).div_(0.5).unsqueeze(0)
            tensors.append(tensor)
    # TraceAnything 的时间编码按实际视角数均匀分布；不能把4帧时间表
    # 写死，否则32帧采集会在第5帧触发 IndexError。
    frame_count = len(tensors)
    times = tuple(
        index / max(frame_count - 1, 1) for index in range(frame_count)
    )
    views = [
        # ManyAR_PatchEmbed 会把卷积结果原地写回输入派生的缓存；若输入仍为
        # FP32、外层 autocast 产生 BF16 卷积结果，官方实现会因两者 dtype
        # 不同而报错。显式 BF16 与官方混合精度推理语义一致。
        {
            "img": tensor.to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            ),
            "time_step": times[index],
        }
        for index, tensor in enumerate(tensors)
    ]
    return views, target_hw


@torch.no_grad()
def extract_final_decoder(model, views, encoder_chunk_size: int):
    encoded, positions, _shapes = model._encode_images(
        views, chunk_size=encoder_chunk_size
    )
    batch_size = encoded[0].shape[0]
    image_ids, image_times = [], []
    patch_counts = []
    for index, features in enumerate(encoded):
        patch_count = features.shape[1]
        patch_counts.append(patch_count)
        image_ids.extend([index] * patch_count)
        image_times.extend([views[index]["time_step"]] * patch_count)
    image_ids = torch.tensor(
        image_ids * batch_size, device=encoded[0].device, dtype=torch.long
    ).reshape(batch_size, -1)
    image_times = torch.tensor(
        image_times * batch_size, device=encoded[0].device, dtype=torch.float32
    ).reshape(batch_size, -1)
    decoder_outputs, _ = model.decoder(encoded, positions, image_ids, image_times)
    return decoder_outputs[-1], patch_counts


@torch.no_grad()
def extract_encoder_final(model, views, encoder_chunk_size: int):
    """提取 Trace CroCo image encoder 的最终归一化 patch tokens。"""
    encoded, _positions, _shapes = model._encode_images(
        views, chunk_size=encoder_chunk_size
    )
    patch_counts = [features.shape[1] for features in encoded]
    return torch.cat(encoded, dim=0), patch_counts


@torch.no_grad()
def extract_encoder_layer(model, views, encoder_chunk_size: int, layer_index: int):
    """提取 CroCo encoder enc_blocks[layer_index-1] 的输出（N=1-based）。"""
    if not 1 <= layer_index <= len(model.encoder.enc_blocks):
        raise ValueError(
            f"Trace encoder 层索引={layer_index} 超出范围1..{len(model.encoder.enc_blocks)}"
        )
    captured = []
    module = model.encoder.enc_blocks[layer_index - 1]
    handle = module.register_forward_hook(
        lambda _m, _i, output: captured.append(output.detach())
    )
    try:
        encoded, _positions, _shapes = model._encode_images(
            views, chunk_size=encoder_chunk_size
        )
    finally:
        handle.remove()
    if len(captured) != len(encoded):
        raise RuntimeError(
            f"Trace encoder hook 捕获数异常：{len(captured)}，帧数={len(encoded)}"
        )
    patch_counts = [features.shape[1] for features in encoded]
    return torch.cat(captured, dim=0), patch_counts


@torch.no_grad()
def extract_decoder_layer(model, views, encoder_chunk_size: int, layer_index: int):
    """提取 decoder outputs[layer_index]，索引24是最终 dec_norm 表示。"""
    encoded, positions, _shapes = model._encode_images(
        views, chunk_size=encoder_chunk_size
    )
    batch_size = encoded[0].shape[0]
    image_ids, image_times = [], []
    patch_counts = []
    for index, features in enumerate(encoded):
        patch_count = features.shape[1]
        patch_counts.append(patch_count)
        image_ids.extend([index] * patch_count)
        image_times.extend([views[index]["time_step"]] * patch_count)
    image_ids = torch.tensor(
        image_ids * batch_size, device=encoded[0].device, dtype=torch.long
    ).reshape(batch_size, -1)
    image_times = torch.tensor(
        image_times * batch_size, device=encoded[0].device, dtype=torch.float32
    ).reshape(batch_size, -1)
    decoder_outputs, _ = model.decoder(encoded, positions, image_ids, image_times)
    if not 0 <= layer_index < len(decoder_outputs):
        raise ValueError(
            f"Trace decoder 输出索引={layer_index} 超出范围0..{len(decoder_outputs)-1}"
        )
    return decoder_outputs[layer_index], patch_counts


@torch.no_grad()
def extract_track_dpt_feature(model, views, encoder_chunk_size: int, refine_level: int):
    """运行官方 Track PixelHead，捕获 DPT refinenet3/refinenet1。"""
    from einops import rearrange

    encoded, positions, shapes = model._encode_images(
        views, chunk_size=encoder_chunk_size
    )
    batch_size, patches_per_view = encoded[0].shape[:2]
    image_ids, image_times = [], []
    for index, features in enumerate(encoded):
        patch_count = features.shape[1]
        image_ids.extend([index] * patch_count)
        image_times.extend([views[index]["time_step"]] * patch_count)
    image_ids = torch.tensor(
        image_ids * batch_size, device=encoded[0].device, dtype=torch.long
    ).reshape(batch_size, -1)
    image_times = torch.tensor(
        image_times * batch_size, device=encoded[0].device, dtype=torch.float32
    ).reshape(batch_size, -1)
    decoder_outputs, _ = model.decoder(encoded, positions, image_ids, image_times)
    frame_count = len(views)
    gathered_outputs = [
        rearrange(
            layer_output,
            "b (n p) d -> (n b) p d",
            n=frame_count,
            p=patches_per_view,
        )
        for layer_output in decoder_outputs
    ]
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        captured.append(output.detach())

    if refine_level not in (1, 2, 3):
        raise ValueError(f"只支持 Track DPT refinenet1/2/3，实际={refine_level}")
    module = getattr(model.track_head.dpt.scratch, f"refinenet{refine_level}")
    handle = module.register_forward_hook(capture)
    try:
        # 必须让官方 DPT 正常 forward，才能严格沿用其多尺度融合、空间布局、
        # act_postprocess 与 layer_rn；hook 在 trajectory regression head 之前捕获。
        model.track_head(gathered_outputs, torch.stack(shapes))
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Trace Track DPT p1 hook 捕获数异常：{len(captured)}")
    return captured[0], patches_per_view


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TraceAnything teacher 提取要求 CUDA。")
    code_dir = Path(args.trace_code_dir)
    checkpoint_path = Path(args.trace_checkpoint)
    config_path = Path(args.trace_config)
    for required in (code_dir / "trace_anything" / "trace_anything.py", checkpoint_path, config_path):
        if not required.is_file():
            raise FileNotFoundError(f"缺少 TraceAnything 文件：{required}")
    if args.long_side % 16:
        raise ValueError("TraceAnything long-side 必须是16的倍数。")
    expected_dim = (
        args.expected_dim
        if args.expected_dim is not None
        else (
            256 if args.feature_source.startswith("track_dpt_") else 1024
        )
    )

    records = unique_records(args.selection_manifest)
    if args.limit:
        records = records[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    # 官方 ManyAR_PatchEmbed 按参数 dtype 分配输出缓存，因此模型与输入都需
    # 显式采用同一 BF16 精度，不能只依赖外层 autocast。
    model = (
        build_feature_model(code_dir, config_path, checkpoint_path, args.feature_source)
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    torch.cuda.reset_peak_memory_stats(device)

    completed = []
    for record in tqdm(records, desc=f"Collecting Trace {args.feature_source} features"):
        source_paths = tuple(record["image_paths"])
        # identifier 必须来自 manifest 原帧组，以便与 VGGT/DA3/RADIO/Wan
        # 的唯一组缓存配对；实际 Trace 输入则固定为期望帧数。
        paths = tuple(normalize_frame_paths(list(source_paths), args.expected_frames))
        identifier = frame_group_id(source_paths)
        output_path = output_dir / f"{identifier}.pt"
        if not output_path.is_file():
            views, input_hw = preprocess_views(list(paths), args.long_side, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.feature_source == "encoder_final":
                    feature, patch_counts = extract_encoder_final(
                        model, views, args.encoder_chunk_size
                    )
                    if len(set(patch_counts)) != 1:
                        raise ValueError(f"Trace encoder各帧 patch 数不一致：{patch_counts}")
                    patch_height, patch_width = input_hw[0] // 16, input_hw[1] // 16
                    feature = feature.reshape(
                        args.expected_frames, patch_height, patch_width, feature.shape[-1]
                    ).permute(0, 3, 1, 2)
                elif args.feature_source.startswith("encoder_block"):
                    encoder_index = int(args.feature_source.removeprefix("encoder_block"))
                    feature, patch_counts = extract_encoder_layer(
                        model, views, args.encoder_chunk_size, encoder_index
                    )
                    if len(set(patch_counts)) != 1:
                        raise ValueError(f"Trace encoder各帧 patch 数不一致：{patch_counts}")
                    patch_height, patch_width = input_hw[0] // 16, input_hw[1] // 16
                    if feature.shape[0] != args.expected_frames or feature.shape[1] != patch_height * patch_width:
                        raise ValueError(
                            f"Trace encoder中间层形状={tuple(feature.shape)}，"
                            f"期望frames={args.expected_frames} tokens={patch_height * patch_width}"
                        )
                    feature = feature.reshape(
                        args.expected_frames, patch_height, patch_width, feature.shape[-1]
                    ).permute(0, 3, 1, 2)
                elif args.feature_source.startswith("decoder_block") or args.feature_source == "final_decoder":
                    decoder_index = {
                        "decoder_block5": 5,
                        "decoder_block11": 11,
                        "decoder_block17": 17,
                        "decoder_block23": 23,
                        "decoder_block12": 12,
                        "decoder_block18": 18,
                        "final_decoder": 24,
                    }[args.feature_source]
                    feature, patch_counts = extract_decoder_layer(
                        model, views, args.encoder_chunk_size, decoder_index
                    )
                    if len(set(patch_counts)) != 1:
                        raise ValueError(f"Trace decoder各帧 patch 数不一致：{patch_counts}")
                    patch_height, patch_width = input_hw[0] // 16, input_hw[1] // 16
                    expected_tokens = args.expected_frames * patch_height * patch_width
                    if tuple(feature.shape[:2]) != (1, expected_tokens):
                        raise ValueError(
                            f"Trace decoder形状={tuple(feature.shape)}，期望token={expected_tokens}"
                        )
                    feature = feature.reshape(
                        args.expected_frames, patch_height, patch_width, feature.shape[-1]
                    ).permute(0, 3, 1, 2)
                elif args.feature_source.startswith("track_dpt_"):
                    refine_level = {
                        "track_dpt_refinenet3": 3,
                        "track_dpt_refinenet2": 2,
                        "track_dpt_refinenet1": 1,
                    }[args.feature_source]
                    feature, _patches_per_view = extract_track_dpt_feature(
                        model, views, args.encoder_chunk_size, refine_level
                    )
                else:
                    raise ValueError(f"未知 Trace feature source={args.feature_source}")
                if feature.ndim != 4 or feature.shape[0] != args.expected_frames:
                    raise ValueError(
                        f"Trace feature形状异常：{tuple(feature.shape)}，"
                        f"期望帧数={args.expected_frames}"
                    )
            aligned = F.interpolate(
                feature.float(),
                size=(args.target_height, args.target_width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).contiguous().to(torch.bfloat16).cpu()
            expected = (args.expected_frames, args.target_height, args.target_width, expected_dim)
            if tuple(aligned.shape) != expected:
                raise ValueError(f"Trace teacher形状={tuple(aligned.shape)}，期望={expected}")
            if not torch.isfinite(aligned.float()).all():
                raise FloatingPointError(f"Trace teacher含NaN/Inf：{identifier}")
            temporary = output_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "trace_feature": aligned,
                    "format": (
                            f"TraceAnything Track PixelHead DPT {args.feature_source}"
                            if args.feature_source.startswith("track_dpt_")
                            else f"TraceAnything {args.feature_source} spatial tokens"
                    ),
                    "stage": (
                        f"after_track_dpt_{args.feature_source.split('refinenet')[-1]}_before_trajectory_regression"
                        if args.feature_source.startswith("track_dpt_")
                        else "final_multiview_encoder_decoder"
                    ),
                    "feature_source": args.feature_source,
                    "input_size": list(input_hw),
                    "target_grid": [args.target_height, args.target_width],
                    "time_steps": [
                        index / max(args.expected_frames - 1, 1)
                        for index in range(args.expected_frames)
                    ],
                    "dtype": "bfloat16",
                    "image_paths": list(paths),
                    "original_image_paths": list(source_paths),
                },
                temporary,
            )
            os.replace(temporary, output_path)
        completed.append(
            {
                "group_id": identifier,
                "feature_file": output_path.name,
                "image_paths": list(source_paths),
                "shape": [args.expected_frames, args.target_height, args.target_width, expected_dim],
                "dtype": "bfloat16",
            }
        )

    manifest_path = output_dir / "manifest.jsonl"
    temporary_manifest = manifest_path.with_suffix(".jsonl.tmp")
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        for record in completed:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_manifest, manifest_path)
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(
        f"Trace teacher采集完成：groups={len(completed)}，"
        f"CUDA峰值={peak_gib:.2f} GiB，目录={output_dir}"
    )


if __name__ == "__main__":
    main()
