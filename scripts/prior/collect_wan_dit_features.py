#!/usr/bin/env python3
"""Extract deterministic Wan2.1 DiT intermediate features for ViPS.

The four ScanNet frames are resized without changing their order, the last frame
is repeated to satisfy Wan's 4n+1 convention, and the VAE latents are mixed with
one global deterministic noise template at a fixed flow-matching timestep.
Only the selected DiT block is executed; the denoising head is never run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(os.environ.get("VIPS_WS", str(PROJECT_DIR.parents[1])))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-code-dir", default=str(WORKSPACE_ROOT / "code/Wan2.1"))
    parser.add_argument("--wan-checkpoint", default=str(WORKSPACE_ROOT / "models/Wan2.1-T2V-1.3B"))
    parser.add_argument("--selection-manifest", default=str(WORKSPACE_ROOT / "prior/scannet_series_5tasks_10pct_4frames_len2048_seed0_manifest.jsonl"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--empty-context-cache", default=str(WORKSPACE_ROOT / "prior/wan2.1_t2v1.3b_empty_prompt_context.pt"))
    parser.add_argument("--block", type=int, default=20, help="1-based DiT block number, 1..30")
    parser.add_argument("--timestep", type=int, default=300)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=14)
    parser.add_argument("--target-width", type=int, default=18)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expected-frames", type=int, default=4)
    return parser.parse_args()


def frame_group_id(paths):
    return hashlib.sha1(json.dumps(paths, ensure_ascii=False).encode()).hexdigest()[:20]


def unique_records(path):
    groups = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                paths = tuple(record["image_paths"])
                groups.setdefault(paths, record)
    return sorted(groups.values(), key=lambda item: tuple(item["image_paths"]))


def normalize_frame_paths(paths, expected_frames):
    """保持时间顺序并输出精确帧数，兼容少数32帧场景仅有12个源帧。"""
    if not paths:
        raise FileNotFoundError("Wan teacher sample has no RGB frames")
    indices = torch.linspace(0, len(paths) - 1, expected_frames).round().long().tolist()
    return [paths[index] for index in indices]


def load_video(paths, height, width, device, expected_frames):
    frames = []
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Wan teacher frame missing: {path}")
        with Image.open(path) as source:
            image = source.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
            frame = pil_to_tensor(image).float().div_(127.5).sub_(1.0)
            frames.append(frame)
    if len(frames) != expected_frames:
        raise ValueError(f"Wan teacher expects {expected_frames} source frames, got {len(frames)}")
    # Wan VAE temporal convention is 4n+1. Preserve all source frames and pad only
    # by repeating the final frame; never resample or reorder the four observations.
    while (len(frames) - 1) % 4:
        frames.append(frames[-1].clone())
    return torch.stack(frames, dim=1).to(device=device)


def build_empty_context(code_dir, checkpoint_dir, config, device):
    from wan.modules.t5 import T5EncoderModel

    encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=str(checkpoint_dir / config.t5_checkpoint),
        tokenizer_path=str(checkpoint_dir / config.t5_tokenizer),
    )
    # The empty prompt keeps the teacher cache a function of the frame group only.
    context = encoder([""], torch.device("cpu"))[0].to(device=device)
    del encoder
    gc.collect()
    return context


@torch.no_grad()
def forward_to_block(model, latent, context, timestep, block_number):
    from wan.modules.model import sinusoidal_embedding_1d

    embedded = model.patch_embedding(latent.unsqueeze(0))
    grid_sizes = torch.tensor([embedded.shape[2:]], dtype=torch.long)
    tokens = embedded.flatten(2).transpose(1, 2)
    seq_len = tokens.shape[1]
    seq_lens = torch.tensor([seq_len], dtype=torch.long)
    # Match Wan's official forward exactly: modulation stays FP32 even though
    # attention/FFN run under outer BF16 autocast.
    with amp.autocast(dtype=torch.float32):
        time_embedding = model.time_embedding(
            sinusoidal_embedding_1d(model.freq_dim, timestep).float()
        )
        modulation = model.time_projection(time_embedding).unflatten(1, (6, model.dim))
    text = model.text_embedding(
        torch.stack([
            torch.cat([
                context,
                context.new_zeros(model.text_len - context.shape[0], context.shape[1]),
            ])
        ])
    )
    if model.freqs.device != tokens.device:
        model.freqs = model.freqs.to(tokens.device)
    kwargs = dict(
        e=modulation,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=model.freqs,
        context=text,
        context_lens=None,
    )
    for index, block in enumerate(model.blocks, start=1):
        tokens = block(tokens, **kwargs)
        if index == block_number:
            break
    feature = tokens[:, :seq_len].reshape(1, *grid_sizes[0].tolist(), model.dim)
    return feature[0]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Wan teacher extraction requires CUDA")
    if not 1 <= args.block <= 30:
        raise ValueError("--block must be in [1, 30]")
    if not 0 <= args.timestep <= 1000:
        raise ValueError("--timestep must be in [0, 1000]")
    if args.height % 16 or args.width % 16:
        raise ValueError("Wan input height/width must be divisible by 16")
    code_dir, checkpoint_dir = Path(args.wan_code_dir), Path(args.wan_checkpoint)
    required = [
        code_dir / "wan" / "modules" / "model.py",
        checkpoint_dir / "config.json",
        checkpoint_dir / "Wan2.1_VAE.pth",
        checkpoint_dir / "models_t5_umt5-xxl-enc-bf16.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Wan files:\n" + "\n".join(missing))
    sys.path.insert(0, str(code_dir))
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.vae import WanVAE

    config = WAN_CONFIGS["t2v-1.3B"]
    device = torch.device("cuda")
    context_cache = Path(args.empty_context_cache)
    if context_cache.is_file():
        context = torch.load(
            context_cache, map_location=device, weights_only=True
        )["context"]
    else:
        context = build_empty_context(code_dir, checkpoint_dir, config, device)
        context_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"context": context.cpu(), "prompt": "", "model": "umt5-xxl"},
            context_cache,
        )
        context = context.to(device)
    vae = WanVAE(
        vae_pth=str(checkpoint_dir / config.vae_checkpoint),
        dtype=torch.bfloat16,
        device=device,
    )
    model = WanModel.from_pretrained(checkpoint_dir).eval().requires_grad_(False)
    # Preserve the checkpoint's mixed parameter dtypes; official Wan also only
    # moves this module to CUDA and relies on BF16 autocast for heavy operators.
    model.to(device=device)

    records = unique_records(args.selection_manifest)
    if args.limit:
        records = records[:args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_cache = {}
    completed = []
    torch.cuda.reset_peak_memory_stats(device)
    for record in tqdm(records, desc=f"Wan block {args.block} teacher"):
        source_paths = tuple(record["image_paths"])
        # 使用原 manifest 的 group id 与其他 prior 配对；实际输入固定为
        # expected_frames，故 12-frame 源样本在32帧模式同样可用。
        paths = tuple(normalize_frame_paths(source_paths, args.expected_frames))
        identifier = frame_group_id(source_paths)
        output_path = output_dir / f"{identifier}.pt"
        if not output_path.is_file():
            video = load_video(paths, args.height, args.width, device, args.expected_frames)
            clean = vae.encode([video])[0]
            shape = tuple(clean.shape)
            if shape not in noise_cache:
                generator = torch.Generator(device=device).manual_seed(args.noise_seed)
                noise_cache[shape] = torch.randn(
                    shape, generator=generator, device=device, dtype=torch.float32
                )
            sigma = args.timestep / 1000.0
            noisy = ((1.0 - sigma) * clean + sigma * noise_cache[shape]).to(torch.bfloat16)
            timestep = torch.tensor([args.timestep], device=device, dtype=torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feature = forward_to_block(model, noisy, context, timestep, args.block)
            # [F_latent,H_latent/2,W_latent/2,C] -> [4,14,18,C].
            aligned = F.interpolate(
                feature.permute(3, 0, 1, 2).unsqueeze(0).float(),
                size=(args.expected_frames, args.target_height, args.target_width),
                mode="trilinear",
                align_corners=False,
            )[0].permute(1, 2, 3, 0).contiguous().to(torch.bfloat16).cpu()
            expected = (args.expected_frames, args.target_height, args.target_width, 1536)
            if tuple(aligned.shape) != expected:
                raise ValueError(f"Wan feature shape={tuple(aligned.shape)}, expected={expected}")
            if not torch.isfinite(aligned.float()).all():
                raise FloatingPointError(f"Wan feature contains NaN/Inf: {identifier}")
            temporary = output_path.with_suffix(".pt.tmp")
            torch.save({
                "wan_feature": aligned,
                "model": "Wan2.1-T2V-1.3B",
                "dit_block_1based": args.block,
                "timestep": args.timestep,
                "noise": {"kind": "shared_fixed_template", "seed": args.noise_seed},
                "text_condition": "empty_prompt",
                "source_frames": args.expected_frames,
                "vae_frames": int(video.shape[1]),
                "frame_padding": "repeat_last",
                "input_size": [args.height, args.width],
                "target_grid": [args.expected_frames, args.target_height, args.target_width],
                "image_paths": list(paths),
                "original_image_paths": list(source_paths),
            }, temporary)
            os.replace(temporary, output_path)
        completed.append(identifier)
    (output_dir / "collection_summary.json").write_text(json.dumps({
        "groups": len(completed),
        "dit_block_1based": args.block,
        "timestep": args.timestep,
        "peak_cuda_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }, indent=2), encoding="utf-8")
    print(f"Completed {len(completed)} groups at {output_dir}")


if __name__ == "__main__":
    main()
