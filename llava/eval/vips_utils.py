"""Shared ViPS evaluation support for the five ScanNet-series evaluators."""

from __future__ import annotations

from pathlib import Path
import math
import random

import torch
from transformers import AutoConfig, AutoTokenizer


def add_vips_eval_arguments(parser):
    parser.add_argument("--vips-proxy-path", type=str, default=None)
    parser.add_argument("--vips-vggt-cache", type=str, default=None)
    parser.add_argument(
        "--torch-dtype", choices=("float16", "bfloat16"), default="bfloat16",
        help="Model and video tensor dtype used during evaluation.",
    )
    parser.add_argument(
        "--eval-percent", type=float, default=100.0,
        help="Deterministically evaluate this percentage of the task split (default: 100).",
    )
    parser.add_argument("--eval-seed", type=int, default=0)


def align_eval_vocab_config(config, base_model_path):
    """Keep the base checkpoint's embedding size when adapter config is stale."""
    base_config = AutoConfig.from_pretrained(base_model_path)
    config["vocab_size"] = max(
        int(config.get("vocab_size", 0)), int(getattr(base_config, "vocab_size", 0))
    )
    return config


def resize_eval_embeddings(model, tokenizer):
    current = model.get_input_embeddings().num_embeddings
    target = max(int(current), int(len(tokenizer)))
    if target != current:
        model.resize_token_embeddings(target)


def resize_eval_embeddings_from_state(model, state_dict):
    """Resize embeddings to the saved non-LoRA state before loading it."""
    target_vocab = None
    for key in ("model.embed_tokens.weight", "embed_tokens.weight"):
        value = state_dict.get(key)
        if value is not None and value.ndim == 2:
            target_vocab = int(value.shape[0])
            break
    if target_vocab is not None and model.get_input_embeddings().num_embeddings != target_vocab:
        model.resize_token_embeddings(target_vocab)


def load_eval_tokenizer(lora_path, base_model_path):
    """Use the adapter tokenizer when saved, otherwise the base tokenizer.

    Trainer's final output directory may omit tokenizer files while numbered
    checkpoints contain them; the base Aquila tokenizer is equivalent for this
    case and has the correct 151936 vocabulary.
    """
    lora_path = Path(lora_path)
    if (lora_path / "tokenizer_config.json").is_file():
        tokenizer_path = lora_path
    else:
        # Final Trainer output directories may omit tokenizer files, while
        # checkpoint-N contains the exact tokenizer with <ground>/<coord>.
        checkpoints = sorted(
            (path for path in lora_path.glob("checkpoint-*") if (path / "tokenizer_config.json").is_file()),
            key=lambda path: int(path.name.split("-")[-1]),
            reverse=True,
        )
        tokenizer_path = checkpoints[0] if checkpoints else Path(base_model_path)
    print(f"Evaluation tokenizer: {tokenizer_path}")
    return AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=False)


def select_eval_subset(records, percent, seed, task_name):
    """Stable random subset for cheap, comparable validation runs."""
    percent = float(percent)
    if not 0.0 < percent <= 100.0:
        raise ValueError(f"--eval-percent must be in (0, 100], got {percent}")
    total = len(records)
    count = total if percent >= 100.0 else max(1, math.ceil(total * percent / 100.0))
    if count == total:
        print(f"Evaluation subset {task_name}: {total}/{total} (100%)")
        return records
    indices = sorted(random.Random(int(seed)).sample(range(total), count))
    print(f"Evaluation subset {task_name}: {count}/{total} ({percent:g}%, seed={seed})")
    return [records[index] for index in indices]


def _frame_key(paths):
    return tuple(
        str(path).replace("\\\\", "/").split("posed_images/")[-1]
        for path in paths
    )


def build_vips_feature_index(cache_dir):
    """Index offline VGGT f_base tensors by their exact sampled frame group."""
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"ViPS VGGT cache directory not found: {cache_dir}")
    index = {}
    for feature_path in cache_dir.rglob("*.pt"):
        record = torch.load(feature_path, map_location="cpu", weights_only=True)
        paths = record.get("image_paths") or record.get("frame_paths")
        if paths and "f_base" in record:
            index[_frame_key(paths)] = feature_path
    if not index:
        raise RuntimeError(f"No valid ViPS f_base cache files under {cache_dir}")
    print(f"Loaded ViPS evaluation VGGT index: {len(index)} frame groups")
    return index


def attach_vips_f_base(video_dict, feature_index):
    """Attach [1,F,14,14,2048] offline f_base to a merged one-sample video dict."""
    frame_files = video_dict.get("frame_files")
    if not frame_files or len(frame_files) != 1:
        raise ValueError("ViPS evaluation expects one merged video sample with frame_files")
    key = _frame_key(frame_files[0])
    feature_path = feature_index.get(key)
    if feature_path is None:
        raise KeyError(f"No ViPS VGGT cache for evaluation frames: {key}")
    feature = torch.load(feature_path, map_location="cpu", weights_only=True)["f_base"]
    if feature.ndim == 3 and feature.shape[1] == 14 * 14:
        feature = feature.view(feature.shape[0], 14, 14, feature.shape[-1])
    if feature.ndim != 4 or tuple(feature.shape[1:]) != (14, 14, 2048):
        raise ValueError(f"Invalid ViPS F_base {tuple(feature.shape)} from {feature_path}")
    video_dict["vips_f_base"] = feature.unsqueeze(0)


def initialize_vips_for_eval(model, args):
    """Materialize Stage-2 modules then load their saved non-LoRA weights.

    The regular LLaVA loader reads ``non_lora_trainables.bin`` before these
    runtime-created ViPS modules exist.  Reloading it after initialization is
    therefore required for an actual ViPS evaluation.
    """
    if not args.vips_proxy_path or not args.vips_vggt_cache:
        raise ValueError(
            "A ViPS checkpoint requires both --vips-proxy-path and "
            "--vips-vggt-cache during evaluation."
        )
    config = model.config
    if not getattr(config, "use_vips", False):
        raise ValueError("This checkpoint is not marked use_vips=True")
    core = model.get_model()
    core.initialize_vips(
        args.vips_proxy_path,
        layers=tuple(config.vips_injection_layers),
        share_gate=bool(config.vips_share_gate),
        experts=tuple(config.vips_experts),
        fusion_mode=getattr(config, "vips_fusion_mode", "gate"),
        include_null=bool(getattr(config, "vips_include_null", False)),
        gate_cond=getattr(config, "vips_gate_cond", "last"),
        gate_bias=getattr(config, "vips_gate_bias", None),
        gate_temp=float(getattr(config, "vips_gate_temp", 1.0) or 1.0),
        aux_routing=float(getattr(config, "vips_aux_routing", 0.0) or 0.0),
        task_embed=bool(getattr(config, "vips_task_embed", False)),
        hybrid_beta_per_dataset=bool(getattr(config, "vips_hybrid_beta_per_dataset", False)),
        layer_gate=bool(getattr(config, "vips_layer_gate", False)),
        layer_gate_per_dataset=bool(getattr(config, "vips_layer_gate_per_dataset", False)),
    )

    lora_path = Path(args.lora_path) if args.lora_path else None
    state_path = lora_path / "non_lora_trainables.bin" if lora_path else None
    if state_path is None or not state_path.is_file():
        raise FileNotFoundError("ViPS evaluation needs --lora-path/non_lora_trainables.bin")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state = {(key[11:] if key.startswith("base_model.") else key): value for key, value in state.items()}
    if any(key.startswith("model.model.") for key in state):
        state = {(key[6:] if key.startswith("model.") else key): value for key, value in state.items()}
    resize_eval_embeddings_from_state(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_vips = [key for key in missing if "vips_fusion" in key]
    unexpected_vips = [key for key in unexpected if "vips_fusion" in key]
    if missing_vips or unexpected_vips:
        raise RuntimeError(
            f"Failed to restore ViPS fusion: missing={missing_vips[:3]}, "
            f"unexpected={unexpected_vips[:3]}"
        )
    print(
        f"ViPS evaluation enabled: experts={tuple(config.vips_experts)}, "
        f"layers={tuple(config.vips_injection_layers)}"
    )
    return build_vips_feature_index(args.vips_vggt_cache)
