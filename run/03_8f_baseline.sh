#!/usr/bin/env bash
set -euo pipefail

# 8f 3DRS baseline without ViPS.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/transformers/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PY="${PY:-/home/user/miniconda3/envs/3drs/bin/python}"
BASE="${BASE:-../../models/Aquila-VL-2B-llava-qwen}"
VT="${VT:-../../models/siglip-so400m-patch14-384}"
OUT="${OUT:-outputs/baseline_8f}"
SMOKE="${SMOKE:-0}"

EXTRA=""
if [ "$SMOKE" = "1" ]; then
  EXTRA="--max_steps 3"
fi

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" llava/train/train_3d.py \
  --model_name_or_path "$BASE" --version qwen_1_5 \
  --data_path scripts/3d/train/multi.yaml --image_folder data --video_folder data \
  --embodiedscan_folder data/embodiedscan --vision_tower "$VT" \
  --mm_tunable_parts mm_language_model --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 --mm_patch_merge_type spatial_unpad \
  --mm_spatial_pool_stride 2 --mm_spatial_pool_mode bilinear --mm_newline_position grid \
  --bits 16 --lora_enable True --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --bf16 True --fp16 False --gradient_checkpointing True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --dataloader_num_workers 2 --model_max_length 2048 --frames_upbound 8 --force_sample True \
  --lazy_preprocess True --vision_chunk_size 8 \
  --world_position_embedding_type avg-discrete-sin3d --object_feature_type patch14-pe \
  --ground_head_type infonce --ground_loss_scale 1 --disable_3d_alignment True \
  --dataset_percent 10 --dataset_seed 0 --seed 0 \
  --num_train_epochs 1 --max_steps -1 --save_steps 2000 --save_strategy steps \
  --logging_steps 1 --report_to none $EXTRA --output_dir "$OUT" 2>&1 | tee "$OUT/train.log"
