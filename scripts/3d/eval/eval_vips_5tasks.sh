#!/usr/bin/env bash
# Evaluate a ViPS LoRA checkpoint on all five ScanNet-series tasks.
# Usage:
#   bash scripts/3d/eval/eval_vips_5tasks.sh OUTPUT_DIR BASE_MODEL PROXY_PATH VGGT_CACHE [EVAL_PERCENT]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${1:?Usage: $0 OUTPUT_DIR BASE_MODEL PROXY_PATH VGGT_CACHE}"
BASE_MODEL="${2:?Missing BASE_MODEL}"
PROXY_PATH="${3:?Missing PROXY_PATH}"
VGGT_CACHE="${4:?Missing VGGT_CACHE}"
EVAL_PERCENT="${5:-${VIPS_EVAL_PERCENT:-100}}"
EVAL_SEED="${VIPS_EVAL_SEED:-0}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"
PY="${VIPS_PY:-python}"
FRAMES="${VIPS_FRAMES:-4}"
PERCENT_TAG="${EVAL_PERCENT//./p}pct"
RESULT_DIR="${OUT_DIR}/evaluation_${FRAMES}frames_${PERCENT_TAG}_seed${EVAL_SEED}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$RESULT_DIR"
export TOKENIZERS_PARALLELISM=false

common=(
  --model-path "$BASE_MODEL"
  --lora-path "$OUT_DIR"
  --vips-proxy-path "$PROXY_PATH"
  --vips-vggt-cache "$VGGT_CACHE"
  --video-folder data
  --embodiedscan-folder data/embodiedscan
  --conv-mode qwen_1_5
  --n_gpu 1
  --frame_sampling_strategy uniform
  --max_frame_num "$FRAMES"
  --force_sample True
  --torch-dtype "$EVAL_DTYPE"
  --eval-percent "$EVAL_PERCENT"
  --eval-seed "$EVAL_SEED"
)

run_task() {
  local name="$1"
  local model_script="$2"
  local question_file="$3"
  local metric_script="$4"
  local answer_file="${RESULT_DIR}/${name}.jsonl"
  if [[ -e "$answer_file" || -e "${RESULT_DIR}/${name}_metrics.txt" ]]; then
    echo "Overwriting existing ${name} evaluation files under ${RESULT_DIR}"
    rm -f -- "$answer_file" "${RESULT_DIR}/${name}_metrics.txt"
  fi
  echo "========== Evaluating ${name} =========="
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$model_script" \
    "${common[@]}" \
    --question-file "$question_file" \
    --answer-file "$answer_file"
  "$PY" "$metric_script" --input-file "$answer_file" | tee "${RESULT_DIR}/${name}_metrics.txt"
}

run_task scanrefer \
  llava/eval/model_scanrefer.py \
  data/processed/scanrefer_vg_val_llava_style.json \
  llava/eval/eval_scanrefer.py

run_task multi3drefer \
  llava/eval/model_multi3drefer.py \
  data/processed/multi3drefer_val_llava_style.json \
  llava/eval/eval_multi3drefer.py

run_task scan2cap \
  llava/eval/model_scan2cap.py \
  data/processed/scan2cap_val_llava_style.json \
  llava/eval/eval_scan2cap.py

run_task scanqa \
  llava/eval/model_scanqa.py \
  data/processed/scanqa_val_llava_style.json \
  llava/eval/eval_scanqa.py

run_task sqa3d \
  llava/eval/model_sqa3d.py \
  data/processed/sqa3d_test_llava_style.json \
  llava/eval/eval_sqa3d.py

echo "========== Standard metrics (Acc/F1/C@0.5/CIDEr/EM) =========="
"$PY" llava/eval/eval_standard_metrics.py \
  --eval-dir "$RESULT_DIR" \
  | tee "${RESULT_DIR}/standard_metrics.txt"

echo "========== Done =========="
echo "Predictions and metrics: ${RESULT_DIR}"
