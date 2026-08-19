#!/usr/bin/env bash
set -euo pipefail

# Evaluate a trained output on ScanRefer/Multi3DRefer/Scan2Cap/ScanQA/SQA3D.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/transformers/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXP="${EXP:-}"
if [ -z "$EXP" ]; then
  echo "usage: EXP=outputs/<experiment> FRAMES=4|8 bash run/04_eval.sh" >&2
  exit 1
fi

PY="${PY:-/home/user/miniconda3/envs/3drs/bin/python}"
BASE="${BASE:-../../models/Aquila-VL-2B-llava-qwen}"
PROXY="${PROXY:-outputs/vips_proxy_selected_v2_4frames/best.pt}"
FRAMES="${FRAMES:-8}"
if [ "$FRAMES" = "8" ]; then
  CACHE="${CACHE:-prior/3drs_vips_eval_8frames/vggt}"
else
  CACHE="${CACHE:-prior/3drs_vips_eval_4frames_agg17/vggt}"
fi

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  DATA_PERCENT="${DATA_PERCENT:-1}"
fi

VIPS_PY="$PY" VIPS_FRAMES="$FRAMES" bash scripts/3d/eval/eval_vips_5tasks.sh \
  "$EXP" "$BASE" "$PROXY" "$CACHE" "${DATA_PERCENT:-10}" 2>&1 | tee "$EXP/eval.log"
