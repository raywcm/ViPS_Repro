#!/usr/bin/env bash
set -euo pipefail

# Train the shared selected_v2 prior proxy (Stage-1).
# Teacher features and manifest live under $PRIOR (default: one level above this repo).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/transformers/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PY="${PY:-/home/user/miniconda3/envs/3drs/bin/python}"
PRIOR="${PRIOR:-$ROOT/../..}/prior"
OUT="${OUT:-outputs/vips_proxy_selected_v2_4frames}"
SMOKE="${SMOKE:-0}"

EXTRA=""
if [ "$SMOKE" = "1" ]; then
  EXTRA="--max-groups 8 --epochs 1 --num-workers 0"
fi

mkdir -p "$OUT"
"$PY" scripts/prior/train_efficient_prior_proxy.py \
  --selection-manifest "$PRIOR/scannet_series_5tasks_100pct_4frames_len2048_seed0_manifest_unique.jsonl" \
  --vggt-dir "$PRIOR/vggt_agg17_grid14x18_scannet_series_100pct_4frames_unique" \
  --da3-dir "$PRIOR/da3_sweep_scannet_series_100pct_4frames/da3_large_1_1_layer9_grid14x18_scannet_series_100pct_4frames_all" \
  --radio-dir "$PRIOR/radio_v2_5_l_block5_spatial_grid14x18_scannet_series_100pct_4frames_unique" \
  --trace-dir "$PRIOR/trace_sweep_scannet_series_100pct_4frames/decoder_block5" \
  --wan-dir "$PRIOR/wan2.1_t2v1.3b_block20_t300_grid14x18_scannet_series_100pct_4frames_unique" \
  --frames 4 --epochs 50 --batch-size 4 --num-workers 4 \
  --lr 1e-4 --weight-decay 0.01 --cosine-weight 0 --val-ratio 0.05 --seed 0 \
  $EXTRA --output-dir "$OUT" 2>&1 | tee "$OUT/train.log"
