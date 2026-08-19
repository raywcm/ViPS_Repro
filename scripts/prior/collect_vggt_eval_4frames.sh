#!/usr/bin/env bash
# Extract only the VGGT F_base needed by ViPS evaluation splits.
# Supports VIPS_AGG_BLOCK (default 24); output dir gets _agg<block> suffix when !=24.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MLLM_ROOT="$(cd "$ROOT/../.." && pwd)"
PY="${PY:-python}"
FRAMES="${VIPS_FRAMES:-4}"
AGG_BLOCK="${VIPS_AGG_BLOCK:-24}"
SUFFIX=""; [ "$AGG_BLOCK" != "24" ] && SUFFIX="_agg${AGG_BLOCK}"
OUT="${VIPS_EVAL_CACHE_ROOT:-$ROOT/prior/3drs_vips_eval_${FRAMES}frames${SUFFIX}}"
MANIFEST="$OUT/manifest.jsonl"
LIMIT="${VIPS_LIMIT:-0}"
PERCENT="${VIPS_EVAL_PERCENT:-100}"
SEED="${VIPS_EVAL_SEED:-0}"
VGGT_CHECKPOINT="${VIPS_VGGT_CHECKPOINT:-$MLLM_ROOT/models/VGGT-1B/model.pt}"

mkdir -p "$OUT"
"$PY" "$ROOT/scripts/prior/build_3drs_vips_manifest.py" \
  --data-root "$ROOT/data" \
  --embodiedscan-folder "$ROOT/data/embodiedscan" \
  --annotation-files \
    "$ROOT/data/processed/scanrefer_vg_val_llava_style.json" \
    "$ROOT/data/processed/multi3drefer_val_llava_style.json" \
    "$ROOT/data/processed/scan2cap_val_llava_style.json" \
    "$ROOT/data/processed/scanqa_val_llava_style.json" \
    "$ROOT/data/processed/sqa3d_test_llava_style.json" \
  --frames "$FRAMES" --percent "$PERCENT" --seed "$SEED" --output "$MANIFEST"

"$PY" "$ROOT/scripts/prior/collect_vggt_3drs_vips.py" \
  --manifest "$MANIFEST" \
  --checkpoint "$VGGT_CHECKPOINT" \
  --output-dir "$OUT/vggt" \
  --frames "$FRAMES" \
  --agg-block "$AGG_BLOCK" \
  --limit "$LIMIT"

echo "Evaluation VGGT cache complete: $OUT/vggt"
