#!/usr/bin/env bash
# Build one 3DRS-aligned ViPS teacher cache.  The model-specific collectors
# live under scripts/prior in this repo; override VIPS_COLLECTOR_DIR if you
# keep them elsewhere.  The common output grid is 14x14.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MLLM_ROOT="$(cd "$ROOT/../.." && pwd)"
LEGACY="${VIPS_COLLECTOR_DIR:-$ROOT/scripts/prior}"
PY="${PY:-python}"
FRAMES="${VIPS_FRAMES:-4}"
OUT="${VIPS_CACHE_ROOT:-$ROOT/prior/3drs_vips_${FRAMES}frames}"
MANIFEST="$OUT/manifest.jsonl"
LIMIT="${VIPS_LIMIT:-0}"
DATASET_TAG="${VIPS_DATASET_TAG:-3drs_vips_${FRAMES}frames}"
RADIO_BLOCK="${VIPS_RADIO_BLOCK:-}"
VGGT_CHECKPOINT="${VIPS_VGGT_CHECKPOINT:-$MLLM_ROOT/models/VGGT-1B/model.pt}"
AGG_BLOCK="${VIPS_AGG_BLOCK:-24}"

test -f "$LEGACY/collect_da3_last_layer_features.py" || { echo "Missing collectors: $LEGACY" >&2; exit 1; }
mkdir -p "$OUT"
"$PY" "$ROOT/scripts/prior/build_3drs_vips_manifest.py" \
  --data-yaml "$ROOT/scripts/3d/train/multi.yaml" --data-root "$ROOT/data" \
  --embodiedscan-folder "$ROOT/data/embodiedscan" --frames "$FRAMES" --output "$MANIFEST"

"$PY" "$ROOT/scripts/prior/collect_vggt_3drs_vips.py" \
  --manifest "$MANIFEST" --checkpoint "$VGGT_CHECKPOINT" \
  --output-dir "$OUT/vggt" --frames "$FRAMES" --agg-block "$AGG_BLOCK" --limit "$LIMIT"

"$PY" "$LEGACY/collect_da3_last_layer_features.py" \
  --input-manifest "$MANIFEST" --output-dir "$OUT/da3" --dataset-tag "$DATASET_TAG" \
  --feature-source backbone_layer --feature-layer 23 --expected-frames "$FRAMES" \
  --target-height 14 --target-width 14 --deduplicate-by-frame-group --limit "$LIMIT"

if [[ -n "$RADIO_BLOCK" ]]; then
  "$PY" "$LEGACY/collect_radio_block_spatial_features.py" \
    --selection-manifest "$MANIFEST" --output-dir "$OUT/radio" \
    --radio-block "$RADIO_BLOCK" --expected-frames "$FRAMES" \
    --target-height 14 --target-width 14 --limit "$LIMIT"
else
  "$PY" "$LEGACY/collect_radio_final_spatial_features.py" \
    --selection-manifest "$MANIFEST" --output-dir "$OUT/radio" \
    --expected-frames "$FRAMES" --target-height 14 --target-width 14 --limit "$LIMIT"
fi

"$PY" "$LEGACY/collect_trace_final_decoder_features.py" \
  --selection-manifest "$MANIFEST" --output-dir "$OUT/trace" --feature-source final_decoder \
  --expected-frames "$FRAMES" --target-height 14 --target-width 14 --limit "$LIMIT"

"$PY" "$LEGACY/collect_wan_dit_features.py" \
  --selection-manifest "$MANIFEST" --output-dir "$OUT/wan" --block 20 \
  --expected-frames "$FRAMES" --target-height 14 --target-width 14 --limit "$LIMIT"

echo "Cache extraction complete. DA3 output is nested under $OUT/da3; use that generated directory for Stage 1."
