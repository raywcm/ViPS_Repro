#!/usr/bin/env bash
# 4帧 feature-layer sweep：VGGT/DA3/RADIO/Trace/Wan 各候选层
# 每层：collect(如缓存缺失) -> train proxy(50 epochs, lr 1e-4, val 5%)
# 用法：bash sweep_expert_feature_layers_4frames.sh [expert]
set -uo pipefail
ROOT="${VIPS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MLLM_ROOT="${MLLM_ROOT:-$(cd "$ROOT/../.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-$ROOT}"
VIPS_DIR="${VIPS_DIR:-$ROOT}"
PRIOR="${PRIOR:-$ROOT/prior}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs}"
PY="${PY:-/home/user/miniconda3/envs/3drs/bin/python}"
MANIFEST="${MANIFEST:-$PRIOR/scannet_series_5tasks_100pct_4frames_len2048_seed0_manifest_unique.jsonl}"
EXPERT=${1:-all}

# 固定参照缓存（其他专家用代表层）
VGGT_FIX=$PRIOR/vggt_14x18_scannet_series_100pct_4frames_unique
DA3_FIX=$PRIOR/da3_sweep_scannet_series_100pct_4frames/da3_large_1_1_layer23_grid14x18_scannet_series_100pct_4frames_all
RADIO_FIX=$PRIOR/radio_v2_5_l_final_spatial_grid14x18_scannet_series_100pct_4frames_unique
TRACE_FIX=$PRIOR/trace_sweep_scannet_series_100pct_4frames/T3-final_decoder
WAN_FIX=$PRIOR/wan2.1_t2v1.3b_block20_t300_grid14x18_scannet_series_100pct_4frames_unique

COMMON_PROXY=(
  --selection-manifest "$MANIFEST"
  --vggt-dir "$VGGT_FIX"
  --da3-dir "$DA3_FIX"
  --radio-dir "$RADIO_FIX"
  --trace-dir "$TRACE_FIX"
  --wan-dir "$WAN_FIX"
  --frames 4 --epochs 50 --batch-size 4 --num-workers 4
  --lr 1e-4 --weight-decay 0.01 --cosine-weight 0
  --val-ratio 0.05 --seed 0
)

run_proxy() {  # $1=expert_name  $2=expert_dir  $3=layer_name  $4=expert_key
  local expert=$1 dir=$2 layer=$3 key=$4 out
  out="$OUT_ROOT/vips_proxy_${expert}_sweep_4frames/$layer"
  [ -f "$out/best.pt" ] && { echo "[skip proxy] $expert/$layer (best.pt exists)"; return 0; }
  mkdir -p "$out"
  echo "===== PROXY $expert/$layer ====="
  "$PY" "$PROJECT_DIR/scripts/prior/train_efficient_prior_proxy.py" \
    "${COMMON_PROXY[@]}" --output-dir "$out" "$key" "$dir" 2>&1 | tail -3
}

# ---------- VGGT ----------
sweep_vggt() {
  local agg_dir
  for agg in 4 11; do
    agg_dir="$PRIOR/vggt_agg${agg}_grid14x18_scannet_series_100pct_4frames_unique"
    if [ ! -f "$agg_dir/manifest.jsonl" ] || [ ! -s "$agg_dir/manifest.jsonl" ]; then
      echo "===== COLLECT VGGT agg$agg ====="
      (cd "$VIPS_DIR" && "$PY" scripts/prior/collect_vggt_3drs_vips.py \
        --manifest "$MANIFEST" --checkpoint "$MLLM_ROOT/models/VGGT-1B/model.pt" \
        --output-dir "$agg_dir" --frames 4 --agg-block "$agg" \
        --target-height 14 --target-width 18)
    fi
    run_proxy vggt "$agg_dir" "agg$agg" --vggt-dir
  done
}

# ---------- DA3 ----------
sweep_da3() {
  local d
  for layer in 7 11 15 19; do
    d="$PRIOR/da3_sweep_scannet_series_100pct_4frames/da3_large_1_1_layer${layer}_grid14x18_scannet_series_100pct_4frames_all"
    if [ ! -f "$d/manifest.jsonl" ] || [ ! -s "$d/manifest.jsonl" ]; then
      echo "===== COLLECT DA3 layer$layer ====="
      (cd "$PROJECT_DIR" && "$PY" scripts/prior/collect_da3_last_layer_features.py \
        --feature-source backbone_layer --feature-layer "$layer" \
        --input-manifest "$MANIFEST" --output-dir "$PRIOR/da3_sweep_scannet_series_100pct_4frames" \
        --target-height 14 --target-width 18 --process-res 504 --expected-frames 4 \
        --dataset-tag scannet_series_100pct_4frames --deduplicate-by-frame-group)
    fi
    run_proxy da3 "$d" "bb$layer" --da3-dir
  done
  # 已有缓存层：bb9/bb23/dpt_preout 只训练 proxy
  run_proxy da3 "$PRIOR/da3_sweep_scannet_series_100pct_4frames/da3_large_1_1_layer9_grid14x18_scannet_series_100pct_4frames_all" bb9 --da3-dir
  run_proxy da3 "$PRIOR/da3_sweep_scannet_series_100pct_4frames/da3_large_1_1_layer23_grid14x18_scannet_series_100pct_4frames_all" bb23 --da3-dir
  run_proxy da3 "$PRIOR/da3_large_1_1_depth_output_conv1_v2_grid14x18_scannet_series_100pct_4frames_all" dpt_preout --da3-dir
}

# ---------- RADIO ----------
sweep_radio() {
  local d
  for blk in 5 6 9 11 17 23; do
    d="$PRIOR/radio_v2_5_l_block${blk}_spatial_grid14x18_scannet_series_100pct_4frames_unique"
    if [ ! -f "$d/manifest.jsonl" ] || [ ! -s "$d/manifest.jsonl" ]; then
      echo "===== COLLECT RADIO block$blk ====="
      (cd "$PROJECT_DIR" && "$PY" scripts/prior/collect_radio_block_spatial_features.py \
        --selection-manifest "$MANIFEST" --output-dir "$d" \
        --radio-block "$blk" --target-height 14 --target-width 18 \
        --expected-frames 4)
    fi
    run_proxy radio "$d" "block$blk" --radio-dir
  done
  run_proxy radio "$RADIO_FIX" final_spatial --radio-dir
}

# ---------- Trace ----------
sweep_trace() {
  local d src
  for n in 5 11 17 23; do
    src="encoder_block$n"; d="$PRIOR/trace_sweep_scannet_series_100pct_4frames/$src"
    if [ ! -f "$d/manifest.jsonl" ] || [ ! -s "$d/manifest.jsonl" ]; then
      echo "===== COLLECT Trace $src ====="
      (cd "$PROJECT_DIR" && "$PY" scripts/prior/collect_trace_final_decoder_features.py \
        --selection-manifest "$MANIFEST" --output-dir "$d" --feature-source "$src" \
        --expected-frames 4 --target-height 14 --target-width 18)
    fi
    run_proxy trace "$d" "enc$n" --trace-dir
  done
  for n in 5 11 17 23; do
    src="decoder_block$n"; d="$PRIOR/trace_sweep_scannet_series_100pct_4frames/$src"
    if [ ! -f "$d/manifest.jsonl" ] || [ ! -s "$d/manifest.jsonl" ]; then
      echo "===== COLLECT Trace $src ====="
      (cd "$PROJECT_DIR" && "$PY" scripts/prior/collect_trace_final_decoder_features.py \
        --selection-manifest "$MANIFEST" --output-dir "$d" --feature-source "$src" \
        --expected-frames 4 --target-height 14 --target-width 18)
    fi
    run_proxy trace "$d" "fusion$n" --trace-dir
  done
  d="$PRIOR/trace_sweep_scannet_series_100pct_4frames/track_dpt_refinenet2"
  if [ ! -f "$d/manifest.jsonl" ] || [ ! -s "$d/manifest.jsonl" ]; then
    echo "===== COLLECT Trace track_dpt_refinenet2 ====="
    (cd "$PROJECT_DIR" && "$PY" scripts/prior/collect_trace_final_decoder_features.py \
      --selection-manifest "$MANIFEST" --output-dir "$d" --feature-source track_dpt_refinenet2 \
      --expected-frames 4 --target-height 14 --target-width 18)
  fi
  run_proxy trace "$d" dpt_mid --trace-dir
  # fusion_postnorm = final_decoder(T3)；dpt_late = refinenet1(T5) 已有缓存
  run_proxy trace "$TRACE_FIX" fusion_postnorm --trace-dir
  run_proxy trace "$PRIOR/trace_sweep_scannet_series_100pct_4frames/T5-track_dpt_refinenet1" dpt_late --trace-dir
}

# ---------- Wan ----------
sweep_wan() {
  local d
  for blk in 4 9 19 24 29; do
    d="$PRIOR/wan2.1_t2v1.3b_block${blk}_t300_grid14x18_scannet_series_100pct_4frames_unique"
    if [ ! -f "$d/manifest.jsonl" ] || [ ! -s "$d/manifest.jsonl" ]; then
      echo "===== COLLECT Wan block$blk ====="
      (cd "$PROJECT_DIR" && "$PY" scripts/prior/collect_wan_dit_features.py \
        --selection-manifest "$MANIFEST" --output-dir "$d" \
        --block "$blk" --expected-frames 4 --target-height 14 --target-width 18)
    fi
    run_proxy wan "$d" "block$blk" --wan-dir
  done
  run_proxy wan "$PRIOR/wan2.1_t2v1.3b_block14_t300_grid14x18_scannet_series_100pct_4frames_unique" block14 --wan-dir
  run_proxy wan "$PRIOR/wan2.1_t2v1.3b_block20_t300_grid14x18_scannet_series_100pct_4frames_unique" block20 --wan-dir
}

case "$EXPERT" in
  vggt) sweep_vggt ;;
  da3) sweep_da3 ;;
  radio) sweep_radio ;;
  trace) sweep_trace ;;
  wan) sweep_wan ;;
  all)
    sweep_vggt
    sweep_da3
    sweep_radio
    sweep_trace
    sweep_wan
    ;;
  *) echo "unknown expert: $EXPERT"; exit 1 ;;
esac
echo "SWEEP DONE: $EXPERT"
