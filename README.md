# ViPS-Repro

Unofficial lightweight reimplementation of ViPS for spatial understanding.

本项目为 ViPS 多视觉先验融合框架的轻量复现，基于 3DRS 代码框架进行开发。
主要实现了 Efficient Prior Proxy和 Dynamic Prior Fusion，
并在 ScanRefer、Multi3DRefer、Scan2Cap、ScanQA 和 SQA3D 上进行训练与评估。

由于计算资源限制，本项目采用较小的基础模型和 4/8 帧输入，

## How to run

## run/ *

| 脚本 | 用途 |
| --- | --- |
| `run/01_proxy.sh` | Stage-1：训练 selected_v2 proxy |
| `run/02_8f_main.sh` | 8f 主结果：五专家 Dynamic Fusion |
| `run/03_8f_baseline.sh` | 8f 主结果：3DRS baseline |
| `run/04_eval.sh` | 五任务评估（`EXP=outputs/<exp> FRAMES=4\|8`） |

其余实验没有单独脚本，直接用下文各节里的命令运行。

## 0. 环境

```bash
conda create -n vips python=3.10
conda activate vips

ROOT=./ViPS_Repro
PY=python3
cd "$ROOT"
pip install -r requirements.txt

export PYTHONPATH="$PWD/transformers/src:$PWD"
export VIPS_PY=$PY
```

通用变量：

```bash
BASE=../../models/Aquila-VL-2B-llava-qwen
VT=../../models/siglip-so400m-patch14-384
PROXY=outputs/vips_proxy_selected_v2_4frames/best.pt
TCACHE4=prior/3drs_vips_4frames_agg17/vggt
TCACHE8=prior/3drs_vips_8frames_agg17/vggt
ECACHE4=prior/3drs_vips_eval_4frames_agg17/vggt
ECACHE8=prior/3drs_vips_eval_8frames/vggt
```

## 1. Feature Layer sweep

用仓库内脚本扫候选层（可选参数 `EXPERT=all|vggt|da3|radio|trace|wan`）：

```bash
cd "$ROOT"
PRIOR="$ROOT/../.."/prior bash scripts/prior/sweep_expert_feature_layers_4frames.sh all
# 单专家补跑：PRIOR="$ROOT/../.."/prior bash scripts/prior/sweep_expert_feature_layers_4frames.sh vggt
```

最终选定层：`vggt=agg17 / da3=bb9 / radio=block5 / trace=fusion5 / wan=block20`。
默认 `PRIOR=$MLLM_ROOT/prior`（`MLLM_ROOT` 默认 `$ROOT/../..`）、
`MANIFEST=$PRIOR/scannet_series_5tasks_100pct_4frames_len2048_seed0_manifest_unique.jsonl`、
`OUT_ROOT=$ROOT/outputs`，均可用环境变量覆盖。sweep 会先检查对应层缓存，缺失时才提取。
若 `MANIFEST` 不存在，先运行 `scripts/prior/build_3drs_vips_manifest.py` 和
`build_unique_frame_groups.py` 生成。

## 2. selected Proxy 训练

Stage-1 的 teacher 特征与 manifest 放在仓库上一级的共享工作区
（如 `./code/prior`）。脚本里用 `STAGE1=../..`（相对仓库根）
定位它，也可以用 `PROXY_PRIOR` 覆盖。

```bash
cd "$ROOT"
STAGE1=../..
$PY scripts/prior/train_efficient_prior_proxy.py \
  --selection-manifest "$STAGE1/prior/scannet_series_5tasks_100pct_4frames_len2048_seed0_manifest_unique.jsonl" \
  --vggt-dir "$STAGE1/prior/vggt_agg17_grid14x18_scannet_series_100pct_4frames_unique" \
  --da3-dir "$STAGE1/prior/da3_sweep_scannet_series_100pct_4frames/da3_large_1_1_layer9_grid14x18_scannet_series_100pct_4frames_all" \
  --radio-dir "$STAGE1/prior/radio_v2_5_l_block5_spatial_grid14x18_scannet_series_100pct_4frames_unique" \
  --trace-dir "$STAGE1/prior/trace_sweep_scannet_series_100pct_4frames/decoder_block5" \
  --wan-dir "$STAGE1/prior/wan2.1_t2v1.3b_block20_t300_grid14x18_scannet_series_100pct_4frames_unique" \
  --frames 4 --epochs 50 --batch-size 4 --num-workers 4 \
  --lr 1e-4 --weight-decay 0.01 --cosine-weight 0 --val-ratio 0.05 --seed 0 \
  --output-dir outputs/vips_proxy_selected_v2_4frames
```

也可以直接跑 `bash run/01_proxy.sh`。

## 2.1 重建 prior 缓存

如果机器上还没有 Stage-1 teacher 特征或评估 VGGT 缓存，可按以下顺序重建：

```bash
# 五路 teacher 缓存（4f/8f 都用 agg17，输出目录名与 TCACHE4/TCACHE8 一致）
VIPS_AGG_BLOCK=17 VIPS_CACHE_ROOT=prior/3drs_vips_4frames_agg17 \
  bash scripts/prior/collect_3drs_vips_4frames.sh
VIPS_FRAMES=8 VIPS_AGG_BLOCK=17 VIPS_CACHE_ROOT=prior/3drs_vips_8frames_agg17 \
  bash scripts/prior/collect_3drs_vips_4frames.sh

# 评估集 VGGT 缓存（4f agg17、8f final agg24，输出目录名与 ECACHE4/ECACHE8 一致）
VIPS_AGG_BLOCK=17 bash scripts/prior/collect_vggt_eval_4frames.sh
VIPS_FRAMES=8 bash scripts/prior/collect_vggt_eval_4frames.sh
```

收集脚本内部依次调用 `build_3drs_vips_manifest.py`、`collect_vggt_3drs_vips.py` 和
`scripts/prior/` 下的五个专家收集器；`VIPS_LIMIT` 可限制帧组数做小样本验证，
`VIPS_RADIO_BLOCK` 可把 RADIO 换成指定 block。DA3/RADIO/Trace/Wan 收集器需要
外部模型代码与权重，默认路径是 `VIPS_WS/code/{Depth-Anything-3,RADIO,TraceAnything,Wan2.1}`
与 `VIPS_WS/models/...`（`VIPS_WS` 默认取仓库上一级共享工作区）。
完整性可用 `validate_3drs_vips_cache.py` 检查。

## 3. Stage-2 公共训练命令

先定义公共参数，后续实验只列差异：

```bash
COMMON="--model_name_or_path $BASE --version qwen_1_5 \
  --data_path scripts/3d/train/multi.yaml --image_folder data --video_folder data \
  --embodiedscan_folder data/embodiedscan --vision_tower $VT \
  --mm_tunable_parts mm_language_model --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 --mm_patch_merge_type spatial_unpad \
  --mm_spatial_pool_stride 2 --mm_spatial_pool_mode bilinear --mm_newline_position grid \
  --bits 16 --lora_enable True --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --bf16 True --fp16 False --gradient_checkpointing True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --dataloader_num_workers 2 --lazy_preprocess True \
  --world_position_embedding_type avg-discrete-sin3d --object_feature_type patch14-pe \
  --ground_head_type infonce --ground_loss_scale 1 --disable_3d_alignment True \
  --use_vips True --vips_proxy_path $PROXY --vips_vggt_cache $TCACHE4 \
  --vips_injection_layers 2,5,8,11,14 --vips_experts vggt,da3,radio,trace,wan \
  --vips_share_gate False --vips_stats_steps 10 \
  --dataset_percent 10 --dataset_seed 0 --seed 0 \
  --num_train_epochs 1 --max_steps -1 --save_steps 2000 --save_strategy steps \
  --logging_steps 1 --report_to none"

run_train() {  # $1=out_dir  其余为附加参数
  local out=$1; shift
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 $PY llava/train/train_3d.py $COMMON "$@" --output_dir "$out" \
    2>&1 | tee "$out/train.log"
}
```

## 4. 注入位置消融

独立 Gate：

```bash
run_train outputs/layers_early   --vips_injection_layers 1,2,3,4,5
run_train outputs/layers_block6  --vips_injection_layers 2,5,8,11,14
run_train outputs/layers_uniform --vips_injection_layers 3,8,13,18,23
run_train outputs/layers_late    --vips_injection_layers 13,16,19,22,25
```

Shared Gate 加 `--vips_share_gate True`。

这些消融没有单独脚本，直接改上面命令的 `--vips_injection_layers` 和
`--vips_share_gate` 即可。

## 5. Gate 稳定性

block6 五专家 DF：

```bash
run_train outputs/5exp_gate_block6_4f
```

low-cos 对照：

```bash
# Stage-1：训练 low-cos proxy（radio-dir 换成 block17）
$PY scripts/prior/train_efficient_prior_proxy.py \
  --selection-manifest "$STAGE1/prior/scannet_series_5tasks_100pct_4frames_len2048_seed0_manifest_unique.jsonl" \
  --vggt-dir "$STAGE1/prior/vggt_agg17_grid14x18_scannet_series_100pct_4frames_unique" \
  --da3-dir "$STAGE1/prior/da3_sweep_scannet_series_100pct_4frames/da3_large_1_1_layer9_grid14x18_scannet_series_100pct_4frames_all" \
  --radio-dir "$STAGE1/prior/radio_v2_5_l_block17_spatial_grid14x18_scannet_series_100pct_4frames_unique" \
  --trace-dir "$STAGE1/prior/trace_sweep_scannet_series_100pct_4frames/decoder_block5" \
  --wan-dir "$STAGE1/prior/wan2.1_t2v1.3b_block20_t300_grid14x18_scannet_series_100pct_4frames_unique" \
  --frames 4 --epochs 50 --batch-size 4 --num-workers 4 \
  --lr 1e-4 --weight-decay 0.01 --cosine-weight 0 --val-ratio 0.05 --seed 0 \
  --output-dir outputs/vips_proxy_lowcos_radio17_4frames

# Stage-2：用 low-cos proxy 训练
run_train outputs/lowcos_radio17_4f --vips_proxy_path outputs/vips_proxy_lowcos_radio17_4frames/best.pt
```

训练时每 10 步写 `vips_gate_stats.csv`，用 Gate Weight / Effective Contribution / Entropy 判断 collapse。

## 6. 8f 主结果

8f 必须加 `--frames_upbound 8 --vision_chunk_size 8 --model_max_length 2048 --force_sample True`
并把 `--vips_vggt_cache` 换成 `$TCACHE8`：

```bash
8F="--frames_upbound 8 --vision_chunk_size 8 --model_max_length 2048 --force_sample True --vips_vggt_cache $TCACHE8"

# No Prior（3DRS baseline，去掉 ViPS 参数）
mkdir -p outputs/baseline_8f
CUDA_VISIBLE_DEVICES=0 $PY llava/train/train_3d.py \
  --model_name_or_path $BASE --version qwen_1_5 \
  --data_path scripts/3d/train/multi.yaml --image_folder data --video_folder data \
  --embodiedscan_folder data/embodiedscan --vision_tower $VT \
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
  --logging_steps 1 --report_to none --output_dir outputs/baseline_8f \
  2>&1 | tee outputs/baseline_8f/train.log

# Direct Addition（mean 融合）
run_train outputs/vanilla_8f $8F --vips_fusion_mode mean

# Single VGGT
run_train outputs/single_vggt_8f $8F --vips_experts vggt

# Dynamic Fusion（五专家 gate）
run_train outputs/df_5exp_8f $8F
```

## 7. 8f Expert 消融

```bash
run_train outputs/subset2_8f $8F --vips_experts vggt,radio
run_train outputs/subset3_8f $8F --vips_experts vggt,da3,radio
run_train outputs/subset4_8f $8F --vips_experts vggt,da3,radio,trace
```

## 8. 4f 主结果与专家消融

```bash
# 4f 主结果（baseline 用第 6 节完整命令去掉 ViPS 参数；run_train 会带 --use_vips True）
run_train outputs/vanilla_4f    --vips_fusion_mode mean
run_train outputs/single_vggt_4f --vips_experts vggt
run_train outputs/early_block6_4f --vips_injection_layers 2,5,8,11,14

# 4f 专家消融
run_train outputs/subset2_4f --vips_experts vggt,radio
run_train outputs/subset3_4f --vips_experts vggt,da3,radio
run_train outputs/subset4_4f --vips_experts vggt,da3,radio,trace
```

## 9. 轻量几何先验

先收集 SigLIP 多层特征，再训练残差适配器并生成伪 VGGT 缓存：

```bash
# 1) 收集 SigLIP 第 6/13/20/26 层（train/eval 各一份）
$PY scripts/3d/prior/collect_siglip_layers.py \
  --vggt-dir prior/3drs_vips_4frames_agg17/vggt \
  --output-dir prior/native_geo_siglip_layers \
  --siglip-model $VT --layers 6,13,20,26 --batch-size 8

# 2) 训练 Geometry Residual Adapter（teacher = VGGT agg17）
$PY scripts/3d/prior/train_geometry_residual_adapter.py \
  --siglip-dir prior/native_geo_siglip_layers \
  --vggt-dir prior/3drs_vips_4frames_agg17/vggt \
  --output-dir outputs/geometry_residual_adapter \
  --epochs 50 --batch-size 4 --lr 1e-4 --val-ratio 0.05 --seed 0 --num-workers 4 \
  --alpha 1.0 --residual-loss 0.1

# 3) 生成 A（native）/ B（native + alpha*delta）训练缓存
$PY scripts/3d/prior/gen_residual_caches.py \
  --siglip-dir prior/native_geo_siglip_layers \
  --output-dir prior/geometry_residual_cache_A --mode A
$PY scripts/3d/prior/gen_residual_caches.py \
  --siglip-dir prior/native_geo_siglip_layers \
  --output-dir prior/geometry_residual_cache_B --mode B \
  --adapter-ckpt outputs/geometry_residual_adapter/best.pt

# 4) Stage-2（单 VGGT 专家，A/B 各一次）
run_train outputs/native_4f    --vips_experts vggt --vips_vggt_cache prior/geometry_residual_cache_A
run_train outputs/residual_4f  --vips_experts vggt --vips_vggt_cache prior/geometry_residual_cache_B

# Real VGGT = 第 6 节的 single_vggt 命令
```

8f Residual Adapter B 用 8f SigLIP cache + 8f teacher（VGGT agg17 8f），Stage-2 加 8f 参数并把 `--vips_vggt_cache` 换成 `prior/geometry_residual_cache_B_8f`。

## 10. 评估

```bash
# 4f
VIPS_PY=$PY bash scripts/3d/eval/eval_vips_5tasks.sh \
  outputs/<out> $BASE $PROXY $ECACHE4 10

# 8f（必须同时设置 VIPS_PY 和 VIPS_FRAMES=8，缺任一都会静默失败）
VIPS_PY=$PY VIPS_FRAMES=8 bash scripts/3d/eval/eval_vips_5tasks.sh \
  outputs/<out> $BASE $PROXY $ECACHE8 10

# 或直接：EXP=outputs/<out> FRAMES=8 bash run/04_eval.sh
```
