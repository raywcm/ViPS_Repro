import os
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# 仓库内 VGGT 是 vggt/vggt 的嵌套源码目录；保证从任意机器、任意 cwd
# 执行本脚本时都优先使用本地代码。
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = os.environ.get("VGGT_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    dev_capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    dtype = torch.bfloat16 if dev_capability[0] >= 8 else torch.float16
else:
    dtype = torch.float32

model = VGGT()
checkpoint = torch.load('checkpoints/model.pt', map_location=device)
msg = model.load_state_dict(checkpoint)
print('loading status:', msg)
# 输入已经使用 BF16/FP16；模型也必须同步转为半精度，否则会以 FP32
# 常驻显存，导致 12GB 显卡在多帧 aggregator 前向时峰值溢出。
model = model.to(device=device, dtype=dtype).eval()

root_dir = 'data/scannet/posed_images'
root_save_3d_feature = 'data/scannet/posed_images_3d_feature_vggt'
if not os.path.exists(root_save_3d_feature):
    os.makedirs(root_save_3d_feature)
    print("create dir", root_save_3d_feature)

scene_list = os.listdir(root_dir)
num_frames_to_sample = int(os.environ.get("VGGT_FRAMES", "32"))
limit = int(os.environ.get("VGGT_LIMIT", "0"))
if limit > 0:
    scene_list = scene_list[:limit]

for scene in tqdm(scene_list):
    scene_dir = os.path.join(root_dir, scene)
    if not os.path.isdir(scene_dir):
        continue

    scene_save_dir = os.path.join(root_save_3d_feature, scene)
    if not os.path.exists(scene_save_dir):
        os.makedirs(scene_save_dir)

    file_names = [file for file in os.listdir(scene_dir) if file.endswith('.jpg')]
    file_names.sort()
    total_frames = len(file_names)
    if total_frames == 0:
        continue
    sampled_indices = np.linspace(0, total_frames - 1, num=num_frames_to_sample, dtype=int)
    sampled_file_list = [os.path.join(scene_dir, file_names[i]) for i in sampled_indices]

    images = load_and_preprocess_images(sampled_file_list)
    images = images.to(device, non_blocking=True)
    images = images.to(dtype)
    images = images.unsqueeze(0)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images)

    feature = aggregated_tokens_list[-1].cpu().numpy()
    ps_idx_np = ps_idx.cpu().numpy() if isinstance(ps_idx, torch.Tensor) else ps_idx

    save_path = os.path.join(scene_save_dir, 'vggt.npz')
    np.savez_compressed(save_path, feature=feature, ps_idx=ps_idx_np)
