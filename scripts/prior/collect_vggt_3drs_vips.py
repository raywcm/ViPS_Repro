#!/usr/bin/env python3
"""Extract the 3DRS-aligned VGGT F_base cache for ViPS Stage 1/2."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vggt"))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--frames", type=int, default=4)
    p.add_argument("--agg-block", type=int, default=24, help="1-based aggregator output index, 24=final")
    p.add_argument("--target-height", type=int, default=14)
    p.add_argument("--target-width", type=int, default=14)
    p.add_argument("--limit", type=int, default=0); a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("VGGT cache extraction requires CUDA")
    raw_records = [json.loads(line) for line in Path(a.manifest).read_text(encoding="utf-8").splitlines() if line]
    groups = {}
    for record in raw_records:
        paths = tuple(record["image_paths"])
        groups.setdefault(paths, record)
    records = sorted(groups.values(), key=lambda r: tuple(r["image_paths"]))
    for record in records:
        if "group_id" not in record:
            payload = json.dumps(record["image_paths"], ensure_ascii=False).encode("utf-8")
            record["group_id"] = hashlib.sha1(payload).hexdigest()[:20]
    if a.limit: records = records[:a.limit]
    device = torch.device("cuda"); dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT().to(device=device, dtype=dtype).eval()
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=False))
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True); done = []
    for record in tqdm(records, desc="VGGT F_base", unit="group"):
        path = out / f"{record['group_id']}.pt"
        if not path.exists():
            paths = record["image_paths"]
            if len(paths) != a.frames: raise ValueError(f"{record['group_id']} expects {a.frames} frames")
            images = load_and_preprocess_images(paths).to(device=device, dtype=dtype).unsqueeze(0)
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                tokens, patch_start = model.aggregator(images)
            if not (1 <= a.agg_block <= len(tokens)):
                raise RuntimeError(f"agg-block {a.agg_block} out of range 1..{len(tokens)}")
            patches = tokens[a.agg_block - 1][0, :, patch_start:, :]
            height, width = images.shape[-2] // 14, images.shape[-1] // 14
            if patches.shape != (a.frames, height * width, 2048):
                raise RuntimeError(f"Unexpected VGGT tokens {tuple(patches.shape)}, input grid={height}x{width}")
            feature = F.interpolate(patches.view(a.frames, height, width, 2048).permute(0, 3, 1, 2).float(), size=(a.target_height, a.target_width), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).cpu().to(torch.bfloat16).contiguous()
            temp = path.with_suffix(".pt.tmp")
            torch.save({"f_base": feature, "image_paths": paths, "group_id": record["group_id"]}, temp); os.replace(temp, path)
        done.append({"group_id": record["group_id"], "feature_file": path.name, "image_paths": record["image_paths"], "shape": [a.frames, a.target_height, a.target_width, 2048]})
    (out / "manifest.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in done), encoding="utf-8")
    print(f"VGGT cache complete: {len(done)} groups -> {out}")
if __name__ == "__main__": main()
