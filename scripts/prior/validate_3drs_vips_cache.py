#!/usr/bin/env python3
"""Fail fast if a five-expert 3DRS-ViPS cache is incomplete or misaligned."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

KEYS = {"vggt": "f_base", "da3": "da3_feature", "radio": "radio_feature", "trace": "trace_feature", "wan": "wan_feature"}
def main():
    p=argparse.ArgumentParser()
    for name in KEYS: p.add_argument(f"--{name}-dir", required=True)
    p.add_argument("--manifest", required=True); p.add_argument("--frames", type=int, default=4); a=p.parse_args()
    records=[json.loads(line) for line in Path(a.manifest).read_text(encoding="utf-8").splitlines() if line]
    dirs={name:Path(getattr(a, f"{name}_dir")) for name in KEYS}
    for record in records:
        for name, key in KEYS.items():
            path=dirs[name] / f"{record['group_id']}.pt"
            if not path.is_file(): raise FileNotFoundError(f"{name}: missing {path}")
            payload=torch.load(path, map_location="cpu", weights_only=True)
            tensor=payload[key]
            if tensor.ndim != 4 or tuple(tensor.shape[:3]) != (a.frames,14,14):
                raise ValueError(f"{name}:{path.name} has invalid shape {tuple(tensor.shape)}")
            paths=payload.get("image_paths")
            if paths is None or list(paths) != record["image_paths"]:
                raise ValueError(f"{name}:{path.name} image path ordering does not equal manifest")
    print(f"Valid ViPS cache: {len(records)} groups, {a.frames} frames, 14x14")
if __name__ == "__main__": main()
