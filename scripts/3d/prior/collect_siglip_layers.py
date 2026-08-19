"""Collect SigLIP hidden states for the native geometry proxy."""
import argparse, os
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

LAYERS_DEFAULT = "6,13,20,26"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vggt-dir", required=True, help="Stage-2 VGGT cache dir whose image_paths define the groups")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--siglip-model", required=True)
    ap.add_argument("--layers", default=LAYERS_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    layers = tuple(int(x) for x in args.layers.split(","))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(args.siglip_model, torch_dtype=torch.bfloat16).eval()
    model.to(args.device)
    proc = CLIPImageProcessor.from_pretrained(args.siglip_model)
    proc.size = {"shortest_edge": 384}
    proc.crop_size = {"height": 384, "width": 384}

    vggt_dir = Path(args.vggt_dir)
    files = sorted(p for p in vggt_dir.rglob("*.pt"))
    if args.limit > 0:
        files = files[:args.limit]
    print(f"groups={len(files)} layers={layers}")

    n = 0
    for fp in files:
        stem = fp.stem
        dst = out / f"{stem}.pt"
        if dst.exists():
            continue
        rec = torch.load(fp, map_location="cpu", weights_only=False)
        paths = rec["image_paths"]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        px = proc(images=imgs, return_tensors="pt").pixel_values.to(args.device, dtype=torch.bfloat16)
        hs = model.vision_model(px, output_hidden_states=True).hidden_states
        data = {"hidden": {str(l): hs[l].float().cpu() for l in layers}, "image_paths": list(paths), "group_id": rec.get("group_id", stem)}
        torch.save(data, dst)
        n += 1
    print(f"collected {n} groups -> {out}")

if __name__ == "__main__":
    main()
