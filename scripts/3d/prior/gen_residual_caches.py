"""Generate pseudo VGGT caches A/B for Stage-2 geometry experiments."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from geometry_residual_adapter import GeometryResidualAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--adapter-ckpt", default=None, help="required for B")
    ap.add_argument("--limit", type=int, default=0,
                    help="0=all; positive=smoke limit on input caches")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.mode == "B":
        if not args.adapter_ckpt:
            raise ValueError("--adapter-ckpt required for mode B")
        ckpt = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
        model = GeometryResidualAdapter(**ckpt["config"])
        model.load_state_dict(ckpt["model"])
        model.eval()
    else:
        # A: native projector only
        model = GeometryResidualAdapter(use_cross_frame=True, seed=0)
        model.eval()
    model.to(args.device).float()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    n = 0
    files = sorted(Path(args.siglip_dir).rglob("*.pt"))
    if args.limit > 0:
        files = files[: args.limit]
    for fp in files:
        dst = out / fp.name
        if dst.exists():
            continue
        s = torch.load(fp, map_location="cpu", weights_only=False)
        keys = sorted(s["hidden"].keys(), key=int)[: model.n_layers]
        hidden = {str(i): s["hidden"][k].float().unsqueeze(0).to(args.device) for i, k in enumerate(keys)}
        if args.mode == "A":
            _, f_native, _ = model(hidden)
            pred = f_native.float().cpu()[0]
        else:
            enhanced, _, _ = model(hidden)
            pred = enhanced.float().cpu()[0]
        torch.save({"f_base": pred, "image_paths": s["image_paths"], "group_id": s.get("group_id", Path(fp).stem)}, dst)
        n += 1
    print(f"generated {n} caches (mode {args.mode}) -> {out}")


if __name__ == "__main__":
    main()
