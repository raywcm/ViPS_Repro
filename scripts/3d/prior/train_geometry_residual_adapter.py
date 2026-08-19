"""Train the geometry residual adapter (SigLIP -> VGGT agg17)."""
from __future__ import annotations
import argparse, random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from geometry_residual_adapter import GeometryResidualAdapter


def suffix_key(paths):
    return tuple(str(p).replace("\\\\", "/").split("posed_images/")[-1] for p in paths)


def index_dir(cache_dir):
    index = {}
    for fp in Path(cache_dir).rglob("*.pt"):
        try:
            rec = torch.load(fp, map_location="cpu", weights_only=False)
        except Exception:
            continue
        paths = rec.get("image_paths") or rec.get("frame_paths")
        if paths:
            index[suffix_key(paths)] = (fp, rec)
    return index


class ResidualGeoDataset(Dataset):
    def __init__(self, siglip_dir, vggt_dir, n_layers=4):
        self.siglip_index = index_dir(siglip_dir)
        self.vggt_index = index_dir(vggt_dir)
        self.n_layers = n_layers
        self.items = []
        for v_path, rec in self.vggt_index.values():
            paths = rec.get("image_paths") or rec.get("frame_paths")
            key = suffix_key(paths)
            s_path = self.siglip_index.get(key)
            s_path = s_path[0] if s_path is not None else None
            if s_path is not None:
                self.items.append((s_path, v_path, paths))
        print(f"paired groups: {len(self.items)} (siglip index {len(self.siglip_index)}, vggt index {len(self.vggt_index)})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s_path, v_path, paths = self.items[idx]
        s = torch.load(s_path, map_location="cpu", weights_only=False)
        v = torch.load(v_path, map_location="cpu", weights_only=False)
        keys = sorted(s["hidden"].keys(), key=int)[: self.n_layers]
        hidden = {str(i): s["hidden"][k].float() for i, k in enumerate(keys)}
        target = v["f_base"].float()
        return hidden, target


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip-dir", required=True)
    ap.add_argument("--vggt-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-groups", type=int, default=0,
                    help="0=all; positive=smoke limit on paired groups")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--residual-loss", type=float, default=0.0,
                    help="weight for direct L2 on delta vs (VGGT - F_native)")
    ap.add_argument("--no-cross-frame", dest="use_cross_frame", action="store_false", default=True)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    ds = ResidualGeoDataset(args.siglip_dir, args.vggt_dir)
    if args.max_groups > 0:
        ds.items = ds.items[: args.max_groups]
    n = len(ds)
    if n < 2:
        raise ValueError(f"not enough paired groups: {n}")
    n_val = max(1, int(n * args.val_ratio))
    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    val_idx = set(idx[:n_val])
    train_idx = [i for i in idx if i not in val_idx]
    train_dl = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=False)
    val_dl = DataLoader(Subset(ds, sorted(val_idx)), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    model = GeometryResidualAdapter(use_cross_frame=args.use_cross_frame,
                                    alpha=args.alpha, seed=args.seed).cuda()
    # freeze native projector; only residual branch trains
    for p in model.native.parameters():
        p.requires_grad_(False)
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for ep in range(args.epochs):
        model.train()
        tl = 0.0; tc = 0.0; tn = 0
        for hidden, target in tqdm(train_dl, desc=f"ep{ep}", leave=False):
            target = target.cuda()
            hidden = {k: v.cuda() for k, v in hidden.items()}
            enhanced, f_native, delta = model(hidden)
            mse = F.mse_loss(enhanced, target)
            cos = F.cosine_similarity(enhanced, target, dim=-1).mean()
            loss = mse + (1.0 - cos)
            if args.residual_loss > 0:
                target_res = (target - f_native).detach()
                loss = loss + args.residual_loss * F.mse_loss(delta, target_res)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += mse.item() * len(target); tc += cos.item() * len(target); tn += len(target)
        model.eval()
        vm = 0.0; vc = 0.0; vn = 0
        with torch.no_grad():
            for hidden, target in val_dl:
                target = target.cuda()
                hidden = {k: v.cuda() for k, v in hidden.items()}
                enhanced, _, _ = model(hidden)
                vm += F.mse_loss(enhanced, target).item() * len(target)
                vc += F.cosine_similarity(enhanced, target, dim=-1).mean().item() * len(target)
                vn += len(target)
        vm /= vn; vc /= vn
        score = vm + (1 - vc)
        print(f"ep{ep} train_mse={tl/tn:.5f} train_cos={tc/tn:.5f} val_mse={vm:.5f} val_cos={vc:.5f} score={score:.5f}")
        if score < best:
            best = score
            torch.save({"model": model.state_dict(),
                        "config": {"in_dim": 1152, "n_layers": 4, "out_dim": 2048,
                                   "use_cross_frame": args.use_cross_frame,
                                   "alpha": args.alpha, "seed": args.seed},
                        "val_mse": vm, "val_cos": vc}, out / "best.pt")
    print(f"done best score={best}")


if __name__ == "__main__":
    main()
