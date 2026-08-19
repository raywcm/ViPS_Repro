#!/usr/bin/env python3
"""Build the single source-of-truth frame manifest for 3DRS-ViPS teachers.

It mirrors ``VideoProcessor.sample_frame_files(..., force_sample=True)`` but
does not instantiate VideoProcessor, therefore it never requires an existing
3DRS VGGT feature cache.  Every teacher must consume this manifest unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import math
from pathlib import Path

import numpy as np
import yaml


def group_id(paths):
    return hashlib.sha1(json.dumps(paths, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", default="scripts/3d/train/multi.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--embodiedscan-folder", default="data/embodiedscan")
    parser.add_argument(
        "--annotation-files", nargs="+", default=None,
        help="Explicit annotation JSON files (for example val/test files). Overrides --data-yaml.",
    )
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--percent", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="prior/3drs_vips_4frames/manifest.jsonl")
    args = parser.parse_args()
    if args.frames < 1: raise ValueError("--frames must be positive")
    if not 0.0 < args.percent <= 100.0: raise ValueError("--percent must be in (0, 100]")
    data_root = Path(args.data_root); yaml_path = Path(args.data_yaml)
    # multi.yaml paths are written relative to the project root, e.g.
    # ``data/processed/...``.  Do not prepend ``data_root`` a second time.
    project_root = data_root.parent
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if args.annotation_files is None else None
    video_ids = set()
    annotation_files = args.annotation_files or [item["json_path"] for item in config["datasets"]]
    for source_index, annotation_file in enumerate(annotation_files):
        annotations = Path(annotation_file)
        if not annotations.is_absolute(): annotations = project_root / annotations
        records = json.loads(annotations.read_text(encoding="utf-8"))
        if args.percent < 100.0:
            count = max(1, math.ceil(len(records) * args.percent / 100.0))
            indices = sorted(random.Random(args.seed + source_index).sample(range(len(records)), count))
            records = [records[index] for index in indices]
        video_ids.update(record["video"] for record in records if record.get("video"))
    scenes = {}
    for split in ("train", "val", "test"):
        metadata_path = Path(args.embodiedscan_folder) / f"embodiedscan_infos_{split}.pkl"
        if not metadata_path.is_file():
            continue
        metadata = pickle.loads(metadata_path.read_bytes())["data_list"]
        scenes.update({record["sample_idx"]: record for record in metadata if record["sample_idx"] in video_ids})
    missing = sorted(video_ids.difference(scenes))
    if missing: raise KeyError(f"EmbodiedScan metadata misses {len(missing)} videos, e.g. {missing[:3]}")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_index, video_id in enumerate(sorted(video_ids)):
        images = scenes[video_id]["images"]
        indices = np.linspace(0, len(images) - 1, args.frames, dtype=int)
        paths = [str((data_root / images[index]["img_path"]).resolve()) for index in indices]
        if not all(Path(path).is_file() for path in paths):
            unavailable = [path for path in paths if not Path(path).is_file()]
            raise FileNotFoundError(f"{video_id} has unavailable RGB frames: {unavailable}")
        records.append({"sample_index": sample_index, "sample_id": video_id, "group_id": group_id(paths), "image_paths": paths})
    with output.open("w", encoding="utf-8") as handle:
        for record in records: handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} unique 3DRS frame groups to {output} ({args.percent:g}%, seed={args.seed})")


if __name__ == "__main__": main()
