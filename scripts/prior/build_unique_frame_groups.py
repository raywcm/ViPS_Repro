#!/usr/bin/env python3
"""将样本manifest去重为唯一帧组manifest，供所有离线prior缓存复用。"""
import argparse, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
a=p.parse_args(); groups={}
for line in Path(a.input).open(encoding='utf-8'):
    if line.strip():
        r=json.loads(line); groups.setdefault(tuple(r['image_paths']),r)
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8') as f:
    for paths,r in sorted(groups.items()):
        r=dict(r); r['group_id']=hashlib.sha1(json.dumps(paths,ensure_ascii=False).encode()).hexdigest()[:20]
        f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f'unique frame groups: {len(groups)} -> {out}')
