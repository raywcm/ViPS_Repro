"""Compute the five-task metrics used in the experiment tables.

This operates on prediction jsonl files produced by the model evaluators, so
model inference is not repeated.  Scan2Cap is a captioning task conditioned on
the supplied reference box in the current 3DRS evaluator; consequently its
C@0.5 is explicitly reported as *reference-box gated* (not autonomous box
localization).
"""
import argparse, json, re
import math
from pathlib import Path
import numpy as np
from llava.eval.box_utils import get_3d_box_corners, box3d_iou

def _box_iou(a, b):
    return box3d_iou(get_3d_box_corners(a[:3], a[3:]),
                     get_3d_box_corners(b[:3], b[3:]))

def _valid_box(box):
    """A valid axis-aligned box is a six-number center/size vector."""
    return (
        isinstance(box, (list, tuple))
        and len(box) == 6
        and all(value is not None for value in box)
    )

def _caption_score(items):
    from llava.eval.caption_eval.cider.cider import Cider
    res, gts = {}, {}
    for x in items:
        k = str(x['sample_id'])
        norm = lambda s: 'sos ' + s.replace('.', ' . ').replace(',', ' , ').lower() + ' eos'
        res[k] = [norm(x['pred_response'])]
        gts[k] = [norm(s) for s in x['gt_response']]
    return Cider().compute_score(gts, res)[0] * 100

def _f1(pred, gt, threshold):
    if not pred and not gt: return 1.0
    if not pred or not gt: return 0.0
    # F1@IoU only needs the maximum number of above-threshold pairs, not a
    # maximum-weight assignment.  Augmenting-path matching is exact here and
    # avoids factorial enumeration when a scene has dozens of proposals.
    edges = [
        [j for j, b in enumerate(gt) if _box_iou(a, b) >= threshold]
        for a in pred
    ]
    matched_gt = [-1] * len(gt)

    def augment(pred_index, visited):
        for gt_index in edges[pred_index]:
            if visited[gt_index]:
                continue
            visited[gt_index] = True
            if matched_gt[gt_index] < 0 or augment(matched_gt[gt_index], visited):
                matched_gt[gt_index] = pred_index
                return True
        return False

    tp = sum(augment(index, [False] * len(gt)) for index in range(len(pred)))
    return 2.0 * tp / (len(pred) + len(gt))

def _clean_sqa(s):
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9,\'\s\-:]+', '', s)
    return re.sub(r'\s+', ' ', s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', required=True)
    ap.add_argument('--scan2cap-annotations', default='data/processed/scan2cap_val_llava_style.json')
    ap.add_argument('--scanqa-annotations', default='data/processed/scanqa_val_llava_style.json')
    ap.add_argument('--sqa3d-annotations', default='data/processed/sqa3d_val_llava_style.json')
    args = ap.parse_args()
    root = Path(args.eval_dir)
    load = lambda n: [json.loads(x) for x in (root/n).read_text().splitlines() if x.strip()]
    out = {}

    d = load('scanrefer.jsonl')
    ious = [_box_iou(x['pred_response'], x['gt_response']) for x in d]
    out['ScanRefer'] = {'Acc@0.25': 100*np.mean(np.array(ious)>=.25), 'Acc@0.5': 100*np.mean(np.array(ious)>=.5), 'count': len(d)}

    d = load('multi3drefer.jsonl'); vals = {}
    # Reproduce the official confidence selection used by eval_multi3drefer.
    for t in (.25, .5):
        fs=[]
        for x in d:
            scores=list(map(float,x['scores'])); objs=x['objects']; pred=[]
            if max(range(len(scores)), key=scores.__getitem__) != len(scores)-1:
                ex=[math.exp((s-max(scores))/.07) for s in scores]; z=sum(ex); p=[v/z for v in ex][:-1]
                order=sorted(range(len(p)), key=p.__getitem__, reverse=True); total=0
                for i in order:
                    pred.append(objs[i]); total += p[i]
                    if total >= .4: break
            fs.append(_f1(pred,x['gt_response'],t))
        vals[f'F1@{t}']=100*np.mean(fs)
    vals['count']=len(d); out['Multi3DRefer']=vals

    d = load('scan2cap.jsonl')
    anns={str(x['id']):x for x in json.loads(Path(args.scan2cap_annotations).read_text())}
    valid_box_input = []
    gated=[]
    for x in d:
        a=anns.get(str(x['sample_id']))
        if a is None or not _valid_box(a.get('box_input')) or not _valid_box(a.get('gt_box')):
            continue
        valid_box_input.append(x)
        if _box_iou(a['box_input'], a['gt_box']) >= .5:
            gated.append(x)
    out['Scan2Cap']={
        'C@0.5': _caption_score(gated) if gated else None,
        'count':len(d),
        'valid_box_input_count':len(valid_box_input),
        'gated_count':len(gated),
        'localization_source':'reference box (box_input)',
    }

    d=load('scanqa.jsonl'); anns={str(x['id']):x for x in json.loads(Path(args.scanqa_annotations).read_text())}
    from llava.eval.caption_eval.cider.cider import Cider
    res={}; gts={}; em=0
    for x in d:
        key='_'.join(x['sample_id'].split('_')[:-1]+['0']); a=anns.get(key) or anns.get(x['sample_id'])
        if a is None: continue
        res[key]=[x['pred_response'].rstrip('.')]; gts[key]=a['metadata']['answers']; em += int(_clean_sqa(x['pred_response']) in {_clean_sqa(y) for y in a['metadata']['answers']})
    out['ScanQA']={'CIDEr':Cider().compute_score(gts,res)[0]*100,'EM':100*em/max(1,len(res)),'count':len(res)}

    d=load('sqa3d.jsonl'); em=0
    for x in d: em += int(_clean_sqa(x['pred_response'])==_clean_sqa(x['gt_response']))
    out['SQA3D']={'EM':100*em/max(1,len(d)),'count':len(d)}
    print(json.dumps(out, ensure_ascii=False, indent=2)); (root/'standard_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__': main()
