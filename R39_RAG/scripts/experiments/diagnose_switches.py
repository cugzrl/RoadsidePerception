#!/usr/bin/env python3
"""Locate CLEAR identity switches and stratify their observed matching gaps.

This diagnostic mirrors the official CLEAR matching and verifies its count
against TrackEval. It never modifies predictions or tunes tracking parameters.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'.rag_deps'),str(ROOT/'R39_RAG/src')]
from r39_rag.config import load_knowledge
from r39_rag.evaluation.tracking import load_reference, load_prediction, make_data, metric_suite


def switch_events(gt,pred,keys):
    data = make_data(gt,pred,keys)
    last_id = np.full(data['num_gt_ids'],np.nan)
    previous = np.full(data['num_gt_ids'],np.nan)
    last_match = np.full(data['num_gt_ids'],-1,dtype=int)
    events=[]
    for t,(gt_ids,pred_ids) in enumerate(zip(data['gt_ids'],data['tracker_ids'])):
        if not len(gt_ids) or not len(pred_ids):
            continue
        similarity = data['similarity_scores'][t]
        score = 1000*(pred_ids[None,:]==previous[gt_ids[:,None]])+similarity
        score[similarity<.5-np.finfo(float).eps]=0
        rows,cols=linear_sum_assignment(-score)
        keep=score[rows,cols]>np.finfo(float).eps
        rows,cols=rows[keep],cols[keep]
        for row,col in zip(rows,cols):
            g,p=int(gt_ids[row]),int(pred_ids[col])
            if np.isfinite(last_id[g]) and p!=last_id[g]:
                reference=gt[keys[t]][int(row)]
                box=reference['bbox_xyxy']
                gap=t-int(last_match[g])-1
                events.append({'camera':keys[t][0],'frame':keys[t][1],'previous_matched_frame':keys[last_match[g]][1],
                               'matched_gap_frames':gap,'reference_global_id':reference['global_track_id'],
                               'class_name':reference['class_name'],'bbox_height':box[3]-box[1],
                               'bbox_xyxy':box,'matched_iou':float(similarity[row,col])})
            last_id[g]=p
            last_match[g]=t
        previous[:]=np.nan
        previous[gt_ids[rows]]=pred_ids[cols]
    official=int(metric_suite()['CLEAR'].eval_sequence(data)['IDSW'])
    if len(events)!=official:
        raise AssertionError(f'Diagnostic differs from TrackEval: {len(events)} vs {official}')
    return events


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--group',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report={}
    for seq in load_knowledge()['holdout_sequences']:
        meta,gt,_=load_reference(ROOT/'R39_RAG/data/normalized/sequences'/seq)
        pred,_=load_prediction(args.run_root/seq/args.group/'tracking.json',meta)
        events=[]
        for camera in meta['camera_ids']:
            events.extend(switch_events(gt,pred,[(camera,t) for t in range(meta['frame_count'])]))
        bins=Counter('consecutive' if e['matched_gap_frames']==0 else
                     'gap_1_to_3' if e['matched_gap_frames']<4 else
                     'gap_4_to_30' if e['matched_gap_frames']<=30 else 'gap_over_30' for e in events)
        report[seq]={'IDSW':len(events),'by_matched_gap':dict(bins),
                     'small_reference_box_height_under_40':sum(e['bbox_height']<40 for e in events),
                     'by_class':dict(Counter(e['class_name'] for e in events)),'events':events}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({seq:{k:v for k,v in r.items() if k!='events'} for seq,r in report.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
