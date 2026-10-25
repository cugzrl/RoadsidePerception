#!/usr/bin/env python3
"""Audit HOTA computation, frame alignment and detector/identity disagreements.

Oracle controls are diagnostics using pseudo labels, not tracking methods or
strict theoretical upper bounds. No predictions or reference files are changed.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'.rag_deps'),str(ROOT/'R39_RAG/src')]
from r39_rag.config import load_knowledge
from r39_rag.evaluation.tracking import load_reference,load_prediction,make_data,iou_matrix,metric_suite


def matches(gt,pred,threshold=.5,aware=True):
    similarity=iou_matrix(gt,pred)
    if aware:
        similarity*=np.asarray([r['class_id'] for r in gt])[:,None]==np.asarray([r['class_id'] for r in pred])[None,:]
    # Maximize valid-pair count first, then IoU. Ignore identities for diagnostics.
    score=(similarity>=threshold).astype(float)*1000+similarity
    a,b=linear_sum_assignment(-score)
    keep=similarity[a,b]>=threshold
    return a[keep],b[keep],similarity


def independent_hota(data):
    ng,npred=data['num_gt_ids'],data['num_tracker_ids']
    gc=np.zeros(ng)
    pc=np.zeros(npred)
    potential=np.zeros((ng,npred))
    for g,p,s in zip(data['gt_ids'],data['tracker_ids'],data['similarity_scores']):
        np.add.at(gc,g,1)
        np.add.at(pc,p,1)
        denominator=s.sum(1)[:,None]+s.sum(0)[None,:]-s
        normalized=np.divide(s,denominator,out=np.zeros_like(s),where=denominator>np.finfo(float).eps)
        np.add.at(potential,(g[:,None],p[None,:]),normalized)
    alignment=np.divide(potential,gc[:,None]+pc[None,:]-potential,
                        out=np.zeros_like(potential),where=gc[:,None]+pc[None,:]-potential>0)
    assignments=[]
    for g,p,s in zip(data['gt_ids'],data['tracker_ids'],data['similarity_scores']):
        a,b=linear_sum_assignment(-(alignment[g[:,None],p[None,:]]*s))
        assignments.append((g[a],p[b],s[a,b]))
    result=defaultdict(list)
    for alpha in np.arange(.05,1,.05):
        counts=np.zeros_like(potential)
        for g,p,s in assignments:
            valid=s>=alpha-np.finfo(float).eps
            np.add.at(counts,(g[valid],p[valid]),1)
        tp=counts.sum()
        association=np.sum(counts*counts/np.maximum(1,gc[:,None]+pc[None,:]-counts))/max(1,tp)
        detection=tp/max(1,data['num_gt_dets']+data['num_tracker_dets']-tp)
        result['HOTA'].append(np.sqrt(association*detection))
        result['AssA'].append(association)
        result['DetA'].append(detection)
        result['HOTA_TP'].append(tp)
    return {k:np.asarray(v) for k,v in result.items()}


def controls(gt,pred,keys,replace_boxes=False):
    output={}
    for key in keys:
        reference,rows=gt.get(key,[]),pred.get(key,[])
        a,b,_=matches(reference,rows)
        assignment={int(j):int(i) for i,j in zip(a,b)}
        new=[]
        for j,row in enumerate(rows):
            copy=dict(row)
            if j in assignment:
                target=reference[assignment[j]]
                copy['global_track_id']=('oracle',target['global_track_id'])
                if replace_boxes:
                    copy['bbox_xyxy']=target['bbox_xyxy']
            else:
                copy['global_track_id']=('unmatched',key,j)
            new.append(copy)
        output[key]=new
    return output


def brief(raw):
    return {key:float(np.mean(raw[key])*100) for key in ['HOTA','DetA','AssA']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,default=ROOT/'outputs/rag_tp_20261025')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    hota=metric_suite()['HOTA']
    all_raw=defaultdict(dict)
    report={'protocol':'class-aware 2D box IoU; same official per-camera aggregation as the original report',
            'oracle_note':'Reference-ID and snapped-box controls use pseudo GT at IoU 0.5, are diagnostic only, and are not strict HOTA upper bounds',
            'sequences':{}}
    for seq in load_knowledge()['holdout_sequences']:
        meta,gt,gt_audit=load_reference(ROOT/'R39_RAG/data/normalized/sequences'/seq)
        pred,pred_audit=load_prediction(args.experiment/'tracking_hybrid'/seq/'stmc/tracking.json',meta)
        source=ROOT/'outputs/r39_ablation/candidates'/f'{seq}_merged.json'
        payload=json.loads(source.read_text())
        raw={(frame['camera_id'],frame['frame_index']):frame['detections'] for frame in payload['frames']}
        tracked_ids={key:{r['detection_id'] for r in rows} for key,rows in pred.items()}
        coverage=Counter()
        heights=defaultdict(Counter)
        cameras={}
        independent_error=0.
        for camera in meta['camera_ids']:
            keys=[(camera,t) for t in range(meta['frame_count'])]
            data=make_data(gt,pred,keys)
            baseline=hota.eval_sequence(data)
            independent=independent_hota(data)
            independent_error=max(independent_error,max(float(np.max(np.abs(independent[k]-baseline[k]))) for k in independent))
            all_raw['baseline'][seq+'|'+camera]=baseline
            cameras[camera]=brief(baseline)
            variants={'reference_self':gt,
                      'reference_ids_same_tracker_boxes':controls(gt,pred,keys),
                      'reference_ids_and_snapped_matched_boxes':controls(gt,pred,keys,True),
                      'all_input_boxes_with_reference_ids':controls(gt,raw,keys)}
            for name,variant in variants.items():
                all_raw[name][seq+'|'+camera]=hota.eval_sequence(make_data(gt,variant,keys))
            for key in keys:
                reference,rows=gt.get(key,[]),raw.get(key,[])
                a,b,sim=matches(reference,rows)
                matched={int(i):int(j) for i,j in zip(a,b)}
                coverage['reference_boxes']+=len(reference)
                coverage['input_boxes']+=len(rows)
                coverage['input_matches_iou_05']+=len(a)
                coverage['tracked_boxes']+=len(pred.get(key,[]))
                c,d,_=matches(reference,pred.get(key,[]))
                coverage['tracked_matches_iou_05']+=len(c)
                for i,row in enumerate(reference):
                    height=row['bbox_xyxy'][3]-row['bbox_xyxy'][1]
                    bucket='under_20px' if height<20 else '20_to_40px' if height<40 else '40_to_80px' if height<80 else '80px_plus'
                    heights[bucket]['reference']+=1
                    if i in matched:
                        j=matched[i]
                        heights[bucket]['matched_in_input']+=1
                        coverage['input_matched_and_assigned_id' if rows[j]['detection_id'] in tracked_ids[key] else 'input_matched_but_no_id']+=1
                        if rows[j]['detection_id'] in tracked_ids[key]:
                            heights[bucket]['input_match_has_tracking_id']+=1
                    if i in set(c):
                        heights[bucket]['matched_in_tracking']+=1
                    best=float(sim[i].max()) if len(rows) else 0.
                    if i not in matched:
                        coverage['unmatched_input_some_overlap_01_to_05' if best>=.1 else 'unmatched_input_no_overlap_01']+=1
            print(f'完成指标控制实验 {seq} {camera}',flush=True)
        # Evaluate a fixed range of offsets; never change the evaluation alignment.
        offsets={}
        for offset in [-20,-5,-2,-1,0,1,2,5,20]:
            matched,total,iou_sum=0,0,0.
            for camera in meta['camera_ids']:
                for t in range(20,meta['frame_count']-20,10):
                    reference=gt.get((camera,t),[])
                    a,b,s=matches(reference,raw.get((camera,t+offset),[]))
                    matched+=len(a)
                    total+=len(reference)
                    iou_sum+=float(s[a,b].sum())
            offsets[str(offset)]={'matches':matched,'reference_boxes':total,'recall':matched/max(total,1),
                                  'mean_matched_iou':iou_sum/max(matched,1)}
        # Diagnose structural inconsistency without declaring either source correct.
        overlaps=Counter()
        steps=[]
        previous={}
        for key,rows in gt.items():
            similarity=iou_matrix(rows,rows)
            overlaps['same_camera_frame_pairs_iou_over_08']+=int(np.triu(similarity>.8,1).sum())
        for key in sorted(gt,key=lambda k:(k[0],k[1])):
            for row in gt[key]:
                identity=(key[0],row['global_track_id'])
                box=np.asarray(row['bbox_xyxy'])
                center=(box[:2]+box[2:])/2
                scale=np.linalg.norm(box[2:]-box[:2])
                if identity in previous and key[1]-previous[identity][0]==1:
                    steps.append(float(np.linalg.norm(center-previous[identity][1])/max(1,scale)))
                previous[identity]=(key[1],center)
        report['sequences'][seq]={'coverage':dict(coverage),'by_reference_box_height':{k:dict(v) for k,v in heights.items()},
                                  'per_camera_baseline':cameras,'frame_offset_sensitivity':offsets,
                                  'independent_hota_max_absolute_error':independent_error,
                                  'reference_structure':{**overlaps,'consecutive_steps':len(steps),
                                                         'center_jumps_over_two_box_diagonals':int(np.sum(np.asarray(steps)>2))},
                                  'reference_audit':gt_audit,'prediction_audit':pred_audit}
    combined={name:hota.combine_sequences(raw) for name,raw in all_raw.items()}
    report['combined']={name:brief(raw) for name,raw in combined.items()}
    report['per_alpha']={name:[{'alpha':float(a),**{k:float(raw[k][i]*100) for k in ['HOTA','DetA','AssA','DetRe','DetPr','LocA']},
                                      **{k:int(raw[k][i]) for k in ['HOTA_TP','HOTA_FN','HOTA_FP']}}
                              for i,a in enumerate(hota.array_labels)] for name,raw in combined.items()}
    original=json.loads((args.experiment/'evaluation/metrics.json').read_text())['methods']['stmc']['combined']
    error=max(abs(report['combined']['baseline'][k]-original[k]) for k in ['HOTA','DetA','AssA'])
    if error>1e-8 or max(r['independent_hota_max_absolute_error'] for r in report['sequences'].values())>1e-8:
        raise AssertionError('HOTA audit failed numerical agreement')
    report['original_summary_max_error']=error
    (args.output/'hota_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['combined'],indent=2),flush=True)


if __name__=='__main__':
    main()
