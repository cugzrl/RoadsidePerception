#!/usr/bin/env python3
"""Render structured pseudo labels without camera averaging or trajectory smoothing.

The left BEV uses original supplied ground points (unverified coordinates).
The right BEV reprojects the original 2D boxes using the STMC calibration.
Neither view runs a tracker or repairs identities.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'R39_RAG/src')]
from pipeline.bev_map import LaneMap,BEVBasemap,make_r39_view,color_for_id
from r39_rag.projection.homography import load_projectors,bottom_center
from r39_rag.evaluation.tracking import valid_box

CAMERA_CODES={'R39_Aw_CamS':'A','R39_Bn_CamW':'B','R39_Ce_CamN':'C','R39_Ds_CamE':'D'}


def statistics(values):
    values=np.asarray(values,dtype=float)
    return {'count':len(values),'median':float(np.median(values)) if len(values) else None,
            'p90':float(np.percentile(values,90)) if len(values) else None,
            'over_8m':int(np.sum(values>8))}


def load_sequence(sequence):
    directory=ROOT/'R39_RAG/data/normalized/sequences'/sequence
    meta=json.loads((directory/'metadata.json').read_text())
    source=directory/'versions/sam3_manual_gid/observations.jsonl'
    projectors=load_projectors(ROOT/'configs/cameras/r39',meta['camera_ids'],1e-6)
    frames=defaultdict(list)
    audit=Counter()
    for line in source.open():
        row=json.loads(line)
        audit['input_observations']+=1
        if not valid_box(row) or row.get('global_track_id') is None:
            audit['invalid_box_or_missing_id']+=1
            continue
        raw=(row.get('geometry_3d') or {}).get('ground_point')
        raw=np.asarray(raw[:2],dtype=float) if raw is not None else np.full(2,np.nan)
        xy,_=projectors[row['camera_id']].image_to_world(np.asarray([bottom_center(row['bbox_xyxy'])]))
        frames[row['frame_index']].append({'id':row['global_track_id'],'camera':row['camera_id'],
                                          'raw':raw,'projected':xy[0],'flags':row['quality_flags']})
    return meta,frames,audit


def draw_panel(base,view,history,rows,mode,frame,fps,radius):
    image=base.copy()
    active=0
    outside=0
    by_id=defaultdict(list)
    for row in rows:
        point=row[mode]
        if not np.isfinite(point).all():
            continue
        if np.linalg.norm(point-view.center)>radius:
            outside+=1
            continue
        key=(row['id'],row['camera'])
        history[key].append((frame,point))
        by_id[row['id']].append(point)
        active+=1
    for key,trail in history.items():
        while trail and trail[0][0]<frame-int(2*fps):
            trail.popleft()
        color=color_for_id(key[0])
        for (t0,p0),(t1,p1) in zip(trail,list(trail)[1:]):
            if t1-t0<=round(.15*fps) and np.linalg.norm(p1-p0)<=15:
                pixels=view.to_px_int(np.stack([p0,p1]))
                cv2.line(image,tuple(pixels[0]),tuple(pixels[1]),tuple(int(c*.65) for c in color),1,cv2.LINE_AA)
    # Red links reveal disagreement instead of hiding it in an averaged point.
    for points in by_id.values():
        for a in range(len(points)):
            for b in range(a):
                if np.linalg.norm(points[a]-points[b])>8:
                    pixels=view.to_px_int(np.stack([points[a],points[b]]))
                    cv2.line(image,tuple(pixels[0]),tuple(pixels[1]),(65,65,230),1,cv2.LINE_AA)
    for row in rows:
        point=row[mode]
        if not np.isfinite(point).all() or np.linalg.norm(point-view.center)>radius:
            continue
        x,y=view.to_px_int(point)[0]
        color=color_for_id(row['id'])
        cv2.circle(image,(x,y),4,color,2,cv2.LINE_AA)
        code=CAMERA_CODES[row['camera']]
        dx,dy={'A':(5,-5),'B':(5,12),'C':(-38,-5),'D':(-38,12)}[code]
        cv2.putText(image,f"{row['id']}{code}",(x+dx,y+dy),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1,cv2.LINE_AA)
    cv2.rectangle(image,(0,0),(image.shape[1],58),(23,23,23),-1)
    title='SOURCE ground_point (coordinate system unverified)' if mode=='raw' else 'Original bbox -> STMC ground homography'
    cv2.putText(image,title,(14,24),cv2.FONT_HERSHEY_SIMPLEX,.57,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(image,f'visible observations: {active}   outside {radius:g}m: {outside}',(14,47),cv2.FONT_HERSHEY_SIMPLEX,.46,(205,205,205),1,cv2.LINE_AA)
    return image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sequence',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--radius-m',type=float,default=90)
    parser.add_argument('--panel-size',type=int,default=900)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    meta,frames,audit=load_sequence(args.sequence)
    graph=LaneMap.load()
    view,directions=make_r39_view(graph,args.panel_size,args.panel_size,args.radius_m)
    basemap=BEVBasemap(graph,view)
    base=basemap.render()
    basemap.draw_arm_labels(base,directions)
    basemap.draw_scale_bar(base)
    width,height=args.panel_size*2,args.panel_size+78
    output=args.output_dir/(args.sequence+'_structured_bev.mp4')
    temporary=output.with_suffix('.render.mp4')
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),meta['sample_fps'],(width,height))
    if not writer.isOpened():
        raise RuntimeError('Cannot open video writer')
    history={mode:defaultdict(deque) for mode in ['raw','projected']}
    spread={mode:[] for mode in history}
    jump={mode:[] for mode in history}
    previous={mode:{} for mode in history}
    totals={mode:Counter() for mode in history}
    for frame in range(meta['frame_count']):
        rows=frames[frame]
        panels=[]
        for mode in history:
            groups=defaultdict(list)
            for row in rows:
                point=row[mode]
                if not np.isfinite(point).all():
                    totals[mode]['missing_or_nonfinite']+=1
                    continue
                totals[mode]['finite']+=1
                in_view=np.linalg.norm(point-view.center)<=args.radius_m
                totals[mode]['inside_view' if in_view else 'outside_view']+=1
                if in_view:
                    groups[row['id']].append(point)
                    key=(row['id'],row['camera'])
                    if key in previous[mode] and frame-previous[mode][key][0]==1:
                        jump[mode].append(float(np.linalg.norm(point-previous[mode][key][1])))
                    previous[mode][key]=(frame,point)
            for points in groups.values():
                for i in range(len(points)):
                    for j in range(i):
                        spread[mode].append(float(np.linalg.norm(points[i]-points[j])))
            panels.append(draw_panel(base,view,history[mode],rows,mode,frame,meta['sample_fps'],args.radius_m))
        canvas=np.full((height,width,3),22,np.uint8)
        canvas[78:,:args.panel_size]=panels[0]
        canvas[78:,args.panel_size:]=panels[1]
        texts=[f'{args.sequence} | structured pseudo labels | frame {frame:04d}/1199 | {frame/meta["sample_fps"]:05.2f}s',
               'ID color shared across cameras. A/B/C/D = camera. Red links = same-ID views disagree by >8m.',
               'No smoothing, no camera averaging. 2s trails; break lines at >15m jumps or >0.15s gaps.']
        for i,text in enumerate(texts):
            cv2.putText(canvas,text,(14,22+i*23),cv2.FONT_HERSHEY_SIMPLEX,.53,(225,225,225),1,cv2.LINE_AA)
        writer.write(canvas)
        if frame in [100,300,600,900]:
            cv2.imwrite(str(args.output_dir/f'{args.sequence}_frame_{frame:04d}.png'),canvas)
    writer.release()
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(temporary),'-c:v','libx264',
                    '-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)
    capture=cv2.VideoCapture(str(output))
    count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if count!=meta['frame_count'] or abs(fps-meta['sample_fps'])>.01:
        raise RuntimeError('Encoded video coverage mismatch')
    temporary.unlink()
    summary={'sequence':args.sequence,'video':str(output),'video_frames':count,'fps':fps,
             'audit':dict(audit),'radius_m':args.radius_m,'no_camera_averaging':True,'no_smoothing':True,
             'coordinate_note':'source points have unverified coordinate system and units; right panel shares STMC calibration',
             'totals':{k:dict(v) for k,v in totals.items()},
             'same_id_cross_camera_distance_inside_view':{k:statistics(v) for k,v in spread.items()},
             'same_camera_consecutive_step_inside_view':{k:statistics(v) for k,v in jump.items()}}
    (args.output_dir/f'{args.sequence}_bev_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'已生成 {output}，{count} 帧，{fps:g} FPS',flush=True)


if __name__=='__main__':
    main()
