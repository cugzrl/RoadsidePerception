#!/usr/bin/env python3
"""Find and export paired identity recovery evidence without changing predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFont
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.visualize_result import draw_frame

SEQUENCES = ['r39_20251021_070327', 'r39_20251021_070428']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def overlap(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    extent = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                        np.maximum(a[:, None, :2], b[None, :, :2]))
    intersection = extent.prod(axis=2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    return intersection / np.maximum(1e-12, area_a[:, None] + area_b[None, :] - intersection)


def load_sequence(sequence):
    paths = [ROOT / 'outputs/r39_ablation' / sequence / group / 'tracking.json' for group in ['A2', 'D', 'B']]
    results = [read_json(path) for path in paths]
    frames = [{(f['camera_id'], f['frame_index']): f for f in result['frames']} for result in results]
    reference = defaultdict(list)
    path = ROOT / 'R39_RAG/data/normalized/sequences' / sequence / 'versions/sam3_manual_gid/observations.jsonl'
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            box = row.get('bbox_xyxy')
            if row.get('local_track_id') is not None and box and box[2] > box[0] and box[3] > box[1]:
                reference[row['camera_id'], row['frame_index']].append(row)
    tracks = defaultdict(list)
    for key, rows in sorted(reference.items()):
        detections = frames[0][key]['detections']
        mappings = [{d['detection_id']: d for d in index[key]['detections']} for index in frames]
        if set(mappings[0]) != set(mappings[1]):
            raise ValueError('Detector coverage differs')
        similarity = overlap([r['bbox_xyxy'] for r in rows], [d['bbox_xyxy'] for d in detections])
        classes = np.asarray([r['class_id'] for r in rows])[:, None] == np.asarray([d['class_id'] for d in detections])[None, :]
        score = np.where(classes & (similarity >= .5), similarity, 0)
        rr, cc = linear_sum_assignment(-score)
        matched = {int(r): int(c) for r, c in zip(rr, cc) if score[r, c] >= .5}
        for r, row in enumerate(rows):
            item = {'frame': key[1], 'reference_box': row['bbox_xyxy'], 'flags': row.get('quality_flags', []),
                    'class_name': row['class_name'], 'detection_id': None, 'ids': [None] * 3, 'iou': None}
            if r in matched:
                c = matched[r]
                detection = detections[c]
                identifier = detection['detection_id']
                copies = [mapping[identifier] for mapping in mappings]
                if any(d['bbox_xyxy'] != detection['bbox_xyxy'] for d in copies):
                    raise ValueError('Detector geometry differs')
                alternatives = np.delete(score[:, c], r)
                item.update(detection_id=identifier, ids=[d.get('global_track_id') for d in copies],
                            iou=float(similarity[r, c]), box=detection['bbox_xyxy'],
                            margin=float(similarity[r, c] - alternatives.max(initial=0)))
            tracks[key[0], row['local_track_id']].append(item)
    return results, frames, tracks, paths


def candidates(sequence, tracks):
    found = []
    for (camera, identity), timeline in tracks.items():
        if len({r['frame'] for r in timeline}) != len(timeline):
            # A repeated local identity cannot establish a unique target.
            continue
        observed = [r for r in timeline if r['ids'][0] is not None and r['ids'][1] is not None]
        for i in range(3, len(observed) - 3):
            before, after = observed[i - 1], observed[i]
            if before['ids'][0] == after['ids'][0] or before['ids'][1] != after['ids'][1]:
                continue
            if not 0 < after['frame'] - before['frame'] <= 60:
                continue
            left, right = observed[i - 3:i], observed[i:i + 4]
            if any(r['ids'][0] != before['ids'][0] for r in left) or any(r['ids'][0] != after['ids'][0] for r in right):
                continue
            start, end = max(0, before['frame'] - 6), min(1199, after['frame'] + 10)
            window = [r for r in timeline if start <= r['frame'] <= end]
            visible = [r for r in window if r['detection_id'] is not None]
            if any(r['ids'][1] != before['ids'][1] for r in visible):
                continue
            if len(visible) < 8 or any('id_reassigned' in flag for r in window for flag in r['flags']):
                continue
            if any(r['margin'] < .10 for r in visible):
                continue
            found.append({'sequence': sequence, 'camera': camera, 'reference_local_id': identity,
                          'event_frame': after['frame'], 'previous_frame': before['frame'],
                          'start': start, 'end': end, 'baseline_transition': [before['ids'][0], after['ids'][0]],
                          'rag_id': before['ids'][1], 'cv_only_transition': [before['ids'][2], after['ids'][2]],
                          'class_name': after['class_name'], 'matched_frames': len(visible),
                          'window_frames': end - start + 1, 'min_iou': min(r['iou'] for r in visible),
                          'mean_iou': float(np.mean([r['iou'] for r in visible])),
                          'mean_box_height': float(np.mean([r['box'][3]-r['box'][1] for r in visible]))})
    return found


def export_case(case, results, frames, tracks, output, ordinal):
    camera, local_id = case['camera'], case['reference_local_id']
    timeline = {r['frame']: r for r in tracks[camera, local_id]}
    name = f"case_{ordinal:02d}_{case['sequence'][-6:]}_{camera}_{case['event_frame']:04d}"
    directory = output / 'cases' / name
    label_font = ImageFont.truetype(str(ROOT / 'third_party/HyperLPR/resource/font/platech.ttf'), 25)
    camera_font = ImageFont.truetype(str(ROOT / 'third_party/HyperLPR/resource/font/platech.ttf'), 30)
    records = []
    for t in range(case['start'], case['end'] + 1):
        row = timeline.get(t)
        match_source = 'reference_iou_at_least_0.5'
        if row is None or row['detection_id'] is None:
            # Use geometry alone to display low-overlap or unassigned detections
            # between verified observations. Never infer or copy an identity.
            anchors = [r for r in timeline.values() if r['detection_id'] is not None]
            anchor = min(anchors, key=lambda r: abs(r['frame'] - t))
            proposals = [d for d in frames[0][camera, t]['detections'] if d['class_name'] == case['class_name']]
            scores = overlap([anchor['box']], [d['bbox_xyxy'] for d in proposals])[0]
            order = np.argsort(-scores)
            if len(order) and scores[order[0]] >= .3 and (len(order) == 1 or scores[order[0]] - scores[order[1]] >= .15):
                detection = proposals[int(order[0])]
                copies = [next(d for d in index[camera, t]['detections'] if d['detection_id'] == detection['detection_id'])
                          for index in frames]
                if copies[1].get('global_track_id') not in [None, case['rag_id']]:
                    raise ValueError('Visual continuity includes a RAG identity change')
                reference_iou = float(overlap([row['reference_box']], [detection['bbox_xyxy']])[0, 0]) if row else None
                row = dict(row or {}, detection_id=detection['detection_id'], box=detection['bbox_xyxy'],
                           ids=[d.get('global_track_id') for d in copies], iou=reference_iou)
                match_source = 'geometry_continuity_for_visual_review'
            else:
                match_source = 'no_target_detection'
        record = frames[0][camera, t]
        image = cv2.imread(record['image_path'])
        if image is None:
            raise FileNotFoundError(record['image_path'])
        height, width = image.shape[:2]
        prepared = (Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), width, height)
        identifier = row['detection_id'] if row else None
        for side, folder in enumerate(['stmc', 'stmc-rag']):
            current = frames[side][camera, t]
            selected = [d for d in current['detections'] if d['detection_id'] == identifier]
            if selected and selected[0].get('global_track_id') is not None:
                identity = selected[0]['global_track_id']
                if sum(d.get('global_track_id') == identity for d in current['detections']) != 1:
                    raise ValueError(f'Duplicate displayed identity: {camera}, {t}, {identity}')
            rendered = draw_frame(dict(current, detections=selected), label_font, camera_font, width, height, 3, prepared)
            # Preserve the full camera image. The header is outside the source pixels.
            canvas = np.full((height + 42, width, 3), 20, np.uint8)
            canvas[42:] = rendered
            message = f'{folder.upper()} | {camera} | frame {t:04d} | {t/results[0]["fps"]:.2f}s'
            if not selected:
                message += ' | target detection unavailable'
            cv2.putText(canvas, message, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2, cv2.LINE_AA)
            target = directory / folder / f'frame_{t:04d}.png'
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), canvas):
                raise IOError(target)
        records.append({'frame': t, 'detection_id': identifier, 'baseline_id': row['ids'][0] if row else None,
                        'rag_id': row['ids'][1] if row else None, 'cv_only_id': row['ids'][2] if row else None,
                        'reference_iou': row['iou'] if row else None, 'bbox_xyxy': row.get('box') if row else None,
                        'display_selection': match_source})
    report = dict(case, folder=str(directory.relative_to(output)), frames=records,
                  frame_index_origin=0, source_frame_stride=3,
                  evidence_scope='Same-camera local reference identity; not a cross-camera ground-truth guarantee.',
                  rendering='Full camera resolution; only the selected target is labeled; no interpolation or ID rewriting.',
                  display_selection='Reference matches seed the example. Geometry-only continuity displays low-IoU or unassigned detections in the intervening frames.')
    (directory / 'case.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/stmc_rag_comparison_20261025')
    parser.add_argument('--select', action='append', default=[], help='sequence,camera,local_id,event_frame')
    args = parser.parse_args()
    selections = {tuple(s.split(',')) for s in args.select}
    reports, provenance = [], {}
    for sequence in SEQUENCES:
        if args.select and not any(selection.split(',')[0] == sequence for selection in args.select):
            continue
        results, frames, tracks, paths = load_sequence(sequence)
        choices = candidates(sequence, tracks)
        print(json.dumps(choices, ensure_ascii=False, indent=2), flush=True)
        for case in choices:
            key = (sequence, case['camera'], str(case['reference_local_id']), str(case['event_frame']))
            if key in selections:
                reports.append(export_case(case, results, frames, tracks, args.output, len(reports) + 1))
                selections.remove(key)
        provenance[sequence] = {group: {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                for group, path in zip(['stmc_A2', 'stmc_rag_D', 'cv_only_B'], paths)}
    if selections:
        raise ValueError(f'Unverified selections: {selections}')
    if reports:
        (args.output / 'cases.json').write_text(json.dumps({'inputs': provenance, 'cases': reports}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'已导出 {len(reports)} 组逐帧对比样例', flush=True)


if __name__ == '__main__':
    main()
