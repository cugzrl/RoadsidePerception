"""Adapt camera observations to the official TrackEval metric API.

Scores describe agreement with SAM3/manual-GID pseudo labels, not human GT.
Camera sequences use ordinary MOT semantics. The optional global stream
serializes (frame, camera) observations and must be reported separately.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def iou_matrix(left, right):
    a = np.asarray([x['bbox_xyxy'] for x in left], dtype=float).reshape(-1, 4)
    b = np.asarray([x['bbox_xyxy'] for x in right], dtype=float).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(hi - lo, 0).prod(-1)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)


def valid_box(row):
    box = row.get('bbox_xyxy')
    return box is not None and len(box) == 4 and np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]


def load_reference(sequence_dir: Path):
    meta = json.loads((sequence_dir / 'metadata.json').read_text())
    versions = meta['source_versions']
    if versions != ['sam3_manual_gid']:
        raise ValueError(f'Expected one explicit pseudo-label version: {versions}')
    source = sequence_dir / 'versions' / versions[0] / 'observations.jsonl'
    frames, audit = defaultdict(list), Counter()
    for line in source.open():
        row = json.loads(line)
        audit['input_observations'] += 1
        if not valid_box(row) or row.get('global_track_id') is None or row.get('class_id') is None:
            audit['invalid_or_unidentified'] += 1
            continue
        frame = int(row['frame_index'])
        if not 0 <= frame < meta['frame_count']:
            raise ValueError('Reference frame outside metadata range')
        expected_ms = frame * 1000.0 / meta['sample_fps']
        if abs(row['timestamp_ms'] - expected_ms) > 1.0:
            raise ValueError('Reference timestamps disagree with sampled frame indices')
        frames[(row['camera_id'], frame)].append(row)
    for key, rows in frames.items():
        ids = [r['global_track_id'] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate reference identity at {key}; resolve labels first')
    audit['valid_observations'] = sum(map(len, frames.values()))
    return meta, frames, dict(audit)


def load_prediction(path: Path, meta):
    payload = json.loads(path.read_text())
    if abs(float(payload['fps']) - meta['sample_fps']) > 1e-6:
        raise ValueError('Prediction FPS differs from reference; explicit resampling is required')
    frames, seen = {}, set()
    audit = Counter()
    for frame in payload['frames']:
        key = (frame['camera_id'], int(frame['frame_index']))
        if key in seen:
            raise ValueError(f'Duplicate prediction frame {key}')
        seen.add(key)
        if abs(float(frame['timestamp_ms']) - key[1] * 1000.0 / meta['sample_fps']) > 1.0:
            raise ValueError(f'Prediction timestamp mismatch at {key}')
        rows = []
        for row in frame['detections']:
            audit['input_detections'] += 1
            if row.get('global_track_id') is None:
                audit['without_track_id'] += 1
                continue
            if not valid_box(row):
                raise ValueError(f'Invalid predicted box at {key}')
            rows.append(row)
        # TrackEval requires one observation per identity per camera/frame.
        # Preserve every detection: duplicate occurrences get singleton IDs.
        # Dropping them would artificially hide false positives.
        used = set()
        unique_rows = []
        for occurrence, row in enumerate(sorted(rows, key=lambda r: -float(r.get('score') or 0))):
            identity = row['global_track_id']
            if identity in used:
                row = dict(row, global_track_id=('duplicate', identity, key, occurrence))
                audit['duplicate_identity_occurrences_split'] += 1
            used.add(identity)
            unique_rows.append(row)
        rows = unique_rows
        frames[key] = rows
    expected = {(cam, t) for cam in meta['camera_ids'] for t in range(meta['frame_count'])}
    if seen != expected:
        raise ValueError(f'Prediction coverage mismatch: missing={len(expected-seen)}, extra={len(seen-expected)}')
    audit['tracked_observations'] = sum(map(len, frames.values()))
    return frames, dict(audit)


def make_data(reference, prediction, keys, class_aware=True):
    gt_map, pred_map = {}, {}
    data = {'gt_ids': [], 'tracker_ids': [], 'similarity_scores': [], 'num_timesteps': len(keys)}
    for key in keys:
        gt, pred = reference.get(key, []), prediction.get(key, [])
        def remap(rows, mapping):
            return np.asarray([mapping.setdefault(r['global_track_id'], len(mapping)) for r in rows], dtype=int)
        data['gt_ids'].append(remap(gt, gt_map))
        data['tracker_ids'].append(remap(pred, pred_map))
        similarity = iou_matrix(gt, pred)
        if class_aware:
            same = np.asarray([r['class_id'] for r in gt])[:, None] == np.asarray([r['class_id'] for r in pred])[None, :]
            similarity *= same
        data['similarity_scores'].append(similarity)
    data.update(num_gt_ids=len(gt_map), num_tracker_ids=len(pred_map),
                num_gt_dets=sum(map(len, data['gt_ids'])), num_tracker_dets=sum(map(len, data['tracker_ids'])))
    return data


def metric_suite():
    import trackeval
    return {'HOTA': trackeval.metrics.HOTA(),
            'CLEAR': trackeval.metrics.CLEAR({'THRESHOLD': 0.5, 'PRINT_CONFIG': False}),
            'Identity': trackeval.metrics.Identity({'THRESHOLD': 0.5, 'PRINT_CONFIG': False})}


def summarize(raw):
    return {'HOTA': 100 * float(np.mean(raw['HOTA']['HOTA'])),
            'DetA': 100 * float(np.mean(raw['HOTA']['DetA'])),
            'AssA': 100 * float(np.mean(raw['HOTA']['AssA'])),
            'MOTA': 100 * float(raw['CLEAR']['MOTA']),
            'IDF1': 100 * float(raw['Identity']['IDF1']),
            **{k: int(raw['CLEAR'][k]) for k in ['IDSW', 'CLR_TP', 'CLR_FN', 'CLR_FP', 'Frag']}}


def evaluate_sequence(sequence_dir, prediction_path, class_aware=True):
    meta, gt, gt_audit = load_reference(sequence_dir)
    pred, pred_audit = load_prediction(prediction_path, meta)
    suite = metric_suite()
    camera_raw = {}
    for camera in meta['camera_ids']:
        keys = [(camera, t) for t in range(meta['frame_count'])]
        data = make_data(gt, pred, keys, class_aware)
        camera_raw[camera] = {name: metric.eval_sequence(data) for name, metric in suite.items()}
    combined = {name: metric.combine_sequences({cam: raw[name] for cam, raw in camera_raw.items()}) for name, metric in suite.items()}
    keys = [(cam, t) for t in range(meta['frame_count']) for cam in sorted(meta['camera_ids'])]
    global_data = make_data(gt, pred, keys, class_aware)
    global_raw = {name: metric.eval_sequence(global_data) for name, metric in suite.items()}
    return {'camera_combined': summarize(combined),
            'per_camera': {cam: summarize(raw) for cam, raw in camera_raw.items()},
            'global_observation_stream': summarize(global_raw),
            'audit': {'reference': gt_audit, 'prediction': pred_audit,
                      'sample_fps': meta['sample_fps'], 'frames_per_camera': meta['frame_count'],
                      'timebase': meta['timebase_source'], 'sync_status': meta['sync_status']}}, combined, global_raw


def combine_runs(raw_by_sequence):
    return summarize({name: metric.combine_sequences({seq: raw[name] for seq, raw in raw_by_sequence.items()})
                      for name, metric in metric_suite().items()})
