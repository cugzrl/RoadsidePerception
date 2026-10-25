#!/usr/bin/env python3
"""Measure future prediction and retrieval latency on untouched test sequences."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / '.rag_deps'), str(ROOT / 'R39_RAG/src'), str(ROOT / 'R39_RAG/scripts/build')]
from build_kv_knowledge import build
from r39_rag.config import load_knowledge
from r39_rag.knowledge.kv_motion import KVIndex, constant_velocity, VALUE_OFFSETS
from r39_rag.knowledge.prediction_eval import summarize, track_macro, assert_library_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_knowledge()
    learned = KVIndex.load(args.bank)
    legacy = KVIndex.load(ROOT / 'R39_RAG/knowledge_base/kv_motion_class.npz')
    assert_library_split(sorted(set(learned.sequence_id)), config['holdout_sequences'], config['holdout_sequences'])
    reference = build(config, config['holdout_sequences'], allow_holdout=True)
    results, tracks, sequences = {}, [], []
    names = ['CV', 'legacy_top1_fallback', 'encoded_top1_fallback', 'encoded_mode_fallback']
    errors = {name: [] for name in names}
    timing = {'legacy': [], 'encoded': []}
    hits = {'legacy': 0, 'encoded': 0}
    for i, row in enumerate(reference['rows']):
        cv = constant_velocity(row['key_xy'], VALUE_OFFSETS)
        forecasts = {'CV': cv}
        for name, bank in [('legacy', legacy), ('encoded', learned)]:
            started = time.perf_counter()
            found = bank.query(row['key_xy'], class_name=row['class_name'], diverse=True)
            timing[name].append((time.perf_counter()-started)*1000)
            hit = bool(found.get('reliable'))
            hits[name] += hit
            forecasts[name + '_top1_fallback'] = found['matches'][0]['future_xy'] if hit else cv
            if name == 'encoded':
                prediction = cv
                if hit:
                    futures = np.stack([m['future_xy'] for m in found['matches']])
                    scores = np.array([m['score'] for m in found['matches']])
                    weights = np.exp((scores-scores.max())/.15)
                    weights /= weights.sum()
                    # Select one coherent trajectory by neighborhood mass, not GT error.
                    pairwise = np.linalg.norm(futures[:,None]-futures[None,:], axis=-1).mean(-1)
                    mass = np.exp(-pairwise**2/8) @ weights
                    candidate = int(np.argmax(mass))
                    if mass[candidate] >= .55:
                        prediction = futures[candidate]
                forecasts['encoded_mode_fallback'] = prediction
        for name in names:
            errors[name].append(np.linalg.norm(forecasts[name] - row['value_xy'], axis=1))
        tracks.append(f"{row['sequence_id']}|{row['camera_id']}|{row['local_track_id']}")
        sequences.append(row['sequence_id'])
        if i % 1000 == 0:
            print(f'检索评估 {i}/{len(reference["rows"])}', flush=True)
    arrays = {name: np.asarray(value) for name,value in errors.items()}
    for name, values in arrays.items():
        results[name] = {**summarize(values), 'track_macro': track_macro(values, tracks),
                         'by_sequence': {seq: summarize(values[np.asarray(sequences)==seq]) for seq in sorted(set(sequences))}}
    report = {'reference_type': 'quality-filtered pseudo trajectories, not human GT',
              'test_sequences': config['holdout_sequences'], 'training_sequences': learned.meta['training_sequences'],
              'selection': 'Top1 or history-only neighborhood mode; no oracle future selection',
              'methods': results, 'hit_count': hits,
              'latency_ms': {name: {'mean': float(np.mean(values)), 'p95': float(np.percentile(values,95))} for name,values in timing.items()},
              'filter_counts': reference['filter_counts']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({name: {k:v for k,v in result.items() if k in ['ade_m','fde_m','count']} for name,result in results.items()}, indent=2))


if __name__ == '__main__':
    main()
