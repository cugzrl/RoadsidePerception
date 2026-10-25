#!/usr/bin/env python3
"""Evaluate complete held-out tracking runs with official TrackEval metrics."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / '.rag_deps'), str(ROOT / 'R39_RAG/src')]
from r39_rag.config import load_knowledge
from r39_rag.evaluation.tracking import evaluate_sequence, combine_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--groups', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--class-agnostic', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_knowledge()
    report = {'reference_type': 'SAM3 structured detections with manual global-ID corrections; pseudo GT',
              'protocol': {'class_aware': not args.class_agnostic, 'clear_iou': 0.5,
                           'hota_alphas': '0.05:0.05:0.95', 'implementation': 'TrackEval 1.3.0',
                           'primary_scope': 'ordinary per-camera MOT, combined by official metric aggregation',
                           'global_scope': 'serialized (frame,camera) observation stream with shared global IDs; supplemental, not a standard MTMC benchmark',
                           'missing_predictions': 'unidentified detections are omitted; unmatched reference boxes count as FN',
                           'duplicate_prediction_ids': 'highest-score box keeps ID; other occurrences keep boxes and receive singleton IDs, counted in audit',
                           'label_filter': 'only invalid boxes or missing identity/class are excluded; no prediction-dependent GT filtering'},
              'methods': {}}
    csv_rows = []
    for group in args.groups:
        results, camera_raw, global_raw = {}, {}, {}
        for sequence in config['holdout_sequences']:
            path = args.run_root / sequence / group / 'tracking.json'
            print(f'评估 {sequence} {group}', flush=True)
            result, camera_raw[sequence], global_raw[sequence] = evaluate_sequence(
                ROOT / 'R39_RAG/data/normalized/sequences' / sequence, path, not args.class_agnostic)
            result['prediction'] = str(path)
            result['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            results[sequence] = result
            csv_rows.append({'method': group, 'sequence': sequence, **result['camera_combined']})
        combined = combine_runs(camera_raw)
        report['methods'][group] = {'sequences': results, 'combined': combined,
                                     'global_observation_stream': combine_runs(global_raw)}
        csv_rows.append({'method': group, 'sequence': 'COMBINED', **combined})
        (args.output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (args.output / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({k: v['combined'] for k,v in report['methods'].items()}, indent=2))


if __name__ == '__main__':
    main()
