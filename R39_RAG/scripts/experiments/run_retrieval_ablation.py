#!/usr/bin/env python3
"""Run matched-input ablations without overwriting historical results."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'R39_RAG/src'))
from r39_rag.config import load_knowledge


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encoded-bank', type=Path, required=True)
    parser.add_argument('--groups', nargs='+', choices=['stmc', 'memory_cv', 'kv_legacy', 'kv_encoded'],
                        default=['stmc', 'memory_cv', 'kv_legacy', 'kv_encoded'])
    parser.add_argument('--sequences', nargs='*')
    parser.add_argument('--device', default='cuda:0', help='cpu builds graphs on CPU while RAMA still solves on CUDA')
    args = parser.parse_args()
    config = load_knowledge()
    sequences = args.sequences or config['holdout_sequences']
    if not set(sequences) <= set(config['holdout_sequences']):
        raise ValueError('Tracking evaluation uses only configured test sequences')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / '.rag_deps') + os.pathsep + env.get('PYTHONPATH', '')
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    for seq in sequences:
        detections = ROOT / 'outputs/r39_ablation/candidates' / f'{seq}_merged.json'
        features = ROOT / 'outputs/r39_ablation/features' / f'{seq}_mot20_low'
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in
                  [detections, features/'features.npy', features/'feature_metadata.json']}
        for group in args.groups:
            output = args.output.resolve() / seq / group
            if output.exists():
                raise FileExistsError(f'Use a new output directory: {output}')
            output.mkdir(parents=True)
            flags = ['tracker.position_state.active_cameras_only=true',
                     'tracker.short_recovery.enabled=true', 'tracker.short_recovery.one_to_one=true',
                     'tracker.link_guard.enabled=true', f'tracker.link_guard.log_path={output}/link_guard.jsonl',
                     'tracker.knowledge.enabled=false', 'tracker.static_pseudo.enabled=false',
                     'tracker.cross_camera_prior.enabled=false', 'tracker.low_confidence_thresh=0.30',
                     'tracker.class_high_thresh=[0.65,0.65,0.65,0.70,0.72,0.72,0.75]',
                     f'tracker.motion_memory.enabled={str(group != "stmc").lower()}',
                     'tracker.motion_memory.min_missed=4', 'tracker.motion_memory.max_missed=30',
                     'tracker.motion_memory.reid_min=0.75', 'tracker.motion_memory.static_reid_min=0.85',
                     'tracker.motion_memory.class_reid_min=[0.80,0.80,0.80,0.75,0.75,0.75,0.78]',
                     'tracker.motion_memory.strict_protect=true',
                     f'tracker.motion_memory.log_path={output}/memory_log.jsonl',
                     'tracker.motion_memory.use_topology=false',
                     f'tracker.motion_memory.use_kv={str(group.startswith("kv_")).lower()}',
                     'tracker.motion_memory.kv_class_aware=true', 'tracker.motion_memory.kv_diverse=true',
                     'tracker.motion_memory.kv_blend=true',
                     f'tracker.motion_memory.kv_fusion={"mixture" if group == "kv_encoded" else "legacy"}',
                     f'tracker.motion_memory.kv_path={args.encoded_bank.resolve() if group == "kv_encoded" else ROOT / "R39_RAG/knowledge_base/kv_motion_class.npz"}']
            command = [sys.executable, str(ROOT / 'scripts/run_tracking.py'), '--detections', str(detections),
                       '--features-dir', str(features), '--output-dir', str(output), '--device', args.device,
                       '--stmc-override', *flags]
            print(f'开始 {seq} {group}', flush=True)
            start = time.perf_counter()
            with (output / 'run.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            meta = {'group': group, 'sequence': seq, 'elapsed_s': time.perf_counter()-start,
                    'input_sha256': hashes, 'command': command, 'overrides': flags,
                    'baseline': 'STMC with common position/short-gap/association fixes; all knowledge and memory disabled',
                    'parameter_policy': 'existing repository working points, fixed before this evaluation',
                    'test_labels_used_for_association_or_tuning': False}
            (output / 'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
            print(f'完成 {seq} {group}: {meta["elapsed_s"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
