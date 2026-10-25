"""Compact RAG-TP-inspired history/scene encoders and exact FAISS retrieval.

Only observed history and static scene attributes enter retrieval keys.
Future trajectories supervise offline training and remain retrieved values.
This is an adaptation for association, not a reproduction of RAG-TP.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .kv_motion import KVIndex, _heading_and_speed, to_heading_frame, map_futures

CLASSES = ['Car', 'Van', 'Bus', 'Tricycle', 'ScooterRider', 'Cyclist', 'Pedestrian', '']


def inputs(keys, classes, history_s=1.0):
    temporal, scene = [], []
    for key, name in zip(np.asarray(keys), classes):
        heading, speed = _heading_and_speed(key, history_s)
        relative = to_heading_frame(key - key[-1], heading)
        velocity = np.gradient(relative, history_s / 4, axis=0)
        acceleration = np.gradient(velocity, history_s / 4, axis=0)
        temporal.append(np.concatenate([relative / 10, velocity / 10, acceleration / 20], axis=1))
        one_hot = np.zeros(len(CLASSES))
        one_hot[CLASSES.index(name) if name in CLASSES else -1] = 1
        scene.append(np.r_[key[-1] / 100, np.cos(heading), np.sin(heading), np.log1p(speed) / 3, one_hot])
    return np.asarray(temporal, dtype=np.float32), np.asarray(scene, dtype=np.float32)


class MotionEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.token = nn.Linear(6, dim)
        self.position = nn.Parameter(torch.randn(1, 5, dim) * .02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, dropout=0.0, activation='gelu', batch_first=True)
        self.history = nn.TransformerEncoder(layer, 2)
        self.scene = nn.Sequential(nn.Linear(13, 48), nn.GELU(), nn.Linear(48, 32))
        self.future = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, dim))
        self.history_decoder = nn.Linear(dim, 30)
        self.future_decoder = nn.Linear(dim, 8)
        self.scene_decoder = nn.Linear(32, 13)
        self.forecast = nn.Sequential(nn.Linear(dim + 32, 96), nn.GELU(), nn.Linear(96, 8))

    def encode(self, temporal, scene):
        h = self.history(self.token(temporal) + self.position).mean(1)
        s = self.scene(scene)
        return h, s

    def key(self, temporal, scene):
        h, s = self.encode(temporal, scene)
        return F.normalize(torch.cat([F.normalize(h, dim=-1), .5 * F.normalize(s, dim=-1)], -1), dim=-1)

    def loss(self, temporal, scene, future):
        h, s = self.encode(temporal, scene)
        f = self.future(future)
        reconstruction = F.smooth_l1_loss(self.history_decoder(h), temporal.flatten(1))
        reconstruction += F.smooth_l1_loss(self.future_decoder(f), future)
        reconstruction += F.smooth_l1_loss(self.scene_decoder(s), scene)
        prediction = F.smooth_l1_loss(self.forecast(torch.cat([h, s], -1)), future)
        # Soft future-neighborhood targets avoid treating similar motion as negatives.
        target = torch.softmax(-torch.cdist(future.detach(), future.detach()).square() / .5, dim=1)
        logits = F.normalize(h, dim=-1) @ F.normalize(f, dim=-1).T / .15
        alignment = F.kl_div(F.log_softmax(logits, dim=1), target, reduction='batchmean')
        return prediction + .3 * reconstruction + .15 * alignment, prediction


def encode_arrays(model, keys, classes, batch_size=512):
    temporal, scene = inputs(keys, classes)
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            output.append(model.key(torch.from_numpy(temporal[start:start+batch_size]),
                                    torch.from_numpy(scene[start:start+batch_size])).numpy())
    return np.ascontiguousarray(np.concatenate(output), dtype=np.float32)


class EmbeddingKVIndex(KVIndex):
    def __init__(self, arrays, meta, model, backend='faiss'):
        super().__init__(arrays, meta)
        self.model = model.eval()
        self.embeddings = np.ascontiguousarray(arrays['embeddings'], dtype=np.float32)
        if len(self.embeddings) != len(self.feature) or not np.isfinite(self.embeddings).all():
            raise ValueError('Embedding rows must match finite knowledge windows')
        self.backend = backend
        self.faiss = None
        if backend == 'faiss':
            import faiss
            faiss.omp_set_num_threads(1)
            self.faiss = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.faiss.add(self.embeddings)
        elif backend != 'numpy':
            raise ValueError(backend)

    @classmethod
    def load(cls, path, backend='faiss'):
        path = Path(path)
        with np.load(path, allow_pickle=True) as archive:
            arrays = {name: archive[name] for name in archive.files}
        meta = json.loads(str(arrays['meta_json']))
        model_path = path.parent / meta['encoder_file']
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != meta['encoder_sha256']:
            raise ValueError('Encoder checksum differs from the indexed encoder')
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model = MotionEncoder(checkpoint['dim'])
        model.load_state_dict(checkpoint['state_dict'])
        return cls(arrays, meta, model, backend)

    def query(self, key_xy, top_k=None, min_score=None, class_name=None, diverse=True):
        key = np.asarray(key_xy, dtype=float)
        if key.shape != (5, 2) or not np.isfinite(key).all() or len(self.feature) == 0:
            return {'reliable': False, 'reason': 'invalid_history', 'matches': []}
        temporal, scene = inputs(key[None], [class_name or ''], float(self.meta.get('history_seconds', 1)))
        with torch.inference_mode():
            vector = self.model.key(torch.from_numpy(temporal), torch.from_numpy(scene)).numpy()
        if self.faiss is not None:
            # The bank is small. Exact full ranking preserves spatial/class recall.
            similarity, chosen = self.faiss.search(vector, len(self.feature))
            similarity, chosen = similarity[0], chosen[0]
        else:
            scores = (self.embeddings @ vector[0])
            chosen = np.argsort(-scores, kind='stable')
            similarity = scores[chosen]
        position = np.linalg.norm(self.end_xy[chosen] - key[-1], axis=1)
        heading, speed = _heading_and_speed(key, float(self.meta.get('history_seconds', 1)))
        direction = np.cos(self.heading[chosen] - heading)
        valid = (position <= float(self.meta.get('spatial_radius_m', 25))) & (direction >= .25)
        if class_name is not None:
            valid &= np.asarray(self.class_name)[chosen] == class_name
        chosen, similarity, position, direction = chosen[valid], similarity[valid], position[valid], direction[valid]
        score = np.exp(-(1 - np.clip(similarity, -1, 1)) / .25) * np.exp(-position / 18) * (.5 + .5 * direction)
        score *= np.exp(-np.abs(self.speed[chosen] - speed) / 8)
        order = np.argsort(-score, kind='stable')
        limit = int(top_k or self.meta.get('top_k', 5))
        floor = float(self.meta.get('min_score', .3) if min_score is None else min_score)
        matches, seen = [], set()
        for idx in order:
            row = int(chosen[idx])
            track = (self.sequence_id[row], self.camera_id[row], int(self.local_track_id[row]))
            if score[idx] < floor or (diverse and track in seen):
                continue
            seen.add(track)
            matches.append({'index': row, 'score': float(score[idx]), 'embedding_similarity': float(similarity[idx]),
                            'sequence_id': self.sequence_id[row], 'camera_id': self.camera_id[row],
                            'local_track_id': int(self.local_track_id[row]), 'track_index': int(self.track_index[row]),
                            'kind': self.kind[row], 'lane_id': self.lane_id[row], 'speed': float(self.speed[row]),
                            'future_xy': map_futures(key[-1], heading, self.value_rel[row]), 'uses_topology_filter': False})
            if len(matches) == limit:
                break
        return {'reliable': len(matches) >= 2, 'reason': 'encoded_history_scene' if len(matches) >= 2 else 'insufficient_independent_support',
                'matches': matches, 'best_score': matches[0]['score'] if matches else 0.0,
                'candidate_count': len(chosen), 'support_tracks': len(matches), 'backend': self.backend,
                'query_heading': heading, 'query_speed': speed, 'uses_topology_filter': False}
