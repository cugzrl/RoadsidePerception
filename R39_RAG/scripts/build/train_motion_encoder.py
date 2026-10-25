#!/usr/bin/env python3
"""Train only on the four build sequences, then write a separate encoded bank."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / '.rag_deps'), str(ROOT / 'R39_RAG/src')]
from r39_rag.config import load_knowledge
from r39_rag.knowledge.embedding import MotionEncoder, inputs, encode_arrays
from r39_rag.knowledge.kv_motion import assert_build_only


def train(model, temporal, scene, future, mask, epochs, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = np.flatnonzero(mask)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        permutation = indices[torch.randperm(len(indices), generator=generator).numpy()]
        for start in range(0, len(permutation), 256):
            batch = permutation[start:start+256]
            loss, _ = model.loss(temporal[batch], scene[batch], future[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        yield epoch + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=20261025)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory to preserve experiments')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = load_knowledge()
    source = ROOT / 'R39_RAG/knowledge_base/kv_motion_class.npz'
    with np.load(source, allow_pickle=True) as bank:
        arrays = {k: bank[k] for k in bank.files}
    sequences = sorted(set(arrays['sequence_id'].tolist()))
    assert_build_only(config, sequences)
    if sequences != sorted(config['build_sequences']):
        raise ValueError('Bank must contain exactly the four configured build sequences')
    temporal, scene = map(torch.from_numpy, inputs(arrays['key_xy'], arrays['class_name']))
    future = torch.from_numpy(arrays['value_rel'].reshape(-1, 8).astype(np.float32) / 10)
    # The final build sequence is validation; neither test sequence is consulted.
    validation_sequence = config['build_sequences'][-1]
    val = arrays['sequence_id'] == validation_sequence
    model = MotionEncoder()
    best, best_epoch, history = float('inf'), 1, []
    for epoch in train(model, temporal, scene, future, ~val, args.epochs, args.seed):
        model.eval()
        with torch.inference_mode():
            h, s = model.encode(temporal[val], scene[val])
            loss = torch.nn.functional.smooth_l1_loss(model.forecast(torch.cat([h,s], -1)), future[val]).item()
        history.append({'epoch': epoch, 'validation_future_loss': loss})
        if loss < best:
            best, best_epoch = loss, epoch
        if epoch % 5 == 0:
            print(f'编码器验证 epoch={epoch} loss={loss:.5f}', flush=True)
    # Refit all build data for the epoch budget selected without test labels.
    torch.manual_seed(args.seed)
    model = MotionEncoder()
    for _ in train(model, temporal, scene, future, np.ones(len(future), dtype=bool), best_epoch, args.seed):
        pass
    encoder_file = args.output / 'motion_encoder.pt'
    torch.save({'dim': model.dim, 'state_dict': model.state_dict()}, encoder_file)
    arrays['embeddings'] = encode_arrays(model, arrays['key_xy'], arrays['class_name'])
    meta = json.loads(str(arrays['meta_json']))
    meta.update(retrieval_type='encoded_faiss_v1', encoder_file=encoder_file.name,
                encoder_sha256=hashlib.sha256(encoder_file.read_bytes()).hexdigest(),
                training_sequences=sequences, validation_sequence=validation_sequence,
                selected_epochs=best_epoch, seed=args.seed, future_used_at_query=False)
    arrays['meta_json'] = np.array(json.dumps(meta))
    np.savez_compressed(args.output / 'encoded_motion.npz', **arrays)
    import faiss
    index = faiss.IndexFlatIP(arrays['embeddings'].shape[1])
    index.add(arrays['embeddings'])
    faiss.write_index(index, str(args.output / 'motion.faiss'))
    report = {'meta': meta, 'validation_history': history, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'index_type': 'IndexFlatIP (exact search; appropriate for this small bank)',
              'model': '2-layer 64-dimensional temporal Transformer + scene MLP, future autoencoder, soft future-neighborhood alignment',
              'adaptation': 'Inspired by RAG-TP; no claim of full paper reproduction'}
    (args.output / 'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'windows': len(future), 'selected_epochs': best_epoch, 'validation_loss': best}, indent=2), flush=True)


if __name__ == '__main__':
    main()
