"""Tokenize enough text for the 50M-token budget into train.bin / val.bin.

Streams the corpus from the Hugging Face Hub (no full download), encodes it
with the GPT-2 BPE (tiktoken), appends <|endoftext|> after every document and
writes uint16 token streams in the same format as data/openwebtext/prepare.py.
The first --val-tokens go to val.bin and the next --train-tokens to train.bin.

  python experiments/assignment13/prepare_data.py                      # FineWeb-Edu
  python experiments/assignment13/prepare_data.py --source tinystories
  python experiments/assignment13/prepare_data.py --source random --train-tokens 200000   # offline smoke test
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SOURCES = {
    'fineweb_edu': dict(path='HuggingFaceFW/fineweb-edu', name='sample-10BT'),
    'tinystories': dict(path='roneneldan/TinyStories', name=None),
}


def stream_tokens(source, limit, docs_per_batch=1024):
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding('gpt2')
    spec = SOURCES[source]
    ds = load_dataset(spec['path'], name=spec['name'], split='train', streaming=True)
    batch, produced = [], 0
    for doc in ds:
        batch.append(doc['text'])
        if len(batch) < docs_per_batch:
            continue
        for ids in enc.encode_ordinary_batch(batch):
            ids.append(enc.eot_token)
            arr = np.asarray(ids, dtype=np.uint16)
            produced += arr.size
            yield arr
        batch = []
        print(f"  {produced / 1e6:6.1f}M / {limit / 1e6:.1f}M tokens", flush=True)
        if produced >= limit:
            return
    for ids in enc.encode_ordinary_batch(batch):
        yield np.asarray(ids + [enc.eot_token], dtype=np.uint16)


def split_stream(chunks, val_tokens, train_tokens):
    buf = np.concatenate(list(chunks))
    assert buf.size >= val_tokens + train_tokens, f"corpus ran out at {buf.size} tokens"
    return buf[:val_tokens], buf[val_tokens:val_tokens + train_tokens]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='fineweb_edu', choices=[*SOURCES, 'random'])
    ap.add_argument('--train-tokens', type=int, default=55_000_000)
    ap.add_argument('--val-tokens', type=int, default=1_000_000)
    ap.add_argument('--out-dir', default=None, help="default: experiments/assignment13/data/<source>")
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else HERE / 'data' / args.source
    out.mkdir(parents=True, exist_ok=True)
    total = args.val_tokens + args.train_tokens

    if args.source == 'random':
        rng = np.random.default_rng(0)
        val = rng.integers(0, 50257, args.val_tokens, dtype=np.uint16)
        train = rng.integers(0, 50257, args.train_tokens, dtype=np.uint16)
    else:
        print(f"streaming {args.source} until {total / 1e6:.1f}M tokens")
        val, train = split_stream(stream_tokens(args.source, total), args.val_tokens, args.train_tokens)

    val.tofile(out / 'val.bin')
    train.tofile(out / 'train.bin')
    meta = {'source': args.source, 'tokenizer': 'gpt2', 'train_tokens': int(train.size),
            'val_tokens': int(val.size)}
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}: {meta}", flush=True)
    if args.source != 'random':
        # the Hub streaming reader leaves a non-daemon thread behind that blocks
        # interpreter shutdown forever once the files are written; don't wait on it
        sys.stderr.flush()
        os._exit(0)


if __name__ == '__main__':
    main()
