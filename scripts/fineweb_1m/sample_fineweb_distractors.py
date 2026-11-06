#!/usr/bin/env python3
"""Sample the 900k FineWeb-Edu distractors for the 1M-document setting (Table 2).

This is the exact script used for the paper. It streams
`HuggingFaceFW/fineweb-edu` config `sample-10BT`, pseudo-shuffles it with
`IterableDataset.shuffle(seed=42, buffer_size=20000)`, skips docs whose stripped
text is shorter than 200 characters, de-duplicates on the FineWeb-Edu `id`, and
writes the first 900,000 docs as (docid, text, url) with docid = fineweb_<i>.
The paper sample was drawn with datasets==4.8.5.

You normally do NOT need to run this. Streaming shuffle depends on the upstream
shard list and the `datasets` version, so a re-run is not guaranteed to give the
same sample. The exact sample used in the paper is on the Hub:
    https://huggingface.co/datasets/Tevatron/browsecomp-plus-fineweb-edu-900k-distractors
and `build_1m_corpus.py` downloads it for you.

Output: corpus/fineweb_distractors_900k/data.parquet (configurable)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm


SCHEMA = pa.schema([
    ("docid", pa.string()),
    ("text", pa.string()),
    ("url", pa.string()),
])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=900_000,
                    help="Number of distractor docs to write.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-buffer", type=int, default=20_000,
                    help="Streaming-shuffle buffer size.")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] / "corpus" / "fineweb_distractors_900k" / "data.parquet")
    ap.add_argument("--min-text-len", type=int, default=200,
                    help="Skip docs shorter than this many characters.")
    ap.add_argument("--row-group-size", type=int, default=10_000)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"streaming {args.dataset}/{args.config} ...")
    print(f"target n={args.n:,}  seed={args.seed}  shuffle_buffer={args.shuffle_buffer:,}")
    print(f"out={args.out}")
    print()

    ds = load_dataset(args.dataset, name=args.config, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    writer = pq.ParquetWriter(args.out, SCHEMA, compression="zstd")
    t0 = time.time()
    n_written = 0
    n_seen = 0
    n_skipped_short = 0
    n_skipped_dup_id = 0
    seen_ids: set[str] = set()

    batch_docid: list[str] = []
    batch_text: list[str] = []
    batch_url: list[str] = []

    def _flush() -> None:
        if not batch_docid:
            return
        tbl = pa.table(
            {"docid": batch_docid, "text": batch_text, "url": batch_url},
            schema=SCHEMA,
        )
        writer.write_table(tbl, row_group_size=args.row_group_size)
        batch_docid.clear(); batch_text.clear(); batch_url.clear()

    pbar = tqdm(total=args.n, unit="doc", smoothing=0.1)
    for rec in ds:
        n_seen += 1
        text = (rec.get("text") or "").strip()
        if len(text) < args.min_text_len:
            n_skipped_short += 1
            continue
        fw_id = rec.get("id") or f"_seen{n_seen}"
        if fw_id in seen_ids:
            n_skipped_dup_id += 1
            continue
        seen_ids.add(fw_id)

        docid = f"fineweb_{n_written}"
        url = rec.get("url") or ""
        batch_docid.append(docid)
        batch_text.append(text)
        batch_url.append(url)
        n_written += 1
        pbar.update(1)

        if len(batch_docid) >= args.row_group_size:
            _flush()

        if n_written >= args.n:
            break

    _flush()
    writer.close()
    pbar.close()

    elapsed = time.time() - t0
    print()
    print(f"wrote {n_written:,} docs in {elapsed/60:.1f} min")
    print(f"  seen {n_seen:,}  skipped_short {n_skipped_short:,}  skipped_dup_id {n_skipped_dup_id:,}")
    print(f"  file: {args.out}  ({args.out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
