#!/usr/bin/env python3
"""Build the 1M-document setting (Table 2): BrowseComp-Plus 100k + 900k FineWeb-Edu.

Prerequisite: the 100k setup from README steps 2-3 (corpus parquet, flat file
tree, and the filename<->docid map).

Steps (each can be skipped if its output already exists):
  1. Get the 900k distractors: download the exact paper sample from
     Tevatron/browsecomp-plus-fineweb-edu-900k-distractors (or pass a local
     parquet with --distractors) and verify its sha256.
  2. Merged corpus parquet: BCP rows first (docids unchanged, so gold labels
     still resolve), then the 900k distractors. 1,000,195 docs.
  3. Flat file tree for the agents: the BCP tree is hardlinked (copied if
     hardlinks are not possible) and each distractor is written to
     fineweb/<docid>.txt.
  4. Filename<->docid map: the 100k map plus "fineweb/<docid>.txt" -> docid.

Then build the BM25 index on the merged parquet:
    python scripts/build_bm25_index.py \\
        --corpus corpus/browsecomp_plus_1m.parquet --out runs/bm25_full_1m
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_REPO = "Tevatron/browsecomp-plus-fineweb-edu-900k-distractors"
HF_FILE = "data/train-00000-of-00001.parquet"
SHA256 = "3cfcfafe1b7b9ac4842addfad3ae6552d5b3847d4559cc79e0b5c01eedeabbae"
N_DISTRACTORS = 900_000

SCHEMA = pa.schema([("docid", pa.string()), ("text", pa.string()), ("url", pa.string())])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def get_distractors(local: Path | None, skip_verify: bool) -> Path:
    if local is None:
        from huggingface_hub import hf_hub_download
        print(f"downloading {HF_REPO}/{HF_FILE} ...")
        local = Path(hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset"))
    print(f"distractors: {local}")
    if not skip_verify:
        digest = sha256(local)
        if digest != SHA256:
            raise SystemExit(f"sha256 mismatch: got {digest}, expected {SHA256}. "
                             "Pass --skip-verify to use a different sample on purpose.")
        print("  sha256 OK (matches the paper sample)")
    n = pq.ParquetFile(local).metadata.num_rows
    if n != N_DISTRACTORS:
        print(f"  WARNING: {n:,} rows (paper uses {N_DISTRACTORS:,})")
    return local


def _as_strings(tbl: pa.Table) -> pa.Table:
    return pa.table({c: tbl.column(c).cast(pa.string()) for c in SCHEMA.names}, schema=SCHEMA)


def merge_parquet(bcp: Path, distractors: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out, SCHEMA, compression="zstd")
    n = 0
    for src in (bcp, distractors):
        pf = pq.ParquetFile(src)
        for rg in tqdm(range(pf.num_row_groups), desc=f"merge {src.name}", unit="rg"):
            tbl = _as_strings(pf.read_row_group(rg, columns=SCHEMA.names))
            writer.write_table(tbl, row_group_size=10_000)
            n += tbl.num_rows
    writer.close()
    print(f"merged corpus: {n:,} docs -> {out}")


def build_tree(bcp_docs: Path, distractors: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    files = [p for p in bcp_docs.rglob("*") if p.is_file()]
    n_copied = 0
    for src in tqdm(files, desc="BCP files", unit="file"):
        dst = out / src.relative_to(bcp_docs)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
            n_copied += 1
    if n_copied:
        print(f"  {n_copied:,} BCP files copied (hardlink not possible)")

    fw_dir = out / "fineweb"
    fw_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(distractors)
    pbar = tqdm(total=pf.metadata.num_rows, desc="FineWeb files", unit="file")
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["docid", "text"])
        for did, txt in zip(tbl.column("docid").to_pylist(), tbl.column("text").to_pylist()):
            (fw_dir / f"{did}.txt").write_text(txt or "", encoding="utf-8")
            pbar.update(1)
    pbar.close()
    print(f"file tree -> {out}")


def build_map(map_100k: Path, distractors: Path, out: Path) -> None:
    base = json.loads(map_100k.read_text())
    rel = dict(base.get("relpath_to_docid", base))
    n_base = len(rel)
    docids = pq.read_table(distractors, columns=["docid"]).column("docid").to_pylist()
    for did in docids:
        rel[f"fineweb/{did}.txt"] = did
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"relpath_to_docid": rel}))
    print(f"filename map: {n_base:,} + {len(docids):,} = {len(rel):,} entries -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distractors", type=Path, default=None,
                    help="Local distractor parquet (default: download from the Hub).")
    ap.add_argument("--skip-verify", action="store_true", help="Skip the sha256 check.")
    ap.add_argument("--bcp-parquet", type=Path, default=REPO_ROOT / "corpus" / "browsecomp_plus.parquet")
    ap.add_argument("--bcp-docs", type=Path, default=REPO_ROOT / "corpus" / "bcp_plus")
    ap.add_argument("--filename-map", type=Path, default=REPO_ROOT / "runs" / "corpus_filename_docid_map.json")
    ap.add_argument("--out-parquet", type=Path, default=REPO_ROOT / "corpus" / "browsecomp_plus_1m.parquet")
    ap.add_argument("--out-docs", type=Path, default=REPO_ROOT / "corpus" / "bcp_plus_1m")
    ap.add_argument("--out-map", type=Path, default=REPO_ROOT / "runs" / "corpus_filename_docid_map_1m.json")
    args = ap.parse_args()

    distractors = get_distractors(args.distractors, args.skip_verify)
    if args.out_parquet.exists():
        print(f"skip merge, exists: {args.out_parquet}")
    else:
        merge_parquet(args.bcp_parquet, distractors, args.out_parquet)
    build_tree(args.bcp_docs, distractors, args.out_docs)
    build_map(args.filename_map, distractors, args.out_map)
    print("\nnext: python scripts/build_bm25_index.py "
          f"--corpus {args.out_parquet} --out runs/bm25_full_1m")


if __name__ == "__main__":
    main()
