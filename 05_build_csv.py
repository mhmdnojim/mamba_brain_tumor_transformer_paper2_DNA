"""
Stage 5 — Merge all JSONL window files into cancer_genes.csv for training.

Reads (default mode):
  sequences/*_windows.jsonl.gz        (label=1, cancer variants)
  sequences/normals_windows.jsonl.gz  (label=0, random normal windows)

Reads (--gc_matched mode):
  sequences/cancer_windows_gcmatched.jsonl.gz
  sequences/normals_windows_gcmatched.jsonl.gz

Writes:
  cancer_genes.csv  —  columns: sequence, chromosome, start, label

Usage:
  python 05_build_csv.py                          # writes ./cancer_genes.csv
  python 05_build_csv.py --out /path/to/out.csv   # custom path
  python 05_build_csv.py --gc_matched --out dataset/cancer_genes_gcmatched.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from pathlib import Path


def main(out_csv: str = "cancer_genes.csv", gc_matched: bool = False,
         negatives: str | None = None) -> int:
    seq_dir = Path("sequences")
    if not seq_dir.exists():
        print("sequences/ not found. Run stages 1-4 first.", file=sys.stderr)
        return 1

    if negatives is not None:
        # Explicit negatives file (e.g. from 04c_generate_negatives_matched.py)
        neg_path = Path(negatives)
        if not neg_path.exists():
            print(f"{neg_path} not found.", file=sys.stderr)
            return 1
        cancer_files = [f for f in sorted(seq_dir.glob("*_windows.jsonl.gz"))
                        if "normals" not in f.name and "matched" not in f.name
                        and "gcmatched" not in f.name]
        jsonl_files = cancer_files + [neg_path]
        print(f"Mode: custom negatives ({neg_path.name})")
    elif gc_matched:
        cancer_file = seq_dir / "cancer_windows_gcmatched.jsonl.gz"
        normal_file = seq_dir / "normals_windows_gcmatched.jsonl.gz"
        for f in [cancer_file, normal_file]:
            if not f.exists():
                print(f"{f} not found. Run 04b_select_gcmatched_negatives.py first.",
                      file=sys.stderr)
                return 1
        jsonl_files = [cancer_file, normal_file]
        print("Mode: GC-matched dataset (04b subsampling)")
    else:
        jsonl_files = [f for f in sorted(seq_dir.glob("*_windows.jsonl.gz"))
                       if "gcmatched" not in f.name and "matched" not in f.name]

    if not jsonl_files:
        print("No *_windows.jsonl.gz files found in sequences/.", file=sys.stderr)
        return 1

    print(f"Reading from: {[f.name for f in jsonl_files]}", flush=True)

    rows: list[tuple[str, str, int, int]] = []
    n_cancer = n_normal = 0

    for jf in jsonl_files:
        with gzip.open(jf, "rt") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seq   = rec["seq"]
                    chrom = rec.get("chromosome") or rec.get("chrom", "")
                    start = int(rec.get("start", -1))
                    label = int(rec["label"])
                    rows.append((seq, chrom, start, label))
                    if label == 1:
                        n_cancer += 1
                    else:
                        n_normal += 1
                except Exception:
                    continue

    if n_normal == 0:
        print("ERROR: No normal sequences (label=0) found. "
              "Run stage 4/4c and make sure the negatives JSONL has records.",
              file=sys.stderr)
        return 1

    # Shuffle before writing so train/val/test splits are random
    random.seed(42)
    random.shuffle(rows)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["sequence", "chromosome", "start", "label"])
        writer.writerows(rows)

    size_mb = out_path.stat().st_size / 1e6
    print("=" * 50)
    print(f"  Output          : {out_path}  ({size_mb:.1f} MB)")
    print(f"  Cancer (label=1): {n_cancer:,}")
    print(f"  Normal (label=0): {n_normal:,}")
    print(f"  Total           : {len(rows):,}")
    balance = n_normal / max(1, n_cancer + n_normal) * 100
    print(f"  Class balance   : {balance:.1f}% normal  "
          f"{'✓ balanced' if 40 < balance < 60 else '⚠ imbalanced'}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="cancer_genes.csv",
                   help="Output CSV path (default: cancer_genes.csv)")
    p.add_argument("--gc_matched", action="store_true",
                   help="Use GC-matched files from 04b_select_gcmatched_negatives.py")
    p.add_argument("--negatives", type=str, default=None,
                   help="Explicit path to negatives JSONL.gz (e.g. from 04c)")
    args = p.parse_args()
    sys.exit(main(args.out, gc_matched=args.gc_matched, negatives=args.negatives))
