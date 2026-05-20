"""
Stage 5 — Merge all JSONL window files into cancer_genes.csv for training.

Reads:
  sequences/*_windows.jsonl.gz   (label=1, cancer variants)
  sequences/normals_windows.jsonl.gz  (label=0, random normal windows)

Writes:
  cancer_genes.csv  —  columns: sequence, label

Usage:
  python 05_build_csv.py                          # writes ./cancer_genes.csv
  python 05_build_csv.py --out /path/to/out.csv   # custom path
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from pathlib import Path


def main(out_csv: str = "cancer_genes.csv") -> int:
    seq_dir = Path("sequences")
    if not seq_dir.exists():
        print("sequences/ not found. Run stages 1–4 first.", file=sys.stderr)
        return 1

    jsonl_files = sorted(seq_dir.glob("*_windows.jsonl.gz"))
    if not jsonl_files:
        print("No *_windows.jsonl.gz files found in sequences/.", file=sys.stderr)
        return 1

    print(f"Reading from: {[f.name for f in jsonl_files]}", flush=True)

    rows: list[tuple[str, int]] = []
    n_cancer = n_normal = 0

    for jf in jsonl_files:
        with gzip.open(jf, "rt") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    rows.append((rec["seq"], int(rec["label"])))
                    if rec["label"] == 1:
                        n_cancer += 1
                    else:
                        n_normal += 1
                except Exception:
                    continue

    if n_normal == 0:
        print("WARNING: No normal sequences (label=0) found. "
              "Run stage 4 (04_generate_negatives.py) first.", file=sys.stderr)

    # Shuffle before writing so train/val/test splits are random
    random.seed(42)
    random.shuffle(rows)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["sequence", "label"])
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
    args = p.parse_args()
    sys.exit(main(args.out))
