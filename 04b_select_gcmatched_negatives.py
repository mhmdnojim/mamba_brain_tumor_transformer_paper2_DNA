"""
Stage 4b — Select GC-content-matched negatives from the existing normal pool.

No new Ensembl calls required. Works entirely from already-fetched JSONL files:
  sequences/TCGA-GBM_windows.jsonl.gz
  sequences/TCGA-BRCA_windows.jsonl.gz
  sequences/normals_windows.jsonl.gz

Strategy
--------
1. Load all cancer sequences and compute their (chromosome, gc_bin) distribution.
2. Load all normal sequences and compute their (chromosome, gc_bin) distribution.
3. For each GC bin: keep min(cancer_count, normal_count) sequences from each class.
   Excess sequences in the larger class are discarded.
4. Write two new GC-matched JSONL files:
     sequences/cancer_windows_gcmatched.jsonl.gz
     sequences/normals_windows_gcmatched.jsonl.gz

The result is a smaller but GC-balanced dataset.  Run 05_build_csv.py with
--gc_matched to build the training CSV from these files.

Usage:
    python 04b_select_gcmatched_negatives.py
    python 04b_select_gcmatched_negatives.py --bin_width 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gc_content(seq: str) -> float:
    s = seq.upper()
    return (s.count("G") + s.count("C")) / max(len(s), 1)


def gc_bin(seq: str, width: float) -> float:
    """Lower edge of the GC bin for this sequence, rounded to avoid float noise."""
    gc = gc_content(seq)
    return round(int(gc / width) * width, 6)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with gzip.open(path, "rt") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"  Wrote {len(records):,} records → {path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(bin_width: float = 0.05, seed: int = 42, chrom_match: bool = True) -> int:
    seq_dir = Path("sequences")
    rng = random.Random(seed)

    # ── Load cancer sequences ─────────────────────────────────────────────────
    cancer_records: list[dict] = []
    for p in sorted(seq_dir.glob("*_windows.jsonl.gz")):
        if "normals" in p.name or "gcmatched" in p.name:
            continue
        recs = load_jsonl(p)
        cancer_records.extend(recs)
        print(f"  Loaded {len(recs):,} cancer records from {p.name}")

    if not cancer_records:
        print("No cancer sequences found. Run stage 3 first.", file=sys.stderr)
        return 1

    # ── Load normal sequences ─────────────────────────────────────────────────
    normals_path = seq_dir / "normals_windows.jsonl.gz"
    if not normals_path.exists():
        print(f"{normals_path} not found. Run stage 4 first.", file=sys.stderr)
        return 1

    normal_records = load_jsonl(normals_path)
    print(f"  Loaded {len(normal_records):,} normal records from {normals_path.name}")

    # ── Add GC bin to every record ────────────────────────────────────────────
    print(f"\nComputing GC bins (width={bin_width:.2f}) ...")
    for rec in cancer_records:
        rec["_gc_bin"] = gc_bin(rec["seq"], bin_width)
        # Normalise chromosome key: ensure 'chr' prefix
        chrom = str(rec.get("chromosome", "unk"))
        rec["_chrom"] = chrom if chrom.startswith("chr") else "chr" + chrom

    for rec in normal_records:
        rec["_gc_bin"] = gc_bin(rec["seq"], bin_width)
        chrom = str(rec.get("chromosome", "unk"))
        rec["_chrom"] = chrom if chrom.startswith("chr") else "chr" + chrom

    # ── Build per-(chrom, gc_bin) buckets ─────────────────────────────────────
    # key = (chrom, gc_bin) if chrom_match else (gc_bin,)
    def slot_key(rec: dict):
        return (rec["_chrom"], rec["_gc_bin"]) if chrom_match else rec["_gc_bin"]

    cancer_buckets:  dict = defaultdict(list)
    normal_buckets:  dict = defaultdict(list)

    for rec in cancer_records:
        cancer_buckets[slot_key(rec)].append(rec)
    for rec in normal_records:
        normal_buckets[slot_key(rec)].append(rec)

    # ── Match: keep min(cancer, normal) per slot ──────────────────────────────
    all_slots = set(cancer_buckets) | set(normal_buckets)
    selected_cancer: list[dict] = []
    selected_normal: list[dict] = []

    n_cancer_dropped = n_normal_dropped = n_slots_perfect = n_slots_limited = 0

    for slot in sorted(all_slots, key=str):
        c_list = cancer_buckets.get(slot, [])
        n_list = normal_buckets.get(slot, [])
        keep = min(len(c_list), len(n_list))

        if keep == 0:
            continue

        rng.shuffle(c_list)
        rng.shuffle(n_list)

        selected_cancer.extend(c_list[:keep])
        selected_normal.extend(n_list[:keep])

        n_cancer_dropped += len(c_list) - keep
        n_normal_dropped += len(n_list) - keep

        if len(c_list) == len(n_list):
            n_slots_perfect += 1
        else:
            n_slots_limited += 1

    total_kept = len(selected_cancer) + len(selected_normal)
    total_orig = len(cancer_records) + len(normal_records)

    print("\n" + "═" * 58)
    print("  GC-MATCHED SELECTION SUMMARY")
    print("═" * 58)
    print(f"  Original cancer  : {len(cancer_records):>10,}")
    print(f"  Original normal  : {len(normal_records):>10,}")
    print(f"  Original total   : {total_orig:>10,}")
    print("─" * 58)
    print(f"  Kept cancer      : {len(selected_cancer):>10,}  "
          f"({100*len(selected_cancer)/max(1,len(cancer_records)):.1f}% retained)")
    print(f"  Kept normal      : {len(selected_normal):>10,}  "
          f"({100*len(selected_normal)/max(1,len(normal_records)):.1f}% retained)")
    print(f"  Kept total       : {total_kept:>10,}  "
          f"({100*total_kept/max(1,total_orig):.1f}% retained)")
    print(f"  Slots balanced   : {n_slots_perfect:>10,}")
    print(f"  Slots limited    : {n_slots_limited:>10,}  (smaller class capped the larger)")
    print("═" * 58)

    # ── GC distribution check ─────────────────────────────────────────────────
    import statistics

    def gc_stats(records):
        vals = [gc_content(r["seq"]) for r in records]
        if not vals:
            return 0.0, 0.0
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    c_mean, c_std = gc_stats(selected_cancer)
    n_mean, n_std = gc_stats(selected_normal)
    print(f"\n  Post-matching GC content:")
    print(f"    Cancer  : mean={c_mean:.4f}  std={c_std:.4f}")
    print(f"    Normal  : mean={n_mean:.4f}  std={n_std:.4f}")
    print(f"    Δ mean  : {c_mean - n_mean:+.4f}  "
          f"(was +0.0916 before matching)")

    # ── Clean up private keys before writing ──────────────────────────────────
    for rec in selected_cancer + selected_normal:
        rec.pop("_gc_bin", None)
        rec.pop("_chrom", None)

    # ── Write GC-matched JSONL files ──────────────────────────────────────────
    print("\nWriting GC-matched files ...")
    cancer_out = seq_dir / "cancer_windows_gcmatched.jsonl.gz"
    normal_out = seq_dir / "normals_windows_gcmatched.jsonl.gz"
    write_jsonl(selected_cancer, cancer_out)
    write_jsonl(selected_normal, normal_out)

    print(f"\nNext step:")
    print(f"  python 05_build_csv.py --gc_matched --out dataset/cancer_genes_gcmatched.csv")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin_width", type=float, default=0.05,
                   help="GC bin width for matching (default 0.05 = 5%% bins)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for subsampling (default 42)")
    p.add_argument("--no_chrom_match", action="store_true",
                   help="Match only by GC bin, ignore chromosome. "
                        "Default: match by (chromosome, GC bin).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(
        bin_width=args.bin_width,
        seed=args.seed,
        chrom_match=not args.no_chrom_match,
    ))
