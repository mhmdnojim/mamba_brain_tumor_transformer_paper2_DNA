"""
Stage 4 — Generate negative-class (label=0) sequences.

Randomly samples genomic positions from GRCh38, excluding a ±512 bp buffer
around every cancer variant found in variants/*.tsv.gz. Fetches 512 bp DNA
windows from Ensembl REST and writes to sequences/normals_windows.jsonl.gz.

Target count: automatically matches the total cancer-variant count so the
final dataset is balanced (50% cancer / 50% normal).

Run after stage 3.  Resumable — restart to continue from where it stopped.
"""

from __future__ import annotations

import bisect
import gzip
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ENSEMBL = "https://rest.ensembl.org"
HEADERS = {"Accept": "application/json", "User-Agent": "tcga-window-fetcher/1.0"}
WINDOW = 512
LEFT = 255
RIGHT = 256
TARGET_RPS = 13.0
MIN_INTERVAL = 1.0 / TARGET_RPS
MAX_RETRIES = 6
BUFFER = 512        # exclude ±BUFFER bp around any known variant
OVERSAMPLE = 3      # sample 3× candidates to absorb fetch failures

# GRCh38 primary assembly chromosome lengths (bp)
CHR_LENGTHS: dict[str, int] = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}


def chr_to_ensembl(c: str) -> str:
    c = c.removeprefix("chr")
    return "MT" if c == "M" else c


def fetch_one(session: requests.Session, chrom: str, start: int) -> str | None:
    s = start - LEFT
    e = start + RIGHT
    if s < 1:
        return None
    url = f"{ENSEMBL}/sequence/region/human/{chrom}:{s}..{e}:1"
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 200:
            return r.json()["seq"]
        if r.status_code == 429:
            retry_after = float(r.headers.get("Retry-After", delay))
            time.sleep(retry_after)
            delay = max(delay * 2, retry_after)
            continue
        if r.status_code in (400, 404):
            return None
        if 500 <= r.status_code < 600:
            time.sleep(delay)
            delay *= 2
            continue
        return None
    return None


def load_blocked(var_dir: Path) -> dict[str, list[int]]:
    """Return {chrom: sorted_list_of_positions} from all variants TSVs."""
    raw: dict[str, list[int]] = {c: [] for c in CHR_LENGTHS}
    for tsv in sorted(var_dir.glob("*_variants.tsv.gz")):
        df = pd.read_csv(tsv, sep="\t", usecols=["chromosome", "start"],
                         dtype={"chromosome": str, "start": "Int64"})
        for row in df.itertuples(index=False):
            if row.chromosome in raw:
                raw[row.chromosome].append(int(row.start))
    blocked = {c: sorted(v) for c, v in raw.items()}
    total = sum(len(v) for v in blocked.values())
    print(f"Loaded {total:,} blocked positions", flush=True)
    return blocked


def is_blocked(positions: list[int], pos: int) -> bool:
    """O(log n) check: any known variant within ±BUFFER of pos."""
    lo = bisect.bisect_left(positions, pos - BUFFER)
    hi = bisect.bisect_right(positions, pos + BUFFER)
    return lo < hi


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            try:
                done.add(json.loads(line)["variant_id"])
            except Exception:
                continue
    return done


def count_cancer_seqs(seq_dir: Path) -> int:
    total = 0
    for p in seq_dir.glob("*_windows.jsonl.gz"):
        if "normals" in p.name:
            continue
        with gzip.open(p, "rt") as f:
            total += sum(1 for _ in f)
    return total


def sample_candidates(
    blocked: dict[str, list[int]], n: int, seed: int = 42
) -> list[tuple[str, int]]:
    """Sample n (chromosome, position) pairs, skipping blocked zones."""
    rng = random.Random(seed)
    chroms = list(CHR_LENGTHS.keys())
    weights = [CHR_LENGTHS[c] for c in chroms]
    candidates: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()

    attempts = 0
    max_attempts = n * 10
    while len(candidates) < n and attempts < max_attempts:
        chrom = rng.choices(chroms, weights=weights, k=1)[0]
        length = CHR_LENGTHS[chrom]
        pos = rng.randint(LEFT + 1, length - RIGHT)
        key = (chrom, pos)
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        if is_blocked(blocked[chrom], pos):
            attempts += 1
            continue
        candidates.append(key)
        attempts += 1

    return candidates


def main() -> int:
    var_dir = Path("variants")
    seq_dir = Path("sequences")
    out_path = seq_dir / "normals_windows.jsonl.gz"
    seq_dir.mkdir(exist_ok=True)

    if not var_dir.exists():
        print("variants/ not found. Run stage 2 first.", file=sys.stderr)
        return 1

    blocked = load_blocked(var_dir)

    target = count_cancer_seqs(seq_dir)
    if target == 0:
        print("No cancer sequences found. Run stage 3 first.", file=sys.stderr)
        return 1
    print(f"Cancer sequences found : {target:,}", flush=True)

    done = load_done(out_path)
    needed = max(0, target - len(done))
    print(f"Negatives already done : {len(done):,}", flush=True)
    print(f"Negatives still needed : {needed:,}", flush=True)

    if needed == 0:
        print("Negatives already complete.")
        return 0

    print(f"Sampling {needed * OVERSAMPLE:,} candidates (3× to absorb failures)...",
          flush=True)
    candidates = sample_candidates(blocked, needed * OVERSAMPLE)

    session = requests.Session()
    last = 0.0
    n_ok = n_fail = 0

    with gzip.open(out_path, "at") as out:
        for chrom, pos in candidates:
            if n_ok >= needed:
                break

            dt = time.monotonic() - last
            if dt < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - dt)
            last = time.monotonic()

            vid = f"{chrom}:{pos}:normal"
            if vid in done:
                continue

            ens_chrom = chr_to_ensembl(chrom)
            seq = fetch_one(session, ens_chrom, pos)
            if seq is None or len(seq) != WINDOW:
                n_fail += 1
                continue

            rec = {
                "variant_id": vid,
                "project": "NORMAL",
                "case_id": "normal",
                "chromosome": chrom,
                "start": pos,
                "ref": ".",
                "alt": ".",
                "variant_type": "normal",
                "gene": ".",
                "seq": seq.upper(),
                "label": 0,
            }
            out.write(json.dumps(rec) + "\n")
            n_ok += 1

            if (n_ok % 200) == 0:
                out.flush()
                print(f"  negatives: ok={n_ok:,}  fail={n_fail:,}", flush=True)

    print(f"DONE  ok={n_ok:,}  fail={n_fail:,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
