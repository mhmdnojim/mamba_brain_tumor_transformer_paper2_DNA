"""
Stage 4c -- Generate properly matched negative sequences.

Replaces 04_generate_negatives.py (random sampling) and
04b_select_gcmatched_negatives.py (subsampling / data loss).

THREE PATHS depending on what files you have locally:

  PATH A -- Ensembl-only (nothing to download, ~5-8 hours)
      python 04c_generate_negatives_matched.py

  PATH B -- Local FASTA (recommended, ~10 min)
      # One-time download in Colab (~1.1 GB):
      # wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/\\
      #   GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/\\
      #   GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz && gunzip *.fna.gz
      python 04c_generate_negatives_matched.py --ref GRCh38.fna

  PATH C -- Local FASTA + GENCODE (~15 min, gold standard)
      # Additional download (~50 MB):
      # wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/\\
      #   release_38/gencode.v38.annotation.gtf.gz
      python 04c_generate_negatives_matched.py \\
          --ref GRCh38.fna \\
          --gencode gencode.v38.annotation.gtf.gz

Matching criteria applied:
  1. Chromosome  -- bedtools shuffle -chrom (same chr as positive)
  2. GC content  -- |gc_neg - gc_pos| <= gc_tol  (default 0.02)
  3. Region type -- exonic/intronic/intergenic must match  (PATH C only)

Also excludes +/- 512bp around all known variants.
No positives are discarded -- one matched negative per positive.

Speed comparison:
  Path A (Ensembl):        ~5-8 hours  (rate-limited API)
  Path B (local FASTA):    ~10 minutes (no API for GC/sequence)
  Path C (FASTA+GENCODE):  ~15 minutes (no API at all)

Requirements (Colab/Linux):
    apt-get install -y bedtools samtools
    pip install requests pandas tqdm pyfaidx
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENSEMBL      = "https://rest.ensembl.org"
HEADERS      = {"Accept": "application/json", "User-Agent": "tcga-matcher/1.0"}
WINDOW       = 512
LEFT         = 255       # bases left of anchor
RIGHT        = 256       # bases right of anchor (LEFT + 1 + RIGHT = 512)
TARGET_RPS   = 12.0
MIN_INTERVAL = 1.0 / TARGET_RPS
MAX_RETRIES  = 6
BUFFER       = 512       # exclude +/- BUFFER around known variants

# GRCh38 primary assembly chromosome lengths (bp)
CHR_LENGTHS: dict[str, int] = {
    "chr1":  248956422, "chr2":  242193529, "chr3":  198295559,
    "chr4":  190214555, "chr5":  181538259, "chr6":  170805979,
    "chr7":  159345973, "chr8":  145138636, "chr9":  138394717,
    "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16":  90338345, "chr17":  83257441, "chr18":  80373285,
    "chr19":  58617616, "chr20":  64444167, "chr21":  46709983,
    "chr22":  50818468, "chrX":  156040895, "chrY":   57227415,
}

UCSC_SIZES_URL = ("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/"
                  "hg38.chrom.sizes")
GENCODE_BED_URL = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
                   "release_46/gencode.v46.annotation.gtf.gz")


# ---------------------------------------------------------------------------
# Helpers -- chromosome normalisation
# ---------------------------------------------------------------------------
def norm_chrom(c: str) -> str:
    c = str(c)
    return c if c.startswith("chr") else "chr" + c


def to_ensembl(c: str) -> str:
    c = c.removeprefix("chr")
    return "MT" if c == "M" else c


def gc_content(seq: str) -> float:
    s = seq.upper()
    return (s.count("G") + s.count("C")) / max(len(s), 1)


# ---------------------------------------------------------------------------
# Local FASTA support (Path B / Path C)
# ---------------------------------------------------------------------------
def ensure_pyfaidx() -> bool:
    try:
        import pyfaidx  # noqa: F401
        return True
    except ImportError:
        print("  Installing pyfaidx ...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyfaidx"],
                           capture_output=True, text=True)
        return r.returncode == 0


def load_fasta(fasta_path: Path):
    """Load FASTA with pyfaidx. Builds .fai index on first use (~30 sec for GRCh38)."""
    try:
        from pyfaidx import Fasta
    except ImportError:
        raise RuntimeError("pyfaidx not installed. Run: pip install pyfaidx")
    print(f"  Loading FASTA index: {fasta_path}")
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
    print(f"  FASTA loaded. Chromosomes: {len(fasta.keys())}")
    return fasta


def fetch_seq_local(fasta, chrom: str, pos: int) -> str | None:
    """Extract 512 bp window from local FASTA. Instant, no API call."""
    s = pos - LEFT
    e = pos + RIGHT
    if s < 1:
        return None
    # pyfaidx uses 0-based half-open for slicing
    try:
        # Try with 'chr' prefix first, then without
        key = chrom
        if key not in fasta:
            key = chrom.removeprefix("chr")
        if key not in fasta:
            return None
        seq = fasta[key][s - 1:e].seq   # pyfaidx is 0-based
        return seq.upper() if len(seq) == WINDOW else None
    except Exception:
        return None


def write_candidates_bed_for_getfasta(candidates_bed: Path,
                                      out_bed: Path) -> None:
    """Convert shuffled BED (4-col) to getfasta-compatible BED with name col."""
    with open(candidates_bed) as fin, open(out_bed, "w") as fout:
        for i, line in enumerate(fin):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            fout.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\tcand_{i}\n")


# ---------------------------------------------------------------------------
# Step 1 -- Load positives and build per-positive target profile
# ---------------------------------------------------------------------------
def load_positives(seq_dir: Path) -> list[dict]:
    records: list[dict] = []
    for p in sorted(seq_dir.glob("*_windows.jsonl.gz")):
        if "normals" in p.name or "matched" in p.name or "gcmatched" in p.name:
            continue
        with gzip.open(p, "rt") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        print(f"  {p.name}: {len(records):,} total so far")
    return records


# ---------------------------------------------------------------------------
# Step 2 -- Assign region type to positives
#           Source A: GENCODE BED (fast, local)
#           Source B: Ensembl REST (fallback)
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, desc: str) -> bool:
    """Download url to dest.  Returns True on success."""
    print(f"  Downloading {desc} ...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"  {desc}", ascii=True, file=sys.stdout
        ) as bar:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def build_gencode_interval_tree(gtf_gz: Path) -> dict[str, list[tuple[int, int, str]]]:
    """
    Parse GENCODE GTF (gzipped) and return per-chromosome sorted interval list:
      {chrom: [(start, end, "exon"|"gene"), ...]}  -- both 0-based half-open
    Only 'exon' and 'gene' features are kept (transcript/CDS etc. are skipped
    to keep memory reasonable).
    """
    print("  Parsing GENCODE GTF (this takes ~2 minutes) ...")
    ivs: dict[str, list] = defaultdict(list)
    with gzip.open(gtf_gz, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in ("exon", "gene"):
                continue
            chrom = norm_chrom(parts[0])
            if chrom not in CHR_LENGTHS:
                continue
            start = int(parts[3]) - 1   # GTF is 1-based
            end   = int(parts[4])        # end inclusive -> half-open
            ivs[chrom].append((start, end, feature))

    # Sort by start for binary search
    for chrom in ivs:
        ivs[chrom].sort(key=lambda x: x[0])
    total = sum(len(v) for v in ivs.values())
    print(f"  Loaded {total:,} GENCODE intervals across {len(ivs)} chromosomes")
    return dict(ivs)


def classify_by_gencode(ivs: dict, chrom: str, pos: int) -> str:
    """
    Classify a single position using pre-loaded GENCODE intervals.
    Returns 'exonic', 'intronic', or 'intergenic'.
    """
    lst = ivs.get(chrom, [])
    if not lst:
        return "intergenic"

    # Binary search: find intervals whose start <= pos
    idx = bisect.bisect_right(lst, (pos, float("inf"), "")) - 1
    in_gene = False
    # Check a window of nearby intervals
    for i in range(max(0, idx - 1), min(len(lst), idx + 10)):
        s, e, ftype = lst[i]
        if s > pos:
            break
        if s <= pos < e:
            if ftype == "exon":
                return "exonic"
            if ftype == "gene":
                in_gene = True
    return "intronic" if in_gene else "intergenic"


def assign_region_gencode(records: list[dict],
                          ivs: dict) -> list[dict]:
    print("  Classifying positives by GENCODE ...")
    for rec in tqdm(records, desc="  GENCODE classify",
                    ascii=True, ncols=70, file=sys.stdout):
        chrom = norm_chrom(rec["chromosome"])
        pos   = int(rec["start"])
        rec["_region"] = classify_by_gencode(ivs, chrom, pos)
    counts = Counter(r["_region"] for r in records)
    print(f"  Region distribution: {dict(counts)}")
    return records


# Ensembl fallback
def _ensembl_overlap(session: requests.Session, chrom: str,
                     start: int, end: int, last: list) -> str:
    url = (f"{ENSEMBL}/overlap/region/human/{chrom}:{start}-{end}"
           f"?feature=gene;feature=exon")
    dt = time.monotonic() - last[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    last[0] = time.monotonic()
    delay = 1.0
    for _ in range(MAX_RETRIES):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException:
            time.sleep(delay); delay *= 2; continue
        if r.status_code == 200:
            data = r.json()
            exons = [f for f in data if f.get("feature_type") == "exon"]
            genes = [f for f in data if f.get("feature_type") == "gene"]
            if exons:   return "exonic"
            if genes:   return "intronic"
            return "intergenic"
        if r.status_code == 429:
            t = float(r.headers.get("Retry-After", delay))
            time.sleep(t); delay = max(delay * 2, t); continue
        if r.status_code in (400, 403, 404):
            return "intergenic"
        time.sleep(delay); delay *= 2
    return "unknown"


def assign_region_ensembl(records: list[dict],
                          cache_path: Path) -> list[dict]:
    """Assign region type via Ensembl REST.  Caches results."""
    cached: dict[str, str] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"  Ensembl region cache: {len(cached):,} entries")

    session = requests.Session()
    last    = [0.0]
    todo    = [r for r in records
               if r.get("variant_id", "") not in cached]

    if todo:
        print(f"  Querying Ensembl region type for {len(todo):,} positives ...")
        for rec in tqdm(todo, desc="  Ensembl region",
                        ascii=True, ncols=70, file=sys.stdout):
            chrom = to_ensembl(norm_chrom(rec["chromosome"]))
            pos   = int(rec["start"])
            region = _ensembl_overlap(session, chrom,
                                      max(1, pos - LEFT), pos + RIGHT, last)
            cached[rec.get("variant_id", f"{chrom}:{pos}")] = region
        cache_path.write_text(json.dumps(cached, indent=2))

    for rec in records:
        key = rec.get("variant_id", "")
        rec["_region"] = cached.get(key, "unknown")

    counts = Counter(r["_region"] for r in records)
    print(f"  Region distribution: {dict(counts)}")
    return records


# ---------------------------------------------------------------------------
# Step 3 -- Build blocked zones from variant TSVs
# ---------------------------------------------------------------------------
def load_blocked(var_dir: Path) -> dict[str, list[int]]:
    raw: dict[str, list[int]] = defaultdict(list)
    for tsv in sorted(var_dir.glob("*_variants.tsv.gz")):
        df = pd.read_csv(tsv, sep="\t", usecols=["chromosome", "start"],
                         dtype={"chromosome": str, "start": "Int64"})
        for row in df.itertuples(index=False):
            c = norm_chrom(str(row.chromosome))
            if c in CHR_LENGTHS:
                raw[c].append(int(row.start))
    blocked = {c: sorted(v) for c, v in raw.items()}
    total = sum(len(v) for v in blocked.values())
    print(f"  Loaded {total:,} blocked positions (+/-{BUFFER}bp)")
    return blocked


def is_blocked(positions: list[int], pos: int) -> bool:
    lo = bisect.bisect_left(positions, pos - BUFFER)
    hi = bisect.bisect_right(positions, pos + BUFFER)
    return lo < hi


# ---------------------------------------------------------------------------
# Step 4 -- Generate shuffled candidate positions via bedtools
# ---------------------------------------------------------------------------
def bedtools_available() -> bool:
    try:
        r = subprocess.run(["bedtools", "--version"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


def install_bedtools() -> bool:
    print("  Installing bedtools ...")
    r = subprocess.run(["apt-get", "install", "-y", "-q", "bedtools"],
                       capture_output=True, text=True)
    return r.returncode == 0


def write_chrom_sizes(dest: Path) -> bool:
    """Write minimal chromosome sizes file (no download needed)."""
    lines = [f"{c}\t{l}" for c, l in CHR_LENGTHS.items()]
    dest.write_text("\n".join(lines) + "\n")
    return True


def write_excl_bed(blocked: dict[str, list[int]], dest: Path) -> None:
    """Write BED of +/- BUFFER zones around all known variants."""
    with open(dest, "w") as f:
        for chrom, positions in blocked.items():
            length = CHR_LENGTHS.get(chrom, 0)
            for pos in positions:
                s = max(0, pos - BUFFER)
                e = min(length, pos + BUFFER)
                f.write(f"{chrom}\t{s}\t{e}\n")


def write_positives_bed(records: list[dict], dest: Path) -> None:
    """One BED line per positive (anchor position +/- window)."""
    with open(dest, "w") as f:
        for i, rec in enumerate(records):
            chrom = norm_chrom(rec["chromosome"])
            pos   = int(rec["start"])
            s     = max(0, pos - LEFT)
            e     = pos + RIGHT
            name  = rec.get("variant_id", f"pos_{i}")
            f.write(f"{chrom}\t{s}\t{e}\t{name}\n")


def run_bedtools_shuffle(pos_bed: Path, excl_bed: Path,
                         sizes_file: Path, out_bed: Path,
                         n_oversample: int = 10,
                         seed: int = 42) -> bool:
    """
    Run bedtools shuffle -chrom to get chromosome-matched random windows.
    Generates n_oversample x more candidates than positives to ensure enough
    survive GC + region-type filtering.
    """
    # Expand positives n_oversample times for the shuffle
    with open(pos_bed) as f:
        lines = f.readlines()
    tmp = pos_bed.parent / "positives_expanded.bed"
    with open(tmp, "w") as f:
        for _ in range(n_oversample):
            f.writelines(lines)

    cmd = [
        "bedtools", "shuffle",
        "-i",    str(tmp),
        "-g",    str(sizes_file),
        "-excl", str(excl_bed),
        "-chrom",
        "-noOverlapping",
        "-seed", str(seed),
    ]
    print(f"  Running: {' '.join(cmd)}")
    with open(out_bed, "w") as fout:
        r = subprocess.run(cmd, stdout=fout, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"  bedtools error: {r.stderr[:500]}")
        return False
    n_lines = sum(1 for _ in open(out_bed))
    print(f"  bedtools shuffle produced {n_lines:,} candidate windows")
    return True


# ---------------------------------------------------------------------------
# Step 5 -- Fetch sequences and match GC + region type
# ---------------------------------------------------------------------------
def fetch_seq(session: requests.Session, chrom: str,
              start: int, last: list) -> str | None:
    s   = max(1, start - LEFT)
    e   = start + RIGHT
    url = f"{ENSEMBL}/sequence/region/human/{chrom}:{s}..{e}:1"
    dt  = time.monotonic() - last[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    last[0] = time.monotonic()
    delay = 1.0
    for _ in range(MAX_RETRIES):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException:
            time.sleep(delay); delay *= 2; continue
        if r.status_code == 200:
            seq = r.json().get("seq", "")
            return seq.upper() if len(seq) == WINDOW else None
        if r.status_code == 429:
            t = float(r.headers.get("Retry-After", delay))
            time.sleep(t); delay = max(delay * 2, t); continue
        if r.status_code in (400, 404):
            return None
        time.sleep(delay); delay *= 2
    return None


def load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with gzip.open(out_path, "rt") as f:
        for line in f:
            try:
                done.add(json.loads(line)["paired_positive_id"])
            except Exception:
                continue
    return done


def match_negatives(
    positives: list[dict],
    candidates_bed: Path,
    blocked: dict[str, list[int]],
    gencode_ivs: dict | None,
    out_path: Path,
    gc_tol: float,
    match_region: bool,
    max_tries: int,
    fasta=None,        # pyfaidx Fasta object (Path B/C) or None (Path A)
) -> None:
    """
    For each positive, find one matching negative from candidates_bed.
    Matching criteria:
      - Same chromosome (guaranteed by bedtools -chrom)
      - |gc(negative) - gc(positive)| <= gc_tol
      - region_type(negative) == region_type(positive)   [if match_region]

    Candidates are consumed in random order; unmatched positives are reported.
    """
    # Build per-chromosome candidate queue from shuffled BED
    print("  Building candidate queues ...")
    chrom_candidates: dict[str, list[int]] = defaultdict(list)
    with open(candidates_bed) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            chrom = norm_chrom(parts[0])
            # BED is 0-based; anchor = midpoint of window
            start = (int(parts[1]) + int(parts[2])) // 2
            if start - LEFT < 1:
                continue
            chrom_candidates[chrom].append(start)

    rng = random.Random(42)
    for chrom in chrom_candidates:
        rng.shuffle(chrom_candidates[chrom])

    # Resume
    done = load_done(out_path)
    print(f"  Already matched: {len(done):,} positives")
    remaining = [p for p in positives
                 if p.get("variant_id", "") not in done]
    print(f"  Positives to match: {len(remaining):,}")

    if not remaining:
        print("  All positives already matched.")
        return

    use_local = fasta is not None
    session   = requests.Session()
    last_api  = [0.0]
    n_ok = n_fail = n_no_candidate = 0
    path_label = "local FASTA" if use_local else "Ensembl REST"
    print(f"  Sequence source: {path_label}")

    cursors: dict[str, int] = defaultdict(int)

    with gzip.open(out_path, "at") as out, \
         tqdm(total=len(remaining), desc="  Matching negatives",
              unit="seq", ascii=True, ncols=80, file=sys.stdout) as pbar:

        for pos_rec in remaining:
            chrom      = norm_chrom(pos_rec["chromosome"])
            pos_gc     = gc_content(pos_rec["seq"])
            pos_region = pos_rec.get("_region", "unknown")
            pos_id     = pos_rec.get("variant_id", "")
            ens_chrom  = to_ensembl(chrom)

            queue  = chrom_candidates.get(chrom, [])
            cursor = cursors[chrom]
            matched = False

            for attempt in range(min(max_tries, len(queue) - cursor)):
                idx = cursor + attempt
                if idx >= len(queue):
                    break
                cand_pos = queue[idx]

                if is_blocked(blocked.get(chrom, []), cand_pos):
                    continue

                # ── Sequence fetch: local FASTA (Path B/C) or Ensembl (Path A)
                if use_local:
                    seq = fetch_seq_local(fasta, chrom, cand_pos)
                else:
                    seq = fetch_seq(session, ens_chrom, cand_pos, last_api)
                if seq is None:
                    n_fail += 1
                    continue

                # ── GC filter (always applied)
                cand_gc = gc_content(seq)
                if abs(cand_gc - pos_gc) > gc_tol:
                    continue

                # ── Region type filter (Path C: GENCODE local, or Ensembl)
                if match_region and pos_region not in ("unknown", "any", ""):
                    if gencode_ivs is not None:
                        cand_region = classify_by_gencode(
                            gencode_ivs, chrom, cand_pos)
                    else:
                        cand_region = _ensembl_overlap(
                            session, ens_chrom,
                            max(1, cand_pos - LEFT), cand_pos + RIGHT,
                            last_api)
                    if cand_region != pos_region:
                        continue

                # ── Accepted
                rec = {
                    "variant_id":         f"{chrom}:{cand_pos}:matched",
                    "paired_positive_id": pos_id,
                    "project":            "NORMAL",
                    "case_id":            "normal",
                    "chromosome":         chrom,
                    "start":              cand_pos,
                    "ref": ".", "alt": ".",
                    "variant_type":       "normal_matched",
                    "gene":               ".",
                    "seq":                seq,
                    "label":              0,
                    "matched_gc":         round(cand_gc, 4),
                    "matched_region":     pos_region if match_region else "any",
                }
                out.write(json.dumps(rec) + "\n")
                cursors[chrom] = idx + 1
                matched = True
                n_ok   += 1
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, fail=n_fail,
                                 no_cand=n_no_candidate, refresh=False)
                break

            if not matched:
                n_no_candidate += 1
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, fail=n_fail,
                                 no_cand=n_no_candidate, refresh=False)

    print(f"\n  DONE  matched={n_ok:,}  fetch_fail={n_fail:,}  "
          f"no_candidate={n_no_candidate:,}")
    if n_no_candidate > 0:
        pct = 100 * n_no_candidate / max(1, n_ok + n_no_candidate)
        print(f"  WARNING: {pct:.1f}% of positives could not be matched.")
        print(f"  Increase --oversample (currently used to generate candidates).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(gc_tol: float = 0.02,
         match_region: bool = True,
         oversample: int = 10,
         max_tries: int = 300,
         use_gencode: bool = True,
         ref_fasta: str | None = None,
         seed: int = 42) -> int:

    var_dir = Path("variants")
    seq_dir = Path("sequences")
    tmp_dir = Path("diagnostics") / "tmp_matching"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = seq_dir / "normals_windows_matched.jsonl.gz"

    if not var_dir.exists():
        print("variants/ not found. Run stage 2 first.", file=sys.stderr)
        return 1

    # ── Detect path ──────────────────────────────────────────────────────────
    fasta = None
    if ref_fasta:
        ref_path = Path(ref_fasta)
        if not ref_path.exists():
            print(f"  ERROR: --ref file not found: {ref_path}", file=sys.stderr)
            return 1
        ensure_pyfaidx()
        fasta = load_fasta(ref_path)
        print(f"  PATH {'C' if use_gencode else 'B'}: local FASTA "
              f"{'+ GENCODE' if use_gencode else '(GC+chr only)'}")
    else:
        print("  PATH A: Ensembl REST only (~5-8 hours)")
        print("  TIP: pass --ref GRCh38.fna to run in ~10 minutes instead.")

    # ── Load positives ───────────────────────────────────────────────────────
    print("\n[1/6] Loading positive sequences ...")
    positives = load_positives(seq_dir)
    if not positives:
        print("No positive sequences found. Run stage 3 first.", file=sys.stderr)
        return 1
    print(f"  Total positives: {len(positives):,}")

    # ── Assign region type ───────────────────────────────────────────────────
    gencode_ivs: dict | None = None
    if match_region:
        print("\n[2/6] Assigning region type to positives ...")
        cache_file = tmp_dir / "positive_regions.json"
        gtf_gz     = tmp_dir / "gencode.v46.annotation.gtf.gz"

        if use_gencode:
            if not gtf_gz.exists():
                ok = _download_file(GENCODE_BED_URL, gtf_gz, "GENCODE GTF")
                if not ok:
                    print("  GENCODE download failed. Falling back to Ensembl REST.")
                    use_gencode = False
            if use_gencode and gtf_gz.exists():
                gencode_ivs = build_gencode_interval_tree(gtf_gz)
                positives   = assign_region_gencode(positives, gencode_ivs)

        if not use_gencode:
            positives = assign_region_ensembl(positives, cache_file)
    else:
        print("\n[2/6] Skipping region-type matching (--no_region)")
        for rec in positives:
            rec["_region"] = "any"

    # ── Load blocked zones ───────────────────────────────────────────────────
    print("\n[3/6] Loading blocked variant zones ...")
    blocked = load_blocked(var_dir)

    # ── Set up bedtools ──────────────────────────────────────────────────────
    print("\n[4/6] Preparing bedtools shuffle ...")
    if not bedtools_available():
        print("  bedtools not found. Attempting install ...")
        if not install_bedtools():
            print("  ERROR: bedtools install failed. "
                  "Run: apt-get install -y bedtools", file=sys.stderr)
            return 1
    print("  bedtools available.")

    sizes_file = tmp_dir / "hg38.chrom.sizes"
    write_chrom_sizes(sizes_file)

    excl_bed = tmp_dir / "variants_excl.bed"
    write_excl_bed(blocked, excl_bed)

    pos_bed  = tmp_dir / "positives.bed"
    write_positives_bed(positives, pos_bed)

    cand_bed = tmp_dir / "candidates_shuffled.bed"
    ok = run_bedtools_shuffle(pos_bed, excl_bed, sizes_file,
                              cand_bed, n_oversample=oversample, seed=seed)
    if not ok:
        print("  bedtools shuffle failed.", file=sys.stderr)
        return 1

    # ── Match and fetch ──────────────────────────────────────────────────────
    print(f"\n[5/6] Fetching and matching negatives ...")
    print(f"  GC tolerance  : +/-{gc_tol}")
    print(f"  Region match  : {match_region}")
    print(f"  Max tries/pos : {max_tries}")
    match_negatives(
        positives      = positives,
        candidates_bed = cand_bed,
        blocked        = blocked,
        gencode_ivs    = gencode_ivs,
        out_path       = out_path,
        gc_tol         = gc_tol,
        match_region   = match_region,
        max_tries      = max_tries,
        fasta          = fasta,
    )

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"\n[6/6] Output summary ...")
    if out_path.exists():
        n = sum(1 for _ in gzip.open(out_path, "rt"))
        mb = out_path.stat().st_size / 1e6
        print(f"  {out_path}  ({n:,} records, {mb:.1f} MB)")
        coverage = 100 * n / max(1, len(positives))
        print(f"  Coverage: {coverage:.1f}% of positives matched")
        print(f"\nNext step:")
        print(f"  python 05_build_csv.py "
              f"--negatives sequences/normals_windows_matched.jsonl.gz "
              f"--out dataset/cancer_genes_matched.csv")

    # Clean private keys before returning
    for rec in positives:
        rec.pop("_region", None)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gc_tol", type=float, default=0.02,
                   help="GC content tolerance for matching (default 0.02 = +/-2%%)")
    p.add_argument("--no_region", action="store_true",
                   help="Skip region-type matching (chromosome + GC only)")
    p.add_argument("--no_gencode", action="store_true",
                   help="Use Ensembl REST for region type instead of GENCODE download")
    p.add_argument("--oversample", type=int, default=10,
                   help="Candidate oversample factor for bedtools shuffle (default 10)")
    p.add_argument("--max_tries", type=int, default=300,
                   help="Max candidate attempts per positive before giving up (default 300)")
    p.add_argument("--ref", type=str, default=None,
                   help="Path to GRCh38 FASTA (Path B/C). "
                        "Makes GC lookup local — 100x faster than Ensembl. "
                        "Example: --ref GRCh38_no_alt.fna")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(
        gc_tol       = args.gc_tol,
        match_region = not args.no_region,
        use_gencode  = not args.no_gencode,
        ref_fasta    = args.ref,
        oversample   = args.oversample,
        max_tries    = args.max_tries,
        seed         = args.seed,
    ))
