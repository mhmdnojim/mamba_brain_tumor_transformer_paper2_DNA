"""
dataset.py — Download, process and save TCGA cancer gene data.

Full control:
    --limit_seqs   Max sequences per cancer type  (default 300, 0 = unlimited)
    --limit_files  Max MAF files to download      (default 5,  0 = unlimited)
    --all          Download everything, ignore both limits

Google Drive (Colab):
    --drive        Mount Google Drive and save everything under
                   /content/drive/MyDrive/dataset/
    --out_dir      Custom save directory (overrides --drive path)

Usage examples:
    # Quick demo (no internet)
    python dataset.py --demo

    # 300 sequences, BRCA only
    python dataset.py --cancer BRCA --limit_seqs 300

    # All sequences, all files, BRCA
    python dataset.py --cancer BRCA --all

    # Multiple cancer types, 1000 sequences each
    python dataset.py --cancer BRCA LUAD PRAD --limit_seqs 1000

    # Save everything to Google Drive (Colab)
    python dataset.py --cancer BRCA --all --drive

    # Custom output directory
    python dataset.py --cancer BRCA --all --out_dir /content/drive/MyDrive/dataset

    # List cancer types
    python dataset.py --list
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from io import StringIO

try:
    from Bio import Entrez, SeqIO
    BIOPYTHON_OK = True
except ImportError:
    BIOPYTHON_OK = False


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
GDC_API    = "https://api.gdc.cancer.gov"
GDC_FILES  = f"{GDC_API}/files"
GDC_DATA   = f"{GDC_API}/data"
ENSEMBL    = "https://rest.ensembl.org"

NO_LIMIT   = 999_999   # sentinel for "unlimited"

CANCER_TYPES = {
    "BRCA": "Breast cancer",
    "LUAD": "Lung adenocarcinoma",
    "LUSC": "Lung squamous cell carcinoma",
    "PRAD": "Prostate cancer",
    "COAD": "Colon adenocarcinoma",
    "STAD": "Stomach adenocarcinoma",
    "BLCA": "Bladder urothelial carcinoma",
    "LIHC": "Liver hepatocellular carcinoma",
    "KIRC": "Kidney renal clear cell carcinoma",
    "GBM":  "Glioblastoma multiforme",
    "OV":   "Ovarian serous cystadenocarcinoma",
    "UCEC": "Uterine corpus endometrial carcinoma",
}

HOUSEKEEPING_GENES = [
    ("GAPDH",  "12", 6534512,   6538371),
    ("ACTB",   "7",  5527148,   5530601),
    ("B2M",    "15", 44711487,  44718851),
    ("HPRT1",  "X",  134460498, 134508555),
    ("SDHA",   "5",  218539731, 218574919),
    ("PPIA",   "7",  44715931,  44721830),
    ("RPL13A", "19", 49468558,  49472480),
    ("UBC",    "12", 124911583, 124916640),
    ("YWHAZ",  "2",  238340226, 238366671),
    ("TBP",    "6",  170554476, 170572870),
]


# ─────────────────────────────────────────────────────────────────────────────
#  GOOGLE DRIVE SETUP
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_BASE = "/content/drive/MyDrive/dataset"

def mount_google_drive() -> bool:
    """Mount Google Drive in Colab. Returns True if successful."""
    try:
        from google.colab import drive
        print("[Drive] Mounting Google Drive...")
        drive.mount("/content/drive", force_remount=False)
        print(f"[Drive] Mounted. Files will be saved to: {DRIVE_BASE}")
        return True
    except ImportError:
        print("[Drive] Not running in Colab — Google Drive not available.")
        print("         Use --out_dir to set a custom save path instead.")
        return False
    except Exception as e:
        print(f"[Drive] Mount failed: {e}")
        return False


def resolve_paths(args) -> tuple[str, str]:
    """
    Decide where to save the CSV and the cache.
    Priority: --out_dir > --drive > current directory

    Returns: (csv_path, cache_dir)
    """
    if args.out_dir:
        base = args.out_dir
    elif args.drive:
        mounted = mount_google_drive()
        base = DRIVE_BASE if mounted else "./dataset"
    else:
        base = "."

    os.makedirs(base, exist_ok=True)
    cache_dir = os.path.join(base, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    csv_path = os.path.join(base, args.out_name)

    print(f"\n[paths] CSV will be saved to  : {csv_path}")
    print(f"[paths] MAF cache directory    : {cache_dir}")
    return csv_path, cache_dir


# ─────────────────────────────────────────────────────────────────────────────
#  GDC — Query MAF file IDs
# ─────────────────────────────────────────────────────────────────────────────
def query_maf_file_ids(cancer_type: str, limit_files: int) -> list:
    """
    Query GDC open-access MAF files for a TCGA cancer type.
    limit_files = 0 means fetch ALL available files.
    """
    size = 1000 if limit_files == 0 else limit_files
    print(f"\n[GDC] Querying MAF files for TCGA-{cancer_type} "
          f"(max files: {'ALL' if limit_files == 0 else limit_files})...")

    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id",
                                     "value": f"TCGA-{cancer_type}"}},
            {"op": "=", "content": {"field": "data_type",
                                     "value": "Masked Somatic Mutation"}},
            {"op": "=", "content": {"field": "data_format",
                                     "value": "MAF"}},
            # Use current GDC workflow; MuTect2/MuSE-only workflows are deprecated
            {"op": "=", "content": {"field": "analysis.workflow_type",
                                     "value": "Aliquot Ensemble Somatic Variant Merging and Masking"}},
            {"op": "=", "content": {"field": "access",
                                     "value": "open"}},
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields":  "file_id,file_name,file_size",
        "format":  "JSON",
        "size":    str(size),
    }

    try:
        resp = requests.get(GDC_FILES, params=params, timeout=30)
        resp.raise_for_status()
        hits = resp.json()["data"]["hits"]
    except Exception as e:
        print(f"  [ERROR] GDC query failed: {e}")
        return []

    total_found = resp.json()["data"]["pagination"]["total"]
    print(f"  Total MAF files available on GDC : {total_found}")
    print(f"  Files to be downloaded           : {len(hits)}")
    for h in hits:
        mb = h.get("file_size", 0) / 1e6
        print(f"    {h['file_id']}  {h['file_name']}  ({mb:.1f} MB)")

    return [h["file_id"] for h in hits]


# ─────────────────────────────────────────────────────────────────────────────
#  GDC — Download a MAF file (cached)
# ─────────────────────────────────────────────────────────────────────────────
def download_maf(file_id: str, cache_dir: str) -> pd.DataFrame:
    """
    Download MAF file from GDC. Caches to cache_dir on first download.
    Second run reads from cache — no re-download.
    """
    cache_path = Path(cache_dir) / f"{file_id}.maf.tsv"

    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1e6
        print(f"  [cache] {cache_path.name}  ({size_mb:.1f} MB)  ← already saved")
        return pd.read_csv(cache_path, sep="\t", comment="#", low_memory=False)

    print(f"  [GDC download] {file_id} ...")
    try:
        resp = requests.post(
            GDC_DATA,
            data=json.dumps({"ids": [file_id]}),
            headers={"Content-Type": "application/json"},
            timeout=180,
            stream=True,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        raise

    raw = b""
    for chunk in resp.iter_content(chunk_size=65536):
        raw += chunk

    if raw[:2] == b"\x1f\x8b":
        import gzip
        raw = gzip.decompress(raw)

    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(text), sep="\t", comment="#", low_memory=False)

    # Save to cache (stays in Google Drive if --drive was used)
    df.to_csv(cache_path, sep="\t", index=False)
    size_mb = cache_path.stat().st_size / 1e6
    print(f"  Saved {len(df):,} mutations → {cache_path}  ({size_mb:.1f} MB)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Ensembl — Fetch DNA sequence for a genomic region
# ─────────────────────────────────────────────────────────────────────────────
_ENSEMBL_LAST_CALL: float = 0.0
_ENSEMBL_MIN_INTERVAL: float = 1.0 / 13.0   # 13 req/sec — Ensembl hard cap is 15


def fetch_sequence_ensembl(chrom: str, start: int, end: int,
                           window: int = 256) -> str | None:
    global _ENSEMBL_LAST_CALL
    # Exact 512 bp window anchored at Start_Position (matches new pipeline spec)
    fetch_start = max(1, start - 255)
    fetch_end   = start + 256
    region = f"{chrom}:{fetch_start}..{fetch_end}"
    url    = f"{ENSEMBL}/sequence/region/human/{region}"

    # Honour Ensembl rate limit
    dt = time.time() - _ENSEMBL_LAST_CALL
    if dt < _ENSEMBL_MIN_INTERVAL:
        time.sleep(_ENSEMBL_MIN_INTERVAL - dt)
    _ENSEMBL_LAST_CALL = time.time()

    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
        except Exception:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            seq = resp.json().get("seq", "")
            return "".join(c for c in seq.upper() if c in "ATGCN") or None
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
            time.sleep(retry_after)
            continue
        if resp.status_code in (400, 404):
            return None
        time.sleep(2 ** attempt)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Ensembl — Fetch normal (housekeeping gene) sequences
# ─────────────────────────────────────────────────────────────────────────────
def fetch_normal_sequences(n_samples: int, seq_len: int) -> list:
    print(f"\n[Ensembl] Fetching {n_samples} normal sequences...")
    seqs = []
    for gene_name, chrom, start, end in HOUSEKEEPING_GENES:
        if len(seqs) >= n_samples:
            break
        seq = fetch_sequence_ensembl(chrom, start, end, window=seq_len)
        if seq and len(seq) >= seq_len:
            for i in range(0, len(seq) - seq_len, seq_len // 2):
                chunk = seq[i: i + seq_len]
                if len(chunk) == seq_len:
                    seqs.append(chunk)
                if len(seqs) >= n_samples:
                    break
        time.sleep(0.3)
    print(f"  Got {len(seqs)} normal sequences")
    return seqs


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN BUILD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(cancer_types: list, seq_len: int,
                  limit_seqs: int, limit_files: int,
                  csv_path: str, cache_dir: str):
    """
    Download TCGA MAF files + Ensembl sequences and save to CSV.

    limit_seqs  = 0 → download ALL sequences
    limit_files = 0 → download ALL MAF files
    """
    eff_seqs  = NO_LIMIT if limit_seqs  == 0 else limit_seqs
    eff_files = NO_LIMIT if limit_files == 0 else limit_files

    cancer_seqs = []

    for cancer_type in cancer_types:
        if cancer_type not in CANCER_TYPES:
            print(f"[SKIP] Unknown: {cancer_type}. Valid: {list(CANCER_TYPES.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f"  TCGA-{cancer_type}  —  {CANCER_TYPES[cancer_type]}")
        print(f"  Sequences limit : {'ALL' if eff_seqs  == NO_LIMIT else eff_seqs}")
        print(f"  Files limit     : {'ALL' if eff_files == NO_LIMIT else eff_files}")
        print(f"{'='*60}")

        file_ids = query_maf_file_ids(cancer_type, limit_files=eff_files)
        if not file_ids:
            print(f"  No files found for {cancer_type}")
            continue

        for fid in file_ids:
            if len(cancer_seqs) >= eff_seqs:
                break
            try:
                maf_df = download_maf(fid, cache_dir)
            except Exception as e:
                print(f"  [SKIP file] {e}")
                continue

            needed = ["Chromosome", "Start_Position", "End_Position"]
            if not all(c in maf_df.columns for c in needed):
                print("  [SKIP] Missing required MAF columns")
                continue

            if "Variant_Type" in maf_df.columns:
                maf_df = maf_df[maf_df["Variant_Type"] == "SNP"]

            remaining  = eff_seqs - len(cancer_seqs)
            n_to_fetch = min(remaining, len(maf_df))
            sample_rows = maf_df.sample(n=n_to_fetch, random_state=42)

            print(f"\n  Fetching {n_to_fetch:,} sequences from Ensembl "
                  f"(total so far: {len(cancer_seqs)})...")

            for _, row in tqdm(sample_rows.iterrows(), total=n_to_fetch,
                               desc=f"  TCGA-{cancer_type}"):
                chrom = str(row["Chromosome"]).replace("chr", "")
                start = int(row["Start_Position"])
                end   = int(row["End_Position"])
                seq = fetch_sequence_ensembl(chrom, start, end)
                if seq and len(seq) >= 50:
                    cancer_seqs.append(seq[:seq_len].ljust(seq_len, "N"))
                time.sleep(0.1)

        print(f"\n  Cancer sequences collected: {len(cancer_seqs):,}")

    if not cancer_seqs:
        print("\n[ERROR] No cancer sequences collected.")
        print("  Check: internet connection, GDC status (status.gdc.cancer.gov)")
        return

    # ── Normal sequences (balanced) ───────────────────────────────────────────
    normal_seqs = fetch_normal_sequences(len(cancer_seqs), seq_len)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = [(s, 1) for s in cancer_seqs] + [(s, 0) for s in normal_seqs]
    random.shuffle(rows)

    print(f"\n[SAVE] Writing {len(rows):,} rows to {csv_path} ...")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "label"])
        writer.writerows(rows)

    size_mb = Path(csv_path).stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"  Dataset saved : {csv_path}  ({size_mb:.1f} MB)")
    print(f"  Total rows    : {len(rows):,}")
    print(f"  Cancer  (1)   : {len(cancer_seqs):,}")
    print(f"  Normal  (0)   : {len(normal_seqs):,}")
    print(f"  Seq length    : {seq_len} bp")
    print(f"{'='*60}")
    print(f"\nTrain with:")
    print(f"  python train.py --data {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO — synthetic data (no internet)
# ─────────────────────────────────────────────────────────────────────────────
def generate_demo_dataset(out_path: str, n_samples: int = 200, seq_len: int = 512):
    print("\n[DEMO] Generating synthetic dataset (no internet required)...")
    random.seed(42)

    def random_seq(length, gc_bias=0.5):
        bases = []
        for _ in range(length):
            r = random.random()
            if r < gc_bias / 2:       bases.append("G")
            elif r < gc_bias:          bases.append("C")
            elif r < gc_bias + (1 - gc_bias) / 2: bases.append("A")
            else:                      bases.append("T")
        return "".join(bases)

    rows = (
        [(random_seq(seq_len, gc_bias=0.65), 1) for _ in range(n_samples // 2)] +
        [(random_seq(seq_len, gc_bias=0.50), 0) for _ in range(n_samples // 2)]
    )
    random.shuffle(rows)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "label"])
        writer.writerows(rows)

    print(f"  Saved {len(rows)} synthetic samples → {out_path}")
    print(f"  Cancer (1): {n_samples//2}    Normal (0): {n_samples//2}")
    print(f"\nTest training with:")
    print(f"  python train.py --data {out_path} --max_steps 100")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # Cancer types
    p.add_argument("--cancer", type=str, nargs="+", default=["BRCA"],
                   metavar="TYPE",
                   help=f"Cancer type(s). Choices: {list(CANCER_TYPES.keys())}")

    # Sequence control
    p.add_argument("--limit_seqs", type=int, default=300,
                   help="Max sequences per cancer type. 0 = ALL sequences.")
    p.add_argument("--limit_files", type=int, default=5,
                   help="Max MAF files to download. 0 = ALL files.")
    p.add_argument("--all", action="store_true",
                   help="Download ALL sequences and ALL files. Overrides "
                        "--limit_seqs and --limit_files.")
    p.add_argument("--seq_len", type=int, default=512,
                   help="DNA window size in base pairs (default 512).")

    # Output paths
    p.add_argument("--out_name", type=str, default="cancer_genes.csv",
                   help="Output CSV filename (default: cancer_genes.csv).")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Directory to save CSV + cache. Overrides --drive.")
    p.add_argument("--drive", action="store_true",
                   help="Mount Google Drive (Colab) and save to "
                        "/content/drive/MyDrive/dataset/")

    # Modes
    p.add_argument("--demo", action="store_true",
                   help="Generate synthetic demo data. No internet needed.")
    p.add_argument("--list", action="store_true",
                   help="Print all available cancer types and exit.")

    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable TCGA cancer types:")
        print(f"  {'Code':<8} Description")
        print(f"  {'-'*40}")
        for code, name in CANCER_TYPES.items():
            print(f"  {code:<8} {name}")
        return

    # Resolve save paths (handles --drive, --out_dir, or default)
    csv_path, cache_dir = resolve_paths(args)

    if args.demo:
        generate_demo_dataset(csv_path, n_samples=200, seq_len=args.seq_len)
        return

    if not BIOPYTHON_OK:
        print("[ERROR] Install biopython: pip install biopython")
        return

    # --all overrides both limits
    limit_seqs  = 0 if args.all else args.limit_seqs
    limit_files = 0 if args.all else args.limit_files

    if args.all:
        print("[INFO] --all: downloading ALL sequences from ALL files.")

    build_dataset(
        cancer_types = args.cancer,
        seq_len      = args.seq_len,
        limit_seqs   = limit_seqs,
        limit_files  = limit_files,
        csv_path     = csv_path,
        cache_dir    = cache_dir,
    )


if __name__ == "__main__":
    main()
