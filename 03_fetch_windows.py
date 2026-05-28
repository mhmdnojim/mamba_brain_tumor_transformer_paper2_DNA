"""
Stage 3 — Fetch a 512 bp DNA window around each variant from Ensembl REST.

Window definition (per user spec): exactly 512 bp, anchored at Start_Position.
  fetch_start = start - 255     (inclusive)
  fetch_end   = start + 256     (inclusive)
  -> 512 bp; the reference base at `start` sits at index 255 (0-indexed).

Strategy: single GET per variant (user choice — slow but simple).
  * Endpoint: GET https://rest.ensembl.org/sequence/region/human/{chr}:{s}..{e}:1
  * Rate cap: 15 req/sec hard. We pace at ~13 req/sec to leave headroom.
  * On HTTP 429: honour Retry-After header, exponential backoff.
  * Resume: appends to JSONL; on restart, skips variants already in output.
  * Label: 1 (cancer) — fixed per user spec. Negative class is a separate task.

Wall-time estimate at 13 req/sec:
  TCGA-GBM ~80k variants  ~1.7 h
  TCGA-BRCA ~130k variants ~2.8 h
  Combined ~4.5 h minimum, more with 429s.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ENSEMBL = "https://rest.ensembl.org"
HEADERS = {"Accept": "application/json", "User-Agent": "tcga-window-fetcher/1.0"}
WINDOW = 512
LEFT = 255      # bases before start
RIGHT = 256     # bases at start + after  (255 + 1 + 256 = 512 incl. anchor)
TARGET_RPS = 13.0
MIN_INTERVAL = 1.0 / TARGET_RPS
MAX_RETRIES = 6


def chr_to_ensembl(c: str) -> str:
    """Ensembl uses '1','2',...,'X','Y','MT' — no 'chr' prefix, 'M' becomes 'MT'."""
    c = c.removeprefix("chr")
    return "MT" if c == "M" else c


def fetch_one(session: requests.Session, chrom: str, start: int) -> str | None:
    s = start - LEFT
    e = start + RIGHT
    if s < 1:  # near chromosome telomere; clamp and let caller pad
        s = 1
    url = f"{ENSEMBL}/sequence/region/human/{chrom}:{s}..{e}:1"

    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as ex:
            if attempt == MAX_RETRIES - 1:
                raise
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
            # Bad region (e.g. past chromosome end). Skip cleanly.
            return None
        if 500 <= r.status_code < 600:
            time.sleep(delay)
            delay *= 2
            continue
        # Other 4xx — give up on this variant
        return None
    return None


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    if path.stat().st_size == 0:
        path.unlink()
        return set()
    done = set()
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["variant_id"])
                except Exception:
                    continue
    except (EOFError, OSError):
        # File was truncated mid-write (session crash). Keep valid records found so far.
        print(f"  WARNING: {path.name} is incomplete (session crash). "
              f"Recovered {len(done):,} records — will re-fetch the rest.")
    return done


def run(variants_tsv: Path, out_jsonl: Path) -> None:
    df = pd.read_csv(variants_tsv, sep="\t")
    df["variant_id"] = (df["chromosome"] + ":" + df["start"].astype(str)
                        + ":" + df["ref"] + ">" + df["alt"]
                        + "@" + df["case_id"])

    done = load_done(out_jsonl)
    todo = df[~df["variant_id"].isin(done)].reset_index(drop=True)
    print(f"[{variants_tsv.name}] total={len(df):,}  done={len(done):,}  todo={len(todo):,}",
          flush=True)

    session = requests.Session()
    last = 0.0
    n_ok = n_fail = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(out_jsonl, "at") as out:
        pbar = tqdm(total=len(todo), unit="var", desc=variants_tsv.stem[:20],
                    dynamic_ncols=False, ascii=True, file=sys.stdout,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for i, row in todo.iterrows():
            dt = time.monotonic() - last
            if dt < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - dt)
            last = time.monotonic()

            chrom = chr_to_ensembl(row["chromosome"])
            try:
                seq = fetch_one(session, chrom, int(row["start"]))
            except Exception as ex:
                seq = None
                tqdm.write(f"  hard fail {row['variant_id']}: {ex}")

            pbar.update(1)
            if seq is None or len(seq) != WINDOW:
                n_fail += 1
                pbar.set_postfix(ok=n_ok, fail=n_fail, refresh=False)
                continue

            rec = {
                "variant_id": row["variant_id"],
                "project": row["project"],
                "case_id": row["case_id"],
                "chromosome": row["chromosome"],
                "start": int(row["start"]),
                "ref": row["ref"],
                "alt": row["alt"],
                "variant_type": row["variant_type"],
                "gene": row["gene"],
                "seq": seq.upper(),
                "label": 1,
            }
            out.write(json.dumps(rec) + "\n")
            n_ok += 1
            pbar.set_postfix(ok=n_ok, fail=n_fail, refresh=False)

            if (n_ok % 500) == 0:
                out.flush()

        pbar.close()

    print(f"[{variants_tsv.name}] DONE  ok={n_ok:,}  fail={n_fail:,}", flush=True)


def main() -> int:
    var_dir = Path("variants")
    out_dir = Path("sequences")
    for project in ["TCGA-GBM", "TCGA-BRCA"]:
        src = var_dir / f"{project}_variants.tsv.gz"
        if not src.exists():
            print(f"missing {src}; run stage 2 first", file=sys.stderr)
            continue
        dst = out_dir / f"{project}_windows.jsonl.gz"
        run(src, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
