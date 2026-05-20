"""
Stage 2 — Parse all .maf.gz files in mafs/{project}/ and emit a unified TSV
of variant coordinates per project.

Output columns:
  project, case_id, chromosome, start, end, ref, alt, variant_type, gene

Notes on MAF format (GDC v1.0+):
  * Header line starts with 'Hugo_Symbol'
  * Comment lines start with '#' (version, filedate, etc.)
  * Coordinates are 1-based, fully closed (GRCh38)
  * Chromosome column already includes the 'chr' prefix (e.g. 'chr1', 'chrX')
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pandas as pd

KEEP_COLS = [
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "Variant_Type",
    "Tumor_Sample_Barcode",
]


def parse_maf(path: Path) -> pd.DataFrame:
    # GDC MAFs are TSV with '#' comment lines at the top.
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        usecols=KEEP_COLS,
        low_memory=False,
        dtype={"Chromosome": str, "Start_Position": "Int64", "End_Position": "Int64"},
    )
    df["case_id"] = df["Tumor_Sample_Barcode"].str.slice(0, 12)  # TCGA-XX-XXXX
    return df


def main(projects: list[str]) -> int:
    root = Path("mafs")
    out = Path("variants")
    out.mkdir(exist_ok=True)

    for project in projects:
        proj_dir = root / project
        mafs = sorted(proj_dir.rglob("*.maf.gz"))
        if not mafs:
            print(f"[{project}] no MAFs found under {proj_dir}", file=sys.stderr)
            continue

        frames = []
        for m in mafs:
            try:
                frames.append(parse_maf(m))
            except Exception as e:
                print(f"[{project}] FAILED to parse {m.name}: {e}", file=sys.stderr)
        if not frames:
            continue

        df = pd.concat(frames, ignore_index=True)
        df = df.rename(columns={
            "Hugo_Symbol": "gene",
            "Chromosome": "chromosome",
            "Start_Position": "start",
            "End_Position": "end",
            "Reference_Allele": "ref",
            "Tumor_Seq_Allele2": "alt",
            "Variant_Type": "variant_type",
        })
        df["project"] = project
        df = df[["project", "case_id", "chromosome", "start", "end",
                 "ref", "alt", "variant_type", "gene"]]

        # Drop non-canonical contigs (alt scaffolds, decoys). Keep chr1..22, X, Y, M.
        canonical = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
        df = df[df["chromosome"].isin(canonical)].reset_index(drop=True)

        out_path = out / f"{project}_variants.tsv.gz"
        df.to_csv(out_path, sep="\t", index=False, compression="gzip")
        print(f"[{project}] {len(df):,} variants across {df['case_id'].nunique()} cases "
              f"-> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(["TCGA-GBM", "TCGA-BRCA"]))
