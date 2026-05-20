"""
Stage 1 — Download Masked Somatic Mutation MAFs from GDC for TCGA-GBM and TCGA-BRCA.

Workflow type used: "Aliquot Ensemble Somatic Variant Merging and Masking"
(this is the current GDC workflow; MuTect2/MuSE-only workflows are deprecated).

Outputs a tarball of .maf.gz files per project under ./mafs/{project}/
"""

from __future__ import annotations

import json
import os
import re
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import requests

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"

PROJECTS = ["TCGA-GBM", "TCGA-BRCA"]
OUT_ROOT = Path("mafs")
OUT_ROOT.mkdir(exist_ok=True)


def build_filter(project: str) -> dict:
    return {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Masked Somatic Mutation"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["MAF"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type",
                                     "value": ["Aliquot Ensemble Somatic Variant Merging and Masking"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }


def list_files(project: str) -> list[dict]:
    params = {
        "filters": json.dumps(build_filter(project)),
        "fields": "file_id,file_name,file_size,md5sum",
        "format": "JSON",
        "size": "1000",
    }
    r = requests.get(GDC_FILES, params=params, timeout=60)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def download_bundle(file_ids: list[str], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"ids": file_ids})
    headers = {"Content-Type": "application/json"}
    with requests.post(GDC_DATA, data=payload, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        cdisp = r.headers.get("Content-Disposition", "")
        body = r.content

    # If single file: server returns the .maf.gz directly with Content-Disposition: filename=...
    # If multi-file: server returns a gzipped tarball.
    if "tar" in ctype or "tar" in cdisp:
        with tarfile.open(fileobj=BytesIO(body), mode="r:gz") as tf:
            tf.extractall(dest_dir)
    else:
        # Single file path. Pull filename from header if present.
        m = re.search(r'filename="?([^";]+)"?', cdisp)
        name = m.group(1) if m else f"{file_ids[0]}.maf.gz"
        (dest_dir / name).write_bytes(body)


def main() -> int:
    for project in PROJECTS:
        print(f"[{project}] querying GDC...", flush=True)
        hits = list_files(project)
        if not hits:
            print(f"[{project}] no files found — abort", file=sys.stderr)
            return 1
        print(f"[{project}] {len(hits)} file(s) matched; total size "
              f"{sum(h['file_size'] for h in hits) / 1e6:.1f} MB", flush=True)
        ids = [h["file_id"] for h in hits]
        download_bundle(ids, OUT_ROOT / project)
        print(f"[{project}] downloaded to {OUT_ROOT / project}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
