"""
Stage 6 -- Diagnostics for positives vs. negatives.

Quantifies five distributional differences between cancer (label=1) and
normal (label=0) windows. The goal is to measure how much of the model's
discriminative signal could come from technical shortcuts vs. real cancer
biology, before retraining on a properly matched dataset.

Inputs (from earlier stages):
    sequences/*_windows.jsonl.gz        cancer windows  (label=1)
    sequences/normals_windows.jsonl.gz  normal windows  (label=0)

Outputs (to diagnostics/):
    gc_stats.json
    chrom_stats.json
    dinuc_stats.json
    region_stats.json        (Ensembl-based, ~2k+2k sample)
    repeat_stats.json        (Ensembl-based, ~2k+2k sample)
    figs/gc_hist.png
    figs/chrom_bars.png
    figs/dinuc_heatmap.png
    figs/region_bars.png
    figs/repeat_bars.png
    summary.md               (human-readable verdict)

Resumable: the two Ensembl-based diagnostics cache results in
diagnostics/cache/*.jsonl.gz so an interrupted run picks up where it left
off (Ensembl queries are slow).

Usage:
    python 06_diagnostics.py                    # all 5 diagnostics
    python 06_diagnostics.py --no_ensembl       # skip region + repeat
    python 06_diagnostics.py --sample_n 2000    # Ensembl sample size per class
    python 06_diagnostics.py --bin_width 0.01   # GC histogram resolution
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENSEMBL = "https://rest.ensembl.org"
HEADERS = {"Accept": "application/json", "User-Agent": "tcga-diagnostics/1.0"}
TARGET_RPS = 13.0
MIN_INTERVAL = 1.0 / TARGET_RPS
MAX_RETRIES = 6

CHROM_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    recs = []
    try:
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
    except (EOFError, OSError, zlib.error) as e:
        print(f"  WARNING: {path.name} is corrupt ({type(e).__name__}) — "
              f"recovered {len(recs):,} records before the corruption.")
    return recs


def collect_records(seq_dir: Path,
                    normals_override: Path | None = None) -> tuple[list[dict], list[dict]]:
    cancer, normal = [], []
    for p in sorted(seq_dir.glob("*_windows.jsonl.gz")):
        if "gcmatched" in p.name:
            continue
        is_normals_file = "normals" in p.name
        if is_normals_file:
            if normals_override is None:
                recs = load_jsonl(p)
                normal.extend(recs)
                print(f"  Loaded {len(recs):,} from {p.name}")
            # else: skip glob normals — will load from override below
        else:
            recs = load_jsonl(p)
            cancer.extend(recs)
            print(f"  Loaded {len(recs):,} from {p.name}")

    if normals_override is not None:
        if not normals_override.exists():
            print(f"  ERROR: --normals file not found: {normals_override}",
                  file=sys.stderr)
        else:
            recs = load_jsonl(normals_override)
            normal.extend(recs)
            print(f"  Loaded {len(recs):,} from {normals_override.name}"
                  f"  [--normals override]")

    return cancer, normal


def normalize_chrom(c: str) -> str:
    c = str(c)
    return c if c.startswith("chr") else "chr" + c


def chr_to_ensembl(c: str) -> str:
    c = c.removeprefix("chr")
    return "MT" if c == "M" else c


# ---------------------------------------------------------------------------
# Diagnostic 1 -- GC content
# ---------------------------------------------------------------------------
def gc_content(seq: str) -> float:
    s = seq.upper()
    return (s.count("G") + s.count("C")) / max(len(s), 1)


def run_gc(cancer: list[dict], normal: list[dict], out_dir: Path,
           bin_width: float = 0.01) -> dict:
    print("\n[1/5] GC content")
    c_gc = np.array([gc_content(r["seq"]) for r in cancer])
    n_gc = np.array([gc_content(r["seq"]) for r in normal])

    ks_stat, ks_p = stats.ks_2samp(c_gc, n_gc)

    res = {
        "cancer": {"n": len(c_gc), "mean": float(c_gc.mean()),
                   "std": float(c_gc.std()), "median": float(np.median(c_gc))},
        "normal": {"n": len(n_gc), "mean": float(n_gc.mean()),
                   "std": float(n_gc.std()), "median": float(np.median(n_gc))},
        "delta_mean": float(c_gc.mean() - n_gc.mean()),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }
    (out_dir / "gc_stats.json").write_text(json.dumps(res, indent=2))

    bins = np.arange(0, 1 + bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(c_gc, bins=bins, alpha=0.55, label=f"Cancer (n={len(c_gc):,})",
            color="#d62728", density=True)
    ax.hist(n_gc, bins=bins, alpha=0.55, label=f"Normal (n={len(n_gc):,})",
            color="#1f77b4", density=True)
    ax.axvline(c_gc.mean(), color="#d62728", lw=1, ls="--")
    ax.axvline(n_gc.mean(), color="#1f77b4", lw=1, ls="--")
    ax.set_xlabel("GC content")
    ax.set_ylabel("Density")
    ax.set_title(f"GC distribution  (delta_mean={res['delta_mean']:+.4f}, "
                 f"KS p={ks_p:.2e})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "gc_hist.png", dpi=150)
    plt.close(fig)

    print(f"  cancer GC : {res['cancer']['mean']:.4f} +/- {res['cancer']['std']:.4f}")
    print(f"  normal GC : {res['normal']['mean']:.4f} +/- {res['normal']['std']:.4f}")
    print(f"  delta_mean: {res['delta_mean']:+.4f}")
    print(f"  KS p      : {ks_p:.3e}")
    return res


# ---------------------------------------------------------------------------
# Diagnostic 2 -- Chromosomal distribution
# ---------------------------------------------------------------------------
def run_chrom(cancer: list[dict], normal: list[dict], out_dir: Path) -> dict:
    print("\n[2/5] Chromosomal distribution")
    c_counts = Counter(normalize_chrom(r["chromosome"]) for r in cancer)
    n_counts = Counter(normalize_chrom(r["chromosome"]) for r in normal)

    chroms = [c for c in CHROM_ORDER if c in c_counts or c in n_counts]
    c_arr = np.array([c_counts.get(c, 0) for c in chroms])
    n_arr = np.array([n_counts.get(c, 0) for c in chroms])

    table = np.vstack([c_arr, n_arr])
    chi2, chi2_p, dof, _ = stats.chi2_contingency(table + 1e-9)

    c_frac = c_arr / max(1, c_arr.sum())
    n_frac = n_arr / max(1, n_arr.sum())

    res = {
        "chromosomes": chroms,
        "cancer_counts": c_arr.tolist(),
        "normal_counts": n_arr.tolist(),
        "cancer_fractions": c_frac.tolist(),
        "normal_fractions": n_frac.tolist(),
        "chi2_statistic": float(chi2),
        "chi2_pvalue": float(chi2_p),
        "dof": int(dof),
    }
    (out_dir / "chrom_stats.json").write_text(json.dumps(res, indent=2))

    x = np.arange(len(chroms))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - w/2, c_frac, w, label="Cancer", color="#d62728", alpha=0.85)
    ax.bar(x + w/2, n_frac, w, label="Normal", color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(chroms, rotation=45, ha="right")
    ax.set_ylabel("Fraction of sequences")
    ax.set_title(f"Chromosomal distribution  (chi2={chi2:.1f}, p={chi2_p:.2e})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "chrom_bars.png", dpi=150)
    plt.close(fig)

    print(f"  chi2 = {chi2:.1f}  dof={dof}  p={chi2_p:.3e}")
    deltas = c_frac - n_frac
    idx_sorted = np.argsort(-np.abs(deltas))
    print("  Top 5 imbalanced chromosomes (cancer_frac - normal_frac):")
    for i in idx_sorted[:5]:
        print(f"    {chroms[i]:>6s}: {deltas[i]:+.4f}  "
              f"({c_frac[i]:.4f} vs {n_frac[i]:.4f})")
    return res


# ---------------------------------------------------------------------------
# Diagnostic 3 -- Dinucleotide composition (esp. CpG)
# ---------------------------------------------------------------------------
DINUCS = [a + b for a in "ACGT" for b in "ACGT"]


def dinuc_freqs(seq: str) -> dict[str, float]:
    s = seq.upper()
    counts = Counter(s[i:i+2] for i in range(len(s) - 1)
                     if s[i] in "ACGT" and s[i+1] in "ACGT")
    total = max(1, sum(counts.values()))
    return {d: counts.get(d, 0) / total for d in DINUCS}


def run_dinuc(cancer: list[dict], normal: list[dict], out_dir: Path) -> dict:
    print("\n[3/5] Dinucleotide composition")
    c_mat = np.array([list(dinuc_freqs(r["seq"]).values()) for r in cancer])
    n_mat = np.array([list(dinuc_freqs(r["seq"]).values()) for r in normal])

    c_mean = c_mat.mean(axis=0)
    n_mean = n_mat.mean(axis=0)
    delta = c_mean - n_mean

    tvals, pvals = stats.ttest_ind(c_mat, n_mat, axis=0, equal_var=False)

    res = {
        "dinucleotides": DINUCS,
        "cancer_mean": c_mean.tolist(),
        "normal_mean": n_mean.tolist(),
        "delta": delta.tolist(),
        "tstat": tvals.tolist(),
        "pvalue": pvals.tolist(),
    }
    (out_dir / "dinuc_stats.json").write_text(json.dumps(res, indent=2))

    delta_grid = delta.reshape(4, 4)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(delta_grid, cmap="RdBu_r",
                   vmin=-np.abs(delta).max(), vmax=np.abs(delta).max())
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(list("ACGT")); ax.set_yticklabels(list("ACGT"))
    ax.set_xlabel("Second base")
    ax.set_ylabel("First base")
    ax.set_title("Delta dinucleotide frequency (cancer - normal)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{delta_grid[i, j]:+.4f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(delta_grid[i, j]) > np.abs(delta).max()/2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "dinuc_heatmap.png", dpi=150)
    plt.close(fig)

    idx_sorted = np.argsort(-np.abs(delta))
    print("  Top 5 dinucleotides by |delta_freq|:")
    for i in idx_sorted[:5]:
        sig = "***" if pvals[i] < 1e-10 else ("**" if pvals[i] < 1e-3 else "")
        print(f"    {DINUCS[i]}: delta={delta[i]:+.5f}  "
              f"(cancer={c_mean[i]:.5f}, normal={n_mean[i]:.5f}, p={pvals[i]:.2e}) {sig}")
    cpg_idx = DINUCS.index("CG")
    print(f"  CpG (CG) ratio cancer/normal = "
          f"{c_mean[cpg_idx]/max(1e-9, n_mean[cpg_idx]):.2f}x")
    return res


# ---------------------------------------------------------------------------
# Diagnostic 4 + 5 -- Ensembl-based: region type & repeat overlap
# ---------------------------------------------------------------------------
def ensembl_overlap(session: requests.Session, chrom: str,
                    start: int, end: int, feature: str) -> list | None:
    url = (f"{ENSEMBL}/overlap/region/human/{chrom}:{start}-{end}"
           f"?feature={feature}")
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(delay); delay *= 2
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            retry_after = float(r.headers.get("Retry-After", delay))
            time.sleep(retry_after); delay = max(delay * 2, retry_after)
            continue
        if r.status_code in (400, 403, 404):
            return []
        if 500 <= r.status_code < 600:
            time.sleep(delay); delay *= 2
            continue
        return None
    return None


def classify_region(exon_hits: list, gene_hits: list) -> str:
    if exon_hits:
        return "exonic"
    if gene_hits:
        return "intronic"
    return "intergenic"


def query_sample(records: list[dict], label_name: str, sample_n: int,
                 cache_path: Path, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    indexed = sorted(records, key=lambda r: r.get("variant_id", ""))
    n = min(sample_n, len(indexed))
    sample = rng.sample(indexed, n)
    print(f"  Sampling {n:,} {label_name} records for Ensembl queries")

    cached: dict[str, dict] = {}
    if cache_path.exists():
        with gzip.open(cache_path, "rt") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cached[rec["variant_id"]] = rec
                except Exception:
                    continue
        print(f"  Resumed: {len(cached):,} already cached")

    todo = [r for r in sample if r.get("variant_id") not in cached]
    if not todo:
        print(f"  All {label_name} samples cached. Skipping queries.")
        return list(cached.values())

    session = requests.Session()
    last = 0.0

    with gzip.open(cache_path, "at") as out:
        for r in tqdm(todo, desc=f"  {label_name} Ensembl",
                      ncols=80, ascii=True, file=sys.stdout):
            dt = time.monotonic() - last
            if dt < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - dt)
            last = time.monotonic()

            chrom = chr_to_ensembl(normalize_chrom(r["chromosome"]))
            pos = int(r["start"])
            s = max(1, pos - 255)
            e = pos + 256

            exon_hits   = ensembl_overlap(session, chrom, s, e, "exon")
            gene_hits   = ensembl_overlap(session, chrom, s, e, "gene")
            repeat_hits = ensembl_overlap(session, chrom, s, e, "repeat")

            if exon_hits is None or gene_hits is None or repeat_hits is None:
                continue

            region    = classify_region(exon_hits, gene_hits)
            in_repeat = len(repeat_hits) > 0

            entry = {
                "variant_id": r.get("variant_id", f"{chrom}:{pos}"),
                "label": r.get("label", -1),
                "chromosome": normalize_chrom(r["chromosome"]),
                "start": pos,
                "region": region,
                "n_exons": len(exon_hits),
                "n_genes": len(gene_hits),
                "in_repeat": in_repeat,
                "n_repeats": len(repeat_hits),
            }
            out.write(json.dumps(entry) + "\n")
            cached[entry["variant_id"]] = entry

    return list(cached.values())


def run_region_repeat(cancer: list[dict], normal: list[dict],
                      out_dir: Path, sample_n: int) -> tuple[dict, dict]:
    print("\n[4-5/5] Region type + repeat overlap (Ensembl, sampled)")
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    c_results = query_sample(cancer, "cancer", sample_n,
                             cache_dir / "cancer_ensembl.jsonl.gz")
    n_results = query_sample(normal, "normal", sample_n,
                             cache_dir / "normal_ensembl.jsonl.gz")

    # -- Region type ----------------------------------------------------------
    REGIONS = ["exonic", "intronic", "intergenic"]
    c_reg = Counter(r["region"] for r in c_results)
    n_reg = Counter(r["region"] for r in n_results)
    c_total = max(1, sum(c_reg.values()))
    n_total = max(1, sum(n_reg.values()))

    c_frac = {k: c_reg.get(k, 0) / c_total for k in REGIONS}
    n_frac = {k: n_reg.get(k, 0) / n_total for k in REGIONS}

    table = np.array([[c_reg.get(k, 0) for k in REGIONS],
                      [n_reg.get(k, 0) for k in REGIONS]])
    chi2, chi2_p, dof, _ = stats.chi2_contingency(table + 1e-9)

    region_res = {
        "sample_n_cancer": c_total,
        "sample_n_normal": n_total,
        "regions": REGIONS,
        "cancer_counts": [c_reg.get(k, 0) for k in REGIONS],
        "normal_counts": [n_reg.get(k, 0) for k in REGIONS],
        "cancer_fractions": [c_frac[k] for k in REGIONS],
        "normal_fractions": [n_frac[k] for k in REGIONS],
        "chi2_statistic": float(chi2),
        "chi2_pvalue": float(chi2_p),
    }
    (out_dir / "region_stats.json").write_text(json.dumps(region_res, indent=2))

    x = np.arange(len(REGIONS)); w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - w/2, [c_frac[k] for k in REGIONS], w, label="Cancer",
           color="#d62728", alpha=0.85)
    ax.bar(x + w/2, [n_frac[k] for k in REGIONS], w, label="Normal",
           color="#1f77b4", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(REGIONS)
    ax.set_ylabel("Fraction of sequences")
    ax.set_title(f"Region type  (chi2={chi2:.1f}, p={chi2_p:.2e})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "region_bars.png", dpi=150)
    plt.close(fig)

    print(f"  Region chi2 = {chi2:.1f}  p={chi2_p:.3e}")
    for k in REGIONS:
        print(f"    {k:<11s}: cancer={c_frac[k]:.4f}  normal={n_frac[k]:.4f}  "
              f"delta={c_frac[k]-n_frac[k]:+.4f}")

    # -- Repeat overlap -------------------------------------------------------
    c_rep = sum(1 for r in c_results if r["in_repeat"])
    n_rep = sum(1 for r in n_results if r["in_repeat"])
    c_rep_frac = c_rep / c_total
    n_rep_frac = n_rep / n_total

    rtable = np.array([[c_rep, c_total - c_rep],
                       [n_rep, n_total - n_rep]])
    rchi2, rchi2_p, _, _ = stats.chi2_contingency(rtable + 1e-9)

    repeat_res = {
        "sample_n_cancer": c_total,
        "sample_n_normal": n_total,
        "cancer_in_repeat": c_rep,
        "normal_in_repeat": n_rep,
        "cancer_in_repeat_frac": c_rep_frac,
        "normal_in_repeat_frac": n_rep_frac,
        "chi2_statistic": float(rchi2),
        "chi2_pvalue": float(rchi2_p),
    }
    (out_dir / "repeat_stats.json").write_text(json.dumps(repeat_res, indent=2))

    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.bar(["Cancer", "Normal"], [c_rep_frac, n_rep_frac],
           color=["#d62728", "#1f77b4"], alpha=0.85)
    ax.set_ylabel("Fraction overlapping any repeat element")
    ax.set_ylim(0, 1)
    ax.set_title(f"Repeat overlap  (chi2={rchi2:.1f}, p={rchi2_p:.2e})")
    fig.tight_layout()
    fig.savefig(out_dir / "figs" / "repeat_bars.png", dpi=150)
    plt.close(fig)

    print(f"  Repeat chi2 = {rchi2:.1f}  p={rchi2_p:.3e}")
    print(f"    cancer in repeat: {c_rep_frac:.4f}  ({c_rep}/{c_total})")
    print(f"    normal in repeat: {n_rep_frac:.4f}  ({n_rep}/{n_total})")
    print(f"    delta = {c_rep_frac - n_rep_frac:+.4f}")

    return region_res, repeat_res


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def severity_gc(delta: float) -> str:
    a = abs(delta)
    if a > 0.05:  return "HIGH"
    if a > 0.02:  return "MEDIUM"
    return "LOW"


def severity_p(p: float) -> str:
    if p < 1e-10: return "HIGH"
    if p < 1e-3:  return "MEDIUM"
    return "LOW"


def severity_frac(delta: float) -> str:
    a = abs(delta)
    if a > 0.30:  return "CRITICAL"
    if a > 0.10:  return "HIGH"
    if a > 0.03:  return "MEDIUM"
    return "LOW"


def write_summary(gc_res, chrom_res, dinuc_res, region_res, repeat_res,
                  out_dir: Path) -> None:
    lines = []
    lines.append("# Diagnostics summary")
    lines.append("")
    lines.append("| Diagnostic | Statistic | p-value | Severity |")
    lines.append("|---|---|---|---|")

    lines.append(f"| GC content | delta_mean = {gc_res['delta_mean']:+.4f} | "
                 f"{gc_res['ks_pvalue']:.2e} | {severity_gc(gc_res['delta_mean'])} |")
    lines.append(f"| Chromosomal | chi2 = {chrom_res['chi2_statistic']:.1f} | "
                 f"{chrom_res['chi2_pvalue']:.2e} | "
                 f"{severity_p(chrom_res['chi2_pvalue'])} |")

    cpg_idx = DINUCS.index("CG")
    cpg_delta = dinuc_res['delta'][cpg_idx]
    cpg_p = dinuc_res['pvalue'][cpg_idx]
    lines.append(f"| CpG dinucleotide | delta = {cpg_delta:+.5f} | {cpg_p:.2e} | "
                 f"{severity_p(cpg_p)} (biological -- keep) |")

    if region_res is not None:
        exo_idx = region_res['regions'].index("exonic")
        exo_delta = (region_res['cancer_fractions'][exo_idx]
                     - region_res['normal_fractions'][exo_idx])
        lines.append(f"| Region type (exonic delta) | delta = {exo_delta:+.4f} | "
                     f"{region_res['chi2_pvalue']:.2e} | "
                     f"{severity_frac(exo_delta)} |")

    if repeat_res is not None:
        rep_delta = (repeat_res['cancer_in_repeat_frac']
                     - repeat_res['normal_in_repeat_frac'])
        lines.append(f"| Repeat overlap | delta = {rep_delta:+.4f} | "
                     f"{repeat_res['chi2_pvalue']:.2e} | "
                     f"{severity_frac(rep_delta)} |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **HIGH/CRITICAL severity = technical shortcut.** Model can hit "
                 "high AUC without learning cancer biology. Must be fixed before "
                 "claiming biological discrimination.")
    lines.append("- **CpG enrichment is real biology** (C->T at methylated CpG is a "
                 "known cancer mutational signature). Do not flatten it via matching; "
                 "the model *should* see this.")
    lines.append("- Recommended action: regenerate negatives with bedtools shuffle "
                 "(or Ensembl-based equivalent) matching on chromosome + GC + "
                 "region type simultaneously. Repeat status filtering only if "
                 "the repeat-overlap delta is HIGH or above.")

    (out_dir / "summary.md").write_text("\n".join(lines))
    print("\n" + "=" * 60)
    print("Summary written to: " + str(out_dir / "summary.md"))
    print("=" * 60)
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(no_ensembl: bool = False, sample_n: int = 2000,
         bin_width: float = 0.01, normals: str | None = None) -> int:
    seq_dir = Path("sequences")
    if not seq_dir.exists():
        print("sequences/ not found. Run stages 1-4 first.", file=sys.stderr)
        return 1

    out_dir = Path("diagnostics")
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    normals_path = Path(normals) if normals else None
    if normals_path:
        print(f"Normals override: {normals_path}")

    print("Loading sequences ...")
    cancer, normal = collect_records(seq_dir, normals_override=normals_path)
    if not cancer or not normal:
        print("Need both cancer and normal records.", file=sys.stderr)
        return 1
    print(f"Total: {len(cancer):,} cancer, {len(normal):,} normal")

    gc_res    = run_gc(cancer, normal, out_dir, bin_width=bin_width)
    chrom_res = run_chrom(cancer, normal, out_dir)
    dinuc_res = run_dinuc(cancer, normal, out_dir)

    if no_ensembl:
        print("\nSkipping Ensembl-based diagnostics (--no_ensembl).")
        region_res = repeat_res = None
    else:
        region_res, repeat_res = run_region_repeat(
            cancer, normal, out_dir, sample_n=sample_n)

    write_summary(gc_res, chrom_res, dinuc_res, region_res, repeat_res, out_dir)
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no_ensembl", action="store_true",
                   help="Skip region-type and repeat-overlap (no network).")
    p.add_argument("--sample_n", type=int, default=2000,
                   help="Sample size per class for Ensembl queries (default 2000)")
    p.add_argument("--bin_width", type=float, default=0.01,
                   help="GC histogram bin width (default 0.01)")
    p.add_argument("--normals", type=str, default=None,
                   help="Path to normals JSONL.gz to use instead of auto-detected "
                        "normals_windows.jsonl.gz.  Use after running 04c to measure "
                        "post-matching bias: "
                        "--normals sequences/normals_windows_matched.jsonl.gz")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(no_ensembl=args.no_ensembl,
                  sample_n=args.sample_n,
                  bin_width=args.bin_width,
                  normals=args.normals))
