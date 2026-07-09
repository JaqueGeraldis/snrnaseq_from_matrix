#!/usr/bin/env python3
"""
02_qc.py
========
Step 2 of the TT4 FAPESP case-control snRNA-seq pipeline (MTLE-HS vs autopsy).

  - Loads per-sample checkpoints from tables/doublet_checkpoints/
  - Runs MAD-based QC per sample (n_genes, total_counts, pct_mito, doublets)
  - Saves QC violin plots (before vs after) — scaled with p99 on y-axis
  - Runs sample-level marker coverage filter
  - Saves approved samples list and QC summary table

Output
------
    figures/qc_before_after.png / .pdf
    tables/qc_summary.csv
    tables/samples_approved.txt
    checkpoints/adatas_qc/   — one .h5ad per approved sample

Run
---
    conda activate scrna
    screen -S step02
    nohup python 02_qc.py > logs/02_qc.log 2>&1 &
    tail -f logs/02_qc.log
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for nohup
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OUT_DIR, FIG_DIR, TABLE_DIR, CHECKPOINT_DIR,
    SAMPLES, QC_PARAMS, ESSENTIAL_MARKERS,
    MIN_CELLS_EXPRESSING, MIN_TYPES_COVERED, SEED,
    SOURCE_ORDER, SOURCE_LABELS,
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sc.settings.verbosity = 2
sc.settings.figdir = str(FIG_DIR)
np.random.seed(SEED)

DOUBLET_DIR  = TABLE_DIR / "doublet_checkpoints"
QC_ADATA_DIR = CHECKPOINT_DIR / "adatas_qc"
QC_ADATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FUNCTIONS
# =============================================================================

def mad_filter(values: np.ndarray, n_mads: int = 3, direction: str = "both"):
    """Return (lower, upper) MAD-based thresholds on log1p-transformed values."""
    median = np.median(values)
    mad    = np.median(np.abs(values - median))
    lower  = median - n_mads * mad if direction in ("both", "lower") else -np.inf
    upper  = median + n_mads * mad if direction in ("both", "upper") else  np.inf
    return lower, upper


def run_qc_per_sample(adata: sc.AnnData, sample_name: str, qc_params: dict):
    """
    Apply MAD-based QC to a single sample.
    Returns (adata_filtered, stats_dict).
    """
    n_before = adata.n_obs

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    mask_min_genes = adata.obs["n_genes_by_counts"] >= qc_params["min_genes"]
    mask_mito      = adata.obs["pct_counts_mt"]     <= qc_params["pct_mito_max"]

    lo_genes, hi_genes = mad_filter(
        np.log1p(adata.obs["n_genes_by_counts"]), n_mads=qc_params["mad_n"]
    )
    mask_ngenes = (
        (np.log1p(adata.obs["n_genes_by_counts"]) >= lo_genes) &
        (np.log1p(adata.obs["n_genes_by_counts"]) <= hi_genes)
    )

    lo_counts, hi_counts = mad_filter(
        np.log1p(adata.obs["total_counts"]), n_mads=qc_params["mad_n"]
    )
    mask_counts = (
        (np.log1p(adata.obs["total_counts"]) >= lo_counts) &
        (np.log1p(adata.obs["total_counts"]) <= hi_counts)
    )

    mask_singlets = adata.obs["doublet_class"] == "singlet"
    mask_pass     = mask_min_genes & mask_mito & mask_ngenes & mask_counts & mask_singlets

    adata_filt = adata[mask_pass].copy()
    sc.pp.filter_genes(adata_filt, min_cells=qc_params["min_cells"])

    stats = {
        "sample":             sample_name,
        "source_type":        adata.obs["source_type"].iloc[0],
        "disease_group":      adata.obs["disease_group"].iloc[0],
        "cells_before":       n_before,
        "cells_after":        adata_filt.n_obs,
        "cells_removed":      n_before - adata_filt.n_obs,
        "pct_removed":        round(100 * (n_before - adata_filt.n_obs) / n_before, 2),
        "doublets_removed":   int((~mask_singlets).sum()),
        "median_genes":       round(adata_filt.obs["n_genes_by_counts"].median(), 1),
        "median_counts":      round(adata_filt.obs["total_counts"].median(), 1),
        "median_pct_mito":    round(adata_filt.obs["pct_counts_mt"].median(), 2),
        "genes_kept":         adata_filt.n_vars,
    }

    print(f"  {sample_name} [{stats['source_type']}/{stats['disease_group']}]: "
          f"{n_before:,} → {adata_filt.n_obs:,} nuclei  "
          f"| removed: {stats['cells_removed']:,} ({stats['pct_removed']:.1f}%)")

    return adata_filt, stats


def plot_qc_before_after(adatas_raw: dict, adatas_qc: dict, sample_names: list,
                         qc_params: dict, save_path_prefix: str):
    """
    3-row × 2-col violin grid: n_genes / total_counts / pct_mito — before vs after.
    Y-axis scaled to p99 so outliers do not flatten the violins.
    Threshold lines drawn for n_genes and pct_mito.
    Violins coloured by source_type (caso vs controle).
    """
    metrics    = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    labels     = ["Number of genes per nucleus", "Total counts (UMI)", "% mitochondrial counts"]
    thresholds = {
        "n_genes_by_counts": qc_params["min_genes"],
        "total_counts":      None,
        "pct_counts_mt":     qc_params["pct_mito_max"],
    }

    sample_ids = list(adatas_raw.keys())

    # Colour by source_type
    source_palette = {"caso": "#4C72B0", "controle": "#DD8452"}
    violin_colors  = [
        source_palette.get(SAMPLES[s]["source_type"], "#999999")
        for s in sample_ids
    ]

    fig, axes = plt.subplots(
        3, 2,
        figsize=(18, 20),
        gridspec_kw={"height_ratios": [1.2, 1.5, 1.0]},
        constrained_layout=True
    )

    for row, (metric, label) in enumerate(zip(metrics, labels)):

        data_before = [adatas_raw[s].obs[metric].values for s in sample_ids]
        data_after  = [adatas_qc[s].obs[metric].values  for s in sample_ids]

        # p99-based y-axis so outliers don't collapse the violins
        ymax = max(
            max(np.percentile(d, 99) for d in data_before),
            max(np.percentile(d, 99) for d in data_after)
        ) * 1.08

        thresh = thresholds[metric]
        if thresh is not None:
            ymax = max(ymax, thresh * 1.15)

        for col, (data, subtitle) in enumerate([
            (data_before, "before QC"),
            (data_after,  "after QC"),
        ]):
            ax = axes[row, col]

            vp = ax.violinplot(
                data,
                positions=range(len(data)),
                widths=0.8,
                showextrema=False,
                showmedians=True,
            )
            for body, color in zip(vp["bodies"], violin_colors):
                body.set_facecolor(color)
                body.set_alpha(0.75)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.2)

            ax.set_xticks(range(len(sample_names)))
            ax.set_xticklabels(sample_names, rotation=60, ha="right", fontsize=8)
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(f"{label}\n({subtitle})", fontsize=14, fontweight="bold")
            ax.set_ylim(0, ymax)
            ax.yaxis.grid(True, alpha=0.3)
            ax.xaxis.grid(False)

            if thresh is not None:
                ax.axhline(
                    y=thresh, color="red", linestyle="--", linewidth=1.5,
                    label=f"threshold = {thresh}"
                )
                ax.legend(fontsize=8, loc="upper right", frameon=False)

    # Legend for source_type colours
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=source_palette["caso"],     alpha=0.75, label="Caso (MTLE-HS)"),
        Patch(facecolor=source_palette["controle"], alpha=0.75, label="Controle (autopsia)"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper right",
        fontsize=11, frameon=True, title="Source type"
    )

    fig.suptitle("QC before vs after filtering", fontsize=20, fontweight="bold")

    plt.savefig(f"{save_path_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_path_prefix}.pdf", bbox_inches="tight")
    plt.close()
    print(f"✓ QC figure saved: {save_path_prefix}.png / .pdf")


def marker_coverage_filter(adatas_qc: dict, essential_markers: dict,
                            min_cells: int, min_types: int):
    """
    Remove samples that do not cover all expected hippocampal cell types.
    Returns (samples_pass, samples_fail).
    """
    print("\n=== Sample-level marker coverage filter ===\n")
    samples_pass, samples_fail = [], []

    for sample_id, adata_s in adatas_qc.items():
        source        = SAMPLES[sample_id]["source_type"]
        disease_group = SAMPLES[sample_id]["disease_group"]
        types_covered = 0
        report        = []

        for cell_type, genes in essential_markers.items():
            markers_found = 0
            for gene in genes:
                if gene not in adata_s.var_names:
                    continue
                X      = adata_s[:, gene].X
                n_expr = (X > 0).sum() if not sp.issparse(X) else X.nnz
                if n_expr >= min_cells:
                    markers_found += 1
            covered = markers_found >= 1
            if covered:
                types_covered += 1
            report.append((cell_type, markers_found, len(genes), covered))

        passed = types_covered >= min_types
        print(f"{'✓' if passed else '✗'} {sample_id} [{source}/{disease_group}] "
              f"— {types_covered}/{min_types} types covered")
        for cell_type, found, total, covered in report:
            mark = "✓" if covered else "✗"
            print(f"    {mark} {cell_type}: {found}/{total} markers with "
                  f"≥{min_cells} nuclei")
        print()

        if passed:
            samples_pass.append(sample_id)
        else:
            samples_fail.append(sample_id)

    return samples_pass, samples_fail


# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("02_qc.py")
print("=" * 60)

# ── 1. Load per-sample doublet checkpoints ─────────────────────────────────

print("\n[1/4] Loading doublet checkpoints...\n")

adatas_raw = {}   # pre-QC (for plotting)
adatas_qc  = {}   # post-QC

qc_stats_list = []

for sample_id, meta in SAMPLES.items():
    chk = DOUBLET_DIR / f"{sample_id}.h5ad"
    if not chk.exists():
        print(f"  WARNING: checkpoint not found for {sample_id} — skipping.")
        continue
    adata = sc.read_h5ad(str(chk))
    # Compute QC metrics so columns exist for plotting
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    print(f"  Loaded checkpoint: {sample_id} [{meta['source_type']}/{meta['disease_group']}]"
          f"  ({adata.n_obs:,} nuclei)")
    adatas_raw[sample_id] = adata

print(f"\n  {len(adatas_raw)} samples loaded from checkpoints.")

# ── 2. MAD QC per sample ───────────────────────────────────────────────────

print("\n[2/4] Running MAD-based QC per sample...\n")

for sample_id, adata in adatas_raw.items():
    qc_chk = QC_ADATA_DIR / f"{sample_id}.h5ad"

    if qc_chk.exists():
        print(f"  [SKIP] {sample_id} - QC checkpoint found.")
        aq = sc.read_h5ad(str(qc_chk))
        # Recompute QC metrics if not stored in checkpoint (needed for plotting)
        if "n_genes_by_counts" not in aq.obs.columns:
            aq.var["mt"] = aq.var_names.str.startswith("MT-")
            sc.pp.calculate_qc_metrics(
                aq, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
            )
        adatas_qc[sample_id] = aq
        qc_stats_list.append({
            "sample":           sample_id,
            "source_type":      SAMPLES[sample_id]["source_type"],
            "disease_group":    SAMPLES[sample_id]["disease_group"],
            "cells_before":     adata.n_obs,
            "cells_after":      aq.n_obs,
            "cells_removed":    adata.n_obs - aq.n_obs,
            "pct_removed":      round(100 * (adata.n_obs - aq.n_obs) / adata.n_obs, 2),
            "doublets_removed": int((adata.obs["doublet_class"] == "doublet").sum()),
            "median_genes":     round(aq.obs["n_genes_by_counts"].median(), 1),
            "median_counts":    round(aq.obs["total_counts"].median(), 1),
            "median_pct_mito":  round(aq.obs["pct_counts_mt"].median(), 2),
            "genes_kept":       aq.n_vars,
        })
        continue

    adata_filt, stats = run_qc_per_sample(
        adata,
        sample_name=sample_id,
        qc_params=QC_PARAMS,
    )
    adatas_qc[sample_id] = adata_filt
    qc_stats_list.append(stats)

    adata_filt.write_h5ad(str(qc_chk))
    print(f"    QC checkpoint saved: {qc_chk.name}")

qc_df = pd.DataFrame(qc_stats_list)
qc_df.to_csv(TABLE_DIR / "qc_summary.csv", index=False)
print(f"\n✓ QC summary saved: {TABLE_DIR / 'qc_summary.csv'}")

# ── 3. QC figure — before vs after ────────────────────────────────────────

print("\n[3/4] Generating QC before/after figure...\n")

sample_names = list(adatas_raw.keys())

plot_qc_before_after(
    adatas_raw=adatas_raw,
    adatas_qc=adatas_qc,
    sample_names=sample_names,
    qc_params=QC_PARAMS,
    save_path_prefix=str(FIG_DIR / "qc_before_after"),
)

# ── 4. Sample-level marker coverage filter ────────────────────────────────

print("\n[4/4] Marker coverage filter...\n")

samples_pass, samples_fail = marker_coverage_filter(
    adatas_qc=adatas_qc,
    essential_markers=ESSENTIAL_MARKERS,
    min_cells=MIN_CELLS_EXPRESSING,
    min_types=MIN_TYPES_COVERED,
)

# Save approved sample list
approved_path = TABLE_DIR / "samples_approved.txt"
with open(approved_path, "w") as f:
    for s in samples_pass:
        f.write(f"{s}\n")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(qc_df[[
    "sample", "source_type", "disease_group", "cells_before", "cells_after",
    "pct_removed", "doublets_removed", "median_genes", "median_counts"
]].to_string(index=False))

print(f"\nTotal nuclei before QC : {qc_df['cells_before'].sum():,}")
print(f"Total nuclei after QC  : {qc_df['cells_after'].sum():,}")

for src in SOURCE_ORDER:
    grp = qc_df[qc_df["source_type"] == src]
    if not grp.empty:
        print(f"\n  {SOURCE_LABELS[src]} ({len(grp)} samples): "
              f"{grp['cells_after'].sum():,} nuclei after QC "
              f"(removed {grp['pct_removed'].mean():.1f}% avg)")
        for g in ["G1", "G2"]:
            sub = grp[grp["disease_group"] == g]
            if not sub.empty:
                print(f"    {g}: {len(sub)} amostras, {sub['cells_after'].sum():,} nuclei")

print(f"\nSamples approved       : {len(samples_pass)} → {samples_pass}")
print(f"Samples removed        : {len(samples_fail)} → {samples_fail}")
print(f"\n✓ Approved list saved  : {approved_path}")
print(f"✓ QC checkpoints in    : {QC_ADATA_DIR}")
print(f"✓ Figure saved in      : {FIG_DIR}")
print("\n>>> Next step: run 03_scvi_integration.py")
