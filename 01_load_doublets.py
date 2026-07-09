#!/usr/bin/env python3
"""
01_load_doublets.py
===================
Step 1 of the TT4 FAPESP case-control snRNA-seq pipeline (MTLE-HS vs autopsy).

  - CASO      : loads FILTERED matrix (barcodes/features/matrix.mtx dir) -- already
                cell-called by CellRanger's own filtering, no extra droplet call needed.
  - CONTROLE  : loads RAW matrix (cellranger raw_feature_bc_matrix.h5) -- includes ALL
                droplets (empty + real cells), so runs emptyDrops (DropletUtils, via
                rpy2) FIRST to call real nuclei, then proceeds the same way.
  - Adds sample metadata (source_type, disease_group, donor_id, tissue, etc.)
  - Runs scDblFinder (via rpy2) per sample independently
  - Saves per-sample checkpoint to tables/doublet_checkpoints/<sample_id>.h5ad
    (skip-if-exists: safe to re-run after a crash)

Output
------
    tables/doublet_checkpoints/<sample_id>.h5ad  — one file per sample
    tables/load_summary.csv                      — cells loaded per sample

Run
---
    conda activate base
    screen -S step01
    nohup python 01_load_doublets.py > logs/01_load_doublets.log 2>&1 &
    tail -f logs/01_load_doublets.log
"""

import os
import sys
import tempfile
import scipy.io
import scipy.sparse as sp
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# --- Hide GPU (outdated NVIDIA driver on server) ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- Import global config (must be in the same directory or on PYTHONPATH) ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASE_DIR, OUT_DIR, TABLE_DIR, CHECKPOINT_DIR, LOG_DIR,
    SAMPLES, SEED, EMPTYDROPS_PARAMS
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sc.settings.verbosity = 2
np.random.seed(SEED)

DOUBLET_DIR = TABLE_DIR / "doublet_checkpoints"
DOUBLET_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# R packages
# =============================================================================

print("Loading R packages...")
importr("scDblFinder")
importr("SingleCellExperiment")
importr("BiocParallel")
importr("DropletUtils")   # emptyDrops — only needed for raw/controle samples
importr("Matrix")
print("R packages loaded.\n")

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_filtered_sample(sample_id: str, meta: dict) -> ad.AnnData:
    """Load a CASO sample from a filtered 10x mtx directory (barcodes/features/matrix.mtx)."""
    mtx_dir = meta["path"]
    adata = sc.read_mtx(mtx_dir / "matrix.mtx").T
    barcodes = pd.read_csv(mtx_dir / "barcodes.tsv", header=None)[0].values
    features = pd.read_csv(mtx_dir / "features.tsv", header=None, sep="\t")

    adata.obs_names = barcodes
    # features.tsv pode ter 1 coluna (symbol) ou 3 (id, symbol, type) -- pega a de symbol
    gene_col = 1 if features.shape[1] >= 2 else 0
    adata.var_names = features[gene_col].values
    adata.var_names_make_unique()

    print(f"  [caso/filtered] Loaded: {sample_id} ({meta['donor_id']})"
          f"  | {adata.n_obs:,} nuclei (ja filtrado pelo CellRanger)"
          f"  | {adata.n_vars:,} genes")
    return adata


def run_emptyDrops(adata: ad.AnnData, sample_id: str, seed: int = SEED) -> ad.AnnData:
    """
    Run emptyDrops (DropletUtils) on a RAW controle sample to call real nuclei
    among all droplets, before scDblFinder.
    """
    print(f"  emptyDrops: {sample_id} ({adata.n_obs:,} droplets, raw)")

    tmpdir       = tempfile.mkdtemp()
    mtx_path     = os.path.join(tmpdir, "raw_counts.mtx")
    result_path  = os.path.join(tmpdir, "emptydrops.csv")

    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    scipy.io.mmwrite(mtx_path, X.T)  # genes x droplets, como DropletUtils espera

    fdr_thr = EMPTYDROPS_PARAMS["fdr_threshold"]
    lower   = EMPTYDROPS_PARAMS["lower"]

    ro.r(f"""
        library(DropletUtils)
        library(Matrix)
        set.seed({seed})
        raw_mat <- readMM("{mtx_path}")
        ed <- emptyDrops(raw_mat, lower = {lower})
        is_cell <- !is.na(ed$FDR) & ed$FDR <= {fdr_thr}
        result <- data.frame(is_cell = is_cell, fdr = ed$FDR)
        write.csv(result, "{result_path}", row.names = FALSE)
    """)

    result_df = pd.read_csv(result_path)
    keep_mask = result_df["is_cell"].fillna(False).values

    n_before = adata.n_obs
    adata = adata[keep_mask].copy()
    print(f"    emptyDrops: {n_before:,} droplets -> {adata.n_obs:,} nuclei chamados "
          f"(FDR <= {fdr_thr})")

    os.remove(mtx_path)
    os.remove(result_path)
    os.rmdir(tmpdir)

    return adata


def load_raw_sample(sample_id: str, meta: dict) -> ad.AnnData:
    """Load a CONTROLE sample from a raw cellranger .h5, then call cells with emptyDrops."""
    h5_path = meta["path"]
    adata = sc.read_10x_h5(str(h5_path))
    adata.var_names_make_unique()

    print(f"  [controle/raw] Loaded: {sample_id} ({meta['donor_id']})"
          f"  | {adata.n_obs:,} droplets (raw, antes do emptyDrops)"
          f"  | {adata.n_vars:,} genes")

    adata = run_emptyDrops(adata, sample_id)
    return adata


def attach_metadata(adata: ad.AnnData, sample_id: str, meta: dict) -> ad.AnnData:
    """Attach common sample metadata regardless of source_type."""
    adata.obs["sample_id"]         = sample_id
    adata.obs["donor_id"]          = meta["donor_id"]
    adata.obs["source_type"]       = meta["source_type"]        # "caso" | "controle"
    adata.obs["disease_group"]     = meta["disease_group"]      # "G1" | "G2"
    adata.obs["matrix_type"]       = meta["matrix_type"]         # "filtered" | "raw"
    adata.obs["tissue"]            = meta["tissue"]
    adata.obs["age"]               = int(meta["age"])
    adata.obs["sex"]               = meta["sex"]
    adata.obs["pmi_h"]             = meta["pmi_h"] if meta["pmi_h"] is not None else np.nan
    adata.obs["disease_duration_y"] = (
        meta["disease_duration_y"] if meta["disease_duration_y"] is not None else np.nan
    )
    adata.obs["batch"]             = meta["batch"]

    # Make barcodes unique across samples
    adata.obs_names = [f"{sample_id}_{bc}" for bc in adata.obs_names]
    return adata


def load_sample(sample_id: str, meta: dict) -> ad.AnnData:
    """Dispatch loader according to matrix_type, then attach shared metadata."""
    if meta["matrix_type"] == "filtered":
        adata = load_filtered_sample(sample_id, meta)
    elif meta["matrix_type"] == "raw":
        adata = load_raw_sample(sample_id, meta)
    else:
        raise ValueError(f"matrix_type desconhecido para {sample_id}: {meta['matrix_type']}")

    adata = attach_metadata(adata, sample_id, meta)
    return adata


def run_scDblFinder(adata: ad.AnnData, sample_name: str, seed: int = SEED) -> ad.AnnData:
    """
    Run scDblFinder on a single sample via a temporary MTX file.
    Avoids deprecated rpy2 converters — uses file I/O only.
    """
    print(f"  scDblFinder: {sample_name} ({adata.n_obs:,} nuclei)")

    tmpdir      = tempfile.mkdtemp()
    mtx_path    = os.path.join(tmpdir, "counts.mtx")
    result_path = os.path.join(tmpdir, "doublets.csv")

    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    scipy.io.mmwrite(mtx_path, X.T)

    ro.r(f"""
        library(scDblFinder)
        library(SingleCellExperiment)
        library(Matrix)
        set.seed({seed})
        counts_mat <- readMM("{mtx_path}")
        sce <- SingleCellExperiment(assays = list(counts = counts_mat))
        sce <- scDblFinder(sce, BPPARAM = BiocParallel::SerialParam())
        result <- data.frame(
            score = sce$scDblFinder.score,
            class = as.character(sce$scDblFinder.class)
        )
        write.csv(result, "{result_path}", row.names = FALSE)
    """)

    result_df = pd.read_csv(result_path)
    adata.obs["doublet_score"] = result_df["score"].values
    adata.obs["doublet_class"] = result_df["class"].values

    n_doublets = (result_df["class"] == "doublet").sum()
    pct        = 100 * n_doublets / adata.n_obs
    print(f"    Doublets: {n_doublets:,} ({pct:.1f}%)")

    os.remove(mtx_path)
    os.remove(result_path)
    os.rmdir(tmpdir)

    return adata


# =============================================================================
# MAIN
# =============================================================================

n_caso     = sum(1 for m in SAMPLES.values() if m["source_type"] == "caso")
n_controle = sum(1 for m in SAMPLES.values() if m["source_type"] == "controle")

print("=" * 60)
print("01_load_doublets.py")
print("=" * 60)
print(f"Total samples : {len(SAMPLES)}")
print(f"  Caso        : {n_caso}  (filtered)")
print(f"  Controle    : {n_controle}  (raw + emptyDrops)")
print(f"Checkpoint dir: {DOUBLET_DIR}\n")

load_summary = []

for sample_id, meta in SAMPLES.items():

    checkpoint = DOUBLET_DIR / f"{sample_id}.h5ad"

    # ----------------------------------------------------------------
    # Skip if checkpoint already exists (crash-safe resume)
    # ----------------------------------------------------------------
    if checkpoint.exists():
        print(f"  [SKIP] {sample_id} ({meta['source_type']}, {meta['disease_group']}) "
              f"— checkpoint found: {checkpoint.name}")
        _tmp = sc.read_h5ad(str(checkpoint))
        n_doublets = (_tmp.obs["doublet_class"] == "doublet").sum()
        load_summary.append({
            "sample_id":        sample_id,
            "source_type":      meta["source_type"],
            "disease_group":    meta["disease_group"],
            "matrix_type":      meta["matrix_type"],
            "tissue":           meta["tissue"],
            "n_nuclei_loaded":  _tmp.n_obs,
            "n_doublets":       n_doublets,
            "pct_doublets":     round(100 * n_doublets / _tmp.n_obs, 2),
        })
        del _tmp
        continue

    # ----------------------------------------------------------------
    # Load (filtered ou raw+emptyDrops) + doublet detection
    # ----------------------------------------------------------------
    print(f"\n--- {sample_id} ({meta['donor_id']}) | {meta['source_type']} | {meta['disease_group']} ---")

    adata = load_sample(sample_id, meta)
    adata = run_scDblFinder(adata, sample_name=sample_id)

    # Save checkpoint
    adata.write_h5ad(str(checkpoint))
    print(f"    Checkpoint saved: {checkpoint.name}")

    n_doublets = (adata.obs["doublet_class"] == "doublet").sum()
    load_summary.append({
        "sample_id":        sample_id,
        "source_type":      meta["source_type"],
        "disease_group":    meta["disease_group"],
        "matrix_type":      meta["matrix_type"],
        "tissue":           meta["tissue"],
        "n_nuclei_loaded":  adata.n_obs,
        "n_doublets":       n_doublets,
        "pct_doublets":     round(100 * n_doublets / adata.n_obs, 2),
    })

    del adata   # free memory before next sample

# =============================================================================
# SUMMARY
# =============================================================================

summary_df = pd.DataFrame(load_summary)
summary_path = TABLE_DIR / "load_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(summary_df[[
    "sample_id", "source_type", "disease_group", "matrix_type",
    "tissue", "n_nuclei_loaded", "n_doublets", "pct_doublets"
]].to_string(index=False))

print(f"\nTotal nuclei loaded : {summary_df['n_nuclei_loaded'].sum():,}")
print(f"Total doublets      : {summary_df['n_doublets'].sum():,} "
      f"({100 * summary_df['n_doublets'].sum() / summary_df['n_nuclei_loaded'].sum():.1f}%)")

for src in ["caso", "controle"]:
    grp = summary_df[summary_df["source_type"] == src]
    if not grp.empty:
        print(f"\n  {src.capitalize()} ({len(grp)} samples): "
              f"{grp['n_nuclei_loaded'].sum():,} nuclei, "
              f"{grp['n_doublets'].sum():,} doublets "
              f"({100 * grp['n_doublets'].sum() / grp['n_nuclei_loaded'].sum():.1f}%)")
    for g in ["G1", "G2"]:
        sub = grp[grp["disease_group"] == g]
        if not sub.empty:
            print(f"    {g}: {len(sub)} amostras, {sub['n_nuclei_loaded'].sum():,} nuclei")

print(f"\n✓ Summary saved to : {summary_path}")
print(f"✓ Checkpoints in   : {DOUBLET_DIR}")
print("\n>>> Next step: run 02_qc.py")
