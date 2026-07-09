#!/usr/bin/env python3
"""
04a_clustree.py
===============
Step 4a of the TT4 FAPESP case-control snRNA-seq pipeline (MTLE-HS vs autopsy).

  - Loads the integrated checkpoint (adata_integrated.h5ad)
  - Runs clustree via rpy2 to visualise cluster stability across resolutions
  - Saves clustree.png for human inspection

*** HUMAN CHECKPOINT ***
After this script finishes:
  1. Open figures/clustree.png
  2. Choose the most stable Leiden resolution
  3. Edit LEIDEN_FINAL in config.py
  4. Run 04b_clustering_final.py

Run
---
    conda activate scrna
    screen -S step04a
    nohup python 04a_clustree.py > logs/04a_clustree.log 2>&1 &
    tail -f logs/04a_clustree.log
"""

import os
import sys
import numpy as np
import scanpy as sc
import rpy2.robjects as ro
import matplotlib
matplotlib.use("Agg")

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FIG_DIR, TABLE_DIR, CHECKPOINT_DIR,
    LEIDEN_RESOLUTIONS, SEED
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sc.settings.verbosity = 2
np.random.seed(SEED)

INTEGRATED_CK  = CHECKPOINT_DIR / "adata_integrated.h5ad"
CLUSTERING_CSV = TABLE_DIR / "clustering_resolutions.csv"
CLUSTREE_PNG   = FIG_DIR / "clustree.png"

# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("04a_clustree.py")
print("=" * 60)

# ── 1. Load integrated checkpoint ─────────────────────────────────────────

print(f"\n[1/2] Loading integrated checkpoint...")
print(f"      {INTEGRATED_CK}")

adata = sc.read_h5ad(str(INTEGRATED_CK))
print(f"  {adata.n_obs:,} nuclei × {adata.n_vars:,} genes")

n_caso     = (adata.obs["source_type"] == "caso").sum()
n_controle = (adata.obs["source_type"] == "controle").sum()
print(f"  Caso     : {n_caso:,} nuclei")
print(f"  Controle : {n_controle:,} nuclei")

for g in ["G1", "G2"]:
    n = (adata.obs["disease_group"] == g).sum()
    print(f"    {g}: {n:,} nuclei")

# Verify all Leiden resolutions exist
clustering_cols = [f"leiden_{r}" for r in LEIDEN_RESOLUTIONS]
missing = [c for c in clustering_cols if c not in adata.obs.columns]
if missing:
    raise ValueError(
        f"Missing Leiden columns in checkpoint: {missing}\n"
        f"Re-run 03_scvi_integration.py."
    )
print(f"  Leiden resolutions found: {len(clustering_cols)}")

# ── 2. Clustree via rpy2 ──────────────────────────────────────────────────

print(f"\n[2/2] Running clustree (R)...")
print(f"      Input CSV : {CLUSTERING_CSV}")
print(f"      Output    : {CLUSTREE_PNG}\n")

# Verify CSV exists (written by 03_scvi_integration.py)
if not CLUSTERING_CSV.exists():
    print("  Clustering CSV not found — writing now from checkpoint...")
    clustering_df = adata.obs[clustering_cols].copy()
    clustering_df.to_csv(CLUSTERING_CSV)
    print(f"  Saved: {CLUSTERING_CSV}")

col_names_r = ", ".join(f'"{c}"' for c in clustering_cols)

ro.r(f"""
    library(clustree)
    library(ggplot2)

    df <- read.csv("{CLUSTERING_CSV}", row.names=1)
    colnames(df) <- c({col_names_r})
    df[] <- lapply(df, as.factor)

    p <- clustree(df, prefix="leiden_") +
        theme(legend.position="right") +
        ggtitle("Clustree — cluster stability across Leiden resolutions")

    ggsave("{CLUSTREE_PNG}", plot=p, width=16, height=12, dpi=600)
    cat("clustree saved\\n")
""")

print(f"\n✓ Clustree saved: {CLUSTREE_PNG}")

# ── Summary ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Clusters per resolution:")
print("=" * 60)
for res in LEIDEN_RESOLUTIONS:
    key = f"leiden_{res}"
    n   = adata.obs[key].nunique()
    print(f"  res={res:<5}  {n:>3} clusters")

print("""
*** HUMAN CHECKPOINT ***

1. Open: figures/clustree.png
2. Identify the resolution where cluster structure stabilises
3. Edit LEIDEN_FINAL in config.py  (e.g. LEIDEN_FINAL = "leiden_1.2")
4. Run: python 04b_clustering_final.py
""")
