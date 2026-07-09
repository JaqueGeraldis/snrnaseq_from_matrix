#!/usr/bin/env python3
"""
04b_clustering_final.py
=======================
Step 4b of the TT4 FAPESP case-control snRNA-seq pipeline (MTLE-HS vs autopsy).

  - Loads integrated checkpoint (adata_integrated.h5ad)
  - Applies the resolution defined in LEIDEN_FINAL (config.py)
  - Saves the final UMAP as hybrid PDF + high-res PNG
  - Saves checkpoint: adata_clustered_final.h5ad

*** Edit LEIDEN_FINAL in config.py before running this script ***

Run
---
    conda activate scrna
    screen -S step04b
    nohup python 04b_clustering_final.py > logs/04b_clustering_final.log 2>&1 &
    tail -f logs/04b_clustering_final.log
"""

import os
import sys
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FIG_DIR, CHECKPOINT_DIR,
    LEIDEN_FINAL, SEED
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sc.settings.verbosity = 2
np.random.seed(SEED)

INTEGRATED_CK = CHECKPOINT_DIR / "adata_integrated.h5ad"
FINAL_CK      = CHECKPOINT_DIR / "adata_clustered_final.h5ad"

# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("04b_clustering_final.py")
print("=" * 60)

if LEIDEN_FINAL is None:
    raise ValueError(
        "LEIDEN_FINAL ainda nao foi definido em config.py.\n"
        "Inspecione figures/clustree.png e defina, ex.: LEIDEN_FINAL = 'leiden_1.4'"
    )

print(f"\nLEIDEN_FINAL = '{LEIDEN_FINAL}'  (from config.py)\n")

# ── 1. Load integrated checkpoint ─────────────────────────────────────────

print(f"[1/3] Loading integrated checkpoint...")
adata = sc.read_h5ad(str(INTEGRATED_CK))
print(f"  {adata.n_obs:,} nuclei × {adata.n_vars:,} genes")

if LEIDEN_FINAL not in adata.obs.columns:
    raise ValueError(
        f"Column '{LEIDEN_FINAL}' not found in adata.obs.\n"
        f"Available Leiden columns: "
        f"{[c for c in adata.obs.columns if c.startswith('leiden_')]}\n"
        f"Check LEIDEN_FINAL in config.py."
    )

n_clusters = adata.obs[LEIDEN_FINAL].nunique()
print(f"  Resolution '{LEIDEN_FINAL}': {n_clusters} clusters")

# ── 2. Final UMAP figures ──────────────────────────────────────────────────

print(f"\n[2/3] Saving final UMAP figures...")

# Panel 1: clusters only
sc.pl.umap(
    adata,
    color=LEIDEN_FINAL,
    title=f"Final clustering — {LEIDEN_FINAL} ({n_clusters} clusters)",
    legend_loc="on data",
    legend_fontsize=11,
    legend_fontoutline=3,
    frameon=False,
    show=False,
)
ax = plt.gca()
for coll in ax.collections:
    coll.set_rasterized(True)

pdf_path = FIG_DIR / f"umap_{LEIDEN_FINAL}_hybrid.pdf"
png_path = FIG_DIR / f"umap_{LEIDEN_FINAL}_600dpi.png"
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.close()
print(f"  Saved: {pdf_path.name}")
print(f"  Saved: {png_path.name}")

# Panel 2: clusters + source_type + disease_group side by side
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
sc.pl.umap(
    adata,
    color=LEIDEN_FINAL,
    title=f"{LEIDEN_FINAL} ({n_clusters} clusters)",
    legend_loc="on data",
    legend_fontsize=9,
    legend_fontoutline=2,
    frameon=False,
    ax=axes[0], show=False,
)
sc.pl.umap(
    adata,
    color="source_type",
    title="Source type (caso / controle)",
    frameon=False,
    ax=axes[1], show=False,
)
sc.pl.umap(
    adata,
    color="disease_group",
    title="Disease group (G1 / G2)",
    frameon=False,
    ax=axes[2], show=False,
)
plt.suptitle("Final clustering overview", fontsize=14, fontweight="bold")
plt.tight_layout()
overview_png = FIG_DIR / f"umap_{LEIDEN_FINAL}_overview.png"
plt.savefig(overview_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {overview_png.name}")

# ── 3. Save final checkpoint ───────────────────────────────────────────────

print(f"\n[3/3] Saving final clustered checkpoint...")
adata.write_h5ad(str(FINAL_CK))

print(f"\n✓ Checkpoint saved : {FINAL_CK}")
print(f"✓ Clusters         : {n_clusters}")
print(f"✓ Resolution used  : {LEIDEN_FINAL}")
print("\n>>> Next step: run 05a_annotation_markers.py")
