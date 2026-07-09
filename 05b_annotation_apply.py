#!/usr/bin/env python3
"""
05b_annotation_apply.py
=======================
Step 5b of the TT4 FAPESP case-control snRNA-seq pipeline (MTLE-HS vs autopsy).

  - Loads the final clustered checkpoint (adata_clustered_final.h5ad)
  - Maps CLUSTER_ANNOTATION (from config.py) to adata.obs['cell_type']
  - Saves annotated UMAP (hybrid PDF + 600dpi PNG)
  - Saves cluster annotation table
  - Saves final annotated object: adata_annotated.h5ad

*** This script uses CLUSTER_ANNOTATION from config.py ***
*** Edit config.py before running if annotation has changed ***

Run
---
    conda activate scrna
    screen -S step05b
    nohup python 05b_annotation_apply.py > logs/05b_annotation_apply.log 2>&1 &
    tail -f logs/05b_annotation_apply.log
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OUT_DIR, FIG_DIR, TABLE_DIR, CHECKPOINT_DIR,
    CLUSTER_ANNOTATION, LEIDEN_FINAL, SEED,
    SAMPLES, SOURCE_LABELS,
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sc.settings.verbosity = 2
np.random.seed(SEED)

FINAL_CK     = CHECKPOINT_DIR / "adata_clustered_final.h5ad"
ANNOTATED_CK = OUT_DIR / "adata_annotated.h5ad"

# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("05b_annotation_apply.py")
print("=" * 60)
print(f"\nLEIDEN_FINAL        = '{LEIDEN_FINAL}'")
print(f"Clusters to annotate: {len(CLUSTER_ANNOTATION)}\n")

# ── 1. Load checkpoint ─────────────────────────────────────────────────────

print("[1/4] Loading final clustered checkpoint...")
adata = sc.read_h5ad(str(FINAL_CK))
print(f"  {adata.n_obs:,} nuclei × {adata.n_vars:,} genes")
print(f"  Clusters in data: {adata.obs[LEIDEN_FINAL].nunique()}")

n_caso     = (adata.obs["source_type"] == "caso").sum()
n_controle = (adata.obs["source_type"] == "controle").sum()
n_samples  = adata.obs["sample_id"].nunique()
print(f"  Caso     : {n_caso:,} nuclei")
print(f"  Controle : {n_controle:,} nuclei")

# ── 2. Apply annotation ────────────────────────────────────────────────────

print("\n[2/4] Applying cell type annotation...")

adata.obs["cell_type"] = adata.obs[LEIDEN_FINAL].map(CLUSTER_ANNOTATION)

n_unannotated = adata.obs["cell_type"].isna().sum()
if n_unannotated > 0:
    missing_clusters = sorted(
        adata.obs[adata.obs["cell_type"].isna()][LEIDEN_FINAL].unique()
    )
    raise ValueError(
        f"{n_unannotated:,} nuclei have no annotation.\n"
        f"Clusters missing from CLUSTER_ANNOTATION in config.py: {missing_clusters}\n"
        f"Add them and re-run."
    )

print("  All clusters annotated.")
print("\n  Cell type distribution:")
ct_counts = adata.obs["cell_type"].value_counts()
for ct, n in ct_counts.items():
    pct = 100 * n / adata.n_obs
    print(f"    {ct:<25}: {n:>7,}  ({pct:.1f}%)")

# ── 3. Figures ────────────────────────────────────────────────────────────

print("\n[3/4] Saving annotated UMAP figures...")

# Panel 1: cell type annotation
sc.pl.umap(
    adata,
    color="cell_type",
    title=f"Human hippocampus — cell type annotation "
          f"({n_samples} samples: MTLE-HS vs autopsy)",
    legend_loc="right margin",
    legend_fontsize=10,
    frameon=False,
    show=False,
)
ax = plt.gca()
for coll in ax.collections:
    coll.set_rasterized(True)

pdf_path = FIG_DIR / "umap_cell_type_annotation_hybrid.pdf"
png_path = FIG_DIR / "umap_cell_type_annotation_600dpi.png"
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.close()
print(f"  Saved: {pdf_path.name}")
print(f"  Saved: {png_path.name}")

# Panel 2: cell type + source_type + disease_group side by side
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
sc.pl.umap(
    adata,
    color="cell_type",
    title="Cell type annotation",
    legend_loc="right margin",
    legend_fontsize=9,
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
plt.suptitle(
    f"Human hippocampus — {n_samples} samples  "
    f"({n_caso:,} caso nuclei  |  {n_controle:,} controle nuclei)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
overview_png = FIG_DIR / "umap_annotation_overview.png"
plt.savefig(overview_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {overview_png.name}")

# Panel 3: cell type composition by source_type x disease_group (stacked bar)
comp = (
    adata.obs.groupby(["source_type", "disease_group", "cell_type"])
    .size()
    .reset_index(name="n")
)
comp["group"] = comp["source_type"].astype(str) + "_" + comp["disease_group"].astype(str)
pivot = comp.pivot_table(index="group", columns="cell_type", values="n", fill_value=0)
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(9, 6))
pivot_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
ax.set_ylabel("% of nuclei")
ax.set_xlabel("")
ax.set_title("Cell type composition by group (QC-level check)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
comp_png = FIG_DIR / "celltype_composition_by_group_qc.png"
plt.savefig(comp_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {comp_png.name}  (checagem visual rapida -- analise formal fica em 06_compositional.py)")

# ── 4. Save annotated checkpoint ──────────────────────────────────────────

print("\n[4/4] Saving annotated checkpoint and annotation table...")

# Cluster annotation table
annotation_df = pd.DataFrame.from_dict(
    CLUSTER_ANNOTATION, orient="index", columns=["cell_type"]
)
annotation_df.index.name = "cluster"
annotation_df.to_csv(TABLE_DIR / "cluster_annotation.csv")
print(f"  Annotation table: {TABLE_DIR / 'cluster_annotation.csv'}")

# Main annotated object
adata.write_h5ad(str(ANNOTATED_CK))
print(f"  Annotated object: {ANNOTATED_CK}")

# ── Summary ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Total nuclei   : {adata.n_obs:,}")
print(f"  Genes          : {adata.n_vars:,}")
print(f"  Samples        : {n_samples}")
print(f"    Caso         : {sum(1 for m in SAMPLES.values() if m['source_type'] == 'caso')}")
print(f"    Controle     : {sum(1 for m in SAMPLES.values() if m['source_type'] == 'controle')}")
print(f"  Cell types     : {adata.obs['cell_type'].nunique()}")
print(f"  Leiden used    : {LEIDEN_FINAL}")
print(f"\n✓ adata_annotated.h5ad saved: {ANNOTATED_CK}")
print("\n>>> Next step: run 06_compositional.py")
