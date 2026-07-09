#!/usr/bin/env python3
"""
05a_annotation_markers.py
=========================
Step 5a of the TT4 FAPESP case-control snRNA-seq pipeline (MTLE-HS vs autopsy).

  - Loads the final clustered checkpoint (adata_clustered_final.h5ad)
  - Verifies presence of canonical marker genes
  - Generates dotplot of marker expression per Leiden cluster

*** HUMAN CHECKPOINT ***
After this script finishes:
  1. Open figures/dotplot_leiden_1.4.pdf  (or .png)
  2. Assign a cell type to each cluster
  3. Edit CLUSTER_ANNOTATION in config.py
  4. Run 05b_annotation_apply.py

Run
---
    conda activate scrna
    screen -S step05a
    nohup python 05a_annotation_markers.py > logs/05a_annotation_markers.log 2>&1 &
    tail -f logs/05a_annotation_markers.log
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
    MARKER_GENES, LEIDEN_FINAL, SEED
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sc.settings.verbosity = 2
np.random.seed(SEED)

FINAL_CK = CHECKPOINT_DIR / "adata_clustered_final.h5ad"

# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("05a_annotation_markers.py")
print("=" * 60)
print(f"\nLEIDEN_FINAL = '{LEIDEN_FINAL}'  (from config.py)\n")

# ── 1. Load checkpoint ─────────────────────────────────────────────────────

print(f"[1/3] Loading final clustered checkpoint...")
adata = sc.read_h5ad(str(FINAL_CK))
print(f"  {adata.n_obs:,} nuclei × {adata.n_vars:,} genes")
print(f"  Clusters: {adata.obs[LEIDEN_FINAL].nunique()}")

# ── 2. Verify marker panel ─────────────────────────────────────────────────

print(f"\n[2/3] Verifying canonical marker panel...\n")
all_found = True
for cell_type, genes in MARKER_GENES.items():
    found   = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    status  = "OK" if not missing else f"WARNING - missing: {missing}"
    print(f"  {cell_type:<22}: {found}  [{status}]")
    if missing:
        all_found = False

if all_found:
    print("\n  All marker genes found in dataset.")
else:
    print("\n  WARNING: Some marker genes missing — dotplot will skip them.")

# ── 3. Dotplot ─────────────────────────────────────────────────────────────

print(f"\n[3/3] Generating marker dotplot...")

# Normalise a copy for visualisation only (raw counts preserved in adata.layers)
adata_vis = adata.copy()
sc.pp.normalize_total(adata_vis, target_sum=1e4)
sc.pp.log1p(adata_vis)

# Filter marker dict to genes present in the dataset
marker_genes_filtered = {
    ct: [g for g in genes if g in adata_vis.var_names]
    for ct, genes in MARKER_GENES.items()
    if any(g in adata_vis.var_names for g in genes)
}

sc.pl.dotplot(
    adata_vis,
    marker_genes_filtered,
    groupby=LEIDEN_FINAL,
    dendrogram=True,
    standard_scale="var",
    colorbar_title="Normalised\nexpression",
    figsize=(24, 10),
    show=False,
)

pdf_path = FIG_DIR / f"dotplot_{LEIDEN_FINAL}.pdf"
png_path = FIG_DIR / f"dotplot_{LEIDEN_FINAL}_600dpi.png"

plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.close()

print(f"\n✓ Dotplot saved:")
print(f"  {pdf_path}")
print(f"  {png_path}")

# ── Summary ───────────────────────────────────────────────────────────────

print(f"\nCluster sizes (top 10):")
cluster_sizes = adata.obs[LEIDEN_FINAL].value_counts().head(10)
for cluster, n in cluster_sizes.items():
    print(f"  Cluster {cluster:>3}: {n:>7,} nuclei")

# Composicao caso/controle e disease_group por cluster (ajuda a checar se
# algum cluster e "artefato" dominado por um unico source_type/amostra)
print(f"\nComposicao caso/controle por cluster:")
comp = (
    adata.obs.groupby([LEIDEN_FINAL, "source_type"])
    .size()
    .unstack(fill_value=0)
)
comp_pct = comp.div(comp.sum(axis=1), axis=0).round(2)
print(comp_pct.to_string())

print(f"""
*** HUMAN CHECKPOINT ***

1. Open: figures/dotplot_{LEIDEN_FINAL}.pdf
2. Assign a cell type to each cluster based on marker expression
   (atencao a clusters muito enviesados p/ um unico source_type na tabela acima --
   podem ser sinal de doublets residuais ou efeito de batch, nao um tipo celular real)
3. Edit CLUSTER_ANNOTATION in config.py
4. Run: python 05b_annotation_apply.py
""")
