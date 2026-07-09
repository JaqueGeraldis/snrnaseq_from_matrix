#!/usr/bin/env python3
"""
11_umap_figures.py
==================
Supplementary QC figures for the TT4 FAPESP case-control snRNA-seq pipeline.

  1. UMAP before (PCA) vs after (scVI) batch integration - coloured by sample_id
     (caso_vs_controle_all, todas as 33 amostras)
  2. UMAP doublet score before/after filtering - painel geral (todas as amostras)
     + um painel para cada subgrupo (Caso G1, Caso G2, Controle G1, Controle G2)
  3. UMAP source_type (caso vs controle) - sem PMI
  4. UMAP QC metric: n_genes_by_counts

All figures saved as hybrid PDF (vectorial axes, rasterised scatter) + 600dpi PNG.

Output
------
    figures/qc_before_after/umap_integration_before_after_hybrid.pdf / .png
    figures/qc_before_after/umap_doublet_score_by_group_hybrid.pdf / .png
    figures/qc_before_after/umap_source_type_hybrid.pdf / .png
    figures/qc_before_after/umap_ngenes_hybrid.pdf / .png

Run
---
    conda activate scrna
    screen -S step11
    nohup python 11_umap_figures.py > logs/11_umap_figures.log 2>&1 &
    tail -f logs/11_umap_figures.log
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OUT_DIR, FIG_DIR, TABLE_DIR, CHECKPOINT_DIR,
    SAMPLES, SOURCE_ORDER, SOURCE_LABELS, SEED,
)

sc.settings.verbosity = 1
np.random.seed(SEED)

INTEGRATED_CK = CHECKPOINT_DIR / "adata_integrated.h5ad"
ANNOTATED_CK  = OUT_DIR / "adata_annotated.h5ad"

QC_FIG_DIR = FIG_DIR / "qc_before_after"
QC_FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI_SAVE = 600

SOURCE_COLORS = {"caso": "#DD8452", "controle": "#4C72B0"}

GROUPS = [
    ("Todas as amostras",        None,          None),
    ("Caso G1 (doenca <20a)",    "caso",        "G1"),
    ("Caso G2 (doenca >=20a)",   "caso",        "G2"),
    ("Controle G1",              "controle",    "G1"),
    ("Controle G2",              "controle",    "G2"),
]

def save_hybrid(fig, path_stem):
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  Saved: {path_stem}.pdf / .png")

def rasterise_axes(fig):
    for ax in fig.get_axes():
        for coll in ax.collections:
            coll.set_rasterized(True)

def group_mask(obs, source_type, disease_group):
    mask = pd.Series(True, index=obs.index)
    if source_type is not None:
        mask &= (obs["source_type"] == source_type)
    if disease_group is not None:
        mask &= (obs["disease_group"] == disease_group)
    return mask

# =============================================================================
print("=" * 60)
print("11_umap_figures.py")
print("=" * 60)

print("\nLoading integrated checkpoint...")
adata_int = sc.read_h5ad(str(INTEGRATED_CK))
for col in ["source_type", "disease_group"]:
    adata_int.obs[col] = adata_int.obs[col].astype(str)
print(f"  {adata_int.n_obs:,} nuclei")

print("Loading annotated checkpoint...")
adata = sc.read_h5ad(str(ANNOTATED_CK))
for col in ["source_type", "disease_group"]:
    adata.obs[col] = adata.obs[col].astype(str)
print(f"  {adata.n_obs:,} nuclei")

# -- Recompute PCA UMAP for pre-integration baseline --

print("\nRecomputing PCA (pre-integration baseline)...")
adata_pca = adata_int.copy()
sc.pp.normalize_total(adata_pca, target_sum=1e4)
sc.pp.log1p(adata_pca)
sc.pp.highly_variable_genes(adata_pca, n_top_genes=3000,
                             flavor="seurat", batch_key="sample_id")
sc.pp.pca(adata_pca, mask_var="highly_variable", n_comps=50)
sc.pp.neighbors(adata_pca, use_rep="X_pca")
sc.tl.umap(adata_pca, random_state=SEED)
adata_int.obsm["X_umap_pca"] = adata_pca.obsm["X_umap"].copy()
del adata_pca
print("  X_umap_pca ready.")

# =============================================================================
# Figure 1 - Before (PCA) vs After (scVI) integration - caso_vs_controle_all
# =============================================================================

print("\n[1/4] Integration before/after UMAP (caso_vs_controle_all)...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax, umap_key, title in zip(
    axes,
    ["X_umap_pca", "X_umap"],
    ["Before integration (PCA)", "After integration (scVI)"],
):
    coords = adata_int.obsm[umap_key]
    cats   = adata_int.obs["sample_id"].astype("category")
    n_cats = len(cats.cat.categories)
    cmap   = plt.get_cmap("tab20", n_cats)
    colors = [cmap(cats.cat.codes[i] % n_cats) for i in range(len(cats))]

    ax.scatter(coords[:, 0], coords[:, 1],
               c=colors, s=0.3, alpha=0.4, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

    handles = [
        Patch(color=cmap(i % n_cats), label=cat)
        for i, cat in enumerate(cats.cat.categories)
    ]
    ax.legend(handles=handles, title="Sample", fontsize=5,
              title_fontsize=6, ncol=3,
              bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=False)

n_caso     = sum(1 for m in SAMPLES.values() if m["source_type"] == "caso")
n_controle = sum(1 for m in SAMPLES.values() if m["source_type"] == "controle")
fig.suptitle(
    f"Batch integration - caso_vs_controle_all - {len(SAMPLES)} amostras "
    f"({n_caso} caso + {n_controle} controle)",
    fontsize=15, fontweight="bold",
)
plt.tight_layout()
rasterise_axes(fig)
save_hybrid(fig, str(QC_FIG_DIR / "umap_integration_before_after_hybrid"))
plt.close(fig)

# =============================================================================
# Figure 2 - Doublet score before/after, geral + por subgrupo
# =============================================================================

print("\n[2/4] Doublet score before/after (geral + por subgrupo)...")

DOUBLET_DIR   = TABLE_DIR / "doublet_checkpoints"
doublet_files = sorted(DOUBLET_DIR.glob("*.h5ad"))

if not doublet_files:
    print("  WARNING: no doublet checkpoints found - skipping figure 2.")
else:
    print(f"  Loading {len(doublet_files)} pre-QC checkpoints...")
    adatas_raw_list = []
    for f in doublet_files:
        a = sc.read_h5ad(str(f))
        for col in ["source_type", "disease_group"]:
            a.obs[col] = a.obs[col].astype(str)
        adatas_raw_list.append(a)

    adata_raw_concat = ad.concat(adatas_raw_list, join="inner")
    adata_raw_concat.obs_names_make_unique()
    del adatas_raw_list

    n_total    = adata_raw_concat.n_obs
    n_doublets = (adata_raw_concat.obs["doublet_class"] == "doublet").sum()
    print(f"  Pre-QC total: {n_total:,} nuclei | {n_doublets:,} doublets "
          f"({100*n_doublets/n_total:.1f}%)")

    print("  Building pre-QC UMAP (uma vez, reusada para todos os paineis)...")
    sc.pp.normalize_total(adata_raw_concat, target_sum=1e4)
    sc.pp.log1p(adata_raw_concat)
    sc.pp.highly_variable_genes(adata_raw_concat, n_top_genes=2000,
                                 flavor="seurat", batch_key="sample_id")
    sc.pp.pca(adata_raw_concat, mask_var="highly_variable", n_comps=30)
    sc.pp.neighbors(adata_raw_concat, n_neighbors=30)
    sc.tl.umap(adata_raw_concat, random_state=SEED)

    coords_pre  = adata_raw_concat.obsm["X_umap"]
    coords_post = adata.obsm["X_umap"]

    n_groups = len(GROUPS)
    fig2, axes2 = plt.subplots(n_groups, 2, figsize=(14, 5.2 * n_groups))

    for row, (label, src, dgrp) in enumerate(GROUPS):
        mask_pre  = group_mask(adata_raw_concat.obs, src, dgrp).values
        mask_post = group_mask(adata.obs, src, dgrp).values

        n_pre_g  = mask_pre.sum()
        n_post_g = mask_post.sum()
        n_dbl_g  = (adata_raw_concat.obs.loc[mask_pre, "doublet_class"] == "doublet").sum()

        # Before - group highlighted, rest in light grey background
        ax = axes2[row, 0]
        ax.scatter(coords_pre[~mask_pre, 0], coords_pre[~mask_pre, 1],
                   c="#eeeeee", s=0.2, alpha=0.3, linewidths=0, rasterized=True, zorder=1)
        sc_plot = ax.scatter(coords_pre[mask_pre, 0], coords_pre[mask_pre, 1],
                             c=adata_raw_concat.obs.loc[mask_pre, "doublet_score"].values,
                             cmap="RdYlGn_r", vmin=0, vmax=1,
                             s=0.4, alpha=0.6, linewidths=0, rasterized=True, zorder=2)
        title_pre = (f"{label} - antes do filtro\n"
                     f"({n_pre_g:,} nuclei | {n_dbl_g:,} doublets = "
                     f"{100*n_dbl_g/n_pre_g:.1f}%)") if n_pre_g else f"{label} - sem dados"
        ax.set_title(title_pre, fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(sc_plot, ax=ax, label="Doublet score", shrink=0.6)

        # After - group highlighted
        ax = axes2[row, 1]
        ax.scatter(coords_post[~mask_post, 0], coords_post[~mask_post, 1],
                   c="#eeeeee", s=0.2, alpha=0.3, linewidths=0, rasterized=True, zorder=1)
        sc_plot = ax.scatter(coords_post[mask_post, 0], coords_post[mask_post, 1],
                             c=adata.obs.loc[mask_post, "doublet_score"].values,
                             cmap="RdYlGn_r", vmin=0, vmax=1,
                             s=0.4, alpha=0.6, linewidths=0, rasterized=True, zorder=2)
        ax.set_title(f"{label} - depois do filtro\n({n_post_g:,} singlets)",
                     fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(sc_plot, ax=ax, label="Doublet score", shrink=0.6)

    fig2.suptitle("Doublet score - antes e depois do filtro (geral e por subgrupo)",
                  fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    rasterise_axes(fig2)
    save_hybrid(fig2, str(QC_FIG_DIR / "umap_doublet_score_by_group_hybrid"))
    plt.close(fig2)
    del adata_raw_concat

# =============================================================================
# Figure 3 - Source type (caso vs controle) - sem PMI
# =============================================================================

print("\n[3/4] Source type UMAP (sem PMI)...")

coords = adata.obsm["X_umap"]
src    = adata.obs["source_type"].astype(str).values

fig, ax = plt.subplots(figsize=(9, 8))
for s in SOURCE_ORDER:
    mask = src == s
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=SOURCE_COLORS[s], s=0.3, alpha=0.5,
               linewidths=0, rasterized=True,
               label=SOURCE_LABELS[s])
ax.set_title("Source type (caso vs controle)", fontsize=13, fontweight="bold")
ax.axis("off")
handles = [Patch(color=SOURCE_COLORS[s], label=SOURCE_LABELS[s]) for s in SOURCE_ORDER]
ax.legend(handles=handles, fontsize=10, loc="lower left", frameon=False)
plt.tight_layout()
rasterise_axes(fig)
save_hybrid(fig, str(QC_FIG_DIR / "umap_source_type_hybrid"))
plt.close(fig)

# =============================================================================
# Figure 4 - QC metric: n_genes_by_counts
# =============================================================================

print("\n[4/4] n_genes_by_counts UMAP...")

if "n_genes_by_counts" not in adata.obs.columns:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"],
                                percent_top=None, log1p=False, inplace=True)

fig, ax = plt.subplots(figsize=(9, 8))
vals = adata.obs["n_genes_by_counts"].values.astype(float)
sc_plot = ax.scatter(coords[:, 0], coords[:, 1],
                     c=vals, cmap="viridis",
                     s=0.3, alpha=0.5, linewidths=0, rasterized=True)
ax.set_title("Number of genes per nucleus", fontsize=13, fontweight="bold")
ax.axis("off")
plt.colorbar(sc_plot, ax=ax, label="N genes", shrink=0.6)
plt.tight_layout()
rasterise_axes(fig)
save_hybrid(fig, str(QC_FIG_DIR / "umap_ngenes_hybrid"))
plt.close(fig)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Figures saved in: {QC_FIG_DIR}")
print("\n>>> Pipeline TT4 FAPESP - figuras de QC concluidas.")
