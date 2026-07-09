#!/usr/bin/env python3
"""
03_scvi_integration.py
======================
Step 3 of the TT4 FAPESP case-control snRNA-seq pipeline (MTLE-HS vs autopsy).

  - Loads approved QC checkpoints from checkpoints/adatas_qc/
  - Concatenates approved samples (inner join)
  - Selects 3,000 HVGs (seurat flavor, per batch)
  - Trains scVI model (batch=sample_id, categorical covariate=source_type,
    continuous covariate=pmi_h -- NOTE: pmi_h is NaN for caso samples, see below)
  - Computes UMAP and Leiden clustering at all resolutions (0.2 → 2.0)
  - Saves QC UMAPs and resolution comparison grid for visual inspection

Output
------
    checkpoints/adata_integrated.h5ad   — full object with latent + all Leiden
    scvi_model/                          — trained scVI model
    figures/scvi_training_curve.png
    figures/umap_qc_metadata.png
    figures/umap_leiden_resolutions.png
    tables/clustering_resolutions.csv    — used by 04a_clustree.py

Run
---
    conda activate scrna
    screen -S step03
    nohup python 03_scvi_integration.py > logs/03_scvi_integration.log 2>&1 &
    tail -f logs/03_scvi_integration.log

Note: this is the longest step (~1–3h depending on server load).
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scvi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OUT_DIR, FIG_DIR, TABLE_DIR, CHECKPOINT_DIR, MODEL_DIR,
    SAMPLES, SOURCE_ORDER, SOURCE_LABELS,
    SCVI_PARAMS, LEIDEN_RESOLUTIONS, N_NEIGHBORS, SEED,
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sc.settings.verbosity = 2
sc.settings.figdir = str(FIG_DIR)
np.random.seed(SEED)
scvi.settings.seed = SEED

QC_ADATA_DIR  = CHECKPOINT_DIR / "adatas_qc"
INTEGRATED_CK = CHECKPOINT_DIR / "adata_integrated.h5ad"

# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("03_scvi_integration.py")
print("=" * 60)

# ── 1. Load approved QC checkpoints ───────────────────────────────────────

print("\n[1/6] Loading approved QC checkpoints...\n")

approved_path = TABLE_DIR / "samples_approved.txt"
with open(approved_path) as f:
    approved_ids = [line.strip() for line in f if line.strip()]

print(f"  Approved samples: {len(approved_ids)}")

adatas_approved = {}
for sample_id in approved_ids:
    chk = QC_ADATA_DIR / f"{sample_id}.h5ad"
    if not chk.exists():
        raise FileNotFoundError(
            f"QC checkpoint not found for {sample_id}. Run 02_qc.py first."
        )
    adata = sc.read_h5ad(str(chk))
    src   = SAMPLES[sample_id]["source_type"]
    grp   = SAMPLES[sample_id]["disease_group"]
    print(f"  Loaded: {sample_id} [{src}/{grp}]"
          f"  ({adata.n_obs:,} nuclei)")
    adatas_approved[sample_id] = adata

# ── 2. Concatenation ──────────────────────────────────────────────────────

print("\n[2/6] Concatenating approved samples...\n")

adata = ad.concat(
    list(adatas_approved.values()),
    join="inner",
    merge="same",
    label="sample_id",
    keys=list(adatas_approved.keys()),
)
adata.obs_names_make_unique()

# Ordered categorical source_type
adata.obs["source_type"] = pd.Categorical(
    adata.obs["source_type"], categories=SOURCE_ORDER, ordered=True
)
adata.obs["disease_group"] = pd.Categorical(
    adata.obs["disease_group"], categories=["G1", "G2"], ordered=True
)

# pmi_h eh NaN para casos (cirurgia, sem intervalo post-mortem).
# scVI nao aceita NaN em continuous_covariate_keys -> imputamos 0 e criamos
# uma flag binaria "has_pmi" para o modelo distinguir "PMI real" de "sem PMI (caso)".
# Isso preserva o sinal do PMI nos controles sem quebrar o treino nem inventar
# um valor plausivel para os casos.
adata.obs["has_pmi"] = (~adata.obs["pmi_h"].isna()).astype(int)
adata.obs["pmi_h_imputed"] = adata.obs["pmi_h"].fillna(0.0).astype(float)

# Preserve raw counts in a layer for scVI
adata.layers["counts"] = adata.X.copy()

n_caso     = (adata.obs["source_type"] == "caso").sum()
n_controle = (adata.obs["source_type"] == "controle").sum()

print(f"  Total nuclei : {adata.n_obs:,}")
print(f"    Caso       : {n_caso:,}")
print(f"    Controle   : {n_controle:,}")
print(f"  Genes        : {adata.n_vars:,}")
print(f"  Samples      : {adata.obs['sample_id'].nunique()}")
print(adata)

del adatas_approved   # free memory

# ── 3. HVG selection ──────────────────────────────────────────────────────

print("\n[3/6] Selecting highly variable genes (HVGs)...\n")

# Normalize a temporary copy for HVG selection
# Note: flavor='seurat' (log-normalized) avoids the integer-count requirement
# of 'seurat_v3'; safe to use after log1p.
adata_norm = adata.copy()
sc.pp.normalize_total(adata_norm, target_sum=1e4)
sc.pp.log1p(adata_norm)

sc.pp.highly_variable_genes(
    adata_norm,
    n_top_genes=SCVI_PARAMS["n_top_genes"],
    flavor="seurat",
    batch_key="sample_id",
    subset=False,
)

adata.var["highly_variable"] = adata_norm.var["highly_variable"]
n_hvg = adata.var["highly_variable"].sum()
print(f"  HVGs selected: {n_hvg:,} of {adata.n_vars:,} genes")

sc.pl.highly_variable_genes(adata_norm, save="_hvg.png", show=False)
print(f"  HVG plot saved: {FIG_DIR}/show_hvg.png")

del adata_norm

# ── 4. scVI training ──────────────────────────────────────────────────────

print("\n[4/6] Training scVI model...\n")

adata_hvg = adata[:, adata.var["highly_variable"]].copy()

# Batch correction: sample_id
# Categorical covariates: source_type (caso vs controle) + has_pmi (caso sempre 0)
# Continuous covariate: pmi_h_imputed (0 para casos -- neutralizado por has_pmi)
scvi.model.SCVI.setup_anndata(
    adata_hvg,
    layer="counts",
    batch_key="sample_id",
    categorical_covariate_keys=["source_type", "has_pmi"],
    continuous_covariate_keys=["pmi_h_imputed"],
)

model = scvi.model.SCVI(
    adata_hvg,
    n_layers=SCVI_PARAMS["n_layers"],
    n_latent=SCVI_PARAMS["n_latent"],
    gene_likelihood=SCVI_PARAMS["gene_likelihood"],
)
print(model)

model.train(
    max_epochs=SCVI_PARAMS["max_epochs"],
    early_stopping=SCVI_PARAMS["early_stopping"],
    early_stopping_patience=SCVI_PARAMS["early_stopping_patience"],
    plan_kwargs={"lr": SCVI_PARAMS["lr"]},
    accelerator="cpu",
)

model.save(str(MODEL_DIR), overwrite=True)
print(f"\n✓ scVI model saved: {MODEL_DIR}")

# Training curve
fig, ax = plt.subplots(figsize=(7, 4))
model.history["elbo_train"].plot(ax=ax)
ax.set_title("scVI training curve (ELBO)")
ax.set_xlabel("Epoch")
ax.set_ylabel("ELBO")
plt.tight_layout()
plt.savefig(FIG_DIR / "scvi_training_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Training curve saved: {FIG_DIR / 'scvi_training_curve.png'}")

# Transfer latent representation to full adata
adata_hvg.obsm["X_scVI"] = model.get_latent_representation()
adata.obsm["X_scVI"]     = adata_hvg.obsm["X_scVI"]
print(f"✓ Latent space: {adata.obsm['X_scVI'].shape}")

del adata_hvg

# ── 5. Neighbours + UMAP + Leiden at all resolutions ─────────────────────

print("\n[5/6] Computing neighbours, UMAP and Leiden clusters...\n")

sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=N_NEIGHBORS, n_pcs=None)
sc.tl.umap(adata, random_state=SEED)
print("  UMAP computed.")

for res in LEIDEN_RESOLUTIONS:
    key = f"leiden_{res}"
    sc.tl.leiden(adata, resolution=res, key_added=key, random_state=SEED)
    print(f"  Leiden res={res}: {adata.obs[key].nunique()} clusters")

# ── 6. Figures ────────────────────────────────────────────────────────────

print("\n[6/6] Saving inspection figures...\n")

# Recompute QC metrics for UMAP coloring (not stored after concat)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)

# QC metadata UMAP grid
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
plot_configs = [
    ("leiden_0.6",        "Leiden clusters (res=0.6)",       None),
    ("source_type",       "Source type (caso / controle)",  None),
    ("disease_group",     "Disease group (G1 / G2)",         None),
    ("pmi_h",             "PMI (hours, controles apenas)",  "viridis"),
    ("doublet_score",     "Doublet score",                  "Reds"),
    ("n_genes_by_counts", "Number of genes per nucleus",    "Blues"),
]
for ax, (color, title, cmap) in zip(axes.flatten(), plot_configs):
    kwargs = dict(color=color, title=title, ax=ax, show=False, frameon=False)
    if cmap:
        kwargs["color_map"] = cmap
    sc.pl.umap(adata, **kwargs)

plt.suptitle("UMAP — Quality control and metadata", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "umap_qc_metadata.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: umap_qc_metadata.png")

# Leiden resolution comparison grid (resolutions 1.0 → 2.0)
high_res = [r for r in LEIDEN_RESOLUTIONS if r >= 1.0]
n_cols   = 3
n_rows   = int(np.ceil(len(high_res) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
for ax, res in zip(axes_flat, high_res):
    key = f"leiden_{res}"
    sc.pl.umap(
        adata,
        color=key,
        title=f"Leiden res={res} ({adata.obs[key].nunique()} clusters)",
        legend_loc="on data",
        legend_fontsize=11,
        legend_fontoutline=3,
        ax=ax, show=False, frameon=False,
    )
# Hide unused axes
for ax in axes_flat[len(high_res):]:
    ax.set_visible(False)
plt.suptitle("Leiden resolution comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "umap_leiden_resolutions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: umap_leiden_resolutions.png")

# ── Save clustering table for clustree (04a) ──────────────────────────────

clustering_cols = [f"leiden_{r}" for r in LEIDEN_RESOLUTIONS]
clustering_df   = adata.obs[clustering_cols].copy()
clustering_csv  = TABLE_DIR / "clustering_resolutions.csv"
clustering_df.to_csv(clustering_csv)
print(f"\n✓ Clustering table saved: {clustering_csv}")

# ── Save checkpoint ───────────────────────────────────────────────────────

print(f"\nSaving integrated checkpoint ({adata.n_obs:,} nuclei × {adata.n_vars:,} genes)...")
adata.write_h5ad(str(INTEGRATED_CK))

print(f"\n✓ Checkpoint saved: {INTEGRATED_CK}")
print(f"✓ Figures saved in: {FIG_DIR}")
print("\n>>> Next step: run 04a_clustree.py")
