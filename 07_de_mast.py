#!/usr/bin/env python3
"""
07_de_mast.py
=============
Step 7 of the TT4 FAPESP case-control snRNA-seq pipeline (MTLE-HS vs autopsy).

Pseudobulk differential expression via MAST (R), run across FOUR comparisons,
each in two moments (Moment 1 = cell_type, Moment 2 = Leiden subcluster):

  PERSPECTIVE I      caso_vs_controle_all  — all casos vs all controles
  PERSPECTIVE I-G1   caso_vs_controle_G1   — caso G1 vs controle G1 only
  PERSPECTIVE I-G2   caso_vs_controle_G2   — caso G2 vs controle G2 only
  PERSPECTIVE II     G1_vs_G2_caso         — caso G1 vs caso G2 (disease duration,
                                              controle samples excluded)

Output
------
    tables/de_MAST/<comparison_tag>/momento1_<test>_vs_<ref>.csv
    tables/de_MAST/<comparison_tag>/momento2_<test>_vs_<ref>.csv

Run
---
    conda activate scrna
    screen -S step07
    nohup python 07_de_mast.py > logs/07_de_mast.log 2>&1 &
    tail -f logs/07_de_mast.log

Note: skip-if-exists logic — safe to re-run after a crash.
"""

import os
import sys
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OUT_DIR, TABLE_DIR,
    LEIDEN_FINAL, SEED
)

sc.settings.verbosity = 1
np.random.seed(SEED)

DE_DIR       = TABLE_DIR / "de_MAST"
ANNOTATED_CK = OUT_DIR   / "adata_annotated.h5ad"

# =============================================================================
# COMPARISON DEFINITIONS
# =============================================================================
# subset_fn: function(adata) -> boolean mask, or None for no subsetting
# group_col: "source_type" | "disease_group"
# group_order: (ref, test)  -- test is what logFC/coef refers to (positive = up in test)

COMPARISONS = [
    dict(
        tag="caso_vs_controle_all",
        title="Perspectiva I — Caso vs Controle (todos os pacientes)",
        subset_fn=None,
        group_col="source_type",
        group_order=("controle", "caso"),
    ),
    dict(
        tag="caso_vs_controle_G1",
        title="Perspectiva I-G1 — Caso G1 vs Controle G1",
        subset_fn=lambda ad: ad.obs["disease_group"] == "G1",
        group_col="source_type",
        group_order=("controle", "caso"),
    ),
    dict(
        tag="caso_vs_controle_G2",
        title="Perspectiva I-G2 — Caso G2 vs Controle G2",
        subset_fn=lambda ad: ad.obs["disease_group"] == "G2",
        group_col="source_type",
        group_order=("controle", "caso"),
    ),
    dict(
        tag="G1_vs_G2_caso",
        title="Perspectiva II — Caso G1 vs Caso G2 (tempo de doenca, apenas MTLE-HS)",
        subset_fn=lambda ad: ad.obs["source_type"] == "caso",
        group_col="disease_group",
        group_order=("G1", "G2"),
    ),
]

# =============================================================================
# FUNCTIONS
# =============================================================================

def make_pseudobulk(adata: sc.AnnData, sample_col: str,
                    cell_type_col: str, group_col: str,
                    count_layer: str = "counts"):
    """
    Aggregate single-nucleus counts into pseudobulk per sample × cell type.
    Only combinations with >= 10 nuclei are retained.
    Returns (counts_df [genes × pseudobulks], meta_df).
    """
    results   = {}
    meta_rows = []

    for sample in adata.obs[sample_col].unique():
        for ctype in adata.obs[cell_type_col].unique():
            mask = (
                (adata.obs[sample_col]    == sample) &
                (adata.obs[cell_type_col] == ctype)
            )
            if mask.sum() < 10:
                continue
            sub = adata[mask]
            X   = sub.layers[count_layer]
            counts_sum = (
                np.array(X.sum(axis=0)).flatten()
                if sp.issparse(X) else X.sum(axis=0)
            )
            key = f"{sample}__{str(ctype).replace(' ', '_')}"
            results[key] = counts_sum
            meta_rows.append({
                "pseudobulk_id": key,
                "sample_id":     sample,
                "cluster":       str(ctype),
                "group":         sub.obs[group_col].iloc[0],
                "source_type":   sub.obs["source_type"].iloc[0],
                "disease_group": sub.obs["disease_group"].iloc[0],
                "pmi_h":         sub.obs["pmi_h"].iloc[0],
                "n_cells":       int(mask.sum()),
            })

    counts_df = pd.DataFrame(results, index=adata.var_names)
    meta_df   = pd.DataFrame(meta_rows).set_index("pseudobulk_id")
    return counts_df, meta_df


def run_mast_de(counts_ct: pd.DataFrame, meta_ct: pd.DataFrame,
                cluster_name: str, group_ref: str, group_test: str):
    """
    Run MAST hurdle model for one cluster — group_ref vs group_test.
    Returns a DataFrame with columns:
        gene, pvalue, logFC, ci_hi, ci_lo, fdr, cluster, comparison
    Returns None if the comparison cannot be run (too few samples).
    """
    if len(meta_ct) < 4:
        return None
    groups_present = meta_ct["group"].unique()
    if group_ref not in groups_present or group_test not in groups_present:
        return None

    # Remove zero-sum genes
    gene_mask      = counts_ct.sum(axis=1) > 0
    counts_ct_filt = counts_ct[gene_mask]

    # log-CPM normalisation
    lib_sizes = counts_ct_filt.sum(axis=0)
    log_cpm   = np.log1p(counts_ct_filt.div(lib_sizes, axis=1) * 1e6)

    tmpdir      = tempfile.mkdtemp()
    expr_path   = os.path.join(tmpdir, "expr.csv")
    meta_path   = os.path.join(tmpdir, "meta.csv")
    result_path = os.path.join(tmpdir, "result.csv")

    log_cpm.to_csv(expr_path)
    meta_ct.reset_index().to_csv(meta_path, index=False)

    ro.r(f"""
        library(MAST); library(data.table)
        expr_mat <- as.matrix(read.csv("{expr_path}", row.names=1))
        cdata    <- read.csv("{meta_path}")
        rownames(cdata) <- cdata$pseudobulk_id
        fdata <- data.frame(primerid=rownames(expr_mat), row.names=rownames(expr_mat))
        sca   <- MAST::FromMatrix(exprsArray=expr_mat, cData=cdata, fData=fdata)
        colData(sca)$cngeneson <- scale(colMeans(assay(sca) > 0))
        colData(sca)$group     <- factor(colData(sca)$group,
                                         levels=c("{group_ref}", "{group_test}"))
        zlm_fit       <- MAST::zlm(~ group + cngeneson, sca, method="glm", ebayes=TRUE)
        contrast_name <- paste0("group", "{group_test}")
        summaryDt     <- MAST::summary(zlm_fit, doLRT=contrast_name)$datatable
        fcHurdle <- merge(
            summaryDt[contrast==contrast_name & component=="H",
                      .(primerid, `Pr(>Chisq)`)],
            summaryDt[contrast==contrast_name & component=="logFC",
                      .(primerid, coef, ci.hi, ci.lo)],
            by="primerid")
        fcHurdle[, fdr := p.adjust(`Pr(>Chisq)`, method="BH")]
        write.csv(fcHurdle[order(fdr)], "{result_path}", row.names=FALSE)
    """)

    result_df = pd.read_csv(result_path)
    result_df.columns = ["gene", "pvalue", "logFC", "ci_hi", "ci_lo", "fdr"]
    result_df["cluster"]    = cluster_name
    result_df["comparison"] = f"{group_test}_vs_{group_ref}"

    for f in [expr_path, meta_path, result_path]:
        os.remove(f)
    os.rmdir(tmpdir)

    n_sig = (result_df["fdr"] < 0.05).sum()
    print(f"      {str(cluster_name):<28}: {len(result_df):>6,} genes tested "
          f"| {n_sig:>4} DE (FDR<0.05)")
    return result_df


def run_de_moment(adata: sc.AnnData, cluster_col: str, moment_label: str,
                  group_col: str, group_ref: str, group_test: str,
                  out_dir, extra_col: str = None):
    """
    Run DE group_ref vs group_test for a given cluster column, within out_dir.
    Saves one CSV per moment; skips if already exists.
    """
    pair_label = f"{group_test}_vs_{group_ref}"
    print(f"\n{'-'*60}")
    print(f"{moment_label} — cluster column: '{cluster_col}' | {pair_label}")
    print(f"{'-'*60}\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{moment_label}_{pair_label}.csv"

    if out_path.exists():
        print(f"  [SKIP] {pair_label} — already saved: {out_path.name}")
        df = pd.read_csv(out_path)
        n_sig = (df["fdr"] < 0.05).sum()
        print(f"  Loaded: {len(df):,} tests | {n_sig:,} DE (FDR<0.05)")
        return df

    parent_map = None
    if extra_col and extra_col in adata.obs.columns:
        parent_map = (
            adata.obs[[cluster_col, extra_col]]
            .drop_duplicates()
            .groupby(cluster_col)[extra_col].first()
        )

    n_ref  = (adata.obs[group_col] == group_ref).sum()
    n_test = (adata.obs[group_col] == group_test).sum()
    print(f"  Nuclei: {group_ref}={n_ref:,} | {group_test}={n_test:,}")

    counts_pb, meta_pb = make_pseudobulk(adata, "sample_id", cluster_col, group_col)

    if parent_map is not None:
        meta_pb["cell_type_parent"] = meta_pb["cluster"].map(parent_map)

    pair_results = []
    clusters_sorted = sorted(
        meta_pb["cluster"].unique(),
        key=lambda x: int(x) if str(x).isdigit() else x
    )

    for cluster in clusters_sorted:
        meta_ct   = meta_pb[meta_pb["cluster"] == cluster]
        counts_ct = counts_pb[meta_ct.index]

        # Minimum expression filter: expressed in >= 20% of pseudobulk samples
        min_s          = max(2, int(counts_ct.shape[1] * 0.20))
        gene_mask      = (counts_ct > 0).sum(axis=1) >= min_s
        counts_ct_filt = counts_ct[gene_mask]

        res = run_mast_de(counts_ct_filt, meta_ct, cluster, group_ref, group_test)
        if res is not None:
            if parent_map is not None and len(meta_ct) > 0:
                res["cell_type_parent"] = meta_ct["cell_type_parent"].iloc[0]
            pair_results.append(res)

    if not pair_results:
        print(f"  WARNING: no results for {pair_label}")
        return None

    result_df = pd.concat(pair_results, ignore_index=True)
    result_df.to_csv(out_path, index=False)

    n_sig = (result_df["fdr"] < 0.05).sum()
    print(f"\n{moment_label} — DONE")
    print(f"  Total gene tests : {len(result_df):,}")
    print(f"  DE (FDR < 0.05)  : {n_sig:,}")
    print(f"  Saved            : {out_path}")

    sig = result_df[result_df["fdr"] < 0.05]
    if len(sig) > 0:
        print(f"\n  DE per cluster:")
        print(sig.groupby("cluster").size().rename("n_DE").to_string())

    return result_df


def run_comparison(adata_full: sc.AnnData, cfg: dict):
    """Run Moment 1 (cell_type) and Moment 2 (Leiden subcluster) for one comparison."""
    print(f"\n{'='*60}")
    print(cfg["title"])
    print(f"{'='*60}")

    if cfg["subset_fn"] is not None:
        mask = cfg["subset_fn"](adata_full)
        adata_sub = adata_full[mask].copy()
    else:
        adata_sub = adata_full

    group_col   = cfg["group_col"]
    ref, test   = cfg["group_order"]
    out_dir     = DE_DIR / cfg["tag"]

    n_ref  = (adata_sub.obs[group_col] == ref).sum()
    n_test = (adata_sub.obs[group_col] == test).sum()
    n_samples_ref  = adata_sub.obs.loc[adata_sub.obs[group_col] == ref,  "sample_id"].nunique()
    n_samples_test = adata_sub.obs.loc[adata_sub.obs[group_col] == test, "sample_id"].nunique()
    print(f"  {ref}: {n_ref:,} nuclei ({n_samples_ref} amostras)")
    print(f"  {test}: {n_test:,} nuclei ({n_samples_test} amostras)")

    df_m1 = run_de_moment(
        adata_sub, cluster_col="cell_type", moment_label="momento1",
        group_col=group_col, group_ref=ref, group_test=test,
        out_dir=out_dir, extra_col=None,
    )
    df_m2 = run_de_moment(
        adata_sub, cluster_col=LEIDEN_FINAL, moment_label="momento2",
        group_col=group_col, group_ref=ref, group_test=test,
        out_dir=out_dir, extra_col="cell_type",
    )

    if cfg["subset_fn"] is not None:
        del adata_sub

    return df_m1, df_m2


# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("07_de_mast.py")
print("=" * 60)
print(f"\nLEIDEN_FINAL : {LEIDEN_FINAL}")
print(f"Comparisons  : {[c['tag'] for c in COMPARISONS]}")

# ── Load R packages ────────────────────────────────────────────────────────

print("\nLoading R packages (MAST, data.table)...")
importr("MAST")
importr("data.table")
print("R packages loaded.")

# ── Load annotated checkpoint ──────────────────────────────────────────────

print(f"\nLoading annotated checkpoint: {ANNOTATED_CK}")
adata = sc.read_h5ad(str(ANNOTATED_CK))
for col in ["source_type", "disease_group"]:
    adata.obs[col] = adata.obs[col].astype(str)

print(f"  {adata.n_obs:,} nuclei × {adata.n_vars:,} genes")
print(f"  Cell types  : {sorted(adata.obs['cell_type'].unique())}")
print(f"  Leiden col  : {LEIDEN_FINAL} ({adata.obs[LEIDEN_FINAL].nunique()} clusters)")
print(f"  Caso        : {(adata.obs['source_type']=='caso').sum():,} nuclei")
print(f"  Controle    : {(adata.obs['source_type']=='controle').sum():,} nuclei")
for g in ["G1", "G2"]:
    print(f"    {g}: {(adata.obs['disease_group']==g).sum():,} nuclei")

# ── Run all comparisons ───────────────────────────────────────────────────

all_results = {}
for cfg in COMPARISONS:
    df_m1, df_m2 = run_comparison(adata, cfg)
    all_results[cfg["tag"]] = {"momento1": df_m1, "momento2": df_m2}

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY — DE genes (FDR < 0.05) per comparison")
print("=" * 60)
for tag, res in all_results.items():
    for moment, df in res.items():
        if df is None:
            print(f"  {tag:<24} {moment}: sem resultados")
            continue
        n_sig = (df["fdr"] < 0.05).sum()
        print(f"  {tag:<24} {moment}: {n_sig:,} genes DE / {len(df):,} testados")

print(f"\n✓ Tabelas salvas em: {DE_DIR}/<comparison_tag>/")
print("\n>>> Next step: run 08_de_analysis.py")
