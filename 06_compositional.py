#!/usr/bin/env python3
"""
06_compositional.py
===================
Step 6 of the TT4 FAPESP case-control snRNA-seq pipeline (MTLE-HS vs autopsy).

Two independent compositional analyses (same statistical machinery, applied twice):

  PERSPECTIVE I  — Caso vs Controle (all patients pooled vs all autopsy controls)
                   "Does having MTLE-HS change hippocampal cell type composition
                   at all, relative to non-diseased tissue?"

  PERSPECTIVE II — Group 1 vs Group 2, CASO SAMPLES ONLY (disease duration <20y
                   vs >=20y within the same disease)
                   "Does longer disease duration further reshape cell type
                   composition among patients who already have MTLE-HS?"

For each perspective:
  1. Counts and proportions per sample
  2. Mann-Whitney U per cell type (BH correction)
  3. Propeller/speckle via rpy2
  4. Arcsin-sqrt + Mann-Whitney fallback (Python, always runs)
  5. Figures: stacked bar + delta_p + heatmap / boxplots / propeller methods

Output
------
    tables/perspective1_caso_vs_controle/*.csv
    tables/perspective2_G1_vs_G2/*.csv
    figures/perspective1_caso_vs_controle/*.png / .pdf
    figures/perspective2_G1_vs_G2/*.png / .pdf

Run
---
    conda activate scrna
    screen -S step06
    nohup python 06_compositional.py > logs/06_compositional.log 2>&1 &
    tail -f logs/06_compositional.log
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scanpy as sc
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OUT_DIR, FIG_DIR, TABLE_DIR,
    SAMPLES, SOURCE_ORDER, SOURCE_LABELS, SEED,
)

np.random.seed(SEED)

ANNOTATED_CK = OUT_DIR / "adata_annotated.h5ad"

CLUSTER_COL = "cell_type"
SAMPLE_COL  = "sample_id"

SOURCE_COLORS = {
    "controle": "#4C72B0",
    "caso":     "#DD8452",
}
DISEASE_GROUP_COLORS = {
    "G1": "#2ca02c",
    "G2": "#9467bd",
}
DISEASE_GROUP_LABELS = {
    "G1": "Caso G1 (doenca <20a)",
    "G2": "Caso G2 (doenca >=20a)",
}

CELLTYPE_COLORS = {
    "Astrocyte":         "#1f77b4",
    "Endothelial Cell":  "#ff7f0e",
    "Ependymal Cell":    "#d62728",
    "Microglia":         "#9467bd",
    "T Cell":            "#17becf",
    "Excitatory Neuron": "#8c564b",
    "Inhibitory Neuron": "#c49c94",
    "Neuron":            "#e377c2",
    "OPC":               "#bcbd22",
    "Oligodendrocyte":   "#d8c36a",
}

# =============================================================================
# HELPERS
# =============================================================================

def py2r(df):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.py2rpy(df)

def r2py(rdf):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.rpy2py(rdf)


def compute_counts_props(adata_sub, group_col):
    """Counts and proportions per sample, joined with the grouping variable."""
    counts = (
        adata_sub.obs.groupby([SAMPLE_COL, CLUSTER_COL])
        .size().reset_index(name="n")
    )
    counts_wide = (
        counts.pivot(index=SAMPLE_COL, columns=CLUSTER_COL, values="n")
        .fillna(0).astype(int)
    )
    meta = (
        adata_sub.obs[[SAMPLE_COL, group_col]]
        .drop_duplicates(SAMPLE_COL)
        .set_index(SAMPLE_COL)
    )
    counts_wide  = counts_wide.join(meta)
    cluster_cols = [c for c in counts_wide.columns if c != group_col]
    props        = counts_wide[cluster_cols].div(counts_wide[cluster_cols].sum(axis=1), axis=0)
    props_meta   = props.join(meta)
    return counts_wide, props, props_meta, cluster_cols, meta


def mann_whitney_by_group(props_meta, cluster_cols, group_col, group_order):
    """Mann-Whitney U per cell type between the two levels of group_order."""
    g_ref, g_test = group_order
    rows = []
    for cl in cluster_cols:
        v_ref  = props_meta[props_meta[group_col] == g_ref][cl].dropna().values
        v_test = props_meta[props_meta[group_col] == g_test][cl].dropna().values
        if len(v_ref) < 2 or len(v_test) < 2:
            continue
        stat, pval = stats.mannwhitneyu(v_ref, v_test, alternative="two-sided")
        delta_p = np.median(v_test) - np.median(v_ref)
        rows.append({
            "cell_type":      cl,
            f"median_{g_ref}":  round(np.median(v_ref), 4),
            f"median_{g_test}": round(np.median(v_test), 4),
            "delta_p":        round(delta_p, 4),  # test - ref
            "MW_stat":        round(stat, 3),
            "pval":           pval,
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    _, df["FDR"], _, _ = multipletests(df["pval"], method="fdr_bh")
    df["sig"] = df["FDR"].apply(
        lambda x: "***" if x < 0.001 else ("**" if x < 0.01 else ("*" if x < 0.05 else "ns"))
    )
    return df.sort_values("FDR")


def run_propeller(adata_sub, group_col, group_order, table_dir, tag):
    """Propeller/speckle via rpy2, plus arcsin+MW fallback. Returns dict of results."""
    all_results = {}

    ro.r("""
        suppressMessages({
            if (!requireNamespace("speckle", quietly=TRUE)) {
                if (!requireNamespace("BiocManager", quietly=TRUE))
                    install.packages("BiocManager", repos="https://cloud.r-project.org")
                BiocManager::install("speckle", ask=FALSE, update=FALSE)
            }
            library(speckle)
            library(limma)
        })
    """)

    obs_input = (
        adata_sub.obs[[SAMPLE_COL, CLUSTER_COL, group_col]]
        .copy().reset_index(drop=True)
    )
    obs_input[group_col] = obs_input[group_col].astype(str)

    ro.globalenv["obs_df"]      = py2r(obs_input)
    ro.globalenv["sample_col"]  = SAMPLE_COL
    ro.globalenv["cluster_col"] = CLUSTER_COL
    ro.globalenv["group_col"]   = group_col
    ro.globalenv["lvl1"]        = group_order[0]
    ro.globalenv["lvl2"]        = group_order[1]

    ro.r("""
        obs_df$grp <- factor(obs_df[[group_col]], levels=c(lvl1, lvl2))
        res_grp <- propeller(
            clusters  = obs_df[[cluster_col]],
            sample    = obs_df[[sample_col]],
            group     = obs_df$grp,
            transform = "asin"
        )
        res_grp$cell_type <- rownames(res_grp)
    """)
    res_grp = r2py(ro.globalenv["res_grp"]).reset_index(drop=True)
    res_grp.to_csv(table_dir / f"propeller_{tag}.csv", index=False)
    all_results[f"Propeller — {group_order[0]} vs {group_order[1]}"] = {
        "df": res_grp, "pval_col": "FDR", "fc_col": None
    }
    print(f"\n  Propeller ({tag}):")
    print(res_grp.to_string(index=False))

    return all_results, res_grp


def arcsin_mw_fallback(props_meta, cluster_cols, group_col, group_order, table_dir, tag):
    g_ref, g_test = group_order
    props_asin = np.arcsin(np.sqrt(props_meta[cluster_cols].clip(0, 1))).join(
        props_meta[[group_col]]
    )
    rows = []
    for cl in cluster_cols:
        v1 = props_asin[props_asin[group_col] == g_ref][cl].dropna().values
        v2 = props_asin[props_asin[group_col] == g_test][cl].dropna().values
        if len(v1) < 2 or len(v2) < 2:
            continue
        stat, pval = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        rows.append({"cell_type": cl, "MW_stat": round(stat, 3), "pval": pval})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    _, df["FDR"], _, _ = multipletests(df["pval"], method="fdr_bh")
    df["sig"] = df["FDR"].apply(
        lambda x: "***" if x < 0.001 else ("**" if x < 0.01 else ("*" if x < 0.05 else "ns"))
    )
    df = df.sort_values("FDR")
    df.to_csv(table_dir / f"propeller_fallback_arcsin_mw_{tag}.csv", index=False)
    return df


def make_figures(props_meta, mw_df, cluster_cols, group_col, group_order,
                  group_labels, group_colors, all_results, fig_dir, tag, title_prefix):
    """Generate the 3 figures (overview, boxplots, propeller methods) for one perspective."""

    n_per_group = props_meta[group_col].value_counts().to_dict()
    ct_colors   = [CELLTYPE_COLORS.get(c, "#888888") for c in cluster_cols]

    # ── Figure 1: Overview ──────────────────────────────────────────────
    fig1 = plt.figure(figsize=(18, 20))
    gs_main = gridspec.GridSpec(
        3, 1, figure=fig1,
        height_ratios=[2.8, 2.2, 2.0],
        hspace=0.9,
        left=0.08, right=0.83, top=0.88, bottom=0.04,
    )

    ax0 = fig1.add_subplot(gs_main[0])
    sorted_meta = props_meta.sort_values(group_col)
    sorted_meta[cluster_cols].plot(
        kind="bar", stacked=True, ax=ax0,
        color=ct_colors, width=0.88, legend=False
    )
    ax0.set_xticks([])
    ax0.set_xlabel("")
    ax0.set_ylabel("Proportion of nuclei", fontsize=11)
    ax0.set_title(f"Cell type composition per sample (grouped by {group_col})",
                  fontsize=12, pad=55)
    ax0.yaxis.grid(True, alpha=0.3)
    ax0.xaxis.grid(False)

    prev_g, label_positions = None, {}
    for i, (_, row) in enumerate(sorted_meta.iterrows()):
        g = row[group_col]
        if g != prev_g:
            if prev_g is not None:
                ax0.axvline(x=i - 0.5, color="black", linewidth=1.5,
                            linestyle="--", alpha=0.6)
            label_positions[g] = i
            prev_g = g

    starts = list(label_positions.values()) + [len(sorted_meta)]
    for j, (g, start) in enumerate(label_positions.items()):
        end = starts[j + 1]
        mid = (start + end) / 2
        ax0.annotate(
            f"{group_labels[g]}\n(n={n_per_group.get(g, '?')})",
            xy=(mid, 1.01), xycoords=("data", "axes fraction"),
            ha="center", fontsize=9, color=group_colors[g], fontweight="bold",
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CELLTYPE_COLORS.get(c, "#888888"))
        for c in cluster_cols
    ]
    ax0.legend(handles, cluster_cols, title="Cell type",
               bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    # Delta-p bar
    ax1 = fig1.add_subplot(gs_main[1])
    mw_sorted  = mw_df.sort_values("delta_p").reset_index(drop=True)
    bar_colors = ["#d73027" if d < 0 else "#4575b4" for d in mw_sorted["delta_p"]]
    ax1.barh(range(len(mw_sorted)), mw_sorted["delta_p"],
             color=bar_colors, edgecolor="white", linewidth=0.5, height=0.65)
    ax1.set_yticks(range(len(mw_sorted)))
    ax1.set_yticklabels(mw_sorted["cell_type"], fontsize=10)
    ax1.axvline(0, color="black", linewidth=0.8)
    for i, row in mw_sorted.iterrows():
        label = row["sig"] if row["sig"] != "ns" else ""
        if label:
            offset = 0.002 if row["delta_p"] >= 0 else -0.002
            ha     = "left"  if row["delta_p"] >= 0 else "right"
            ax1.text(row["delta_p"] + offset, i, label,
                     va="center", ha=ha, fontsize=11, fontweight="bold")
    ax1.set_xlabel(f"Δ proportion ({group_order[1]} − {group_order[0]})", fontsize=10)
    ax1.set_title(
        f"Difference in cell type proportion: {group_order[1]} vs {group_order[0]}\n"
        "* FDR<0.05   ** FDR<0.01   *** FDR<0.001  (Mann-Whitney U, BH correction)",
        fontsize=9, loc="left"
    )
    ax1.spines[["top", "right"]].set_visible(False)

    # Heatmap
    ax2 = fig1.add_subplot(gs_main[2])
    med_matrix = pd.DataFrame({
        g: props_meta[props_meta[group_col] == g][cluster_cols].median()
        for g in group_order
    })
    med_z = (med_matrix
             .subtract(med_matrix.mean(axis=1), axis=0)
             .divide(med_matrix.std(axis=1).replace(0, 1), axis=0))
    col_labels    = [f"{group_labels[g]}\n(n={n_per_group.get(g,'?')})" for g in group_order]
    med_z.columns = col_labels
    annot_vals    = med_matrix.copy()
    annot_vals.columns = col_labels
    sns.heatmap(
        med_z, cmap="RdBu_r", center=0,
        linewidths=0.4, ax=ax2,
        annot=annot_vals.round(3), fmt=".3f", annot_kws={"size": 9},
        cbar_kws={"label": "Z-score per cell type\n(relative variation across groups)",
                  "shrink": 0.7},
    )
    ax2.set_title(
        "Median proportion per cell type and group\n"
        "Color = row-normalised z-score | Value = actual median proportion",
        fontsize=9, loc="left"
    )
    ax2.set_xlabel("Group", fontsize=10)
    ax2.tick_params(axis="y", labelsize=9, rotation=0)
    ax2.tick_params(axis="x", labelsize=9)

    fig1.suptitle(
        f"{title_prefix}\n"
        f"(n={len(props_meta)} samples: "
        + ", ".join(f"{n_per_group.get(g,'?')} {group_labels[g]}" for g in group_order) + ")",
        fontsize=14, fontweight="bold", y=0.97,
    )
    plt.savefig(fig_dir / f"fig_compositional_overview_{tag}.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / f"fig_compositional_overview_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig_compositional_overview_{tag}.pdf / .png")

    # ── Figure 2: Facet boxplots ─────────────────────────────────────────
    n_cl  = len(cluster_cols)
    ncols = 3
    nrows = int(np.ceil(n_cl / ncols))

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    fig2.subplots_adjust(hspace=0.8, wspace=0.4, top=0.88, bottom=0.08,
                         left=0.07, right=0.97)

    mw_dict    = mw_df.set_index("cell_type")[["FDR", "sig", "delta_p"]].to_dict(orient="index")
    axes2_flat = axes2.flatten() if hasattr(axes2, "flatten") else [axes2]

    for idx, cl in enumerate(cluster_cols):
        ax = axes2_flat[idx]
        data_groups = [
            props_meta[props_meta[group_col] == g][cl].dropna().values
            for g in group_order
        ]
        bp = ax.boxplot(
            data_groups, patch_artist=True,
            medianprops=dict(color="black", linewidth=1.8),
            flierprops=dict(marker="o", markersize=3, alpha=0.4, markerfacecolor="gray"),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            boxprops=dict(linewidth=0.8),
        )
        for patch, g in zip(bp["boxes"], group_order):
            patch.set_facecolor(group_colors[g])
            patch.set_alpha(0.82)
        for j, (g, d) in enumerate(zip(group_order, data_groups)):
            jit = np.random.uniform(-0.15, 0.15, size=len(d))
            ax.scatter(j + 1 + jit, d, color=group_colors[g], s=28, alpha=0.8,
                       zorder=3, edgecolors="white", linewidths=0.3)
        ax.set_xticks(range(1, len(group_order) + 1))
        ax.set_xticklabels([group_labels[g] for g in group_order], fontsize=8)
        ax.set_ylabel("Proportion", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)

        mw_info     = mw_dict.get(cl, {})
        fdr_val     = mw_info.get("FDR", 1.0)
        sig_str     = mw_info.get("sig", "ns")
        title_color = "#b30000" if sig_str != "ns" else "black"
        ax.set_title(
            f"{cl}\nMW FDR={fdr_val:.3f}  {sig_str}",
            fontsize=9, color=title_color,
            fontweight="bold" if sig_str != "ns" else "normal", pad=6,
        )

    for idx in range(n_cl, len(axes2_flat)):
        axes2_flat[idx].set_visible(False)

    fig2.suptitle(
        f"{title_prefix} — cell type proportion\n"
        "MW = Mann-Whitney U (BH correction)",
        fontsize=11, fontweight="bold", y=0.98,
    )
    plt.savefig(fig_dir / f"fig_compositional_boxplots_{tag}.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / f"fig_compositional_boxplots_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig_compositional_boxplots_{tag}.pdf / .png")

    # ── Figure 3: Propeller methods comparison ───────────────────────────
    n_methods      = len(all_results)
    method_palette = ["#4575b4", "#1a9850"]
    fig3, axes3    = plt.subplots(1, n_methods, figsize=(6.5 * n_methods, 6), sharey=False)
    if n_methods == 1:
        axes3 = [axes3]

    for ax, (method_name, info), color in zip(axes3, all_results.items(), method_palette):
        df       = info["df"].copy()
        pval_col = info["pval_col"]
        ct_col   = "cell_type" if "cell_type" in df.columns else df.columns[0]
        df["-log10p"] = -np.log10(df[pval_col].clip(lower=1e-10))
        df = df.sort_values("-log10p", ascending=True).reset_index(drop=True)
        bar_c = [color if p < 0.05 else "#cccccc" for p in df[pval_col]]
        ax.barh(range(len(df)), df["-log10p"],
                color=bar_c, edgecolor="white", linewidth=0.4, height=0.65)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df[ct_col], fontsize=10)
        ax.axvline(-np.log10(0.05), color="gray", linestyle="--",
                   linewidth=0.9, label="FDR = 0.05")
        ax.set_xlabel("–log₁₀(FDR)", fontsize=10)
        ax.set_title(method_name, fontsize=10, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        for i, row in df.iterrows():
            p = row[pval_col]
            ax.text(row["-log10p"] + 0.02, i,
                    f"{p:.3f}" if p >= 0.001 else f"{p:.2e}",
                    va="center", fontsize=8)

    fig3.suptitle(
        f"Propeller (speckle) — {title_prefix}\n"
        "Filled = FDR < 0.05 | Grey = not significant",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(fig_dir / f"fig_propeller_methods_{tag}.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / f"fig_propeller_methods_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: fig_propeller_methods_{tag}.pdf / .png")


def run_perspective(adata_sub, group_col, group_order, group_labels, group_colors,
                     tag, title_prefix, table_dir, fig_dir):
    """Run the full compositional pipeline for one comparison and return the MW table."""

    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n{title_prefix}\n{'='*60}")

    counts_wide, props, props_meta, cluster_cols, meta = compute_counts_props(
        adata_sub, group_col
    )
    n_per_group = props_meta[group_col].value_counts().to_dict()
    print(f"  Samples: {len(props_meta)} | Cell types: {len(cluster_cols)}")
    print(f"  Per group: {n_per_group}")

    print("\n  Mann-Whitney U...")
    mw_df = mann_whitney_by_group(props_meta, cluster_cols, group_col, group_order)
    print(mw_df.to_string(index=False))
    mw_df.to_csv(table_dir / f"mannwhitney_{tag}.csv", index=False)

    print("\n  Propeller / speckle (R)...")
    all_results, res_grp = run_propeller(adata_sub, group_col, group_order, table_dir, tag)

    mw_asin = arcsin_mw_fallback(props_meta, cluster_cols, group_col, group_order,
                                  table_dir, tag)
    all_results["arcsin + Mann-Whitney (Python fallback)"] = {
        "df": mw_asin, "pval_col": "FDR", "fc_col": None
    }

    print("\n  Generating figures...")
    make_figures(props_meta, mw_df, cluster_cols, group_col, group_order,
                 group_labels, group_colors, all_results, fig_dir, tag, title_prefix)

    return mw_df, counts_wide


# =============================================================================
# MAIN
# =============================================================================

print("=" * 60)
print("06_compositional.py")
print("=" * 60)

print("\nLoading annotated checkpoint...")
adata = sc.read_h5ad(str(ANNOTATED_CK))
print(f"  {adata.n_obs:,} nuclei × {adata.n_vars:,} genes")
print(f"  Cell types  : {sorted(adata.obs[CLUSTER_COL].unique())}")
print(f"  Samples     : {adata.obs[SAMPLE_COL].nunique()}")

for col in ["source_type", "disease_group"]:
    adata.obs[col] = adata.obs[col].astype(str)

n_caso     = (adata.obs["source_type"] == "caso").sum()
n_controle = (adata.obs["source_type"] == "controle").sum()
print(f"  Caso     : {n_caso:,} nuclei "
      f"({sum(1 for m in SAMPLES.values() if m['source_type']=='caso')} samples)")
print(f"  Controle : {n_controle:,} nuclei "
      f"({sum(1 for m in SAMPLES.values() if m['source_type']=='controle')} samples)")

# =============================================================================
# PERSPECTIVE I — Caso vs Controle (all patients pooled vs all controls)
# =============================================================================

mw_p1, counts_p1 = run_perspective(
    adata_sub=adata,
    group_col="source_type",
    group_order=["controle", "caso"],
    group_labels={"controle": "Controle (autopsia)", "caso": "Caso (MTLE-HS)"},
    group_colors=SOURCE_COLORS,
    tag="caso_vs_controle",
    title_prefix="Perspectiva I — Caso vs Controle (todos os pacientes vs todos os controles)",
    table_dir=TABLE_DIR / "perspective1_caso_vs_controle",
    fig_dir=FIG_DIR / "perspective1_caso_vs_controle",
)

# =============================================================================
# PERSPECTIVE II — G1 vs G2, CASO SAMPLES ONLY
# =============================================================================

adata_caso = adata[adata.obs["source_type"] == "caso"].copy()
print(f"\nSubconjunto caso apenas: {adata_caso.n_obs:,} nuclei, "
      f"{adata_caso.obs[SAMPLE_COL].nunique()} amostras")

mw_p2, counts_p2 = run_perspective(
    adata_sub=adata_caso,
    group_col="disease_group",
    group_order=["G1", "G2"],
    group_labels=DISEASE_GROUP_LABELS,
    group_colors=DISEASE_GROUP_COLORS,
    tag="G1_vs_G2",
    title_prefix="Perspectiva II — Caso G1 vs Caso G2 (tempo de doenca, apenas MTLE-HS)",
    table_dir=TABLE_DIR / "perspective2_G1_vs_G2",
    fig_dir=FIG_DIR / "perspective2_G1_vs_G2",
)

del adata_caso

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for name, mw_df in [
    ("Perspectiva I  (Caso vs Controle)", mw_p1),
    ("Perspectiva II (G1 vs G2, caso apenas)", mw_p2),
]:
    sig = mw_df[mw_df["sig"] != "ns"] if len(mw_df) else mw_df
    print(f"\n{name}")
    print(f"  Mann-Whitney significant (FDR<0.05): {len(sig)} cell types")
    if len(sig):
        print(sig[["cell_type", "delta_p", "FDR", "sig"]].to_string(index=False))
    else:
        print("  None — composicao celular robusta entre os grupos comparados.")

print(f"\n✓ Tabelas Perspectiva I  em: {TABLE_DIR / 'perspective1_caso_vs_controle'}")
print(f"✓ Tabelas Perspectiva II em: {TABLE_DIR / 'perspective2_G1_vs_G2'}")
print(f"✓ Figuras Perspectiva I  em: {FIG_DIR / 'perspective1_caso_vs_controle'}")
print(f"✓ Figuras Perspectiva II em: {FIG_DIR / 'perspective2_G1_vs_G2'}")
print("\n>>> Next step: run 07_de_mast.py")
