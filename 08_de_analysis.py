#!/usr/bin/env python3
"""
08_de_analysis.py
=================
Step 8 of the TT4 FAPESP case-control snRNA-seq pipeline (MTLE-HS vs autopsy).

Runs the full DE annotation/catalog/figure pipeline across the FOUR comparisons
produced by 07_de_mast.py:

  caso_vs_controle_all   — Perspectiva I
  caso_vs_controle_G1    — Perspectiva I-G1
  caso_vs_controle_G2    — Perspectiva I-G2
  G1_vs_G2_caso          — Perspectiva II

Steps (per comparison):
  1. Load DE results (Moment 1 = cell_type, Moment 2 = Leiden subcluster)
  2. Recalculate FDR per cluster (BH, Python -- overrides MAST per-cluster FDR)
  3. Apply Level 1 (exploratory) and Level 2 (robust) thresholds
  4. Save catalogued tables
  5. Figures: volcano grid (L1 + L2), bubbleplot M1 vs M2, heatmap M1 vs M2

Gene biotype annotation (MyGene.info) is done ONCE for the union of genes
across all 4 comparisons, to avoid redundant API calls.

Output
------
    tables/de_cataloged/<tag>/nivel1_exploratorio_todos.csv
    tables/de_cataloged/<tag>/nivel2_robusto_todos.csv
    tables/de_cataloged/<tag>/de_all_annotated.csv
    tables/de_cataloged/<tag>/momento{1,2}_summary.csv
    figures/de_analysis/<tag>/volcano_grid_L1.pdf / .png
    figures/de_analysis/<tag>/volcano_grid_L2.pdf / .png
    figures/de_analysis/<tag>/heatmap_DE_p001_M1_M2.pdf / .png
    figures/de_analysis/<tag>/heatmap_DE_FDR005_M1_M2.pdf / .png
    figures/de_analysis/<tag>/bubbleplot_DE_M1_M2.pdf / .png
    figures/de_analysis/<tag>/bubbleplot_DE_M1_M2_L2.pdf / .png
    figures/de_analysis/cross_comparison_summary.png   -- n DE genes across all 4

Run
---
    conda activate scrna
    screen -S step08
    nohup python 08_de_analysis.py > logs/08_de_analysis.log 2>&1 &
    tail -f logs/08_de_analysis.log
"""

import os
import sys
import re
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from statsmodels.stats.multitest import multipletests as _mt
import mygene

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OUT_DIR, FIG_DIR, TABLE_DIR, DE_PARAMS, LEIDEN_FINAL
)

DE_DIR       = TABLE_DIR / "de_MAST"
CATALOG_DIR  = TABLE_DIR / "de_cataloged"
FIG_DE_DIR   = FIG_DIR / "de_analysis"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# COMPARISONS (must match 07_de_mast.py)
# =============================================================================

COMPARISONS = [
    dict(tag="caso_vs_controle_all", ref="controle", test="caso",
         title="Perspectiva I — Caso vs Controle (todos)"),
    dict(tag="caso_vs_controle_G1", ref="controle", test="caso",
         title="Perspectiva I-G1 — Caso G1 vs Controle G1"),
    dict(tag="caso_vs_controle_G2", ref="controle", test="caso",
         title="Perspectiva I-G2 — Caso G2 vs Controle G2"),
    dict(tag="G1_vs_G2_caso", ref="G1", test="G2",
         title="Perspectiva II — Caso G1 vs Caso G2 (tempo de doenca)"),
]

CELL_TYPE_ORDER = [
    "Oligodendrocyte", "Astrocyte", "Inhibitory Neuron",
    "Microglia", "OPC", "Excitatory Neuron",
    "T Cell", "Endothelial Cell", "Ependymal Cell",
]
HEATMAP_CT_ORDER = CELL_TYPE_ORDER

COLOR_NS  = "#d9d9d9"
COLOR_SIG = "#2166AC"

l1_pval  = DE_PARAMS["l1_pval"]
l1_logfc = DE_PARAMS["l1_logfc"]
l2_fdr   = DE_PARAMS["l2_fdr"]
l2_logfc = DE_PARAMS["l2_logfc"]

LNCRNA_PATTERNS = [
    r"^LINC\d", r"-AS\d*$", r"-IT\d*$",
    r"^MALAT", r"^NEAT", r"^XIST", r"^HOTAIR",
    r"^MIAT", r"^KCNQ1OT", r"SNHG",
]
MIRNA_PATTERNS = [r"^MIR\d", r"^hsa-mir", r"^MIRLET"]

print("=" * 60)
print("08_de_analysis.py")
print("=" * 60)
print(f"Comparisons: {[c['tag'] for c in COMPARISONS]}\n")

# =============================================================================
# 1. LOAD ALL DE RESULTS
# =============================================================================

print("[1/6] Loading DE results for all comparisons...")

loaded = {}
for cfg in COMPARISONS:
    tag, ref, test = cfg["tag"], cfg["ref"], cfg["test"]
    pair_label = f"{test}_vs_{ref}"
    m1_path = DE_DIR / tag / f"momento1_{pair_label}.csv"
    m2_path = DE_DIR / tag / f"momento2_{pair_label}.csv"
    if not m1_path.exists() or not m2_path.exists():
        raise FileNotFoundError(
            f"DE results not found for '{tag}': {m1_path} / {m2_path}\n"
            f"Run 07_de_mast.py first."
        )
    m1 = pd.read_csv(m1_path); m1["momento"] = 1
    m2 = pd.read_csv(m2_path); m2["momento"] = 2
    if "cell_type_parent" not in m2.columns:
        m2["cell_type_parent"] = m2["cluster"]
    loaded[tag] = dict(m1=m1, m2=m2, pair_label=pair_label, **cfg)
    print(f"  {tag:<24}: M1={len(m1):,} rows | M2={len(m2):,} rows")

# =============================================================================
# 2. MYGENE.INFO BIOTYPE ANNOTATION (once, union of all genes)
# =============================================================================

print("\n[2/6] Annotating gene biotypes via MyGene.info (union across comparisons)...")

all_genes = set()
for tag, d in loaded.items():
    all_genes.update(d["m1"]["gene"].dropna().unique())
    all_genes.update(d["m2"]["gene"].dropna().unique())
unique_genes = sorted(all_genes)
print(f"  Unique genes across all comparisons: {len(unique_genes):,}")

mg = mygene.MyGeneInfo()

def annotate_genes(gene_list: list, batch_size: int = 500,
                   max_retries: int = 3, delay: int = 10) -> pd.DataFrame:
    ensg_mask  = [g.startswith("ENSG") for g in gene_list]
    ensg_genes = [g for g, m in zip(gene_list, ensg_mask) if m]
    sym_genes  = [g for g, m in zip(gene_list, ensg_mask) if not m]

    rows = []
    for genes, scope in [(ensg_genes, "ensembl.gene"), (sym_genes, "symbol,alias")]:
        if not genes:
            continue
        batches = [genes[i:i + batch_size] for i in range(0, len(genes), batch_size)]
        print(f"  Querying {len(genes):,} genes ({scope}) in {len(batches)} batches...")
        for bi, batch in enumerate(batches):
            for attempt in range(max_retries):
                try:
                    res = mg.querymany(
                        batch, scopes=scope,
                        fields="symbol,ensembl.gene,type_of_gene,biotype",
                        species="human", as_dataframe=True, returnall=False,
                    )
                    res = res.reset_index().rename(columns={"query": "query_id"})
                    rows.append(res)
                    print(f"    batch {bi+1}/{len(batches)} OK")
                    time.sleep(1)
                    break
                except Exception as e:
                    print(f"    batch {bi+1} attempt {attempt+1} failed: {e}")
                    if attempt < max_retries - 1:
                        print(f"    waiting {delay}s before retry...")
                        time.sleep(delay)
                    else:
                        print(f"    batch {bi+1} skipped after {max_retries} attempts.")

    if not rows:
        return pd.DataFrame(columns=["query_id", "symbol", "ensembl_id_annot", "biotype"])

    annot = pd.concat(rows, ignore_index=True)
    if "biotype" not in annot.columns:
        annot["biotype"] = np.nan
    if "type_of_gene" in annot.columns:
        annot["biotype"] = annot["biotype"].fillna(annot["type_of_gene"])
    if "notfound" in annot.columns:
        annot = annot[annot["notfound"].fillna(False) != True]
    annot = annot.drop_duplicates(subset="query_id", keep="first")

    def extract_ensembl(x):
        if isinstance(x, dict): return x.get("gene", np.nan)
        if isinstance(x, list): return x[0].get("gene", np.nan) if x else np.nan
        return x

    annot["ensembl_id_annot"] = (
        annot["ensembl.gene"].apply(extract_ensembl)
        if "ensembl.gene" in annot.columns else np.nan
    )
    return annot[["query_id", "symbol", "ensembl_id_annot", "biotype"]].copy()

annot_cache_path = TABLE_DIR / "gene_biotype_annotation_cache.csv"
if annot_cache_path.exists():
    print(f"  [CACHE] Usando anotacao ja salva: {annot_cache_path.name}")
    annot_df = pd.read_csv(annot_cache_path)
else:
    annot_df = annotate_genes(unique_genes)
    annot_df.to_csv(annot_cache_path, index=False)
    print(f"  Anotacao salva em cache: {annot_cache_path.name}")

coverage = 100 * len(annot_df) / len(unique_genes) if unique_genes else 0
print(f"  Annotated : {len(annot_df):,} | Coverage: {coverage:.1f}%")

def classify_biotype(row):
    bt  = str(row.get("biotype",  "") or "").lower().strip()
    sym = str(row.get("symbol",   "") or row.get("gene", "") or "").upper().strip()
    if bt == "protein-coding":
        return "protein_coding"
    if bt in ("ncrna", "noncoding"):
        for p in MIRNA_PATTERNS:
            if re.search(p, sym, re.IGNORECASE):
                return "miRNA"
        for p in LNCRNA_PATTERNS:
            if re.search(p, sym, re.IGNORECASE):
                return "lncRNA"
        return "lncRNA"
    return None

# =============================================================================
# 3-6. PER-COMPARISON: FDR RECALC, LEVELS, TABLES, FIGURES
# =============================================================================

def recalc_fdr_per_cluster(df, cluster_col):
    df = df.copy()
    fdr_new = np.ones(len(df))
    for cl in df[cluster_col].unique():
        idx = df.index[df[cluster_col] == cl]
        pv  = df.loc[idx, "pvalue"].fillna(1.0).values
        _, q, _, _ = _mt(pv, method="fdr_bh")
        fdr_new[df.index.get_indexer(idx)] = q
    df["fdr"] = fdr_new
    return df


def make_volcano(df_ct, ax, title, x_abs, p_thr, fc_thr, y_col="pvalue",
                 y_label="-log$_{10}$(p-value)", top_n=10):
    df = df_ct[df_ct["logFC"].notna()].copy()
    df["-log10p"] = -np.log10(df[y_col].clip(lower=1e-10))
    sig = (df[y_col] < p_thr) & (df["logFC"].abs() > fc_thr)
    ns  = ~sig

    ax.scatter(df.loc[ns, "logFC"], df.loc[ns, "-log10p"],
               s=5, color=COLOR_NS, alpha=0.30, linewidths=0, zorder=1)
    ax.scatter(df.loc[sig, "logFC"], df.loc[sig, "-log10p"],
               s=14, color=COLOR_SIG, alpha=0.80, linewidths=0, zorder=2)
    ax.axhline(-np.log10(p_thr), color=COLOR_SIG, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(-fc_thr, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.axvline( fc_thr, color="gray", linestyle=":", linewidth=0.6, alpha=0.4)

    sig_df = df[sig].copy()
    def get_label(row):
        sym = row.get("symbol", None)
        if pd.notna(sym) and sym and not str(sym).startswith("ENSG"):
            return str(sym)
        return None
    sig_df["_label"] = sig_df.apply(get_label, axis=1)
    sig_df = sig_df[sig_df["_label"].notna()].sort_values("-log10p", ascending=False).head(top_n)

    for side_df, side in [(sig_df[sig_df["logFC"] < 0].reset_index(drop=True), "left"),
                          (sig_df[sig_df["logFC"] > 0].reset_index(drop=True), "right")]:
        y_used = []
        for idx, row in side_df.iterrows():
            x0, y0 = row["logFC"], row["-log10p"]
            ha = "right" if (side == "left") == (idx % 2 == 0) else "left"
            x_txt = x0 + (0.22 if ha == "left" else -0.22)
            y_txt = y0
            for yp in y_used:
                if abs(y_txt - yp) < 0.40:
                    y_txt = yp + 0.40
            y_used.append(y_txt)
            ax.annotate(
                row["_label"], xy=(x0, y0), xytext=(x_txt, y_txt),
                fontsize=6.5, fontweight="bold", color="#053061", ha=ha,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                arrowprops=dict(arrowstyle="-", color="#053061", lw=0.5, alpha=0.5),
                zorder=5,
            )

    n_sig  = sig.sum()
    n_up   = (sig & (df["logFC"] > 0)).sum()
    n_down = (sig & (df["logFC"] < 0)).sum()
    ax.set_title(f"{title}\n(n={n_sig}: {n_up} up, {n_down} down)",
                 fontsize=9, fontweight="bold", pad=4)
    ax.set_xlabel("log$_2$ Fold Change", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_xlim(-x_abs, x_abs)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


def volcano_grid(df_plot, cell_types_present, title, out_path, p_thr, fc_thr,
                 y_col="pvalue", y_label="-log$_{10}$(p-value)"):
    if not cell_types_present:
        return
    x_abs = np.percentile(df_plot["logFC"].abs().dropna(), 99) * 1.2 if len(df_plot) > 1 else 4.0
    ncols = 2
    nrows = int(np.ceil(len(cell_types_present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.8),
                              constrained_layout=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, ct in enumerate(cell_types_present):
        make_volcano(df_plot[df_plot["cluster"] == ct], axes_flat[i], title=ct,
                     x_abs=x_abs, p_thr=p_thr, fc_thr=fc_thr, y_col=y_col, y_label=y_label)
    for j in range(len(cell_types_present), len(axes_flat)):
        axes_flat[j].set_visible(False)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NS,
               markersize=7, label="Not significant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SIG,
               markersize=8, label=f"{y_col} < {p_thr}  &  |logFC| > {fc_thr}"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=8.5, framealpha=0.8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(f"{out_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}.png / .pdf")


def build_heatmap_matrix(df, cluster_col, pval_col, thr, col_label):
    sub = df[df[pval_col] < thr].copy()
    return (
        sub.groupby(cluster_col).size()
        .reindex(HEATMAP_CT_ORDER, fill_value=0).astype(int)
        .to_frame(name=col_label)
    )


def make_bubble_data(df, cluster_col, biotype_map, pval_col, thr):
    df = df.copy()
    df["biotype_clean"] = df["gene"].map(biotype_map)
    sub = df[(df[pval_col] < thr) & df["biotype_clean"].notna()].copy()
    sub["abs_logFC"] = sub["logFC"].abs()
    return (
        sub.groupby(cluster_col)
        .agg(n_genes=("gene", "count"), mean_abs_lfc=("abs_logFC", "mean"))
        .reset_index().rename(columns={cluster_col: "cell_type"})
    )


def bubbleplot(bdf_m1, bdf_m2, pair_label, title, out_path, pval_label):
    bdf_m1 = bdf_m1[bdf_m1["cell_type"].isin(CELL_TYPE_ORDER)]
    bdf_m2 = bdf_m2[bdf_m2["cell_type"].isin(CELL_TYPE_ORDER)]
    if len(bdf_m1) == 0 and len(bdf_m2) == 0:
        print(f"    (sem dados para {out_path.name})")
        return
    n_max    = max(bdf_m1["n_genes"].max() if len(bdf_m1) else 1,
                   bdf_m2["n_genes"].max() if len(bdf_m2) else 1)
    s_max    = 1200
    lfc_vmax = max(bdf_m1["mean_abs_lfc"].max() if len(bdf_m1) else 1,
                   bdf_m2["mean_abs_lfc"].max() if len(bdf_m2) else 1)
    warm_cmap = LinearSegmentedColormap.from_list(
        "warm_lfc", ["#FFFFCC", "#FED976", "#FEB24C", "#FD8D3C",
                     "#FC4E2A", "#E31A1C", "#B10026"], N=256)
    lfc_norm = Normalize(vmin=0, vmax=lfc_vmax if lfc_vmax > 0 else 1)
    y_map    = {ct: i for i, ct in enumerate(CELL_TYPE_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={"wspace": 0.6})
    for ax, bdf, subtitle_m in zip(axes, [bdf_m1, bdf_m2], ["Moment 1", "Moment 2"]):
        for _, row in bdf.iterrows():
            if row["cell_type"] not in y_map:
                continue
            y     = y_map[row["cell_type"]]
            size  = (row["n_genes"] / n_max) * s_max
            color = warm_cmap(lfc_norm(row["mean_abs_lfc"]))
            ax.scatter(0, y, s=size, color=color, edgecolors="#555555",
                       linewidths=0.5, zorder=3, alpha=0.92)
            ax.text(0, y, str(int(row["n_genes"])), ha="center", va="center",
                    fontsize=8, fontweight="600", color="#1a1a1a", zorder=4)
        ax.set_facecolor("#F9F9F9")
        ax.set_xticks([0]); ax.set_xticklabels([pair_label.replace("_", " ")], fontsize=9)
        ax.set_yticks(range(len(CELL_TYPE_ORDER)))
        ax.set_yticklabels(CELL_TYPE_ORDER, fontsize=9.5)
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, len(CELL_TYPE_ORDER) - 0.4)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#CCCCCC")
        ax.set_title(subtitle_m, fontsize=12, fontweight="bold", pad=10, loc="left")

    legend_ns = [n for n in [5, 20, 50, 100, 200] if n <= n_max]
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#AAAAAA",
               markeredgecolor="#555555", markersize=np.sqrt((n / n_max) * s_max) * 0.55,
               label=f"{n} genes")
        for n in legend_ns
    ]
    axes[1].legend(handles=legend_elems, title=f"No. genes\n({pval_label})",
                   title_fontsize=8.5, fontsize=8, loc="upper left",
                   bbox_to_anchor=(1.02, 1), frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    sm = ScalarMappable(cmap=warm_cmap, norm=lfc_norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], shrink=0.6, pad=0.18, anchor=(0.0, 0.3))
    cbar.set_label("Mean |logFC|", fontsize=8.5); cbar.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=11, y=1.03, color="#2C2C2A")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{out_path}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"    Saved: {out_path.name}.pdf / .png")


# ── Main loop over comparisons ────────────────────────────────────────────

cross_summary = []  # for the cross-comparison figure at the end

for tag, d in loaded.items():
    cfg        = {k: d[k] for k in ["title", "ref", "test"]}
    m1, m2     = d["m1"], d["m2"]
    pair_label = d["pair_label"]

    print(f"\n{'='*60}\n{cfg['title']}  ({pair_label})\n{'='*60}")

    tag_table_dir = CATALOG_DIR / tag
    tag_fig_dir   = FIG_DE_DIR / tag
    tag_table_dir.mkdir(parents=True, exist_ok=True)
    tag_fig_dir.mkdir(parents=True, exist_ok=True)

    print("  Recalculating FDR per cluster (BH, Python)...")
    m1 = recalc_fdr_per_cluster(m1, "cluster")
    m2_cluster_col = "cell_type_parent" if "cell_type_parent" in m2.columns else "cluster"
    m2 = recalc_fdr_per_cluster(m2, m2_cluster_col)

    de_all = pd.concat([m1, m2], ignore_index=True)
    de_all = de_all.merge(
        annot_df.rename(columns={"query_id": "gene"}), on="gene", how="left"
    )
    de_all["biotype_clean"] = de_all.apply(classify_biotype, axis=1)
    de_filt = de_all[de_all["biotype_clean"].notna()].copy()
    de_filt["direction"] = de_filt["logFC"].apply(
        lambda x: "up" if (pd.notna(x) and x > 0) else ("down" if (pd.notna(x) and x < 0) else "NA")
    )

    nivel1_mask = (
        (de_filt["pvalue"] < l1_pval) & (de_filt["logFC"].abs() > l1_logfc) &
        (de_filt["biotype_clean"].isin(["protein_coding", "lncRNA", "miRNA"]))
    )
    nivel2_mask = (
        (de_filt["pvalue"] < DE_PARAMS["l2_pval"]) & (de_filt["fdr"] < l2_fdr) &
        (de_filt["logFC"].abs() > l2_logfc) & (de_filt["biotype_clean"] == "protein_coding")
    )
    de_filt["level"] = None
    de_filt.loc[nivel1_mask, "level"] = "level1_exploratory"
    de_filt.loc[nivel2_mask, "level"] = "level2_robust"
    de_filt.loc[nivel1_mask & nivel2_mask, "level"] = "level1_and_level2"

    de_nivel1 = de_filt[nivel1_mask].copy()
    de_nivel2 = de_filt[nivel2_mask].copy()

    print(f"  Level 1: {len(de_nivel1):,}  |  Level 2: {len(de_nivel2):,}")

    COLS = ["gene", "symbol", "biotype_clean", "comparison", "cluster",
            "momento", "level", "direction", "logFC", "pvalue", "fdr", "ci_hi", "ci_lo"]
    COLS_OUT = [c for c in COLS if c in de_filt.columns]

    de_nivel1[COLS_OUT].to_csv(tag_table_dir / "nivel1_exploratorio_todos.csv", index=False)
    de_nivel2[COLS_OUT].to_csv(tag_table_dir / "nivel2_robusto_todos.csv", index=False)
    de_filt[COLS_OUT].to_csv(tag_table_dir / "de_all_annotated.csv", index=False)

    for momento, df_m, cluster_col_m, label in [
        (1, m1, "cluster", "momento1"), (2, m2, m2_cluster_col, "momento2")
    ]:
        rows = []
        for ct in sorted(df_m[cluster_col_m].unique()):
            sub = df_m[df_m[cluster_col_m] == ct]
            rows.append({
                "cell_type": ct, "comparison": pair_label, "genes_tested": len(sub),
                "p_lt_0.05": (sub["pvalue"] < 0.05).sum(),
                "p_lt_0.001": (sub["pvalue"] < 0.001).sum(),
                "FDR_lt_0.05": (sub["fdr"] < 0.05).sum(),
                "FDR_min": round(sub["fdr"].min(), 4) if len(sub) else np.nan,
            })
        pd.DataFrame(rows).to_csv(tag_table_dir / f"{label}_summary.csv", index=False)
        cross_summary.append({
            "comparison": tag, "momento": label,
            "n_level1": len(de_nivel1[de_nivel1["momento"] == momento]),
            "n_level2": len(de_nivel2[de_nivel2["momento"] == momento]),
        })

    print(f"  Tabelas salvas em: {tag_table_dir}")

    # ── Figures ──────────────────────────────────────────────────────────
    print("  Gerando figuras...")

    catalog = de_filt[["gene", "symbol"]].drop_duplicates()
    m1_plot = m1[m1["logFC"].notna()].merge(catalog, on="gene", how="left")
    cell_types_present = [c for c in CELL_TYPE_ORDER if c in m1_plot["cluster"].unique()]

    volcano_grid(
        m1_plot, cell_types_present,
        title=f"{cfg['title']} — Moment 1, Level 1 exploratory",
        out_path=tag_fig_dir / "volcano_grid_L1",
        p_thr=l1_pval, fc_thr=l1_logfc, y_col="pvalue",
        y_label="-log$_{10}$(p-value)",
    )
    volcano_grid(
        m1_plot, cell_types_present,
        title=f"{cfg['title']} — Moment 1, Level 2 robust (FDR)",
        out_path=tag_fig_dir / "volcano_grid_L2",
        p_thr=l2_fdr, fc_thr=l2_logfc, y_col="fdr",
        y_label="-log$_{10}$(FDR)",
    )

    mat_m1_p001 = build_heatmap_matrix(m1, "cluster", "pvalue", 0.001, pair_label.replace("_", " "))
    mat_m2_p001 = build_heatmap_matrix(m2, m2_cluster_col, "pvalue", 0.001, pair_label.replace("_", " "))
    mat_m1_fdr  = build_heatmap_matrix(m1, "cluster", "fdr", l2_fdr, pair_label.replace("_", " "))
    mat_m2_fdr  = build_heatmap_matrix(m2, m2_cluster_col, "fdr", l2_fdr, pair_label.replace("_", " "))

    for mats, thr_label, fname in [
        ((mat_m1_p001, mat_m2_p001), "p < 0.001", "heatmap_DE_p001_M1_M2"),
        ((mat_m1_fdr, mat_m2_fdr), f"FDR < {l2_fdr}", "heatmap_DE_FDR005_M1_M2"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={"wspace": 0.6})
        for ax, mat, subtitle_m in zip(axes, mats, ["Moment 1", "Moment 2"]):
            vmax = int(mat.values.max()) if mat.values.max() > 0 else 1
            sns.heatmap(mat, ax=ax, cmap="YlGn", annot=True, fmt="d",
                        linewidths=0.4, linecolor="#dddddd",
                        cbar_kws={"label": f"genes ({thr_label})", "shrink": 0.6},
                        vmin=0, vmax=vmax, annot_kws={"size": 10})
            ax.set_title(subtitle_m, fontsize=12, fontweight="bold", pad=8, loc="left")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.tick_params(axis="y", rotation=0, labelsize=9)
        fig.suptitle(f"{cfg['title']}\ngenes with {thr_label}", fontsize=12,
                     fontweight="bold", y=1.02)
        fig.savefig(tag_fig_dir / f"{fname}.pdf", bbox_inches="tight")
        fig.savefig(tag_fig_dir / f"{fname}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"    Saved: heatmap_DE_p001_M1_M2 / heatmap_DE_FDR005_M1_M2")

    biotype_map = de_filt.drop_duplicates("gene").set_index("gene")["biotype_clean"]
    bdf_m1_p = make_bubble_data(m1, "cluster", biotype_map, "pvalue", l1_pval)
    bdf_m2_p = make_bubble_data(m2, m2_cluster_col, biotype_map, "pvalue", l1_pval)
    bubbleplot(bdf_m1_p, bdf_m2_p, pair_label,
              title=f"DE genes per cell type — {cfg['title']}\n(size = n genes p<{l1_pval} | colour = mean |logFC|)",
              out_path=tag_fig_dir / "bubbleplot_DE_M1_M2", pval_label=f"p < {l1_pval}")

    bdf_m1_f = make_bubble_data(m1, "cluster", biotype_map, "fdr", l2_fdr)
    bdf_m2_f = make_bubble_data(m2, m2_cluster_col, biotype_map, "fdr", l2_fdr)
    bubbleplot(bdf_m1_f, bdf_m2_f, pair_label,
              title=f"DE genes per cell type (Level 2) — {cfg['title']}\n(size = n genes FDR<{l2_fdr} | colour = mean |logFC|)",
              out_path=tag_fig_dir / "bubbleplot_DE_M1_M2_L2", pval_label=f"FDR < {l2_fdr}")

# =============================================================================
# CROSS-COMPARISON SUMMARY FIGURE
# =============================================================================

print("\n[cross-comparison] Gerando resumo entre as 4 comparacoes...")

cross_df = pd.DataFrame(cross_summary)
cross_df.to_csv(CATALOG_DIR / "cross_comparison_summary.csv", index=False)

pivot_l1 = cross_df.pivot(index="comparison", columns="momento", values="n_level1")
pivot_l2 = cross_df.pivot(index="comparison", columns="momento", values="n_level2")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, pivot, label in zip(axes, [pivot_l1, pivot_l2], ["Level 1 (exploratory)", "Level 2 (robust, FDR)"]):
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "n DE genes"}, linewidths=0.5, linecolor="#ddd")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
fig.suptitle("Resumo de genes DE por comparacao (Momento 1 vs Momento 2)",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(FIG_DE_DIR / "cross_comparison_summary.png", dpi=300, bbox_inches="tight")
plt.savefig(FIG_DE_DIR / "cross_comparison_summary.pdf", bbox_inches="tight")
plt.close()
print(f"  Saved: cross_comparison_summary.png / .pdf")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY — genes DE por comparacao")
print("=" * 60)
print(cross_df.to_string(index=False))

print(f"\n  Tabelas  : {CATALOG_DIR}/<tag>/")
print(f"  Figuras  : {FIG_DE_DIR}/<tag>/")
print("\n>>> Next step: run 09_enrichment.py")
