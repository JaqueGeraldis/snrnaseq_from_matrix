"""
config.py
=========
Global configuration for the TT4 FAPESP snRNA-seq case-control pipeline
(MTLE-HS hippocampus vs age-matched public autopsy controls).
All scripts import from this file — edit parameters here, nowhere else.

Analysis design:
    - Caso  (source_type = "caso")     — MTLE-HS hippocampal sclerosis, surgical, FILTERED matrix
    - Controle (source_type = "controle") — public autopsy, hippocampus w/o lesion, RAW matrix
    - Stratified by disease_group ("G1": disease duration <20y / "G2": disease duration >=20y),
      with controls age-matched within each group (no formal 1:1 pairing — see note below)
    - 14 caso samples (G1 n=8, G2 n=6)  +  19 controle samples (G1 n=8, G2 n=11)
      -> unbalanced n on purpose (kept as-is per PI decision), not 1:1 paired
    - NOTE: controle samples are RAW cellranger matrices (not CellBender-cleaned, not
      pre-filtered) -> require an emptyDrops step before scDblFinder (see 01_load_doublets.py)
    - NO trajectory analysis in this pipeline (unlike the biopsy x autopsy paper pipeline)

Usage (every other script starts with):
    from config import *          # or
    import config as cfg

How to run each step (inside a screen session):
    conda activate base
    nohup python 01_load_doublets.py > logs/01_load_doublets.log 2>&1 &
    tail -f logs/01_load_doublets.log
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path("/Bioinf12-HD2/jaqueline/tt4_fapesp")

# Onde estão as matrizes de origem (não ficam dentro do BASE_DIR do projeto)
CASO_DIR     = Path("/Bioinf12-HD2/jaqueline/v2_amostras_danibruno_separadas_only_RNA")
CONTROLE_DIR = Path("/Bioinf12-HD2/jaqueline/GSE278576_Zemke2024_autopsy_tese")

OUT_DIR = BASE_DIR / "outputs"

FIG_DIR        = OUT_DIR / "figures"
TABLE_DIR      = OUT_DIR / "tables"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
LOG_DIR        = OUT_DIR / "logs"
MODEL_DIR      = OUT_DIR / "scvi_model"

for _d in [FIG_DIR, TABLE_DIR, CHECKPOINT_DIR, LOG_DIR, MODEL_DIR,
           TABLE_DIR / "de_MAST",
           TABLE_DIR / "enrichment",
           TABLE_DIR / "doublet_checkpoints"]:
    _d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42

# =============================================================================
# SAMPLES
# =============================================================================
# Keys   : ID Tese (identificador usado nas figuras/tabelas)
# Values : metadata dict
#   donor_id        — pasta (caso) ou ID numerico do h5 (controle)
#   source_type     — "caso" | "controle"
#   disease_group   — "G1" (doenca <20a / pareado por idade) | "G2" (doenca >=20a / pareado por idade)
#   matrix_type     — "filtered" (dir com barcodes/features/matrix.mtx) | "raw" (.h5 cellranger)
#   path            — Path completo pra pasta (filtered) ou arquivo .h5 (raw)
#   tissue          — "hippocampus_sclerosis" | "hippocampus_lesion_free"
#   age             — idade do doador (anos)
#   sex             — "Female" | "Male"
#   pmi_h           — apenas controles (IPM em horas); None para casos (cirurgia)
#   disease_duration_y — apenas casos; None para controles
#   batch           — "caso_batch" | "controle_batch"

SAMPLES = {

    # ---------------- CASOS (MTLE-HS, filtered, cirurgia) ----------------
    # Grupo G1 — tempo de doenca < 20 anos
    "ELTM1":  {"donor_id": "G240", "source_type": "caso", "disease_group": "G1",
               "matrix_type": "filtered", "path": CASO_DIR / "G240",
               "tissue": "hippocampus_sclerosis", "age": 38, "sex": "Female",
               "pmi_h": None, "disease_duration_y": 15, "batch": "caso_batch"},
    "ELTM2":  {"donor_id": "G144", "source_type": "caso", "disease_group": "G1",
               "matrix_type": "filtered", "path": CASO_DIR / "G144",
               "tissue": "hippocampus_sclerosis", "age": 31, "sex": "Male",
               "pmi_h": None, "disease_duration_y": 18, "batch": "caso_batch"},
    "ELTM3":  {"donor_id": "G2", "source_type": "caso", "disease_group": "G1",
               "matrix_type": "filtered", "path": CASO_DIR / "G2",
               "tissue": "hippocampus_sclerosis", "age": 27, "sex": "Female",
               "pmi_h": None, "disease_duration_y": 24, "batch": "caso_batch"},
    "ELTM7":  {"donor_id": "G186", "source_type": "caso", "disease_group": "G1",
               "matrix_type": "filtered", "path": CASO_DIR / "G186",
               "tissue": "hippocampus_sclerosis", "age": 38, "sex": "Female",
               "pmi_h": None, "disease_duration_y": 32, "batch": "caso_batch"},
    "ELTM8":  {"donor_id": "G149", "source_type": "caso", "disease_group": "G1",
               "matrix_type": "filtered", "path": CASO_DIR / "G149",
               "tissue": "hippocampus_sclerosis", "age": 49, "sex": "Male",
               "pmi_h": None, "disease_duration_y": 36, "batch": "caso_batch"},
    "ELTM9":  {"donor_id": "G140", "source_type": "caso", "disease_group": "G1",
               "matrix_type": "filtered", "path": CASO_DIR / "G140",
               "tissue": "hippocampus_sclerosis", "age": 39, "sex": "Male",
               "pmi_h": None, "disease_duration_y": 38.84, "batch": "caso_batch"},
    "ELTM10": {"donor_id": "G253", "source_type": "caso", "disease_group": "G1",
               "matrix_type": "filtered", "path": CASO_DIR / "G253",
               "tissue": "hippocampus_sclerosis", "age": 33, "sex": "Male",
               "pmi_h": None, "disease_duration_y": 31, "batch": "caso_batch"},
    "ELTM11": {"donor_id": "G1", "source_type": "caso", "disease_group": "G1",
               "matrix_type": "filtered", "path": CASO_DIR / "G1",
               "tissue": "hippocampus_sclerosis", "age": 50, "sex": "Male",
               "pmi_h": None, "disease_duration_y": 39, "batch": "caso_batch"},

    # Grupo G2 — tempo de doenca >= 20 anos
    "ELTM14": {"donor_id": "G259", "source_type": "caso", "disease_group": "G2",
               "matrix_type": "filtered", "path": CASO_DIR / "G259",
               "tissue": "hippocampus_sclerosis", "age": 62, "sex": "Female",
               "pmi_h": None, "disease_duration_y": 43, "batch": "caso_batch"},
    "ELTM15": {"donor_id": "G232", "source_type": "caso", "disease_group": "G2",
               "matrix_type": "filtered", "path": CASO_DIR / "G232",
               "tissue": "hippocampus_sclerosis", "age": 60, "sex": "Male",
               "pmi_h": None, "disease_duration_y": 46, "batch": "caso_batch"},
    "ELTM16": {"donor_id": "G188", "source_type": "caso", "disease_group": "G2",
               "matrix_type": "filtered", "path": CASO_DIR / "G188",
               "tissue": "hippocampus_sclerosis", "age": 54, "sex": "Female",
               "pmi_h": None, "disease_duration_y": 52, "batch": "caso_batch"},
    "ELTM17": {"donor_id": "G114", "source_type": "caso", "disease_group": "G2",
               "matrix_type": "filtered", "path": CASO_DIR / "G114",
               "tissue": "hippocampus_sclerosis", "age": 48, "sex": "Male",
               "pmi_h": None, "disease_duration_y": 44, "batch": "caso_batch"},
    "ELTM18": {"donor_id": "G212", "source_type": "caso", "disease_group": "G2",
               "matrix_type": "filtered", "path": CASO_DIR / "G212",
               "tissue": "hippocampus_sclerosis", "age": 49, "sex": "Male",
               "pmi_h": None, "disease_duration_y": 46, "batch": "caso_batch"},
    "ELTM19": {"donor_id": "G211", "source_type": "caso", "disease_group": "G2",
               "matrix_type": "filtered", "path": CASO_DIR / "G211",
               "tissue": "hippocampus_sclerosis", "age": 44, "sex": "Female",
               "pmi_h": None, "disease_duration_y": 44, "batch": "caso_batch"},

    # ---------------- CONTROLES (autopsia publica, raw) ----------------
    # CT1 — pareado por idade com o grupo G1 de casos
    "Autopsia1":  {"donor_id": "5579", "source_type": "controle", "disease_group": "G1",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc5579_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 25, "sex": "Female",
                   "pmi_h": 14.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia2":  {"donor_id": "76", "source_type": "controle", "disease_group": "G1",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc76_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 26, "sex": "Female",
                   "pmi_h": 12.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia3":  {"donor_id": "29", "source_type": "controle", "disease_group": "G1",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc29_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 28, "sex": "Male",
                   "pmi_h": 6.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia4":  {"donor_id": "6052", "source_type": "controle", "disease_group": "G1",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc6052_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 28, "sex": "Male",
                   "pmi_h": 15.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia5":  {"donor_id": "5614", "source_type": "controle", "disease_group": "G1",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc5614_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 31, "sex": "Male",
                   "pmi_h": 27.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia6":  {"donor_id": "13344", "source_type": "controle", "disease_group": "G1",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc13344_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 33, "sex": "Female",
                   "pmi_h": 12.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia7":  {"donor_id": "935", "source_type": "controle", "disease_group": "G1",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc935_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 38, "sex": "Female",
                   "pmi_h": 19.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia8":  {"donor_id": "937", "source_type": "controle", "disease_group": "G1",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc937_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 38, "sex": "Female",
                   "pmi_h": 9.0, "disease_duration_y": None, "batch": "controle_batch"},

    # CT2 — pareado por idade com o grupo G2 de casos
    "Autopsia9":  {"donor_id": "1134", "source_type": "controle", "disease_group": "G2",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc1134_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 41, "sex": "Male",
                   "pmi_h": 15.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia10": {"donor_id": "13414", "source_type": "controle", "disease_group": "G2",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc13414_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 41, "sex": "Male",
                   "pmi_h": 9.85, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia11": {"donor_id": "5021", "source_type": "controle", "disease_group": "G2",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc5021_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 43, "sex": "Female",
                   "pmi_h": 11.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia12": {"donor_id": "5087", "source_type": "controle", "disease_group": "G2",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc5087_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 44, "sex": "Male",
                   "pmi_h": 4.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia13": {"donor_id": "1745", "source_type": "controle", "disease_group": "G2",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc1745_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 46, "sex": "Female",
                   "pmi_h": 20.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia14": {"donor_id": "4781", "source_type": "controle", "disease_group": "G2",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc4781_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 46, "sex": "Male",
                   "pmi_h": 17.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia15": {"donor_id": "81", "source_type": "controle", "disease_group": "G2",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc81_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 48, "sex": "Female",
                   "pmi_h": 10.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia16": {"donor_id": "5610", "source_type": "controle", "disease_group": "G2",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc5610_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 50, "sex": "Female",
                   "pmi_h": 11.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia17": {"donor_id": "5551", "source_type": "controle", "disease_group": "G2",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc5551_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 54, "sex": "Male",
                   "pmi_h": 9.0, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia18": {"donor_id": "6021", "source_type": "controle", "disease_group": "G2",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc6021_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 55, "sex": "Female",
                   "pmi_h": 9.5, "disease_duration_y": None, "batch": "controle_batch"},
    "Autopsia19": {"donor_id": "13394", "source_type": "controle", "disease_group": "G2",
                   "matrix_type": "raw", "path": CONTROLE_DIR / "GSE278576_hc13394_raw_feature_bc_matrix.h5",
                   "tissue": "hippocampus_lesion_free", "age": 65, "sex": "Female",
                   "pmi_h": 16.82, "disease_duration_y": None, "batch": "controle_batch"},
}

# Ordered groups for the case-control comparison
SOURCE_ORDER  = ["controle", "caso"]
SOURCE_LABELS = {
    "controle": "Autopsy control",
    "caso":     "MTLE-HS (surgical)",
}

# Comparisons of interest — stratified by disease_group (avoids confounding
# disease duration / age with case-control status)
ALL_PAIRS = [
    ("controle", "caso"),   # global, all groups pooled (use with covariate disease_group)
]

COMPARISONS_BY_GROUP = [
    {"name": "G1_controle_vs_caso", "disease_group": "G1", "levels": ["controle", "caso"]},
    {"name": "G2_controle_vs_caso", "disease_group": "G2", "levels": ["controle", "caso"]},
]

# =============================================================================
# QC PARAMETERS
# =============================================================================

QC_PARAMS = {
    "pct_mito_max": 10,   # maximum mitochondrial read fraction (%)
                          # NOTE: revisar por source_type -- tecido de autopsia
                          # tende a ter %mito mais alta por degradacao post-mortem
    "mad_n":        3,    # number of MADs for outlier filtering
    "min_genes":    200,  # minimum genes per nucleus
    "min_cells":    3,    # minimum cells expressing a gene
}

# emptyDrops (apenas para amostras raw/controle -- ver 01_load_doublets.py)
EMPTYDROPS_PARAMS = {
    "fdr_threshold": 0.01,
    "lower":         100,   # UMI count threshold to define the ambient RNA pool
}

MIN_CELLS_EXPRESSING = 10
MIN_TYPES_COVERED    = 10

# =============================================================================
# CANONICAL MARKERS
# =============================================================================

ESSENTIAL_MARKERS = {
    "Neuron (general)":  ["GRIN2B"],
    "Inhibitory Neuron": ["GAD1"],
    "Excitatory Neuron": ["SLC17A7"],
    "Astrocyte":         ["AQP4", "GFAP"],
    "Oligodendrocyte":   ["MOBP", "MOG", "OPALIN"],
    "OPC":               ["PDGFRA", "VCAN"],
    "Microglia":         ["PTPRC", "APBB1IP"],
    "T Cell":            ["PTPRC", "THEMIS"],
    "Ependymal Cell":    ["DNAH7", "DNAH8", "SPAG17"],
    "Endothelial Cell":  ["VIM", "EPAS1", "TIMP3"],
}

MARKER_GENES = ESSENTIAL_MARKERS.copy()

# =============================================================================
# scVI PARAMETERS
# =============================================================================

SCVI_PARAMS = {
    "n_top_genes":             3000,
    "n_layers":                2,
    "n_latent":                30,
    "gene_likelihood":         "nb",
    "max_epochs":              400,
    "early_stopping":          True,
    "early_stopping_patience": 20,
    "lr":                      1e-3,
}

N_NEIGHBORS = 30

LEIDEN_RESOLUTIONS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# =============================================================================
# *** HUMAN CHECKPOINT 1 ***
# After running 04a_clustree.py, inspect figures/clustree.png,
# then set the resolution below and run 04b_clustering_final.py.
# =============================================================================
LEIDEN_FINAL = "leiden_1.4"   # <-- edit this after inspecting clustree

# =============================================================================
# *** HUMAN CHECKPOINT 2 ***
# After running 05a_annotation_markers.py, inspect the dotplot,
# then fill the dictionary below and run 05b_annotation_apply.py.
# =============================================================================
CLUSTER_ANNOTATION = {
    # --- Oligodendrocyte ---
    "14": "Oligodendrocyte", "28": "Oligodendrocyte", "9":  "Oligodendrocyte",
    "4":  "Oligodendrocyte", "10": "Oligodendrocyte", "1":  "Oligodendrocyte",
    "0":  "Oligodendrocyte", "12": "Oligodendrocyte", "6":  "Oligodendrocyte",
    "11": "Oligodendrocyte", "13": "Oligodendrocyte", "2":  "Oligodendrocyte",
    # --- Astrocyte ---
    "22": "Astrocyte", "3":  "Astrocyte", "8":  "Astrocyte",
    # --- OPC ---
    "31": "OPC", "7":  "OPC",
    # --- Inhibitory Neuron ---
    "18": "Inhibitory Neuron", "17": "Inhibitory Neuron",
    "25": "Inhibitory Neuron", "30": "Inhibitory Neuron",
    # --- Excitatory Neuron ---
    "15": "Excitatory Neuron", "19": "Excitatory Neuron", "16": "Excitatory Neuron",
    "23": "Excitatory Neuron", "29": "Excitatory Neuron",
    # --- Microglia ---
    "27": "Microglia", "5":  "Microglia", "24": "Microglia",
    # --- T Cell ---
    "26": "T Cell",
    # --- Endothelial Cell ---
    "21": "Endothelial Cell", "20": "Endothelial Cell",
    # --- Ependymal Cell ---
    "32": "Ependymal Cell",
}

# =============================================================================
# DE PARAMETERS
# =============================================================================

DE_PARAMS = {
    "l1_pval":  0.001,
    "l1_logfc": 0.5,
    "l2_pval":  0.001,
    "l2_fdr":   0.05,
    "l2_logfc": 0.5,
}

ENRICHMENT_GENE_SETS = ["GO_Biological_Process_2023", "Reactome_2022"]

# =============================================================================
# FIGURE DEFAULTS
# =============================================================================

FIG_DPI    = 300
FIG_DPI_HI = 600

# =============================================================================
# QUICK SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    print("=== config.py — sanity check ===\n")
    print(f"OUT_DIR    : {OUT_DIR}")
    print(f"Samples    : {len(SAMPLES)}")
    n_caso     = sum(1 for m in SAMPLES.values() if m["source_type"] == "caso")
    n_controle = sum(1 for m in SAMPLES.values() if m["source_type"] == "controle")
    print(f"  Caso      : {n_caso}")
    print(f"  Controle  : {n_controle}")
    for g in ["G1", "G2"]:
        nc = sum(1 for m in SAMPLES.values() if m["disease_group"] == g and m["source_type"] == "caso")
        nctl = sum(1 for m in SAMPLES.values() if m["disease_group"] == g and m["source_type"] == "controle")
        print(f"    {g}: {nc} casos / {nctl} controles")
    print()
    print("Checking input files...")
    missing = []
    for sample_id, meta in SAMPLES.items():
        p = meta["path"]
        if meta["matrix_type"] == "filtered":
            ok = (p / "matrix.mtx").exists()
        else:
            ok = p.exists()
        status = "✓" if ok else "✗ NOT FOUND"
        print(f"  {status}  {sample_id}  |  {meta['source_type']:9s}"
              f"  |  {meta['disease_group']}  |  {meta['matrix_type']}  |  {p}")
        if not ok:
            missing.append(sample_id)
    print()
    if missing:
        print(f"WARNING: {len(missing)} file(s) not found — check CASO_DIR / CONTROLE_DIR.")
    else:
        print("All input files found.")
