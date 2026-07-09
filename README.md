# TT4 FAPESP — snRNAseq Pipeline (MTLE-HS vs. Control)

Single-nucleus RNA-seq (snRNAseq) analysis pipeline for human hippocampal tissue, comparing surgical samples from patients with mesial temporal lobe epilepsy with hippocampal sclerosis (MTLE-HS) against public autopsy samples without lesion (GSE278576, Zemke et al., 2024).

Developed as part of the TT4 FAPESP grant project. Adapted from a prior biopsy-vs-autopsy pipeline, with a case-control design and no trajectory analysis step.

## Study design

| Group | Material | Origin | N samples |
|---|---|---|---|
| Case G1 | MTLE-HS hippocampus | Surgery | 8 (disease duration < 40 years) |
| Case G2 | MTLE-HS hippocampus | Surgery | 6 (disease duration ≥ 40 years) |
| Control G1 | Lesion-free hippocampus | Public autopsy | 8 (age-matched to G1) |
| Control G2 | Lesion-free hippocampus | Public autopsy | 11 (age-matched to G2) |

**Two analytical perspectives** are maintained throughout the pipeline (cell composition, differential expression, and enrichment):

- **Perspective I** — all cases vs. all controls (primary), with confirmatory sub-analyses **I-G1** (case G1 vs. control G1) and **I-G2** (case G2 vs. control G2)
- **Perspective II** — case G1 vs. case G2 (hypothesis-generating; isolates the effect of disease duration within the same condition)

Structural differences from the original biopsy-vs-autopsy pipeline: no trajectory analysis; case-control comparison instead of tissue-type comparison; differential handling of filtered (cases) vs. raw (controls) count matrices.

## Flowchart

<img width="2720" height="3920" alt="pipeline_tt4_fapesp_overview" src="https://github.com/user-attachments/assets/0ae1501f-daaa-437f-a8ac-a241a3690ece" />


## Requirements

- `scrna` conda environment (Python 3.11 + R packages via rpy2)
- **Python**: `scanpy`, `anndata`, `scvi-tools`, `gseapy`, `mygene`, `statsmodels`, `seaborn`, `rpy2`
- **R** (via rpy2): `scDblFinder`, `SingleCellExperiment`, `BiocParallel`, `DropletUtils`, `clustree`, `speckle`, `limma`, `MAST`, `data.table`

```bash
conda activate scrna
```

## Directory structure

```
tt4_fapesp/
├── config.py                    # global parameters, samples, annotations
├── 01_load_doublets.py
├── 02_qc.py
├── 03_scvi_integration.py
├── 04a_clustree.py
├── 04b_clustering_final.py
├── 05a_annotation_markers.py
├── 05b_annotation_apply.py
├── 06_compositional.py
├── 07_de_mast.py
├── 08_de_analysis.py
├── 09_enrichment.py
├── 11_umap_figures.py
├── logs/
└── outputs/
    ├── checkpoints/
    ├── figures/
    └── tables/
```

## Pipeline steps

### `config.py`
Global parameters: paths, sample table (`SAMPLES`, with `source_type`, `disease_group`, `matrix_type`), QC thresholds, canonical markers, scVI/Leiden parameters, cluster annotation (`CLUSTER_ANNOTATION`), and DE/enrichment parameters. Edited at both human checkpoints in the pipeline (Leiden resolution and cell type annotation).

### `01_load_doublets.py`
Loads count matrices — **filtered** (mtx) for cases, **raw** (h5, with `emptyDrops`/DropletUtils for real-cell calling) for controls — and runs `scDblFinder` per sample. Per-sample checkpoints (`tables/doublet_checkpoints/`), safe to resume after interruption.

### `02_qc.py`
MAD-based (median absolute deviation) QC filter on number of genes, total counts, and % mitochondrial reads, per sample, plus doublet removal. Additional per-sample canonical marker coverage filter. Produces before/after violin plots and the list of approved samples.

### `03_scvi_integration.py`
HVG selection, scVI model training (batch = sample; covariates = `source_type`, `has_pmi`, `pmi_h_imputed`), UMAP, and Leiden clustering across multiple resolutions (0.2–2.0).

### `04a_clustree.py` / `04b_clustering_final.py`
Clustree generation to inspect cluster stability across resolutions **(human checkpoint: set `LEIDEN_FINAL` in `config.py`)**, followed by applying the final resolution and generating the definitive UMAP.

### `05a_annotation_markers.py` / `05b_annotation_apply.py`
Canonical marker dotplot per cluster **(human checkpoint: fill in `CLUSTER_ANNOTATION` in `config.py`)**, followed by applying the annotation and generating the final annotated object (`adata_annotated.h5ad`).

### `06_compositional.py`
Cell type composition analysis for Perspectives I and II: per-sample counts/proportions, Mann-Whitney U (FDR correction), `propeller`/speckle (R), and an arcsin + Mann-Whitney fallback (Python).

### `07_de_mast.py`
Pseudobulk differential expression via MAST (sample × cluster, minimum 10 nuclei), across four comparisons (Perspectives I, I-G1, I-G2, II), at two moments (cell type and Leiden subcluster).

### `08_de_analysis.py`
Gene biotype annotation (MyGene.info, cached), per-cluster FDR recalculation, cataloguing into Level 1 (exploratory) and Level 2 (robust), and generation of volcano plots, heatmaps, and bubble plots per comparison.

### `09_enrichment.py`
**Strategic** functional enrichment (GO Biological Process, Reactome via Enrichr): mitochondrial genes removed beforehand (PMI confounder); based on Level 2 genes, at the cell-type level; Perspective I run in full for the cell types most biologically relevant to MTLE-HS; Perspectives I-G1/I-G2 restricted to cell types already significant in Perspective I (confirmatory); Perspective II not submitted to formal enrichment (only a Level 1 gene list is reported, hypothesis-generating).

### `11_umap_figures.py`
Supplementary quality-control figures: UMAP before/after batch integration, doublet score before/after filtering (overall and by subgroup), case/control distribution, and number of genes per nucleus.

## Important methodological notes

- **Mitochondrial genes (MT-\*)** dominate the top of the DE gene lists in nearly every comparison — identified as a technical artifact linked to the post-mortem interval difference between cases (surgical, PMI≈0) and controls (autopsy, PMI in hours), not disease biology. For this reason, they are removed before functional enrichment.
- **PMI** is modeled both as an imputed continuous covariate (`pmi_h_imputed`, NaN→0 for cases) and as a binary flag (`has_pmi`), so the scVI model can distinguish a true absence of PMI from PMI=0.
- **Perspective II** is strictly hypothesis-generating: no Level 2 (robust) genes are found in any cell type, and it is therefore not submitted to formal enrichment.
- Sample size is intentionally unbalanced across groups (project team decision), not a strict 1:1 pairing.

## References

- Zemke, N.R. et al. (2024). *GSE278576* — snRNAseq of post-mortem human hippocampus.
- Pipeline adapted from a prior biopsy-vs-autopsy workflow (same research group).
