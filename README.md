# RIDGE: Regional Integration of Deconvolution with Guided Estimation
Code accompanying our study of cortex-wide cell-type mapping and multiscale cortical cellular organization.

## Overview
<p align="center">
  <img src="assets/overview.png" alt="Overview of the analysis workflow" width="900">
</p>

This repository provides preprocessing and analysis scripts supporting reproduction of the main and selected supplementary analyses in our study.

The code is organized into two major parts: `preprocessing/` and `analysis/`. 

`preprocessing/` scripts implement the core RIDGE workflow, including transcriptomic input preparation, regional reference anchoring, region-matched deconvolution, and geometry-guided field reconstruction of regional cell-type estimates into cortex-wide maps.

`analysis/` scripts use the RIDGE-derived cell-type maps to characterize cortical cellular organization into global, DMN, M1, and visual modules following the multiscale structure of the manuscript.

The processed cell-type maps used throughout the analyses are available in `results/celltype_map`. Parcellated versions of these maps are also provided in `results/parcellated_Cammoun033_no_corpuscallosum`,`results/parcellated_Schaefer400`and `results/parcellated_Glasser360`.

The `materials/` folder provides supporting materials for reproducing the analyses, including intermediate data, annotation files, and other input resources used by the preprocessing and analysis scripts.

The repository already includes the full code for the major analytical steps and will be further updated and refined as the project continues to develop.

## Repository structure

```text
RIDGE/
├── preprocessing/
│   ├── P00_filtering_downstream analysis.R
│   ├── P01_celllabel_reannotation.R
│   ├── P02_rds_to_h5ad.R
│   ├── P03_1_integrate_data.py
│   ├── P03_2_DEG_acrossroi.py
│   ├── P04_roi_h5ad_to_roi_rds.R
│   ├── P05_roi_rds_for_deconvolution.R
│   ├── P06_integrate_roi_deconvolution.py
├── analysis/
│   ├── global_celltype_correlation.py
│   ├── global_pr_ai_index_align.py
│   ├── global_celltype_pls_receptors.py
│   ├── global_disease_exposure.py
│   ├── dmn_cyto.py
│   ├── dmn_enav.py
│   ├── m1_region_effector.py
│   ├── m1_scan.py
│   ├── visual_pc.py
│   └── visual_vtc.py
├── materials/    
└── results/
    ├── celltype_map/
    ├── parcellated_Cammoun033_no_corpuscallosum/
    ├── parcellated_Schaefer400/
    └── parcellated_Glasser360/
```
## Code for the paper
The repository is organized into two major code folders: `preprocessing/` and `analysis/`.

### Preprocessing scripts  from overview (A)

The preprocessing scripts implement the RIDGE workflow shown in overview (A): transcriptomic input preparation, regional reference anchoring, region-matched deconvolution, and geometry-guided field reconstruction of regional estimates into cortex-wide cell-type maps.

#### Input data

The deconvolution framework uses two types of input data.

**Single-nucleus RNA-seq reference data**

- Jorstad et al. https://doi.org/10.1126/science.adf6812 
- Siletti et al. https://doi.org/10.1126/science.add7046

Processed single-nucleus RNA-seq data can be accessed from CELLxGENE:  
`https://cellxgene.cziscience.com`

**Spatially aligned gene-expression reference profiles**

- Wagstyl et al. MAGICC gene-expression maps https://github.com/kwagstyl/MAGICC
- Ecker et al. gene-expression maps https://gin.g-node.org/sphache/DecodeGABAData

Other spatially aligned cortical gene-expression maps, such as maps generated using alternative AHBA preprocessing pipelines, can also be used after matching gene names, cortical space, and regional or parcellation definitions.

#### Scripts

- **`preprocessing/P00_filtering_downstream_analysis.R`**  
  Performs initial filtering of single-nucleus RNA-seq data before downstream annotation, integration, and deconvolution reference construction.

- **`preprocessing/P01_celllabel_reannotation.R`**  
  Reannotates cell-type labels across datasets using Seurat label transfer, with the Jorstad et al. 8-region, 24-cell-type annotation as the reference framework.

- **`preprocessing/P02_rds_to_h5ad.R`**  
  Provides basic code for converting Seurat `.rds` objects to `.h5ad` format for Python-based analysis.

- **`preprocessing/P03_1_integrate_data.py`**  
  Integrates single-nucleus RNA-seq datasets across 33 independent cortical regions for downstream region-specific reference construction.

- **`preprocessing/P03_2_DEG_acrossroi.py`**  
  Uses Memento to identify cell-type-specific differentially expressed genes across cortical regions, mainly for robustness checks and supplementary analyses.

- **`preprocessing/P04_roi_h5ad_to_roi_rds.R`**  
  Provides basic code for converting region-specific `.h5ad` files back to Seurat `.rds` format.

- **`preprocessing/P05_roi_rds_for_deconvolution.R`**  
  Prepares region-specific single-nucleus reference matrices for deconvolution by organizing cell-type-specific expression profiles.

- **`preprocessing/P06_integrate_roi_deconvolution.py`**  
  Integrates region-wise deconvolution results into unified cortex-wide cell-type maps for downstream analyses.

### Analysis scripts from overview (B–D)

The analysis scripts use the processed cortex-wide cell-type maps to examine how cellular composition is organized across nested spatial scales of the human cortex. The analytical framework follows the multiscale structure of the manuscript and overview figure, progressing from global cortex-wide organization (overview B), to system-level organization within the default mode, motor, and visual systems (overview C), and finally to local cellular organization within specific cortical territories (overview D).

Scripts with the prefix `global_` correspond to the global-scale analyses in overview (B). These analyses evaluate cortex-wide cellular organization and its relationships with established cortical features and large-scale organizational axes.

The `dmn`, `m1`, and `visual` modules span the system and local scales shown in overview (C–D). At the system level, these analyses characterize cellular organization within the default mode network, primary motor cortex, and visual cortex. At the local level, they further resolve cellular organization across DMN subregions, M1 inter-effector regions, and ventral temporal cortex, respectively.

#### Global-scale analyses from overview (B)

- **`analysis/global_celltype_correlation.py`**  
  Examines cortex-wide relationships among the 24 cell-type maps and summarizes their large-scale organizational structure.

- **`analysis/global_pr_ai_index_align.py`**  
  Tests the alignment between cell-type distributions and the canonical cortical Pr-Al axis, providing a cortex-wide benchmark of cellular organization.

- **`analysis/global_celltype_pls_receptors.py`**  
  Relates cell-type spatial patterns to neurotransmitter receptor density maps using partial least squares analysis.

- **`analysis/global_disease_exposure.py`**  
  Evaluates associations between cell-type maps and spatial patterns of disease-related cortical vulnerability or exposure.

#### DMN analyses across system and local scales (overview C–D)

These analyses examine cellular organization within the default mode network at the system level and its further differentiation across local DMN subregions and cytoarchitectural types. The analyses were conceptually informed by recent work characterizing the default mode network through cytoarchitecture, wiring, and signal flow, particularly Paquola et al. (2025).

- **`analysis/dmn_cyto.py`**  
  Characterizes cell-type organization within the default mode network and relates cellular patterns to cytoarchitectural differentiation across DMN subregions.

- **`analysis/dmn_enav.py`**  
  Examines cell-type variation along complementary dimensions of DMN organization, highlighting cellular heterogeneity within the network.

#### M1 analyses across system and local scales (overview C–D)

These analyses examine continuous cellular organization across primary motor cortex and further resolve the cellular organization of local inter-effector regions. The analyses were motivated by recent evidence that primary motor cortex contains inter-effector regions linked to the somato-cognitive action network, as described by Gordon et al. (2023).

- **`analysis/m1_region_effector.py`**  
  Characterizes continuous cell-type profiles across primary motor cortex and examines their organization along the mouth–hand–foot somatotopic axis.

- **`analysis/m1_scan.py`**  
  Examines the cellular organization of inter-effector regions and their differentiation from neighboring effector territories.

#### Visual cortex analyses across system and local scales (overview C–D)

These analyses examine cellular organization along large-scale visual processing and eccentricity-related axes and further resolve how these patterns are organized locally within ventral temporal cortex. The analyses were informed by recent work linking visual cortical organization, ventral temporal cortex architecture, eccentricity, and category-selective organization, including Kubota et al. (2025).

- **`analysis/visual_pc.py`**  
  Examines major axes of cellular variation across visual cortex and relates them to hierarchical processing and eccentricity-related organization.

- **`analysis/visual_vtc.py`**  
  Examines local cellular organization within ventral temporal cortex and its relationship to eccentricity-gradient reversals and category-selective territories.

### Key conceptual references for analyses
- Huang, Z. et al. (2026). An opposing molecular gradient axis underlies primate cortical organization. Science 392, eaea2673. https://doi.org/10.1126/science.aea2673.
- Paquola, C. et al. (2025). The architecture of the human default mode network explored through cytoarchitecture, wiring and signal flow. *Nature Neuroscience*. https://doi.org/10.1038/s41593-024-01868-0
- Gordon, E. M. et al. (2023). A somato-cognitive action network alternates with effector regions in motor cortex. *Nature*, 617, 351–359. https://doi.org/10.1038/s41586-023-05964-2
- Kubota, E. et al. (2025). White matter connections of human ventral temporal cortex are organized by cytoarchitecture, eccentricity and category-selectivity from birth. *Nature Human Behaviour*. https://doi.org/10.1038/s41562-025-02116-6
