# A Fine-Grained Cellular Atlas Maps Nested Organizational Axes of Human Cortex

Code accompanying the manuscript on cortical cell-type maps, analysis across nested organizational axes.

## Overview
<p align="center">
  <img src="assets/overview.png" alt="Overview of the analysis workflow" width="900">
</p>

This repository contains preprocessing and analysis scripts used to generate the main and supplementary results in our study.

The code is organized into two major parts: preprocessing and analysis. Analysis scripts are further grouped into global, DMN, visual, and M1 modules following the structure of the manuscript.

The processed cell-type maps used throughout the analyses are available in `results/celltype_map`. Parcellated versions of these maps are also provided in `results/parcellated_Cammoun033_no_corpuscallosum` and `results/parcellated_Schaefer400`.

The repository already includes the full code for the major analytical steps and will be further updated and refined as the project continues to develop.

## Repository structure

```text
fine-grained-cellular-atlas-cortex/
├── preprocessing/
│   ├── P00_filtering_downstream analysis.R
│   ├── P01_celllabel_reannotation.R
│   ├── P02_rds_to_h5ad.R
│   ├── P03_1_integrate_data.py
│   ├── P03_2_DEG_acrossroi.py
│   ├── P04_roi_h5ad_to_roi_rds.R
│   ├── P05_roi_rds_for_deconvolution.R
│   ├── P06_integrate_roi_deconvolution.R
├── analysis/
│   ├── global_celltype_correlation.py
│   ├── global_celltype_pls_receptors.py
│   ├── global_disease_exposure.py
│   ├── dmn_cyto.py
│   ├── dmn_enav.py
│   ├── m1_region_effector.py
│   ├── m1_scan.py
│   ├── visual_pc.py
│   ├── visual_vtc.py
│   └── celltype_in_fc_gradient.py
├── materials/
└── results/
    ├── celltype_map/
    ├── parcellated_Cammoun033_no_corpuscallosum/
    └── parcellated_Schaefer400/
    └── parcellated_GLasser360/

## code for the paper



