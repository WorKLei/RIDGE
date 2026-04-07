# A Fine-Grained Cellular Atlas Maps Nested Organizational Axes of Human Cortex

Code accompanying the manuscript on cortical cell-type maps, analysis in nested organizational axes, and functional gradient embedding.

## Overview
This repository contains preprocessing and analysis scripts used to generate the main and supplementary results in our study.

The code is organized into two major parts: preprocessing and analysis. Analysis scripts are further grouped into global, DMN, visual, and M1 modules following the structure of the manuscript.

The processed cell-type maps used throughout the analyses are available in results/celltype_map.

The repository already includes the full code for the major analytical steps and will be further updated and refined as the project continues to develop.

## Repository structure

```text
fine-grained-cellular-atlas-cortex/
├── preprocessing/(currently being organized and refined)
├── analysis/
│   ├── global_celltype_correlation.py
│   ├── global_celltype_pls_receptors.py
│   ├── global_disease_exposure.py
│   ├── dmn_cyto.py
│   ├── dmn_enav.py
│   ├── visual_pc.py
│   ├── visual_vtc.py
│   ├── m1_region_effector.py
│   ├── m1_scan.py
│   └── celltype_in_fc_gradient.py
├── materials/
└── results/
  └── celltype_map
