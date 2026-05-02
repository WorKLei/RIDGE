# ============================================================
#P2: Convert Jorstad geneID Seurat objects and relabeled Siletti Seurat objects to h5ad format for downstream integration and analysis.
# Note:
# Large Seurat objects may occasionally cause memory or object-size
# limitations during downstream integration and analysis. Therefore,
# these relabeled Seurat objects are converted to h5ad format for
# subsequent integration using AnnData/Scanpy-based workflows.
# ============================================================
library(Seurat)
library(SeuratObject)
library(sceasy)
library(reticulate)
# ============================================================
# Set input paths
# ============================================================
jorstad_rds_path <- list.files(
  path = "/mnt/data0/home/wklei/docker_mount_flie/material/cell_single_data/Jorstad_et_al/",
  pattern = "Jorstad_et_al_.*geneID\\.rds$",
  full.names = TRUE
)

siletti_relabel_rds_path <- list.files(
  path = "/mnt/data0/home/wklei/docker_mount_flie/material/cell_single_data/filter_data/siletti_material_Jorstad_transfer",
  pattern = "siletti_.*\\.rds$",
  full.names = TRUE
)

# ============================================================
# Set output paths
# ============================================================
jorstad_h5ad_dir <- "/mnt/data0/home/wklei/docker_mount_flie/material/cell_single_data/Jorstad_et_al_h5ad"
siletti_h5ad_dir <- "/mnt/data0/home/wklei/docker_mount_flie/material/cell_single_data/original_data/siletti_material_Jorstad_transfer_h5ad"

dir.create(jorstad_h5ad_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(siletti_h5ad_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Convert Jorstad geneID Seurat objects to h5ad
# ============================================================
for (rds_file in jorstad_rds_path) {

  seurat_object <- readRDS(rds_file)

  out_file <- file.path(
    jorstad_h5ad_dir,
    gsub("\\.rds$", ".h5ad", basename(rds_file))
  )

  convertFormat(
    seurat_object,
    from = "seurat",
    to = "anndata",
    drop_single_values = FALSE,
    outFile = out_file
  )
}

# ============================================================
# Convert relabeled Siletti Seurat objects to h5ad
# ============================================================
for (rds_file in siletti_relabel_rds_path) {

  seurat_object <- readRDS(rds_file)

  out_file <- file.path(
    siletti_h5ad_dir,
    gsub("\\.rds$", ".h5ad", basename(rds_file))
  )

  convertFormat(
    seurat_object,
    from = "seurat",
    to = "anndata",
    drop_single_values = FALSE,
    outFile = out_file
  )
}