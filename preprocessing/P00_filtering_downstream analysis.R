#P0 -1. Filter the data for downstream analysis
# remove low-quality cells and rare genes, and remove miscellaneous or non-target cell populations
#Note: Abnormal samples have already been removed from the original data.

library(Seurat)
base_input_path <- "/mnt/data0/home/wklei/docker_mount_flie/material/cell_single_data/origin_data/siletti_material"
base_output_path <- "/mnt/data0/home/wklei/docker_mount_flie/material/cell_single_data/filter_data/siletti_material"
input_dir <- file.path(base_input_path)
output_dir <- file.path(base_output_path)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

file_list <- list.files(
  path = input_dir,
  pattern = "\\.rds$",
  full.names = TRUE
)
# Miscellaneous clusters include L5 ET-related cell types.
# These cluster IDs were selected according to the Siletti et al. supplementary materials.
miscellaneous_cluster_id_list <- c(
  "0", "1", "2", "3",
  "164", "165", "166", "167",
  "168", "170", "176", "177",
  "178", "401", "404"
)
# These supercluster terms were excluded because they do not correspond to
# cortical cell types. In addition, some of these cell types showed low cortical
# representation, with less than 50% of cells located in the cortex according to the siletti et al. supplementary materials.
supercluster_term_list <- c(
  "Hippocampal CA1-3",
  "Eccentric medium spiny neuron",
  "Medium spiny neuron",
  "Upper rhombic lip",
  "Thalamic excitatory",
  "Hippocampal CA4",
  "Lower rhombic lip",
  "Hippocampal dentate gyrus",
  "Midbrain-derived inhibitory",
  "Cerebellar inhibitory",
  "Mammillary body",
  "Choroid plexus",
  "Ependymal"
)

for (file in file_list) {

  file_name <- basename(file)
  print(paste("Processing", file_name))

  rds_data <- readRDS(file)


  # Remove low-quality cells and rare genes
  counts_mat <- GetAssayData(
    object = rds_data,
    assay = "RNA",
    slot = "counts"
  )
  rds_data <- CreateSeuratObject(
    counts = counts_mat,
    meta.data = rds_data@meta.data,
    min.cells = 5,
    min.features = 100
  )
  # Remove miscellaneous or non-target cell populations
  rds_data <- rds_data[
    ,
    !rds_data$cluster_id %in% miscellaneous_cluster_id_list
  ]
  rds_data <- rds_data[
    ,
    !rds_data$supercluster_term %in% supercluster_term_list
  ]
  output_file <- file.path(output_dir, file_name)
  saveRDS(rds_data, output_file)
  print(paste("Saved", output_file))
  rm(rds_data, counts_mat)
  gc()
}