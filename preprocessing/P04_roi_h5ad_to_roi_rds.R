# ============================================================
#Convert ROI-level h5ad files to RDS files
# ------------------------------------------------------------
# The integrated dataset has been split into ROI-level .h5ad files.
# This script converts each roi.h5ad file into a corresponding roi.rds file
# for downstream analysis in Seurat/R.
#
# Input : ROI-level .h5ad files
# Output: ROI-level .rds files
#
# Usage:
# 1. Set base_input_path to the folder containing ROI-level .h5ad files.
# 2. Set base_output_path to the folder where converted .rds files will be saved.
# 3. Run this script in R after loading sceasy and reticulate.
# ============================================================
library(sceasy)
library(reticulate)

data_version <- "V4_0"

base_input_path <- "/mount_data_file/materials/integrated_whole_brain_celltype_h5ad/V4_0"
base_output_path <- "/mount_data_file/materials/integrated_whole_brain_celltype_rds/V4_0"

input_dir <- file.path(base_input_path)
output_dir <- file.path(base_output_path)

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  message(sprintf("Output directory created: %s", output_dir))
}

h5ad_files <- list.files(
  path = input_dir,
  pattern = "\\.h5ad$",
  full.names = TRUE
)

if (length(h5ad_files) == 0) {
  stop(sprintf(
    "Error: No .h5ad files found in directory: %s. Please check the path.",
    input_dir
  ))
}

message(sprintf(
  "Version: %s | Total files to process: %d",
  data_version,
  length(h5ad_files)
))

for (h5ad_file in h5ad_files) {

  file_name <- basename(h5ad_file)
  base_name <- tools::file_path_sans_ext(file_name)

  rds_file <- file.path(output_dir, paste0(base_name, ".rds"))

  message(sprintf("\n[Processing] %s ...", file_name))

  tryCatch({

    convertFormat(
      h5ad_file,
      from = "anndata",
      to = "seurat",
      main_layer = "counts",
      outFile = rds_file
    )

    message(sprintf("[Success] Saved to: %s", rds_file))

  }, error = function(e) {

    message(sprintf(
      "[Failed] Error converting file %s: %s",
      file_name,
      e$message
    ))
  })
}

message(sprintf("\nAll %s file processing completed.", data_version))