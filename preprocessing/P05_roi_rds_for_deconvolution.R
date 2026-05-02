# This script prepares the RData files for deconvolution and runs dtangle deconvolution for the whole-cortex MAGICC data.
#NOTE
# The whole-cortex gene expression map used here was generated with MAGICC,
# which is available at https://github.com/kwagstyl/MAGICC.
# Other cortical gene expression maps, such as those preprocessed with abagen,
# can also be used as input for the deconvolution analysis.

library(Seurat)
library(SingleCellExperiment)
library(gridExtra)
library(dplyr)
library(ggplot2)
library(ggridges)
library(GEOquery)
library(dtangle)
library(AnnotationDbi)
library(limma)

# ============================================================
# Set paths
# ============================================================
celltype_rds_dir <- "/mount_data_file/materials/integrated_whole_brain_celltype_rds/V4_0/"
C

preprocessed_out_dir <- "/mount_data_file/materials/deconvoluation_content/preprocessed_data_for_deconvolution/"
dtangle_result_dir <- "/mount_data_file/materials/deconvoluation_content/dtangle_result_wholecortex_MAGICC/"

dir.create(preprocessed_out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dtangle_result_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Read whole-cortex MAGICC matrix
# ============================================================
wholecortex_MAGICC <- read.csv(
  wholecortex_MAGICC_file,
  check.names = FALSE
)

rownames(wholecortex_MAGICC) <- wholecortex_MAGICC$gene.symbol
wholecortex_MAGICC <- wholecortex_MAGICC[, -1]

wholecortex_MAGICC_matrix <- as.matrix(
  wholecortex_MAGICC
)

well_id_labels <- colnames(
  wholecortex_MAGICC_matrix
)

# ============================================================
# Generate preprocessed RData files for deconvolution
# ============================================================
cell_atlas_rds_path <- list.files(
  path = celltype_rds_dir,
  pattern = "\\.rds$",
  full.names = TRUE
)

for (rds_file in cell_atlas_rds_path) {

  cell_atlas <- tools::file_path_sans_ext(
    basename(rds_file)
  )

  cell_atlas <- gsub(
    "^integrated_whole_counts_V4_0_",
    "",
    cell_atlas
  )

  region_cell_single <- readRDS(
    file = rds_file
  )

  common_genes <- intersect(
    rownames(wholecortex_MAGICC_matrix),
    rownames(region_cell_single)
  )

  region_cell_filtered_single <- subset(
    region_cell_single,
    features = common_genes
  )

  rm(region_cell_single)

  region_cell_filtered_single@assays[["RNA"]]@data <-
    region_cell_filtered_single@assays[["RNA"]]@counts

  region_cell_filtered_single <- NormalizeData(
    region_cell_filtered_single
  )

  region_cell_filtered_single <- ScaleData(
    region_cell_filtered_single,
    features = common_genes
  )

  cell_cluster_labels <- as.character(
    region_cell_filtered_single@meta.data[["CrossArea_subclass_edited"]]
  )

  cell_cluster_labels <- gsub(" ", "_", cell_cluster_labels)
  cell_cluster_labels <- gsub("/", "_", cell_cluster_labels)

  region_sample_filtered_AHBA_expression_matrix <-
    wholecortex_MAGICC_matrix[common_genes, , drop = FALSE]

  component_region_matrix <-
    region_cell_filtered_single@assays[["RNA"]]@scale.data[
      common_genes,
      ,
      drop = FALSE
    ]

  gc()

  out_file <- file.path(
    preprocessed_out_dir,
    paste0(
      cell_atlas,
      "_wholecortex_MAGICC_layer.RData"
    )
  )

  save(
    well_id_labels,
    component_region_matrix,
    region_sample_filtered_AHBA_expression_matrix,
    cell_cluster_labels,
    file = out_file
  )

  rm(
    region_cell_filtered_single,
    component_region_matrix,
    region_sample_filtered_AHBA_expression_matrix,
    cell_cluster_labels
  )

  gc()
}

# ============================================================
# Note
#baseline were anchored. In future analyses, the baseline may be adjusted according to empirical or
# expected true cell-type proportions.
# ============================================================

est_phats <- function(Y, markers, baseline_ests, gamma,
                      summary_fn = mean,
                      inv_scale = function(x) 2^x) {

  K <- length(markers)
  markers <- dtangle:::get_marker_list(markers)

  contrib_est <- function(i) {
    Y_i <- Y[, markers[[i]], drop = FALSE]
    baseline_adj <- sweep(Y_i, 2, baseline_ests[[i]])
    amt <- inv_scale(apply(baseline_adj / gamma, 1, summary_fn))
    return(amt)
  }

  contribs <- sapply(1:K, contrib_est)
  phats <- contribs
  colnames(phats) <- names(markers)

  return(phats)
}

modified_dtangle <- function(Y,
                             references = NULL,
                             pure_samples = NULL,
                             n_markers = NULL,
                             data_type = NULL,
                             gamma = NULL,
                             markers = NULL,
                             marker_method = "ratio",
                             summary_fn = mean) {

  stopifnot(all(n_markers > 0))
  stopifnot(!is.null(c(references, pure_samples)))

  cmbd <- dtangle:::combine_Y_refs(
    Y,
    references,
    pure_samples
  )

  Y <- cmbd$Y
  pure_samples <- cmbd$pure_samples

  prc <- dtangle:::process_markers(
    Y,
    pure_samples,
    n_markers,
    data_type,
    gamma,
    markers,
    marker_method
  )

  n_markers <- prc$n_markers
  mrkrs <- prc$mrkrs
  gamma <- prc$gamma

  baseline <- dtangle:::baseline_exprs(
    Y,
    pure_samples,
    mrkrs,
    summary_fn = summary_fn
  )

  names(baseline) <- names(pure_samples)

  # baseline <- lapply(baseline, function(x) {
  #   rep(0, length(x))
  # })

  phats <- est_phats(
    Y,
    mrkrs,
    baseline,
    gamma,
    summary_fn = summary_fn
  )
  return(list(
    estimates = phats,
    markers = mrkrs,
    n_markers = n_markers,
    gamma = gamma,
    marker_baseline = baseline
  ))
}

# ============================================================
# Run dtangle deconvolution
# ============================================================
rdata_files <- list.files(
  path = preprocessed_out_dir,
  pattern = "_wholecortex_MAGICC_layer\\.RData$",
  full.names = TRUE
)
for (files in rdata_files) {

  extracted_content <- gsub(
    ".*/(.*)_wholecortex_MAGICC_layer\\.RData$",
    "\\1",
    files
  )

  load(files)

  commongenes <- intersect(
    rownames(component_region_matrix),
    rownames(region_sample_filtered_AHBA_expression_matrix)
  )
  component_region_matrix <- component_region_matrix[
    commongenes,
    ,
    drop = FALSE
  ]
  region_sample_filtered_AHBA_expression_matrix <-
    region_sample_filtered_AHBA_expression_matrix[
      commongenes,
      ,
      drop = FALSE
    ]

  sce <- component_region_matrix
  me <- region_sample_filtered_AHBA_expression_matrix

  all_cell_type <- unique(
    cell_cluster_labels
  )

  y <- cbind(
    sce,
    me
  )

  y <- normalizeBetweenArrays(
    y
  )

  y <- t(
    y
  )

  y_1 <- normalizeBetweenArrays(
    me
  )

  y_1 <- t(
    y_1
  )

  references <- t(
    sce
  )

  pure_samples <- lapply(
    all_cell_type,
    function(ct) {
      which(cell_cluster_labels == ct)
    }
  )

  names(pure_samples) <- all_cell_type

  marker_list <- find_markers(
    y,
    pure_samples = pure_samples,
    data_type = "rna-seq",
    marker_method = "ratio"
  )

  q <- 0.1

  quantiles <- lapply(
    marker_list$V,
    function(x) {
      quantile(x, 1 - q)
    }
  )

  K <- length(
    pure_samples
  )

  n_markers <- sapply(
    1:K,
    function(i) {
      max(which(marker_list$V[[i]] > quantiles[[i]]))
    }
  )

  marks <- marker_list$L

  dc <- modified_dtangle(
    y_1,
    references = references,
    pure_samples = pure_samples,
    n_markers = n_markers,
    data_type = "microarray-gene",
    markers = marks
  )

  final_est <- dc$estimates[
    (ncol(sce) + 1):(ncol(sce) + nrow(y_1)),
    ,
    drop = FALSE
  ]

  colnames(final_est) <- all_cell_type
  rownames(final_est) <- well_id_labels

  dtangle_path <- file.path(
    dtangle_result_dir,
    paste0(
      extracted_content,
      "_wholecortex_MAGICC_deconvlution.csv"
    )
  )

  write.csv(
    final_est,
    dtangle_path
  )

  rm(
    component_region_matrix,
    region_sample_filtered_AHBA_expression_matrix,
    sce, me, y, y_1, references, pure_samples, marker_list, quantiles, n_markers, marks, dc, final_est
  )

  gc()
}

gc()
