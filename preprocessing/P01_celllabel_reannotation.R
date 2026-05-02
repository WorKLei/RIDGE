#P01. Cell label reannotation for Siletti et al. dataset using Jorstad et al. dataset as reference
#Notes：
# For Siletti regions that anatomically correspond to a Jorstad ROI,
# the matched Jorstad ROI prediction is used directly. This matching
# is determined from the Siletti file name using predefined ROI
# patterns. If no anatomical ROI match is detected, the consensus
# label is used instead.

library(Seurat)
library(SeuratObject)
library(dplyr)
# ============================================================
# Set paths
# ============================================================
refer_rds_path <- list.files(
  path = "/mnt/data0/home/wklei/docker_mount_flie/material/cell_single_data/Jorstad_et_al/",
  pattern = "Jorstad_et_al_.*geneID\\.rds$",
  full.names = TRUE
)
query_rds_path <- list.files(
  path = "/mnt/data0/home/wklei/docker_mount_flie/material/cell_single_data/filter_data/siletti_material",
  pattern = "siletti_.*\\.rds$",
  full.names = TRUE
)

out_dir <- "/mnt/data0/home/wklei/docker_mount_flie/material/cell_single_data/filter_data/siletti_material_Jorstad_transfer"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Set consensus rule
# Options:
# "majority": assign label if one label appears in more than 50% of references
# "all": assign label only if all references give the same label
# ============================================================
consensus_rule <- "majority"

# ============================================================
# anatomical ROI matching rule
# query_pattern is detected from the Siletti
# reference_suffix is the corresponding Jorstad ROI suffix
# ============================================================
roi_match_table <- data.frame(
  reference_suffix = c("ACC", "A1",  "M1",  "S1",  "MTG", "V1"),
  query_pattern    = c("A24", "A1C", "M1C", "S1C", "MTG", "V1C"),
  stringsAsFactors = FALSE
)

# ============================================================
# Run label transfer for each Siletti query dataset
# ============================================================
for (query_rds_chosen in query_rds_path) {

  query_rds_full <- readRDS(query_rds_chosen)
  prediction_cols <- c()
  for (refer_rds_chosen in refer_rds_path) {
    extracted_content <- gsub(
      ".*/Jorstad_et_al_(.*)_geneID\\.rds$",
      "\\1",
      refer_rds_chosen
    )
    CrossArea_subclass_NAME <- paste0("CrossArea_subclass_", extracted_content)
    region_cell_single <- readRDS(refer_rds_chosen)
    common_genes <- intersect(
      rownames(query_rds_full),
      rownames(region_cell_single)
    )
    region_cell_single <- region_cell_single[common_genes, ]
    query_rds_transfer <- query_rds_full[common_genes, ]
    # Normalize and reduce the reference dataset
    region_cell_single <- NormalizeData(region_cell_single)
    region_cell_single <- FindVariableFeatures(
      region_cell_single,
      nfeatures = 3000
    )
    region_cell_single <- ScaleData(region_cell_single)
    region_cell_single <- RunPCA(region_cell_single, npcs = 30)
    query_rds_transfer <- NormalizeData(query_rds_transfer)
    cell_anchors <- FindTransferAnchors(
      reference = region_cell_single,
      query = query_rds_transfer,
      dims = 1:30,
      reference.reduction = "pca"
    )
    # Transfer subclass labels from the reference to the query
    predictions <- TransferData(
      anchorset = cell_anchors,
      refdata = region_cell_single$CrossArea_subclass,
      dims = 1:30
    )
    predicted_id <- predictions$predicted.id
    names(predicted_id) <- rownames(predictions)
    query_rds_full <- AddMetaData(
      object = query_rds_full,
      metadata = predicted_id,
      col.name = CrossArea_subclass_NAME
    )
    prediction_cols <- c(prediction_cols, CrossArea_subclass_NAME)
  }
  # ==========================================================
  # Generate consensus labels based on all transferred labels
  # ==========================================================
  pred_df <- query_rds_full@meta.data[, prediction_cols, drop = FALSE]
  pred_df <- as.data.frame(
    lapply(pred_df, as.character),
    stringsAsFactors = FALSE
  )
  CrossArea_subclass_consensus <- apply(pred_df, 1, function(x) {

    x <- x[!is.na(x) & x != ""]
    if (length(x) == 0) {
      return(NA_character_)
    }
    tb <- sort(table(x), decreasing = TRUE)
    top_label <- names(tb)[1]
    top_n <- as.numeric(tb[1])
    total_n <- length(x)
    if (consensus_rule == "all") {
      if (top_n == total_n) {
        return(top_label)
      } else {
        return(NA_character_)
      }
    } else if (consensus_rule == "majority") {

      if (top_n > total_n * 0.5) {
        return(top_label)
      } else {
        return(NA_character_)
      }

    } else {
      stop("Invalid consensus_rule. Use 'majority' or 'all'.")
    }
  })
  query_rds_full$CrossArea_subclass_consensus <- CrossArea_subclass_consensus

  # ==========================================================
  # Apply anatomical ROI matching rule
  # If the Siletti file name matches a predefined ROI pattern,
  # use the corresponding Jorstad ROI prediction directly
  # ==========================================================
  query_file_name <- basename(query_rds_chosen)
  matched_reference_suffix <- NA_character_
  for (i in seq_len(nrow(roi_match_table))) {

    if (grepl(roi_match_table$query_pattern[i], query_file_name)) {
      matched_reference_suffix <- roi_match_table$reference_suffix[i]
      break
    }
  }

  matched_prediction_col <- paste0(
    "CrossArea_subclass_",
    matched_reference_suffix
  )

  if (
    !is.na(matched_reference_suffix) &&
    matched_prediction_col %in% colnames(query_rds_full@meta.data)
  ) {

    query_rds_full$CrossArea_subclass_edited <- query_rds_full@meta.data[[matched_prediction_col]]

  } else {
    query_rds_full$CrossArea_subclass_edited <- query_rds_full$CrossArea_subclass_consensus
  }
  # ==========================================================
  # Save updated query object
  # ==========================================================
  out_file <- file.path(
    out_dir,
    basename(query_rds_chosen)
  )
  saveRDS(query_rds_full, file = out_file)
}