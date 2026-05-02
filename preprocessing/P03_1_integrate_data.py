import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.path import Path


# ============================================================
# Set paths
# ============================================================
input_directory = "/mnt/disk1/docker_mount_file/materials/h5ad_v4_0"

merged_output_file = (
    "/mnt/disk1/docker_mount_file/materials/"
    "integrated_whole_counts_data_V4_0.h5ad"
)

filtered_output_file = (
    "/mnt/disk1/docker_mount_file/materials/"
    "integrated_whole_counts_data_V4_0_filtered.h5ad"
)

roi_count_output_file = (
    "/mnt/disk1/docker_mount_file/materials/"
    "siletti_filtered_roi_celltype_counts.csv"
)

figure_dir = "/mnt/leiwk/neuro_work/V4_0_code/results/figures"
os.makedirs(figure_dir, exist_ok=True)

sc.settings.figdir = figure_dir
sc.set_figure_params(dpi=300, dpi_save=300)


# ============================================================
# Merge h5ad files using common genes
# ============================================================
h5ad_files = glob.glob(os.path.join(input_directory, "*.h5ad"))

if len(h5ad_files) == 0:
    raise FileNotFoundError(f"No h5ad files found in {input_directory}")

adatas = []
gene_sets = []

for i, file in enumerate(h5ad_files):

    adata = sc.read_h5ad(file)

    # Make cell names unique across datasets
    adata.obs_names = [
        f"{cell}_batch{i}"
        for cell in adata.obs_names
    ]

    # Convert metadata column to string if present
    if "development_stage" in adata.obs.columns:
        adata.obs["development_stage"] = (
            adata.obs["development_stage"].astype(str)
        )

    adatas.append(adata)
    gene_sets.append(set(adata.var_names))

common_genes = sorted(set.intersection(*gene_sets))

if len(common_genes) == 0:
    raise ValueError("No common genes found across h5ad files.")

adatas = [
    adata[:, common_genes].copy()
    for adata in adatas
]

data_combined = sc.concat(
    adatas,
    join="inner",
    merge="unique"
)

# ============================================================
# Basic preprocessing and Harmony
# ============================================================
data_combined.layers["counts"] = data_combined.X.copy()

sc.pp.normalize_total(data_combined)
sc.pp.log1p(data_combined)
sc.pp.highly_variable_genes(
    data_combined,
    n_top_genes=3000
)

sc.tl.pca(data_combined)

# Run Harmony integration using dataset and sample information
sce.pp.harmony_integrate(
    data_combined,
    key=["batch_harmony", "sample_id"]
)
# Build neighborhood graph using Harmony-corrected PCA space
sc.pp.neighbors(
    data_combined,
    use_rep="X_pca_harmony"
)

sc.tl.umap(data_combined)
# Save integrated data after Harmony
data_combined.write_h5ad(merged_output_file)
sc.pl.umap(
    data_combined,
    color="batch_harmony",
    size=2,
    save="_integrated_whole_counts_data_V4_0_harmony_batch.pdf"
)
sc.pl.umap(
    data_combined,
    color="CrossArea_subclass_edited",
    size=2,
    save="_integrated_whole_counts_data_V4_0_harmony_celltype.pdf"
)

# ============================================================
# This section is kept as comments only. It can be uncommented
# if marker-gene dotplot visualization is needed later.
# ============================================================
# marker_select = {
#     'L2/3 IT': ['CUX2', 'TESPA1', 'CBLN2'],
#     'L4 IT': ['GRM7', 'GRIK4', 'SATB2', 'POU6F2'],
#     'L5 IT': ['RORB', 'TLL1', 'LRRK1', 'HS3ST2'],
#     'L6 IT': ['THEMIS', 'ADAMTS3'],
#     'L6 IT Car3': ['GNAO1', 'AK5', 'CADPS2'],
#     'L5 ET': ['COL5A2', 'CLEC2L'],
#     'L5/6 NP': ['SEMA5B', 'CPNE4', 'SATB1'],
#     'L6 CT': ['FOXP2', 'PRKCB', 'ATP2B1'],
#     'L6b': ['SLIT3', 'SEMA3A', 'CELF2'],
#     'Pvalb': ['PVALB'],
#     'Sst': ['ROBO1', 'STXBP6', 'NRXN3'],
#     'Vip': ['ADARB2', 'CSMD1', 'SNTG1', 'OPCML'],
#     'Pax6': ['PLCXD3', 'PAX6', 'NRXN3'],
#     'Sncg': ['INPP4B', 'CHRNA7', 'ADRA1A', 'SDK1'],
#     'Chandelier': ['THSD7A', 'UNC5B', 'NACC2', 'PLEKHH2', 'C1QL1', 'COL15A1'],
#     'Sst Chodl': ['GRIK1', 'NXPH1', 'GRIA1', 'XKR4'],
#     'Lamp5': ['LAMP5'],
#     'Lamp5 Lhx6': ['FGF13', 'EYA4', 'SV2C', 'PDGFD'],
#     'Astro': ['SLC1A2', 'GFAP', 'AQP4'],
#     'Micro/PVM': ['SPP1', 'APBB1IP', 'CD74'],
#     'Oligo': ['PLP1', 'MOG', 'MAG'],
#     'Microglia': ['CD68', 'CX3CR1', 'P2RY12', 'TMEM119'],
#     'OPC': ['PDGFRA', 'OLIG1', 'OLIG2', 'SOX10'],
#     'VLMC': ['LAMA2', 'ATP1A2', 'RBMS3'],
#     'Endo': ['FLT1', 'CLDN5', 'ABCG2', 'EBF1', 'LEF1']
# }
#
# order = [
#     "L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L6 IT Car3",
#     "L5 ET", "L5/6 NP", "L6 CT", "L6b",
#     "Pvalb", "Sst", "Vip", "Pax6", "Sncg",
#     "Lamp5", "Lamp5 Lhx6", "Chandelier", "Sst Chodl",
#     "Astro", "Micro/PVM", "Oligo", "OPC", "VLMC", "Endo"
# ]
# markersDictSorted = {
#     k: marker_select[k]
#     for k in order
# }
# plt.rcParams.update({
#     'font.size': 20,
#     'font.weight': 'normal',
#     'legend.fontsize': 10,
#     'legend.handlelength': 0.7,
#     'legend.labelspacing': 1
# })
# sc.pl.dotplot(
#     data_combined,
#     markersDictSorted,
#     groupby='CrossArea_subclass_edited',
#     categories_order=order,
#     save='AdultCelldotplot_revised_bold.pdf',
#     colorbar_title='Mean\nexpression',
#     size_title='Fraction\nof cells'
# )

# ============================================================
# Filter Siletti cells based on valid Siletti-Jorstad label mapping
# ============================================================
data_combined_siletti = data_combined[
    data_combined.obs["batch_harmony"] == "Siletti_whole_brain"
].copy()

target_combinations = [
    ("Amygdala excitatory", "L2/3 IT"),
    ("Astrocyte", "Astro"),
    ("CGE interneuron", "Lamp5"),
    ("CGE interneuron", "Lamp5 Lhx6"),
    ("CGE interneuron", "Pax6"),
    ("CGE interneuron", "Sncg"),
    ("CGE interneuron", "Vip"),
    ("Committed oligodendrocyte precursor", "OPC"),
    ("Committed oligodendrocyte precursor", "Oligo"),
    ("Deep-layer corticothalamic and 6b", "L6 CT"),
    ("Deep-layer corticothalamic and 6b", "L6b"),
    ("Deep-layer corticothalamic and 6b", "L5/6 NP"),
    ("Deep-layer intratelencephalic", "L5 IT"),
    ("Deep-layer intratelencephalic", "L6 IT"),
    ("Deep-layer intratelencephalic", "L6 IT Car3"),
    ("Deep-layer near-projecting", "L5/6 NP"),
    ("Fibroblast", "VLMC"),
    ("Fibroblast", "Endo"),
    ("LAMP5-LHX6 and Chandelier", "Chandelier"),
    ("LAMP5-LHX6 and Chandelier", "Lamp5 Lhx6"),
    ("MGE interneuron", "Pvalb"),
    ("MGE interneuron", "Sst"),
    ("MGE interneuron", "Sst Chodl"),
    ("Microglia", "Micro/PVM"),
    ("Miscellaneous", "L5 ET"),
    ("Oligodendrocyte", "Oligo"),
    ("Oligodendrocyte precursor", "OPC"),
    ("Splatter", "L4 IT"),
    ("Splatter", "Sst Chodl"),
    ("Upper-layer intratelencephalic", "L2/3 IT"),
    ("Upper-layer intratelencephalic", "L4 IT"),
    ("Vascular", "Endo"),
    ("Vascular", "VLMC"),
]

target_set = set(target_combinations)

valid_siletti_obs = data_combined_siletti.obs[
    data_combined_siletti.obs[
        ["supercluster_term", "CrossArea_subclass_edited"]
    ].apply(tuple, axis=1).isin(target_set)
]

invalid_siletti_cells = data_combined_siletti.obs.index.difference(
    valid_siletti_obs.index
)

data_combined_filtered = data_combined[
    ~data_combined.obs.index.isin(invalid_siletti_cells)
].copy()

filtered_data = data_combined_siletti[
    valid_siletti_obs.index
].copy()


# ============================================================
# Save filtered integrated data
# ============================================================
data_combined_filtered.write_h5ad(filtered_output_file)


# ============================================================
# Plot filtered UMAP
# ============================================================
sc.pl.umap(
    data_combined_filtered,
    color="batch_harmony",
    size=2,
    save="_integrated_whole_counts_data_V4_0_filtered_harmony_batch.pdf"
)

sc.pl.umap(
    data_combined_filtered,
    color="CrossArea_subclass_edited",
    size=2,
    save="_integrated_whole_counts_data_V4_0_filtered_harmony_celltype.pdf"
)


# ============================================================
# Optional marker-gene ranking for filtered integrated data
# ============================================================
# sc.tl.rank_genes_groups(
#     data_combined_filtered,
#     groupby="CrossArea_subclass_edited",
#     method="wilcoxon"
# )
#
# sc.pl.rank_genes_groups_dotplot(
#     data_combined_filtered,
#     groupby="CrossArea_subclass_edited",
#     standard_scale="var",
#     n_genes=5,
#     save="_integrated_whole_counts_data_V4_0_filtered_rank_gene_dotplot.pdf"
# )


# ============================================================
# Save ROI-by-cell-type count table for filtered Siletti cells
# ============================================================
count_df = pd.crosstab(
    filtered_data.obs["roi"],
    filtered_data.obs["CrossArea_subclass_edited"]
)

count_df.to_csv(roi_count_output_file)


# ============================================================
# Define compact two-level Sankey plotting function
# ============================================================
def plot_two_level_sankey(
    obs,
    source_col,
    target_col,
    output_file,
    figsize=(10, 18),
    min_node_value=None,
    label_mode="inside",
    node_width=0.04,
    node_pad=0.008,
    curve=0.35,
    font_size=12,
    flow_alpha=0.6
):
    flow_counts = (
        obs.groupby([source_col, target_col])
        .size()
        .reset_index(name="count")
    )

    flow_counts = flow_counts[flow_counts["count"] > 0].copy()

    source_values = obs[source_col].value_counts()
    target_values = obs[target_col].value_counts()

    source_labels = source_values.index.tolist()
    target_labels = target_values.index.tolist()

    source_colors = dict(
        zip(
            source_labels,
            sns.color_palette("tab20", len(source_labels))
        )
    )

    target_colors = dict(
        zip(
            target_labels,
            sns.color_palette("tab20", len(target_labels))
        )
    )

    source_display_values = source_values.copy()
    target_display_values = target_values.copy()

    if min_node_value is not None:
        source_display_values = source_display_values.apply(
            lambda x: max(x, min_node_value)
        )
        target_display_values = target_display_values.apply(
            lambda x: max(x, min_node_value)
        )

    def compute_node_layout(values):
        n_nodes = len(values)
        total_value = values.sum()
        total_pad = node_pad * max(n_nodes - 1, 0)
        scale = total_value / (1 - total_pad)

        layout = {}
        y_top = 1.0

        for label, value in values.items():
            height = value / scale
            y_bottom = y_top - height

            layout[label] = {
                "y0": y_bottom,
                "y1": y_top,
                "height": height,
                "value": value
            }

            y_top = y_bottom - node_pad

        return layout, scale

    source_layout, source_scale = compute_node_layout(
        source_display_values
    )

    target_layout, target_scale = compute_node_layout(
        target_display_values
    )

    scale = max(source_scale, target_scale)

    source_cursor = {
        label: source_layout[label]["y1"]
        for label in source_labels
    }

    target_cursor = {
        label: target_layout[label]["y1"]
        for label in target_labels
    }

    fig, ax = plt.subplots(figsize=figsize)

    x_source = 0.0
    x_target = 1.0

    # Draw flows
    for _, row in flow_counts.iterrows():

        source = row[source_col]
        target = row[target_col]
        value = row["count"]

        flow_height = value / scale

        y_source_top = source_cursor[source]
        y_source_bottom = y_source_top - flow_height
        source_cursor[source] = y_source_bottom

        y_target_top = target_cursor[target]
        y_target_bottom = y_target_top - flow_height
        target_cursor[target] = y_target_bottom

        x1 = x_source + node_width
        x2 = x_target

        c1 = x1 + curve * (x2 - x1)
        c2 = x2 - curve * (x2 - x1)

        path = Path(
            [
                (x1, y_source_top),
                (c1, y_source_top),
                (c2, y_target_top),
                (x2, y_target_top),
                (x2, y_target_bottom),
                (c2, y_target_bottom),
                (c1, y_source_bottom),
                (x1, y_source_bottom),
                (x1, y_source_top)
            ],
            [
                Path.MOVETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.LINETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CLOSEPOLY
            ]
        )

        if source_values[source] <= target_values[target]:
            flow_color = source_colors[source]
        else:
            flow_color = target_colors[target]

        flow_color = mcolors.to_rgba(
            flow_color,
            alpha=flow_alpha
        )

        patch = mpatches.PathPatch(
            path,
            facecolor=flow_color,
            edgecolor="none"
        )

        ax.add_patch(patch)

    # Draw source nodes
    for label in source_labels:

        y0 = source_layout[label]["y0"]
        height = source_layout[label]["height"]

        rect = mpatches.Rectangle(
            (x_source, y0),
            node_width,
            height,
            facecolor=source_colors[label],
            edgecolor="none"
        )

        ax.add_patch(rect)

        if label_mode == "inside":
            ax.text(
                x_source + node_width + 0.01,
                y0 + height / 2,
                label,
                ha="left",
                va="center",
                fontsize=font_size
            )
        else:
            ax.text(
                x_source - 0.01,
                y0 + height / 2,
                label,
                ha="right",
                va="center",
                fontsize=font_size
            )

    # Draw target nodes
    for label in target_labels:

        y0 = target_layout[label]["y0"]
        height = target_layout[label]["height"]

        rect = mpatches.Rectangle(
            (x_target, y0),
            node_width,
            height,
            facecolor=target_colors[label],
            edgecolor="none"
        )

        ax.add_patch(rect)

        if label_mode == "inside":
            ax.text(
                x_target - 0.01,
                y0 + height / 2,
                label,
                ha="right",
                va="center",
                fontsize=font_size
            )
        else:
            ax.text(
                x_target + node_width + 0.01,
                y0 + height / 2,
                label,
                ha="left",
                va="center",
                fontsize=font_size
            )

    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )

    plt.close()
# ============================================================
# Plot Siletti-to-Jorstad Sankey diagram
# ============================================================
plot_two_level_sankey(
    obs=filtered_data.obs,
    source_col="supercluster_term",
    target_col="CrossArea_subclass_edited",
    output_file=os.path.join(
        figure_dir,
        "siletti_to_jorstad_celltypelabel.pdf"
    ),
    figsize=(10, 18),
    label_mode="inside",
    font_size=12
)

plot_two_level_sankey(
    obs=filtered_data.obs,
    source_col="supercluster_term",
    target_col="CrossArea_subclass_edited",
    output_file=os.path.join(
        figure_dir,
        "siletti_to_jorstad_celltypelabel_labelout.pdf"
    ),
    figsize=(10, 20),
    min_node_value=2000,
    label_mode="outside",
    font_size=18
)


# ============================================================
# Plot Siletti-to-Jorstad cell-type proportion within each ROI
# ============================================================
jorstad_order = [
    "L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L6 IT Car3",
    "L5 ET", "L5/6 NP", "L6 CT", "L6b",
    "Pvalb", "Sst", "Vip", "Pax6", "Sncg",
    "Lamp5", "Lamp5 Lhx6", "Chandelier", "Sst Chodl",
    "Astro", "Micro/PVM", "Oligo", "OPC", "VLMC", "Endo"
]

siletti_order = [
    "Amygdala excitatory",
    "Upper-layer intratelencephalic",
    "Deep-layer intratelencephalic",
    "Miscellaneous",
    "Deep-layer near-projecting",
    "Deep-layer corticothalamic and 6b",
    "MGE interneuron",
    "CGE interneuron",
    "LAMP5-LHX6 and Chandelier",
    "Splatter",
    "Astrocyte",
    "Microglia",
    "Oligodendrocyte",
    "Committed oligodendrocyte precursor",
    "Oligodendrocyte precursor",
    "Fibroblast",
    "Vascular"
]

color_bar = [
    "#f59b00", "#f6a113", "#f6a627", "#f7ac3a", "#f7b24d",
    "#f8b760", "#f8bd74", "#f9c287", "#f9c89a",
    "#1297fd", "#23a0fd", "#34a8fd", "#46b1fd", "#57b9fd",
    "#68c2fe", "#79cbfe", "#8ad3fe", "#9cdcfe",
    "#3b910d", "#4b9b20", "#5aa532", "#6aaf45", "#79ba58",
    "#89c46a"
]

for roi_name in filtered_data.obs["roi"].unique():

    filtered_data_roi = filtered_data[
        filtered_data.obs["roi"] == roi_name
    ].copy()

    counts = (
        filtered_data_roi.obs
        .groupby(["supercluster_term", "CrossArea_subclass_edited"])
        .size()
    )

    total_counts = (
        filtered_data_roi.obs
        .groupby("supercluster_term")
        .size()
    )

    proportions = counts / total_counts

    proportions_df = proportions.reset_index(name="proportion")

    proportions_df_filtered = proportions_df[
        proportions_df["proportion"] > 0
    ].copy()

    proportions_df_filtered["CrossArea_subclass_edited"] = pd.Categorical(
        proportions_df_filtered["CrossArea_subclass_edited"],
        categories=jorstad_order,
        ordered=True
    )

    proportions_df_filtered["supercluster_term"] = pd.Categorical(
        proportions_df_filtered["supercluster_term"],
        categories=siletti_order,
        ordered=True
    )

    proportions_df_filtered = proportions_df_filtered.rename(
        columns={
            "CrossArea_subclass_edited": "Jorstad Cell types",
            "supercluster_term": "Siletti Cell types"
        }
    )

    sns.set(style="whitegrid")

    plt.figure(figsize=(15, 10))

    sns.scatterplot(
        data=proportions_df_filtered,
        x="Jorstad Cell types",
        y="Siletti Cell types",
        size="proportion",
        hue="Jorstad Cell types",
        sizes=(20, 400),
        legend=True,
        alpha=0.8,
        palette=color_bar
    )

    plt.grid(
        linestyle="--",
        linewidth=1
    )

    plt.title(
        f"Cell type proportion in {roi_name}",
        fontsize=20,
        pad=20
    )

    plt.xlabel(
        "Jorstad Cell types",
        fontsize=14
    )

    plt.ylabel(
        "Siletti Cell types",
        fontsize=14
    )

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=12
    )

    plt.yticks(fontsize=12)

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        fontsize=10,
        title=None
    )

    plt.tight_layout()

    safe_roi_name = (
        str(roi_name)
        .replace("/", "_")
        .replace(" ", "_")
    )

    plt.savefig(
        os.path.join(
            figure_dir,
            f"siletti_to_jorstad_proportion_in_{safe_roi_name}.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )

    plt.close()