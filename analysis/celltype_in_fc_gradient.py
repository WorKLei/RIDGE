import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from neuromaps.stats import compare_images


# =========================
# Configuration
# =========================

SPIN_PATH = "analysis/ROI/spin_template/32k/32klh_spins_1000_32k_original.npy"
CELLTYPE_DIR = "project_whole_brain_mapping/brainmap_sm6_gii"
GRAD1_PATH = (
    "multi_maps/fslr32k/neuromaps/"
    "source-margulies2016_desc-fcgradient01_space-fsLR_den-32k_hemi-L_feature.func.gii"
)
GRAD2_PATH = "multi_maps/fslr32k/neuromaps/fs_LR_hemi-L_32k_FC_gradient2_func.gii"
OUTPUT_DIR = "project_whole_brain_mapping/final_code_to_supp_and_figure"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CORR_METHOD = "pearsonr"
EDGE_THRESHOLD = 0.6


# =========================
# Cell-type definitions
# =========================
LAYER_EX_CATEGORIES = [
    "L2/3 IT", "L4 IT", "L5 ET", "L5 IT", "L5/6 NP",
    "L6 CT", "L6 IT", "L6 IT Car3", "L6b"
]
LAYER_IN_CATEGORIES = [
    "Chandelier", "Lamp5", "Lamp5 Lhx6", "Pax6", "Pvalb",
    "Sncg", "Sst", "Sst Chodl", "Vip"
]
LAYER_NON_NEURON_CATEGORIES = [
    "Astro", "Micro/PVM", "OPC", "Oligo", "VLMC", "Endo"
]

ALL_ORIGINAL_CATEGORIES = (
    LAYER_EX_CATEGORIES + LAYER_IN_CATEGORIES + LAYER_NON_NEURON_CATEGORIES
)
ALL_FILE_CATEGORIES = [
    name.replace("/", "_").replace(" ", "_") for name in ALL_ORIGINAL_CATEGORIES
]

FOCUS_CELLS = [
    "L2/3 IT", "L4 IT", "L5 IT", "Vip",
    "L5 ET", "Pvalb", "Sst", "Chandelier", "Oligo"
]

SELECTED_9_CELLS = [
    "L2/3 IT", "L4 IT", "L5 IT", "Vip",
    "L5 ET", "Pvalb", "Sst", "Chandelier", "Oligo"
]

ALL_24_CELLS = ALL_ORIGINAL_CATEGORIES


# =========================
# Helper functions
# =========================
def sanitize_celltype_name(name):
    """Convert cell-type name to file-friendly format."""
    return name.replace("/", "_").replace(" ", "_")


def get_cell_group(celltype_name):
    """Return high-level group label for a cell type."""
    if celltype_name in LAYER_EX_CATEGORIES:
        return "Excitatory"
    if celltype_name in LAYER_IN_CATEGORIES:
        return "Inhibitory"
    return "Non-Neuron"


def get_node_color(celltype_name):
    """Return node color based on cell-type group."""
    if celltype_name in LAYER_EX_CATEGORIES:
        return "#E74C3C"
    if celltype_name in LAYER_IN_CATEGORIES:
        return "#3498DB"
    return "#2ECC71"


def load_gifti_data(path):
    """Load a GIFTI file and return its data as a NumPy array."""
    return nib.load(path).darrays[0].data.copy()


def load_celltype_map(celltype_name, celltype_dir):
    """Load a cell-type surface map."""
    file_name = sanitize_celltype_name(celltype_name)
    file_path = (
        f"{celltype_dir}/lh_{file_name}_"
        "fslr_each_region_ratioed_integrating_cell2map.func.gii"
    )
    return load_gifti_data(file_path)


def mask_zero_intersection(map_a, map_b):
    """Mask entries where either map is zero."""
    a = map_a.copy()
    b = map_b.copy()
    zero_mask = np.logical_or(a == 0, b == 0)
    a[zero_mask] = 0
    b[zero_mask] = 0
    return a, b


def generate_spin_nulls(data, spins):
    """Generate spin-based null maps."""
    rotated = np.zeros(spins.shape)
    for i in range(spins.shape[1]):
        rotated[:, i] = data[spins[:, i]]
    return rotated


def compute_gradient_correlations(celltype_names, gradient_data, spins, celltype_dir):
    """Compute spin-test correlations between cell-type maps and one gradient."""
    results = []

    for celltype_name in celltype_names:
        try:
            cell_data = load_celltype_map(celltype_name, celltype_dir)
            cell_data_masked, gradient_masked = mask_zero_intersection(cell_data, gradient_data)
            rotated_nulls = generate_spin_nulls(cell_data_masked, spins)

            corr, pval = compare_images(
                cell_data_masked,
                gradient_masked,
                nulls=rotated_nulls,
                metric=CORR_METHOD,
            )

            results.append({
                "Cell Type": celltype_name,
                "Correlation": corr,
                "P-value": pval,
                "Category_Group": get_cell_group(celltype_name),
            })

        except Exception as e:
            print(f"Error in {celltype_name}: {e}")

    return pd.DataFrame(results)


def load_multiple_cell_maps(celltype_names, celltype_dir):
    """Load multiple cell-type maps into a list."""
    maps = []
    for celltype_name in celltype_names:
        maps.append(np.nan_to_num(load_celltype_map(celltype_name, celltype_dir)))
    return maps


def compute_celltype_correlation_matrix(celltype_names, celltype_dir):
    """Compute pairwise Pearson correlation matrix across cell-type maps."""
    cell_data_list = load_multiple_cell_maps(celltype_names, celltype_dir)
    n_cells = len(celltype_names)
    corr_matrix = np.zeros((n_cells, n_cells))

    for i in range(n_cells):
        for j in range(i, n_cells):
            r, _ = pearsonr(cell_data_list[i], cell_data_list[j])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r

    return corr_matrix


def build_gradient_positions(celltype_names, df_grad1, df_grad2, flip_grad2=True):
    """Build 2D positions from Gradient 1 and Gradient 2 correlations."""
    pos = {}
    for i, label in enumerate(celltype_names):
        r_g1 = df_grad1.loc[df_grad1["Cell Type"] == label, "Correlation"].values[0]
        r_g2 = df_grad2.loc[df_grad2["Cell Type"] == label, "Correlation"].values[0]
        if flip_grad2:
            r_g2 = -r_g2
        pos[i] = (r_g2, r_g1)
    return pos


def plot_gradient_network(
    corr_mat,
    labels,
    pos_dict,
    threshold=0.6,
    focus_labels=None,
    output_path=None,
):
    """Plot cell types in gradient space with edges weighted by spatial similarity."""
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    graph = nx.Graph()

    if focus_labels is None:
        focus_labels = set()
    else:
        focus_labels = set(focus_labels)

    for i, label in enumerate(labels):
        graph.add_node(i, label=label)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if corr_mat[i, j] > threshold:
                graph.add_edge(i, j, weight=corr_mat[i, j])

    normal_edges = []
    normal_widths = []
    focus_edges = []
    focus_widths = []

    for u, v in graph.edges():
        label_u = labels[u]
        label_v = labels[v]
        width = graph[u][v]["weight"] ** 3 * 15

        if (label_u in focus_labels) and (label_v in focus_labels):
            focus_edges.append((u, v))
            focus_widths.append(width)
        else:
            normal_edges.append((u, v))
            normal_widths.append(width)

    if normal_edges:
        nx.draw_networkx_edges(
            graph,
            pos_dict,
            edgelist=normal_edges,
            width=normal_widths,
            edge_color="gray",
            alpha=0.18,
        )

    if focus_edges:
        nx.draw_networkx_edges(
            graph,
            pos_dict,
            edgelist=focus_edges,
            width=focus_widths,
            edge_color="#F39C12",
            alpha=0.5,
        )

    nx.draw_networkx_nodes(
        graph,
        pos_dict,
        node_size=1000,
        node_color=[get_node_color(label) for label in labels],
        edgecolors="white",
        linewidths=1.2,
    )

    for i, label in enumerate(labels):
        x, y = pos_dict[i]
        plt.text(
            x,
            y + 0.015,
            label,
            fontsize=18,
            ha="center",
            va="bottom",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    plt.xlabel("Spatial Correlation with Gradient 1 (r)", fontsize=22)
    plt.ylabel("Spatial Correlation with Gradient 2 (r)", fontsize=22)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Excitatory",
               markerfacecolor="#E74C3C", markersize=20),
        Line2D([0], [0], marker="o", color="w", label="Inhibitory",
               markerfacecolor="#3498DB", markersize=20),
        Line2D([0], [0], marker="o", color="w", label="Non-neuronal",
               markerfacecolor="#2ECC71", markersize=20),
        Line2D([0], [0], color="gray", lw=4, alpha=0.4, label="Other connections"),
    ]

    if focus_labels:
        legend_elements.append(
            Line2D([0], [0], color="#F39C12", lw=4, alpha=0.9, label="Focus-cell connections")
        )

    plt.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=18,
        bbox_to_anchor=(1.22, 1.05),
        frameon=False,
        labelspacing=1.2,
        handletextpad=0.5,
    )

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()


# =========================
# Load shared data
# =========================
lh_spins = np.load(SPIN_PATH)
grad1_data = load_gifti_data(GRAD1_PATH)
grad2_data = load_gifti_data(GRAD2_PATH)

# =========================
# Gradient correlation analysis
# =========================
df_results_g1 = compute_gradient_correlations(
    ALL_ORIGINAL_CATEGORIES,
    grad1_data,
    lh_spins,
    CELLTYPE_DIR,
)
df_results_g2 = compute_gradient_correlations(
    ALL_ORIGINAL_CATEGORIES,
    grad2_data,
    lh_spins,
    CELLTYPE_DIR,
)

# =========================
# 9-cell network plot
# =========================
corr_matrix_9 = compute_celltype_correlation_matrix(
    SELECTED_9_CELLS,
    CELLTYPE_DIR,
)
gradient_pos_9 = build_gradient_positions(
    SELECTED_9_CELLS,
    df_results_g1,
    df_results_g2,
    flip_grad2=True,
)

plot_gradient_network(
    corr_mat=corr_matrix_9,
    labels=SELECTED_9_CELLS,
    pos_dict=gradient_pos_9,
    threshold=EDGE_THRESHOLD,
    focus_labels=None,
    output_path=os.path.join(OUTPUT_DIR, "celltype_9_in_gradient.svg"),
)

# =========================
# 24-cell network plot
# =========================
corr_matrix_24 = compute_celltype_correlation_matrix(
    ALL_24_CELLS,
    CELLTYPE_DIR,
)
gradient_pos_24 = build_gradient_positions(
    ALL_24_CELLS,
    df_results_g1,
    df_results_g2,
    flip_grad2=True,
)

plot_gradient_network(
    corr_mat=corr_matrix_24,
    labels=ALL_24_CELLS,
    pos_dict=gradient_pos_24,
    threshold=EDGE_THRESHOLD,
    focus_labels=FOCUS_CELLS,
    output_path=os.path.join(OUTPUT_DIR, "celltype_24_in_gradient_focus_edge.svg"),
)