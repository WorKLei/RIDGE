import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from matplotlib.colors import LinearSegmentedColormap
from palettable.colorbrewer.sequential import GnBu_8, YlOrRd_6, Blues_9, Oranges_9


# =========================
# Configuration
# =========================
PATH = "project_whole_brain_mapping/final_code"
OUTPUT_DIR = os.path.join(PATH, "results/visual/figure")
NULL_DATA_PATH = f"{PATH}/materials/visual"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CELLTYPES = [
    "L2/3 IT", "L4 IT", "L5 IT", "L6 IT", "L6 IT Car3", "L5 ET", "L6b", "L6 CT", "L5/6 NP",
    "Pvalb", "Chandelier", "Sst", "Sst Chodl", "Pax6", "Lamp5", "Vip", "Sncg", "Lamp5 Lhx6",
    "Oligo", "OPC", "Astro", "Endo", "VLMC", "Micro/PVM"
]


# =========================
# Data loading
# =========================
def load_data():
    """Load visual-stream vertex indices and corresponding cell-type data."""
    indices = np.load(f"{PATH}/materials/visual/lh.visual_stream_indices.npy")
    df = pd.read_csv(f"{PATH}/results/celltype_map/lh_fslr32k_zscored_24celltype.csv")
    visual_df = df[df["label"].isin(indices)].reset_index(drop=True)
    return visual_df, indices


# =========================
# PCA analysis
# =========================
def run_pca(df, celltypes):
    """Run PCA on transposed cell-type data within the visual stream."""
    scaler = StandardScaler()
    X = df[celltypes].values.T
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=5)
    pca.fit(X_scaled)

    scores = pca.transform(-X_scaled)
    return pca, scores


# =========================
# Plotting functions
# =========================
def plot_explained_variance(pca):
    """Plot explained variance ratios for the first five principal components."""
    variance_ratios = pca.explained_variance_ratio_[:5]
    x_labels = [f"PC{i + 1}" for i in range(len(variance_ratios))]
    plot_data = pd.DataFrame({"PC": x_labels, "Variance Ratio": variance_ratios})

    point_colors = ["skyblue", "#fc8d59"]

    plt.figure(figsize=(5, 5), dpi=300)
    sns.lineplot(
        x="PC",
        y="Variance Ratio",
        data=plot_data,
        marker="o",
        linewidth=2.5,
        color="gray",
    )

    for i in range(2):
        x = plot_data["PC"][i]
        y = plot_data["Variance Ratio"][i]
        plt.plot(x, y, "o", markersize=15, color=point_colors[i])
        plt.text(x, y + 0.01, f"{y * 100:.1f}%", ha="center", va="bottom", fontsize=25)

    plt.ylabel("Explained Variance Ratio", fontsize=25)
    plt.xlabel("")
    plt.ylim(0, max(variance_ratios) * 1.1)
    plt.grid(False)

    ax = plt.gca()
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=25)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("outward", 25))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Visual_PC_explained.svg"), dpi=300)
    plt.show()


def calculate_permutation_p(empirical_data, target_variable, null_matrix):
    """Compute one-tailed permutation p-value based on the sign of the empirical correlation."""
    rho_emp, _ = spearmanr(empirical_data, target_variable)

    null_correlations = []
    for i in range(null_matrix.shape[0]):
        r_null, _ = spearmanr(null_matrix[i, :], target_variable)
        null_correlations.append(r_null)

    null_correlations = np.array(null_correlations)
    n_permutations = len(null_correlations)

    if rho_emp > 0:
        p_perm = (np.sum(null_correlations >= rho_emp) + 1) / (n_permutations + 1)
        direction = "Positive (right-tailed)"
    else:
        p_perm = (np.sum(null_correlations <= rho_emp) + 1) / (n_permutations + 1)
        direction = "Negative (left-tailed)"

    print(
        f"Permutation test | Rho: {rho_emp:.3f} | p-perm: {p_perm:.3f} | Direction: {direction}"
    )
    return rho_emp, p_perm


def plot_correlation(x_data, y_data, x_label, y_label, cmap, color, save_name, p_value_display):
    """Plot a KDE-based correlation figure with regression line and annotated statistics."""
    rho, _ = spearmanr(x_data, y_data)

    plt.figure(figsize=(6, 4.5))
    sns.kdeplot(x=x_data, y=y_data, fill=True, cmap=cmap, thresh=0.005, levels=20)
    sns.regplot(
        x=x_data,
        y=y_data,
        scatter=False,
        color=color,
        line_kws={"linewidth": 2},
    )

    if p_value_display < 0.001:
        p_str = r"$P$ < 0.001"
    else:
        p_str = f"$P$ = {p_value_display:.3f}"

    text_str = f"$\\rho$ = {rho:.2f}\n{p_str}"
    plt.text(
        0.20,
        0.95,
        text_str,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=25,
    )

    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)

    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=25)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("outward", 20))
    ax.spines["bottom"].set_position(("outward", 20))

    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=300, bbox_inches="tight")
    plt.show()


def plot_diverging_bars(df_scores):
    """Plot positive PC1 and PC2 scores as diverging horizontal bars."""
    df_sorted = df_scores.reset_index()
    df_pc1 = (
        df_sorted[df_sorted["PC1 Score"] > 0]
        .sort_values("PC1 Score", ascending=False)
        .reset_index(drop=True)
    )
    df_pc2 = (
        df_sorted[df_sorted["PC2 Score"] > 0]
        .sort_values("PC2 Score", ascending=False)
        .reset_index(drop=True)
    )

    n = max(len(df_pc1), len(df_pc2))
    df_pc1 = df_pc1.reindex(range(n))
    df_pc2 = df_pc2.reindex(range(n))

    cmap_pc1 = LinearSegmentedColormap.from_list("GnBu_8", GnBu_8.mpl_colors[::-1])
    cmap_pc2 = LinearSegmentedColormap.from_list("YlOrRd_6", YlOrRd_6.mpl_colors[::-1])

    pc1_colors = cmap_pc1(np.linspace(0, 1, len(df_pc1)))
    pc2_colors = cmap_pc2(np.linspace(0, 1, len(df_pc2)))

    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.barh(range(n), -df_pc1["PC1 Score"].fillna(0), color=pc1_colors)
    ax.barh(range(n), df_pc2["PC2 Score"].fillna(0), color=pc2_colors)

    ax.set_yticks([])
    m = max(df_sorted["PC1 Score"].max(), df_sorted["PC2 Score"].max())

    for i, (score, label) in enumerate(zip(df_pc1["PC1 Score"], df_pc1["CellType"])):
        if pd.notna(score):
            ax.text(-score - m * 0.02, i, label, ha="right", va="center", fontsize=20)

    for i, (score, label) in enumerate(zip(df_pc2["PC2 Score"], df_pc2["CellType"])):
        if pd.notna(score):
            ax.text(score + m * 0.02, i, label, ha="left", va="center", fontsize=20)

    xticks = ax.get_xticks()
    ax.set_xticklabels([str(int(abs(x))) for x in xticks], fontsize=25)
    ax.invert_yaxis()

    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "PC1_PC2_bars_plot.svg"), dpi=300)
    plt.show()


def plot_polar(df_scores, pc_col, colorscale, save_name):
    """Plot polar bar chart of PCA scores for a selected component."""
    polar_df = pd.DataFrame({
        "direction": df_scores.index,
        "frequency": df_scores[pc_col],
    })

    custom_colorscale = [
        [i / (len(colorscale.colors) - 1), color]
        for i, color in enumerate(colorscale.hex_colors)
    ]

    fig = px.bar_polar(
        polar_df,
        r="frequency",
        theta="direction",
        color="frequency",
        color_continuous_scale=custom_colorscale,
    )

    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        polar=dict(radialaxis=dict(tickfont=dict(size=20))),
        font=dict(size=20),
    )

    fig.write_image(os.path.join(OUTPUT_DIR, save_name), width=800, height=800)
    fig.show()


# =========================
# Main execution
# =========================
visual_df, roi_indices = load_data()

pca, scores = run_pca(visual_df, CELLTYPES)
plot_explained_variance(pca)

calc_dist_img = nib.load(f"{PATH}/materials/visual/lh_Calcarine_sulcus_gc_distance.func.gii")
dist_vals = calc_dist_img.darrays[0].data[roi_indices]

null_dist_path = os.path.join(
    NULL_DATA_PATH,
    "lh_Calcarine_sulcus_gc_distance_32fslr_brainmash_surrogates.npy",
)
null_gc_distance_1000 = np.load(null_dist_path)

target_pc1 = pca.components_[0, :]
rho_dist, p_perm_dist = calculate_permutation_p(
    dist_vals, target_pc1, null_gc_distance_1000
)

plot_correlation(
    dist_vals,
    -target_pc1,
    "Geodesic Distance",
    "PC1 Score",
    Blues_9.mpl_colormap,
    "skyblue",
    "Visual_stream_celltype_PC1_gd_corr.svg",
    p_value_display=p_perm_dist,
)

print("\n--- Processing PC2 vs Eccentricity ---")

ecc_img = nib.load(
    f"{PATH}/materials/visual/lh.S1200_7T_Retinotopy181.Fit1_Eccentricity_MSMAll.func.gii"
)
ecc_vals = ecc_img.darrays[0].data[roi_indices]

null_ecc_path = os.path.join(
    NULL_DATA_PATH,
    "lh_Fit1_Eccentricity_32fslr_brainmash_surrogates.npy",
)
lh_fit1_eccentricity_1000 = np.load(null_ecc_path)

target_pc2 = pca.components_[1, :]
rho_ecc, p_perm_ecc = calculate_permutation_p(
    ecc_vals, target_pc2, lh_fit1_eccentricity_1000
)

plot_correlation(
    ecc_vals,
    -target_pc2,
    "Eccentricity",
    "PC2 Score",
    Oranges_9.mpl_colormap,
    "#df5106",
    "Visual_stream_celltype_PC2_Eccentricity_corr.svg",
    p_value_display=p_perm_ecc,
)

df_cell_scores = pd.DataFrame({
    "CellType": CELLTYPES,
    "PC1 Score": scores[:, 0],
    "PC2 Score": scores[:, 1],
}).set_index("CellType")

plot_diverging_bars(df_cell_scores)
plot_polar(df_cell_scores, "PC1 Score", GnBu_8, "PC1_polar_plot.svg")
plot_polar(df_cell_scores, "PC2 Score", YlOrRd_6, "PC2_polar_plot.svg")