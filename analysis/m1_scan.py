import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.stats import zscore, pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import minmax_scale
from neuromaps.datasets import fetch_fslr
from surfplot import Plot
from palettable.colorbrewer.diverging import RdYlGn_10, RdYlGn_11


# =========================
# Configuration
# =========================
BASE_PATH = "project_whole_brain_mapping/final_code"

CELLTYPE_PATH = os.path.join(
    BASE_PATH, "results/celltype_map/lh_fslr32k_zscored_24celltype.csv"
)
# This matrix can be derived from the Human Connectome Project (HCP) dataset. It is not included in this repository because of its large file size.
VERTEX_MATRIX_PATH = os.path.join(
    BASE_PATH, "materials/M1C/HCP_S1200_812_zcorr_recon2_vertex_matrix.npy"
)
GLASSER_PATH = os.path.join(
    BASE_PATH, "materials/ROI/fslr32k/lh_Glasser360.32k_fs_LR.label.gii"
)
SCAN_PATH = os.path.join(
    BASE_PATH, "materials/ROI/fslr32k/lh.Gordon17_with_SCAN_conte69_lh.label.gii"
)
MOTOR_ROI_PATH = os.path.join(
    BASE_PATH, "materials/M1C/lh.Gordon17_with_SCAN_HCPMMP1_3a.label.gii"
)

OUTPUT_DIR = os.path.join(BASE_PATH, "results/M1C/figures")
OUTPUT_SUBDIR = os.path.join(OUTPUT_DIR, "inter_effector")
os.makedirs(OUTPUT_SUBDIR, exist_ok=True)


# =========================
# Helper functions
# =========================
def load_data():
    """Load input datasets and prepare CCA inputs."""
    print("Loading datasets...")

    celltype_df = pd.read_csv(CELLTYPE_PATH)
    vertex_matrix = np.load(VERTEX_MATRIX_PATH)
    glasser = nib.load(GLASSER_PATH)
    scan = nib.load(SCAN_PATH)
    motor_roi = nib.load(MOTOR_ROI_PATH)

    scan_data = scan.darrays[0].data
    glasser_data = glasser.darrays[0].data
    motor_roi_data = motor_roi.darrays[0].data

    scan_con_idx = (scan_data == 9).nonzero()[0]
    glasser_m1c_idx = (glasser_data == 188).nonzero()[0]

    roi_indices = {
        "top": (motor_roi_data == 18).nonzero()[0],
        "middle": (motor_roi_data == 19).nonzero()[0],
        "bottom": (motor_roi_data == 20).nonzero()[0],
        "foot": (motor_roi_data == 17).nonzero()[0],
        "hand": (motor_roi_data == 10).nonzero()[0],
        "mouth": (motor_roi_data == 11).nonzero()[0],
    }

    con_zfc = vertex_matrix[np.ix_(scan_con_idx, glasser_m1c_idx)]
    col_avg = np.mean(con_zfc, axis=0)

    celltype_m1c = celltype_df[celltype_df["label"].isin(glasser_m1c_idx)]
    x_data = celltype_m1c.drop(columns=["label"])

    return {
        "X": x_data,
        "Y": col_avg,
        "celltype_df": celltype_df,
        "vertex_matrix": vertex_matrix,
        "glasser_data": glasser_data,
        "scan_data": scan_data,
        "roi_indices": roi_indices,
        "glasser_m1c_idx": glasser_m1c_idx,
    }


def create_custom_colormap():
    """Create a segmented colormap for projection maps."""
    base_cmap = cm.get_cmap(RdYlGn_11.mpl_colormap)
    temp_linspace = np.linspace(0, 1, 11)
    colors_hex = [mcolors.to_hex(base_cmap(p)) for p in temp_linspace]

    points = [0.0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 0.80, 1.0]
    gradient_points = [(p, colors_hex[i]) for i, p in enumerate(points)]

    return LinearSegmentedColormap.from_list(
        "my_colormap", gradient_points, N=256
    )


def setup_plot_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply a consistent plotting style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    ax.tick_params(axis="both", which="major", labelsize=14)

    if title:
        ax.set_title(title, fontsize=16)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=16)


def run_iterative_cca(X, Y, stop_threshold=0.5):
    """Run iterative CCA and prune the weakest feature at each step."""
    x_curr = X.copy()
    history = {"columns": [], "corr": [], "weights": []}

    while x_curr.shape[1] > 1:
        cca = CCA(n_components=1)
        cca.fit(x_curr, Y)
        x_c, y_c = cca.transform(x_curr, Y)
        curr_corr = np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1]

        if curr_corr < stop_threshold:
            print(f"Stopping: correlation {curr_corr:.3f} < threshold.")
            break

        weights = cca.x_weights_.ravel()
        history["columns"].append(x_curr.columns.tolist())
        history["corr"].append(curr_corr)
        history["weights"].append(dict(zip(x_curr.columns, weights)))

        idx_remove = np.argmin(np.abs(weights))
        x_curr = x_curr.drop(columns=[x_curr.columns[idx_remove]])

    return history


def plot_brain_surface(data_map, percentile_thresh=90, cmap="viridis", save_path=None):
    """Project vertex data to the surface and outline top percentile regions."""
    nonzero_data = data_map[data_map != 0]
    if len(nonzero_data) == 0:
        print("Warning: no non-zero data to plot.")
        return

    percentile_val = np.percentile(nonzero_data, percentile_thresh)
    marker_mask = np.zeros_like(data_map)
    marker_mask[data_map >= percentile_val] = 1

    ranks_normalized = minmax_scale(
        pd.Series(nonzero_data).rank(method="min"),
        feature_range=(0, 1),
    )

    normalized_full = np.zeros_like(data_map, dtype=float)
    normalized_full[data_map != 0] = ranks_normalized

    surfaces = fetch_fslr(density="32k")
    lh, _ = surfaces["inflated"]

    p = Plot(surf_lh=lh, size=(800, 400), zoom=1.2)
    p.add_layer({"left": normalized_full}, cbar=True, cmap=cmap)
    p.add_layer({"left": marker_mask}, cbar=False, as_outline=True, cmap="gray")

    cbar_kws = {
        "location": "right",
        "label_direction": 45,
        "decimals": 1,
        "fontsize": 8,
        "n_ticks": 2,
        "shrink": 0.25,
        "aspect": 8,
        "draw_border": False,
    }
    p.build(cbar_kws=cbar_kws)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def get_top_indices(data_map, percentile=90):
    """Return indices above the specified percentile among non-zero vertices."""
    nonzero = data_map[data_map != 0]
    thresh = np.percentile(nonzero, percentile)
    return np.where(data_map >= thresh)[0]


# =========================
# Load data
# =========================
data = load_data()
X_full = data["X"]
Y_target = data["Y"]


# =========================
# Iterative CCA
# =========================
cca_history = run_iterative_cca(X_full, Y_target)

plt.figure(figsize=(8, 5))
num_vars = [len(c) for c in cca_history["columns"]]
plt.plot(num_vars, cca_history["corr"], marker="o", color="purple")

ax = plt.gca()
ax.invert_xaxis()
setup_plot_style(ax, xlabel="Number of Cell Types", ylabel="Canonical Correlation")
plt.show()

SELECTED_IDX = 12 #chosen based on correlation profile and interpretability
selected_cols = cca_history["columns"][SELECTED_IDX]
print(f"Selected iteration {SELECTED_IDX}: {len(selected_cols)} variables.")

X_final = X_full[selected_cols]
X_z = zscore(X_final, axis=0)
Y_z = zscore(Y_target, axis=0)

cca_final = CCA(n_components=1)
cca_final.fit(X_z, Y_z)
X_c, Y_c = cca_final.transform(X_z, Y_z)
final_weights = cca_final.x_weights_.ravel()


# =========================
# CCA scatter plot
# =========================
plt.figure(figsize=(6, 6))
plt.scatter(
    X_c[:, 0],
    Y_c[:, 0],
    c=X_c[:, 0],
    cmap=RdYlGn_10.mpl_colormap.reversed(),
    alpha=0.95,
    s=200,
)
sns.regplot(
    x=X_c[:, 0],
    y=Y_c[:, 0],
    scatter=False,
    line_kws={"color": "black", "linewidth": 5},
)
setup_plot_style(
    plt.gca(),
    title=f"CCA Corr = {cca_history['corr'][SELECTED_IDX]:.3f}",
    xlabel="Component 1 X",
    ylabel="Component 1 Y",
)
plt.tick_params(axis="both", which="major", labelsize=28)
plt.savefig(
    os.path.join(OUTPUT_SUBDIR, f"celltype_cca_num_{SELECTED_IDX}_scatter.svg"),
    dpi=300,
)
plt.show()


# =========================
# CCA weight bar plot
# =========================
x_weights = cca_final.x_weights_.ravel()
indices = np.argsort(x_weights)[::-1]
sorted_weights = x_weights[indices]
sorted_celltypes = np.array(selected_cols)[indices]

max_abs = max(abs(min(sorted_weights)), abs(max(sorted_weights)))
norm = Normalize(vmin=-max_abs, vmax=max_abs)
cmap = cm.get_cmap(RdYlGn_11.mpl_colormap.reversed())
colors = [cmap(norm(w)) for w in sorted_weights]

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=sorted_celltypes, y=sorted_weights, color="gray")
for bar, color in zip(ax.patches, colors):
    bar.set_color(color)

ax.tick_params(axis="x", labelsize=25, rotation=45)
ax.tick_params(axis="y", labelsize=25)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_position(("outward", 25))
ax.spines["bottom"].set_position(("outward", 25))
plt.xlabel("Cell Type", fontsize=25)
plt.ylabel("X weights", fontsize=25)
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_SUBDIR, f"celltype_cca_num_{SELECTED_IDX}.svg"
    )
)
plt.show()


# =========================
# Spatial permutation test for CCA correlation
# =========================
lh_m1_con = np.load(
    os.path.join(
        BASE_PATH, "materials/M1C/lh_M1_CON_zFC_32fslr_brainmash_surrogates.npy"
    )
)

X = X_c.flatten()
Y = Y_target.flatten()
n_rois = X.shape[0]

if lh_m1_con.shape[0] != n_rois and lh_m1_con.shape[1] == n_rois:
    lh_m1_con = lh_m1_con.T

r_emp, _ = pearsonr(X, Y)
print(f"Empirical r: {abs(r_emp):.4f}")

n_perms = lh_m1_con.shape[1]
null_rs = np.zeros(n_perms)

for i in range(n_perms):
    y_null = lh_m1_con[:, i]
    r_null, _ = pearsonr(X, y_null)
    null_rs[i] = r_null

n_extreme = np.sum(np.abs(null_rs) >= np.abs(r_emp))
p_spatial = (n_extreme + 1) / (n_perms + 1)
print(f"Spatial permutation p: {p_spatial:.4f}")

if p_spatial < 0.001:
    p_label = r"$P < 0.001$"
else:
    p_label = f"$P$ = {p_spatial:.4f}"

plt.figure(figsize=(6, 4.5))
sns.histplot(
    null_rs,
    kde=True,
    color="lightgray",
    edgecolor="w",
    stat="density",
    label="Null Distribution",
)
plt.axvline(
    np.abs(r_emp),
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"$r$ = {np.abs(r_emp):.3f}\n{p_label}",
)
sns.despine()
plt.xlabel("Correlation r", fontsize=25)
plt.ylabel("Density", fontsize=25)
plt.tick_params(axis="both", which="major", labelsize=25)
plt.legend(frameon=False, fontsize=25)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_SUBDIR, "celltype_cca_8_perm.svg"), dpi=300)
plt.show()


# =========================
# Brain surface projection
# =========================
custom_cmap = create_custom_colormap()

lh_data_celltype = np.zeros(len(data["scan_data"]))
full_cell_data = data["celltype_df"][selected_cols].values
vertex_indices = data["celltype_df"]["label"].values

projection_scores = full_cell_data @ (-final_weights)
lh_data_celltype[vertex_indices] = projection_scores

selected_labels_scan = [10, 11, 17, 18]
scan_roi_mask = np.isin(data["scan_data"], selected_labels_scan)
lh_data_celltype[scan_roi_mask] = 0

print("Plotting cell-type projection...")
plot_brain_surface(
    lh_data_celltype,
    percentile_thresh=90,
    cmap=custom_cmap.reversed(),
    save_path=os.path.join(OUTPUT_SUBDIR, "celltype_weights.svg"),
)

print("Calculating inter-effector connectivity...")
scan_intereffector_indices = (data["scan_data"] == 18).nonzero()[0]
scan_roi_selected_indices = np.where(np.isin(data["scan_data"], selected_labels_scan))[0]
nonzero_indices_glasser = np.where(data["glasser_data"] != 0)[0]

target_indices = np.setdiff1d(nonzero_indices_glasser, scan_roi_selected_indices)

con_inter = data["vertex_matrix"][np.ix_(scan_intereffector_indices, target_indices)]
col_avg_inter = np.mean(con_inter, axis=0)

lh_data_intereffector = np.zeros(len(data["glasser_data"]))
lh_data_intereffector[target_indices] = col_avg_inter

mask_combined = np.zeros_like(lh_data_intereffector)
mask_combined[vertex_indices] = lh_data_intereffector[vertex_indices]
lh_data_intereffector = mask_combined

print("Plotting inter-effector projection...")
plot_brain_surface(
    lh_data_intereffector,
    percentile_thresh=90,
    cmap=custom_cmap.reversed(),
    save_path=os.path.join(OUTPUT_SUBDIR, "intereffector_to_other_90.png"),
)


# =========================
# Overlap analysis
# =========================
print("Running overlap analysis...")

top10_celltype = get_top_indices(lh_data_celltype, 90)
top10_intereffector = get_top_indices(lh_data_intereffector, 90)

overlap_indices = np.intersect1d(top10_celltype, top10_intereffector)
overlap_count = len(overlap_indices)
print(f"Observed overlap count: {overlap_count}")


# =========================
# Overlap permutation test
# =========================
intereffector_1000 = np.load(
    os.path.join(
        BASE_PATH,
        "materials/M1C/lh_intereffector_other_regions_32fslr_brainmash_surrogates.npy",
    )
)

n_permutations = 1000
non_intereffector_zero_indices = np.nonzero(lh_data_intereffector)[0]
n_vertices = len(data["glasser_data"])

lh_data_null = np.zeros((n_permutations, n_vertices))
lh_data_null[:, non_intereffector_zero_indices] = intereffector_1000

nonzero_indices = np.where(lh_data_celltype != 0)[0]
sample_size = int(len(nonzero_indices) * 0.1)

lh_data_null = np.zeros((n_permutations, n_vertices))
null_overlap_counts = np.zeros(n_permutations)

for i in range(n_permutations):
    permuted_indices = np.random.choice(
        nonzero_indices,
        size=sample_size,
        replace=False,
    )
    lh_data_null[i, permuted_indices] = 1
    current_overlap = len(np.intersect1d(permuted_indices, top10_celltype))
    null_overlap_counts[i] = current_overlap

success_count = np.sum(null_overlap_counts >= overlap_count)
p_value_corrected = (success_count + 1) / (n_permutations + 1)

sns.set_theme(style="white")
plt.figure(figsize=(6, 4.5))

if p_value_corrected < 0.001:
    p_str = r"$P$ < 0.001"
else:
    p_str = f"$P$ = {p_value_corrected:.4f}"

text_str = f"{p_str}\nPermutations = {n_permutations}"

sns.histplot(
    null_overlap_counts,
    bins=20,
    kde=True,
    color="skyblue",
    edgecolor="white",
    linewidth=1.2,
    label="Null Distribution",
)

plt.axvline(
    overlap_count,
    color="#C44E52",
    linestyle="--",
    linewidth=2.5,
    label=f"n = {overlap_count}",
)

plt.gca().text(
    0.05,
    0.95,
    text_str,
    transform=plt.gca().transAxes,
    fontsize=20,
    verticalalignment="top",
)

plt.xlabel("Number of Overlapping Vertices", fontsize=20, fontweight="bold")
plt.ylabel("Frequency", fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=20)
plt.legend(frameon=False, loc="upper right", fontsize=20)

sns.despine()
plt.tight_layout()
plt.show()