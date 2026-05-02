import os
import pickle

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from scipy.stats import zscore
from neuromaps.stats import compare_images
from neuromaps import images
from neuromaps.datasets import fetch_fslr, fetch_atlas
from neuromaps.nulls.spins import gen_spinsamples, get_parcel_centroids
from surfplot import Plot
from netneurotools import plotting


# ============================================================
# Set paths
# ============================================================
PARC = "glasser360"
BASE_PATH = "project_whole_brain_mapping/final_code/"

path_parc = os.path.join(
    BASE_PATH,
    f"results/parcellated_{PARC}/"
)

out_dir = os.path.join(
    path_parc,
    "surface_gradients",
    "all_celltype_pca"
)

os.makedirs(out_dir, exist_ok=True)

lh_label_gii_path = os.path.join(
    BASE_PATH,
    "materials/ROI/fslr32k/glasser360_conte69_lh.label.gii"
)

rh_label_gii_path = os.path.join(
    BASE_PATH,
    "materials/ROI/fslr32k/glasser360_conte69_rh.label.gii"
)


# ============================================================
# Helper functions
# ============================================================
def to_parcel_by_celltype_df(obj, name="obj"):
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif isinstance(obj, dict):
        df = pd.DataFrame(obj)
    else:
        raise TypeError(
            f"{name} must be dict or DataFrame, but got {type(obj)}"
        )

    if df.shape[1] > df.shape[0]:
        df = df.T

    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def safe_pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() < 3:
        return np.nan

    a = a[mask]
    b = b[mask]

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


def normalize_min_mean_max(v):
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, np.nan, dtype=float)

    finite = np.isfinite(v)

    if finite.sum() == 0:
        return out

    mn = np.nanmin(v)
    mu = np.nanmean(v)
    mx = np.nanmax(v)

    low = finite & (v <= mu)
    high = finite & (v > mu)

    if np.isclose(mu, mn):
        out[low] = 0.5
    else:
        out[low] = 0.5 * (v[low] - mn) / (mu - mn)

    if np.isclose(mx, mu):
        out[high] = 1.0
    else:
        out[high] = 0.5 + 0.5 * (v[high] - mu) / (mx - mu)

    return np.clip(out, 0, 1)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def map_parcel_values_to_lh_surface(values, lh_labels, valid_labels):
    lh_data = np.zeros(len(lh_labels), dtype=float)

    for lab, value in zip(valid_labels, values):
        lh_data[lh_labels == lab] = value

    return lh_data


# ============================================================
# Load cell-type parcel data
# ============================================================
lh_layer_EX_index = load_pickle(
    os.path.join(path_parc, "lh_layer_EX_index.pkl")
)

lh_layer_IN_index = load_pickle(
    os.path.join(path_parc, "lh_layer_IN_index.pkl")
)

lh_layer_NO_neuron_index = load_pickle(
    os.path.join(path_parc, "lh_layer_NO_neuron_index.pkl")
)

df_ex = to_parcel_by_celltype_df(
    lh_layer_EX_index,
    "lh_layer_EX_index"
)

df_in = to_parcel_by_celltype_df(
    lh_layer_IN_index,
    "lh_layer_IN_index"
)

df_no = to_parcel_by_celltype_df(
    lh_layer_NO_neuron_index,
    "lh_layer_NO_neuron_index"
)

if not (len(df_ex) == len(df_in) == len(df_no)):
    raise ValueError(
        f"Parcel number mismatch: "
        f"EX={len(df_ex)}, IN={len(df_in)}, NO={len(df_no)}"
    )

all_cell_df = pd.concat(
    [df_ex, df_in, df_no],
    axis=1
)

all_cell_df = all_cell_df.loc[
    :,
    ~all_cell_df.columns.duplicated()
]

all_cell_df = all_cell_df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# Load LH parcellation
# ============================================================
lh_gii_parcellation_img = nib.load(lh_label_gii_path)
lh_labels = lh_gii_parcellation_img.darrays[0].data.astype(int)

valid_labels = np.sort(np.unique(lh_labels))
valid_labels = valid_labels[valid_labels > 0]

if len(valid_labels) != all_cell_df.shape[0]:
    raise ValueError(
        f"Number of valid LH labels ({len(valid_labels)}) "
        f"!= number of parcel rows ({all_cell_df.shape[0]})."
    )


# ============================================================
# PCA on all cell-type maps
# ============================================================
X = all_cell_df.values
X_z = zscore(
    X,
    axis=0,
    nan_policy="omit"
)

X_z = np.nan_to_num(
    X_z,
    nan=0.0
)

PCA_number = 10

pca = PCA(
    n_components=PCA_number
)

pcs = pca.fit_transform(X_z)

pc_scores_df = pd.DataFrame(
    pcs,
    index=all_cell_df.index,
    columns=[f"PC{i + 1}" for i in range(PCA_number)]
)

pc_scores_df.to_csv(
    os.path.join(out_dir, "parcel_pc_scores_top10.csv")
)

loadings_df = pd.DataFrame(
    pca.components_.T,
    index=all_cell_df.columns,
    columns=[f"PC{i + 1}" for i in range(PCA_number)]
)

loadings_df.to_csv(
    os.path.join(out_dir, "celltype_loadings_top10.csv")
)

evr_df = pd.DataFrame({
    "PC": [f"PC{i + 1}" for i in range(PCA_number)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "explained_variance_percent": pca.explained_variance_ratio_ * 100,
    "cumulative_variance_percent": (
        np.cumsum(pca.explained_variance_ratio_) * 100
    )
})

evr_df.to_csv(
    os.path.join(out_dir, "explained_variance_ratio_top10.csv"),
    index=False
)

print("Explained variance ratio:")
print(evr_df)


# ============================================================
# Plot PCA explained variance
# ============================================================
plt.figure(figsize=(4, 4))

x = np.arange(1, PCA_number + 1)

plt.plot(
    x,
    evr_df["explained_variance_percent"],
    marker="o",
    linewidth=2
)

plt.xticks(x)
plt.xlabel("Principal Component", fontsize=20)
plt.ylabel("Explained Variance (%)", fontsize=20)
plt.title("PCA Scree Plot")

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=16)

plt.tight_layout()

plt.savefig(
    os.path.join(out_dir, "pca_explained_variance_lineplot.svg"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Save PCA surface maps
# ============================================================
cmap = "Spectral_r"

for pc_index in range(PCA_number):

    pc_name = f"PC{pc_index + 1}"

    lh_data = map_parcel_values_to_lh_surface(
        -pcs[:, pc_index],
        lh_labels,
        valid_labels
    )

    save_fig_path = os.path.join(
        out_dir,
        f"{pc_name}_left.png"
    )

    plotting.pv_plot_surface(
        lh_data,
        template="fslr32k",
        surf="inflated",
        hemi="L",
        cmap=cmap,
        lighting_style="threelights",
        zoom_ratio=1,
        plotter_kws={
            "window_size": [2000, 1500],
        },
        cbar_kws={
            "label_font_size": 60,
            "vertical": True,
            "width": 0.15,
            "height": 0.2,
            "position_x": 0.8,
            "position_y": 0.1,
        },
        mesh_kws={
            "opacity": 1,
            "diffuse": 1,
            "specular": 0.2,
        },
        save_fig=save_fig_path
    )

    print(f"Saved: {save_fig_path}")


# ============================================================
# Plot PC1 on flat fsLR surface
# ============================================================
lh_data_pc1 = map_parcel_values_to_lh_surface(
    -pcs[:, 0],
    lh_labels,
    valid_labels
)

surfaces = fetch_fslr(density="32k")
lh_flat, rh_flat = surfaces["flat"]

p = Plot(
    surf_lh=lh_flat,
    size=(400, 200),
    views=["dorsal"],
    zoom=1.2
)

p.add_layer(
    {"left": lh_data_pc1},
    cbar=True,
    cmap="Spectral_r"
)

fig = p.build()

plt.savefig(
    os.path.join(out_dir, "pc_1_flat_left.png"),
    dpi=450,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Compute Pr-Al index
# ============================================================
all_cell_df.index = np.arange(1, len(all_cell_df) + 1)

PR_PARCELS = [1, 9, 24]
AL_PARCELS = [110, 118]

PREPROCESS_MODE = "none"

X = all_cell_df.copy()
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

if PREPROCESS_MODE == "row_sum_log1p":
    row_sum = X.sum(axis=1).replace(0, np.nan)
    X = np.log1p(
        X.div(row_sum, axis=0) * 1e4
    )

X = X.fillna(0.0)

for pid in PR_PARCELS + AL_PARCELS:
    if pid not in X.index:
        raise ValueError(
            f"Parcel ID {pid} is not in X.index."
        )

pr_corr_mat = pd.DataFrame(
    index=X.index,
    columns=PR_PARCELS,
    dtype=float
)

al_corr_mat = pd.DataFrame(
    index=X.index,
    columns=AL_PARCELS,
    dtype=float
)

for pid in X.index:

    xi = X.loc[pid].values

    for pr in PR_PARCELS:
        pr_corr_mat.loc[pid, pr] = safe_pearson(
            xi,
            X.loc[pr].values
        )

    for al in AL_PARCELS:
        al_corr_mat.loc[pid, al] = safe_pearson(
            xi,
            X.loc[al].values
        )

CorPr_raw = pr_corr_mat.max(axis=1).values
CorAl_raw = al_corr_mat.max(axis=1).values

CorPr = normalize_min_mean_max(CorPr_raw)
CorAl = normalize_min_mean_max(CorAl_raw)

Pr_Al_diff = CorPr - CorAl

Pr_Al_index = (
    (CorPr - CorAl)
    / ((1 - CorAl) + (1 - CorPr))
)

best_Pr_anchor_id = pr_corr_mat.idxmax(axis=1).values
best_Al_anchor_id = al_corr_mat.idxmax(axis=1).values

pr_al_df = pd.DataFrame(
    {
        "parcel_id": X.index,
        "CorPr_raw": CorPr_raw,
        "CorAl_raw": CorAl_raw,
        "CorPr_norm": CorPr,
        "CorAl_norm": CorAl,
        "Pr_Al_diff": Pr_Al_diff,
        "Pr_Al_index": Pr_Al_index,
        "best_Pr_anchor_id": best_Pr_anchor_id,
        "best_Al_anchor_id": best_Al_anchor_id,
    },
    index=X.index
)

out_csv = os.path.join(
    out_dir,
    "glasser360_Pr_Al_index_maxcorr_explicitID.csv"
)

pr_al_df.to_csv(
    out_csv,
    encoding="utf-8-sig"
)

print(f"Saved: {out_csv}")
print(pr_al_df.head())


# ============================================================
# Plot Pr-Al index histogram
# ============================================================
plt.figure(figsize=(6, 4))

plt.hist(
    pr_al_df["Pr_Al_index"].dropna(),
    bins=30
)

plt.xlabel("Pr-Al index")
plt.ylabel("Count")
plt.title("Distribution of Pr-Al index")

plt.tight_layout()

out_hist = os.path.join(
    out_dir,
    "glasser360_Pr_Al_index_hist_explicitID.png"
)

plt.savefig(
    out_hist,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved histogram: {out_hist}")


# ============================================================
# Plot Pr-Al index on flat fsLR surface
# ============================================================
lh_data_pr_al = map_parcel_values_to_lh_surface(
    pr_al_df["Pr_Al_index"].values,
    lh_labels,
    valid_labels
)

p = Plot(
    surf_lh=lh_flat,
    size=(400, 200),
    views=["dorsal"],
    zoom=1.2
)

p.add_layer(
    {"left": lh_data_pr_al},
    cbar=True,
    cmap="Spectral_r"
)

fig = p.build()

plt.savefig(
    os.path.join(out_dir, "pir_Al_index_flat.png"),
    dpi=450,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Spin-test correlation between PC1 and Pr-Al index
# ============================================================
current_out_dir = os.path.join(
    path_parc,
    "Pr_Al_index_celltype_PC1"
)

os.makedirs(
    current_out_dir,
    exist_ok=True
)

parcellation = images.relabel_gifti(
    (
        lh_label_gii_path,
        rh_label_gii_path
    )
)

sphere_surfaces = fetch_atlas(
    "fsLR",
    "32k"
)["sphere"]

coords, hemi = get_parcel_centroids(
    sphere_surfaces,
    parcellation=parcellation,
    method="surface"
)

half_length = len(coords) // 2

coords = coords[:half_length]
hemi = hemi[:half_length]

lh_spins = gen_spinsamples(
    coords,
    hemi,
    n_rotate=1000,
    seed=1234,
    method="vasa"
)

data_a = -pcs[:, 0]
data_b = pr_al_df["Pr_Al_index"].values

n_perm = lh_spins.shape[1]

rotated = np.zeros(
    (len(data_a), n_perm),
    dtype=float
)

for i in range(n_perm):
    rotated[:, i] = data_a[lh_spins[:, i]]

corr, pval = compare_images(
    data_a,
    data_b,
    nulls=rotated,
    metric="pearsonr"
)

print(f"Correlation: {corr:.3f}, p-spin: {pval:.3f}")


# ============================================================
# Plot PC1-Pr-Al correlation
# ============================================================
plt.figure(figsize=(5, 5))

x = np.asarray(data_b, dtype=float)
y = np.asarray(data_a, dtype=float)

plt.scatter(
    x,
    y,
    c=x,
    cmap="Spectral_r",
    alpha=0.95,
    s=300
)

sns.regplot(
    x=x,
    y=y,
    scatter=False,
    color="black",
    line_kws={"linewidth": 5}
)

if pval < 0.001:
    pval_text = r"$P$ < 0.001"
else:
    pval_text = rf"$P$ = {pval:.3f}"

plt.text(
    0.05,
    1.05,
    f"$r$ = {corr:.3f}\n{pval_text}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    fontsize=22
)

plt.xlabel("")
plt.ylabel("")

ax = plt.gca()
ax.tick_params(axis="both", labelsize=20)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_position(("outward", 20))
ax.spines["bottom"].set_position(("outward", 20))

plt.grid(False)
plt.tight_layout()

plt.savefig(
    os.path.join(out_dir, "PC1_Pr_Al_corr.svg"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()