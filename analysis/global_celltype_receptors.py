import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from PIL import Image
from scipy.stats import zscore, pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from pyls import behavioral_pls
from netneurotools import plotting
from neuromaps.datasets import fetch_fslr
from palettable.colorbrewer.diverging import RdBu_8, RdBu_9


# =========================
# Helper functions
# =========================
def create_and_save_pv_plot_surface(data, hemisphere, output_path, cmap="viridis"):
    """Project parcel-level data to the cortical surface and save the figure."""
    data_array = np.zeros(len(hemisphere.darrays[0].data))
    for i, value in enumerate(data):
        data_array[hemisphere.darrays[0].data == i + 1] = value

    plotting.pv_plot_surface(
        data_array,
        template="fslr32k",
        surf="inflated",
        hemi="L",
        cmap=cmap,
        lighting_style="threelights",
        show_colorbar=True,
        zoom_ratio=1.2,
        plotter_kws={"window_size": [2000, 1500]},
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
            "interpolate_before_map": False,
            "diffuse": 0.95,
            "specular": 0.2,
        },
        save_fig=output_path,
    )

    if os.path.exists(output_path):
        im = Image.open(output_path)
        im.save(output_path, dpi=(450, 450))


def plot_loadings(
    loadings,
    ci_low,
    ci_high,
    names,
    title,
    output_path,
    palette,
    top_n=20,
    custom_labels=None,
):
    """Plot top feature loadings with bootstrap confidence intervals."""
    err = (ci_high - ci_low) / 2
    df = pd.DataFrame({"name": names, "loading": loadings, "err": err})

    df_sorted = df.sort_values("loading", ascending=False)
    df_top = df_sorted.head(top_n).copy()

    plt.figure(figsize=(7, 7), dpi=300)
    sns.barplot(data=df_top, x="loading", y="name", palette=palette, ci=None)

    plt.errorbar(
        x=df_top["loading"],
        y=df_top["name"],
        xerr=df_top["err"],
        fmt="none",
        ecolor="black",
        lw=3,
        capsize=5,
        capthick=3,
        alpha=0.8,
    )

    ax = plt.gca()
    plt.title(title, fontsize=24)
    plt.xlabel("Loading Value", fontsize=20)
    plt.ylabel("Feature", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)

    if custom_labels and len(custom_labels) == len(df_top):
        ax.set_yticklabels(custom_labels, fontsize=20)

    sns.despine(offset=20, trim=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


# =========================
# Configuration
# =========================
BASE_PATH = "project_whole_brain_mapping/final_code/"
RESULT_DIR = os.path.join(BASE_PATH, "results", "PLS_receptors")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
ATLAS_DIR = "analysis/ROI/gii_atlas/fslr32k/"

PARC = "Schaefer400"
N_PERM = 10000
N_BOOT = 10000
N_ROIS = 200
LV_IDX = 0

PALETTE_CV = ["#ff6f91"]
PALETTE_BIPOLAR = ["#7db6d6", "#de7563"]

EX_IN_COLUMNS = [
    "5HT2a", "5HT4", "5HT6", "A4B2", "D1", "M1", "mGluR5", "NMDA",
    "5HT1a", "5HT1b", "CB1", "D2", "GABAa", "H3", "MU"
]

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# =========================
# Load data
# =========================
fetch_fslr(density="32k")

lh_parc_atlas = nib.load(f"{ATLAS_DIR}{PARC}_conte69_lh.label.gii")
lh_spins = np.load(f"{BASE_PATH}materials/lh_spins_{N_PERM}.npy")

path_parc = os.path.join(BASE_PATH, f"results/parcellated_{PARC}/")
with open(f"{path_parc}lh_layer_EX_index.pkl", "rb") as f:
    lh_layer_ex_index = pickle.load(f)
with open(f"{path_parc}lh_layer_IN_index.pkl", "rb") as f:
    lh_layer_in_index = pickle.load(f)

lh_layer_ex_in_index = {**lh_layer_ex_index, **lh_layer_in_index}
celltype_df = pd.DataFrame({k: v[:N_ROIS] for k, v in lh_layer_ex_in_index.items()})

with open(f"{BASE_PATH}materials/receptor_summary/receptor_densities_s400.pickle", "rb") as f:
    receptor_data_raw = pickle.load(f)

receptor_map = {
    name: receptor_data_raw["data"][:, i]
    for i, name in enumerate(receptor_data_raw["names"])
}
receptor_df = pd.DataFrame({k: v[:N_ROIS] for k, v in receptor_map.items()})
receptor_df = receptor_df[EX_IN_COLUMNS]

X = zscore(receptor_df, axis=0, ddof=1)
Y = zscore(celltype_df, axis=0, ddof=1)

X_df = pd.DataFrame(X, index=receptor_df.index, columns=receptor_df.columns)
Y_df = pd.DataFrame(Y, index=celltype_df.index, columns=celltype_df.columns)


# =========================
# Run PLS
# =========================
pls_pkl_path = os.path.join(RESULT_DIR, "pls_result.pkl")
if os.path.exists(pls_pkl_path):
    with open(pls_pkl_path, "rb") as f:
        pls_result = pickle.load(f)
else:
    pls_result = behavioral_pls(
        X_df,
        Y_df,
        n_perm=N_PERM,
        n_boot=N_BOOT,
        permsamples=lh_spins,
    )
    with open(pls_pkl_path, "wb") as f:
        pickle.dump(pls_result, f)


# =========================
# Covariance explained
# =========================
singvals = pls_result["singvals"]
cv = singvals ** 2 / np.sum(singvals ** 2)
null_singvals = pls_result["permres"]["permsamples"]
cv_spins = null_singvals ** 2 / np.sum(null_singvals ** 2, axis=0)

nlv = len(singvals)
palette_dist = PALETTE_CV + ["#d5cabd"] * (cv_spins.shape[0] - 1)

fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(
    data=cv_spins.T * 100,
    ax=ax,
    palette=palette_dist,
    zorder=1,
    width=0.8,
    linewidth=4,
    showfliers=False,
    fill=False,
)

for i in range(nlv):
    ax.scatter(i, cv[i] * 100, s=80, color="orange" if i == 0 else "gray", zorder=2)

ax.set_ylabel("Covariance explained %", fontsize=16)
ax.set_xlabel("Latent variables", fontsize=16)
ax.set_xticks(range(nlv))
ax.set_xticklabels([str(i + 1) for i in range(nlv)], fontsize=18)
ax.tick_params(axis="both", labelsize=18)
sns.despine(offset=20, trim=False)

y_pos = cv_spins.max() * 100
ax.scatter(nlv - 1.2, y_pos, s=80, color="orange", zorder=3)
ax.text(nlv - 0.8, y_pos, "$P$ < 0.05", fontsize=20, va="center")

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "EX-IN_PLS_explained.pdf"))
plt.show()
plt.close()


# =========================
# Surface projection of PLS scores
# =========================
cmap_rev = RdBu_9.mpl_colormap.reversed()
brainmap_data = {
    "PLS1_receptors_x_scores": pls_result["x_scores"][:, LV_IDX],
    "PLS1_celltype_y_scores": pls_result["y_scores"][:, LV_IDX],
}

for key, scores in brainmap_data.items():
    create_and_save_pv_plot_surface(
        scores,
        lh_parc_atlas,
        os.path.join(FIGURE_DIR, f"{key}_pv_plot_surface.png"),
        cmap=cmap_rev,
    )


# =========================
# Correlation between PLS scores
# =========================
xscore = pls_result["x_scores"][:, LV_IDX]
yscore = pls_result["y_scores"][:, LV_IDX]
r_obs, _ = pearsonr(xscore, yscore)

null_x_path = os.path.join(BASE_PATH, "materials/receptor_summary/NULL_x_scores_pls1.npy")
null_y_path = os.path.join(BASE_PATH, "materials/receptor_summary/NULL_y_scores_pls1.npy")

null_x = np.load(null_x_path)
null_y = np.load(null_y_path)
n_iter = null_x.shape[1]
null_corrs = [pearsonr(null_x[:, i], null_y[:, i])[0] for i in range(n_iter)]
p_spin = (np.sum(np.abs(null_corrs) >= np.abs(r_obs)) + 1) / (len(null_corrs) + 1)

plt.figure(figsize=(7, 6))
sns.regplot(
    x=xscore,
    y=yscore,
    color="gray",
    scatter_kws={"s": 0},
    line_kws={"linewidth": 5},
)
sns.scatterplot(
    x=xscore,
    y=yscore,
    hue=xscore,
    palette=RdBu_8.mpl_colormap.reversed(),
    s=350,
    legend=False,
    edgecolor="none",
)

plt.text(
    0.05,
    0.95,
    f"$r$ = {r_obs:.2f}\n$P$ = {p_spin:.2f}",
    transform=plt.gca().transAxes,
    fontsize=24,
    va="top",
    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
)

sns.despine(offset=20, trim=True)
plt.title("PLS1 scores", fontsize=25)
plt.xlabel("Receptor scores", fontsize=25)
plt.ylabel("Celltype scores", fontsize=25)
plt.tick_params(labelsize=25)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "EX-IN_PLS1_scores_corr.pdf"))
plt.show()
plt.close()


# =========================
# Reverse PLS for receptor loadings
# =========================
pls_pkl_r_path = os.path.join(RESULT_DIR, "pls_result_r.pkl")
if os.path.exists(pls_pkl_r_path):
    with open(pls_pkl_r_path, "rb") as f:
        pls_result_r = pickle.load(f)
else:
    pls_result_r = behavioral_pls(
        Y_df,
        X_df,
        n_perm=N_PERM,
        n_boot=N_BOOT,
        permsamples=lh_spins,
    )
    with open(pls_pkl_r_path, "wb") as f:
        pickle.dump(pls_result_r, f)


# =========================
# Cell-type loadings
# =========================
palette_celltype = [
    "#8d0c25", "#991027", "#b2182b", "#be3036", "#c2383a", "#f4a582", "#fddbc7",
    "#e1e1e1",
    "#d1e5f0", "#abd2e5", "#4393c3", "#4393c3", "#3b89be", "#2c75b4",
    "#2166ac", "#154e8b", "#09386c", "#053061",
]

plot_loadings(
    loadings=pls_result["y_loadings"][:, LV_IDX],
    ci_low=pls_result["bootres"]["y_loadings_ci"][:, LV_IDX, 0],
    ci_high=pls_result["bootres"]["y_loadings_ci"][:, LV_IDX, 1],
    names=Y_df.columns,
    title=f"PLS{LV_IDX + 1} Neuron Loadings",
    output_path=os.path.join(FIGURE_DIR, f"EX-IN_PLS{LV_IDX + 1}_celltype_loadings.pdf"),
    palette=palette_celltype,
    top_n=24,
)


# =========================
# Receptor loadings
# =========================
palette_receptor = [
    "#be3036", "#e1e1e1", "#e1e1e1",
    "#cfe4ef", "#b9d9e9", "#a3cee2", "#8dc3dc", "#56a0c9",
    "#3b8fbf", "#347fb7", "#2c70b0", "#2560a8", "#255fa0", "#1d4e8f", "#143c76",
]

plot_loadings(
    loadings=pls_result_r["y_loadings"][:, LV_IDX],
    ci_low=pls_result_r["bootres"]["y_loadings_ci"][:, LV_IDX, 0],
    ci_high=pls_result_r["bootres"]["y_loadings_ci"][:, LV_IDX, 1],
    names=X_df.columns,
    title=f"PLS{LV_IDX + 1} Receptor Loadings",
    output_path=os.path.join(FIGURE_DIR, f"EX-IN_PLS{LV_IDX + 1}_receptor_loadings.pdf"),
    palette=palette_receptor,
    top_n=19,
)


# =========================
# Split-half style cross-validation
# =========================
n_splits = 1000
train_scores = np.zeros(n_splits)
test_scores = np.zeros(n_splits)

X_arr = X_df.to_numpy()
Y_arr = Y_df.to_numpy()

for i in range(n_splits):
    train_idx, test_idx = train_test_split(
        np.arange(X_arr.shape[0]),
        test_size=0.7,
        random_state=i,
    )
    Xtrain, Ytrain = X_arr[train_idx], Y_arr[train_idx]
    Xtest, Ytest = X_arr[test_idx], Y_arr[test_idx]

    pls_cv = behavioral_pls(Xtrain, Ytrain, n_boot=0, n_perm=0, test_split=0)

    train_scores[i], _ = spearmanr(
        pls_cv["x_scores"][:, LV_IDX],
        pls_cv["y_scores"][:, LV_IDX],
    )

    test_x_score = Xtest @ pls_cv["x_weights"][:, LV_IDX]
    test_y_score = Ytest @ pls_cv["y_weights"][:, LV_IDX]
    test_scores[i], _ = spearmanr(test_x_score, test_y_score)

fig, ax = plt.subplots(figsize=(4, 6), dpi=300)
sns.boxplot(
    data=[train_scores, test_scores],
    zorder=1,
    width=0.6,
    linewidth=5,
    fill=False,
    palette=PALETTE_BIPOLAR,
    flierprops=dict(
        marker="o",
        markersize=7,
        alpha=0.4,
        linestyle="none",
        markeredgewidth=0,
    ),
)

ax.set_xticklabels(["Train", "Test"])
plt.ylabel("Score correlation", fontsize=18)
ax.tick_params(axis="both", labelsize=18)
sns.despine(trim=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "EX-IN_PLS_cv_scores.pdf"), dpi=300)
plt.show()