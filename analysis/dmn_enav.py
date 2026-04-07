import os
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
from neuromaps.datasets import fetch_fslr
from neuromaps.stats import compare_images
from surfplot import Plot
from palettable.colorbrewer.diverging import RdBu_8


# =========================
# Configuration
# =========================
PATH = "project_whole_brain_mapping/final_code"
MATERIALS_DIR = f"{PATH}/materials"
RESULTS_DIR = f"{PATH}/results"

sch400_atlas7_idx = loadmat(
    f"{MATERIALS_DIR}/DMN/sch400_atlas7_cort.mat"
)["sch400_atlas7_cort"][0, :]
lh_sch400_atlas7_cort_dmn_idx = np.where(sch400_atlas7_idx[:200] == 8)[0]
lh_spins = np.load(f"{MATERIALS_DIR}/fslr32k_spins1000_Schaefer400.npy")[:200, :]

bigbrain_g1 = nib.load(
    "retest/DMN_cyto/tpl-fs_LR_hemi-L_den-32k_desc-Hist_G1.shape.gii"
)

# These navigation-efficiency profiles were computed based on the code associated with Paquola et al. (2025), Nature Neuroscience.
mat_hcp = loadmat(f"{MATERIALS_DIR}/DMN/enav_inter_intra_all_hcp.mat")
enav_profiles = {
    "intra": mat_hcp["intra_conn_extrinsic"][0, :],
    "inter": mat_hcp["inter_conn_extrinsic"][0, :],
    "all": mat_hcp["all_conn_extrinsic"][0, :],
}

parc = "Schaefer400"
lh_gii_parcellation_img = nib.load(
    f"{MATERIALS_DIR}/ROI/fslr32k/{parc}_conte69_lh.label.gii"
)
lh_bigbrain_g1_parc = np.load(f"{MATERIALS_DIR}/DMN/lh_bigbrain_G1_parc.npy")
path_parc = f"{RESULTS_DIR}/parcellated_{parc}/"


# =========================
# Load cell-type profiles
# =========================
lh_layer_all_index = {}
for name in ["EX", "IN", "NO_neuron"]:
    with open(f"{path_parc}lh_layer_{name}_index.pkl", "rb") as f:
        lh_layer_all_index.update(pickle.load(f))


# =========================
# Load surfaces
# =========================
surfaces = fetch_fslr(density="32k")
lh, _ = surfaces["inflated"]


# =========================
# BigBrain G1 correlation across all cell types
# =========================
celltype_dmn = np.zeros(200)
celltype_dmn[lh_sch400_atlas7_cort_dmn_idx] = lh_bigbrain_g1_parc[
    lh_sch400_atlas7_cort_dmn_idx
]

results = []
for key, arr in lh_layer_all_index.items():
    profile_dmn = np.array(arr)
    rotated = profile_dmn[lh_spins]
    corr, pval = compare_images(
        profile_dmn,
        celltype_dmn,
        nulls=rotated,
        metric="pearsonr",
    )
    results.append({"term": key, "corr": corr, "pval": pval})

df_results = pd.DataFrame(results)
reject, pval_fdr, _, _ = multipletests(df_results["pval"], method="fdr_bh")
df_results["pval_fdr"] = pval_fdr
df_results["reject_H0"] = reject
print(df_results)


# =========================
# Heatmap of correlations and FDR-corrected p-values
# =========================
df_plot = df_results.sort_values(by="corr", ascending=False).reset_index(drop=True)
heat_data = df_plot.set_index("term")[["corr"]]

annot_data = []
for _, row in df_plot.iterrows():
    txt = f"{row['pval_fdr']:.3f}"
    if row["reject_H0"]:
        txt += "*"
    annot_data.append([txt])
annot_data = np.array(annot_data)

plt.figure(figsize=(3, 8.8))
ax = sns.heatmap(
    heat_data,
    annot=annot_data,
    fmt="",
    cmap="RdBu_r",
    vmin=df_plot["corr"].min(),
    vmax=df_plot["corr"].max(),
    linewidths=1.2,
    linecolor="black",
    cbar=True,
    annot_kws={"fontsize": 12, "color": "black"},
    cbar_kws={
        "orientation": "horizontal",
        "shrink": 0.7,
        "pad": 0.08,
    },
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels([])
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
ax.tick_params(axis="both", length=0)

cbar = ax.collections[0].colorbar
cbar.ax.set_title("Corr", fontsize=14, pad=8)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/DMN/corr_heatmap.pdf", bbox_inches="tight")
plt.show()


# =========================
# Scatter plots for selected cell types against BigBrain G1
# =========================
celltype_dmn_z = np.zeros(200)
celltype_dmn_z[lh_sch400_atlas7_cort_dmn_idx] = zscore(lh_bigbrain_g1_parc)[
    lh_sch400_atlas7_cort_dmn_idx
]

results_z = []
for key in ["L2/3 IT", "L4 IT", "L5 IT"]:
    profile_dmn = zscore(np.array(lh_layer_all_index[key]))
    rotated = profile_dmn[lh_spins]
    corr, pval = compare_images(
        profile_dmn,
        celltype_dmn_z,
        nulls=rotated,
        metric="pearsonr",
    )
    results_z.append({"term": key, "corr": corr, "pval": pval})

    x_vals = profile_dmn[lh_sch400_atlas7_cort_dmn_idx]
    y_vals = celltype_dmn_z[lh_sch400_atlas7_cort_dmn_idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(
        x_vals,
        y_vals,
        c=y_vals,
        cmap=RdBu_8.mpl_colormap.reversed(),
        alpha=0.95,
        s=300,
    )
    sns.regplot(
        x=x_vals,
        y=y_vals,
        scatter=False,
        color="gray",
        line_kws={"linewidth": 5},
    )

    pval_text = (
        r"$\it{P}$ < 0.001"
        if pval < 0.001
        else rf"$\it{{P}}$ = {pval:.3f}"
    )
    plt.text(
        0.05,
        1.05,
        f"$r$ = {corr:.3f}\n{pval_text}",
        transform=plt.gca().transAxes,
        va="top",
        fontsize=20,
    )

    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=24)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("outward", 20))
    ax.spines["bottom"].set_position(("outward", 20))

    plt.tight_layout()
    plt.show()
    plt.close()

print(pd.DataFrame(results_z))


# =========================
# Correlation with eNav profiles
# =========================
results_enav = []

for enav_name, enav_data in enav_profiles.items():
    celltype_dmn_enav = np.zeros(200)
    celltype_dmn_enav[lh_sch400_atlas7_cort_dmn_idx] = enav_data[:200][
        lh_sch400_atlas7_cort_dmn_idx
    ]

    for key in ["L2/3_IT", "L4 IT", "L5 IT"]:
        lookup_key = key.replace("_", " ") if key not in lh_layer_all_index else key

        profile_dmn = np.zeros(200)
        profile_dmn[lh_sch400_atlas7_cort_dmn_idx] = zscore(
            np.array(lh_layer_all_index[lookup_key])
        )[lh_sch400_atlas7_cort_dmn_idx]

        rotated = profile_dmn[lh_spins]
        corr, pval = compare_images(
            profile_dmn,
            celltype_dmn_enav,
            nulls=rotated,
            metric="pearsonr",
        )
        results_enav.append(
            {"enav": enav_name, "celltype": key, "corr": corr, "pval": pval}
        )

        x_vals = celltype_dmn_enav[lh_sch400_atlas7_cort_dmn_idx]
        y_vals = profile_dmn[lh_sch400_atlas7_cort_dmn_idx]

        plt.figure(figsize=(5, 5))
        plt.scatter(
            x_vals,
            y_vals,
            c=x_vals,
            cmap=RdBu_8.mpl_colormap.reversed(),
            alpha=0.95,
            s=300,
        )
        sns.regplot(
            x=x_vals,
            y=y_vals,
            scatter=False,
            color="gray",
            line_kws={"linewidth": 5},
        )

        if pval < 0.001:
            pval_text = r"$\it{P}$ < 0.001*"
        elif pval < 0.05:
            pval_text = rf"$\it{{P}}$ = {pval:.3f}*"
        else:
            pval_text = rf"$\it{{P}}$ = {pval:.3f}"

        plt.text(
            0.05,
            1.05,
            f"$r$ = {corr:.3f}\n{pval_text}",
            transform=plt.gca().transAxes,
            va="top",
            fontsize=20,
        )

        ax = plt.gca()
        ax.tick_params(axis="both", labelsize=25)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_position(("outward", 20))
        ax.spines["bottom"].set_position(("outward", 20))

        plt.tight_layout()
        clean_key = key.replace("/", "_").replace(" ", "_").lower()
        plt.savefig(f"{RESULTS_DIR}/DMN/{enav_name}_{clean_key}.pdf")
        plt.show()
        plt.close()

print(pd.DataFrame(results_enav))


# =========================
# Surface plots for eNav profiles
# =========================
for profile_name, profile_data in enav_profiles.items():
    lh_parc = np.zeros(200)
    lh_parc[lh_sch400_atlas7_cort_dmn_idx] = profile_data[
        lh_sch400_atlas7_cort_dmn_idx
    ]

    lh_data = np.zeros(len(lh_gii_parcellation_img.darrays[0].data))
    for i, value in enumerate(lh_parc):
        lh_data[lh_gii_parcellation_img.darrays[0].data == i + 1] = value

    p = Plot(surf_lh=lh, size=(1200, 600), zoom=1.2)
    p.add_layer({"left": lh_data}, cbar=True, cmap=RdBu_8.mpl_colormap.reversed())
    fig = p.build()

    plt.savefig(
        f"{RESULTS_DIR}/{profile_name}_DMN_profile.png",
        dpi=450,
        bbox_inches="tight",
    )
    plt.show()