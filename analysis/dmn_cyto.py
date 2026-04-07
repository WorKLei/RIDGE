import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
import nibabel as nib
from scipy.stats import zscore
from neuromaps.stats import compare_images

# =========================
# Configuration
# =========================
PATH = "project_whole_brain_mapping/final_code"
OUT_DIR = f"{PATH}/results/DMN"
os.makedirs(OUT_DIR, exist_ok=True)

EXIST_CELLTYPES = ["L2/3 IT", "L4 IT", "L5 IT"]
EXIST_CELLTYPES_EDITED = [
    cell_type.replace("/", "_").replace(" ", "_")
    for cell_type in EXIST_CELLTYPES
]

# =========================
# Load reference data
# =========================
lh_economo = nib.load(f"{PATH}/materials/ROI/fslr32k/economo_conte69_lh.label.gii")
bigbrain_g1 = nib.load(f"{PATH}/materials/DMN/tpl-fs_LR_hemi-L_den-32k_desc-Hist_G1.shape.gii")
lh_spins = np.load(f"{PATH}/materials/lh_spins_1000_32k.npy")
lh_rsn_networks = nib.load(f"{PATH}/materials/ROI/fslr32k/lh_RSN-networks_fs_LR.label.gii")
lh_rsn_networks_dmn = np.where(lh_rsn_networks.darrays[0].data == 40)[0]

bb_arr = bigbrain_g1.darrays[0].data
bb_masked = np.zeros(len(bb_arr))
bb_masked[lh_rsn_networks_dmn] = bb_arr[lh_rsn_networks_dmn]


# =========================
# Load cell-type maps
# =========================
celltype_imgs = {}
for celltype in EXIST_CELLTYPES_EDITED:
    celltype_imgs[celltype] = nib.load(
        f"{PATH}/results/celltype_map/lh_{celltype}_fslr_cell2map.func.gii"
    )

lh_gii_l2_3_it_img = celltype_imgs["L2_3_IT"]
lh_gii_l4_it_img = celltype_imgs["L4_IT"]
lh_gii_l5_it_img = celltype_imgs["L5_IT"]


# =========================
# Spatial correlation analysis with spin nulls
# =========================
results = []

for celltype in EXIST_CELLTYPES_EDITED:
    celltype_img = celltype_imgs[celltype]
    arr = celltype_img.darrays[0].data

    arr_masked = np.zeros(len(arr))
    arr_masked[lh_rsn_networks_dmn] = arr[lh_rsn_networks_dmn]

    rotated = np.zeros(lh_spins.shape, dtype=float)
    for i in range(lh_spins.shape[1]):
        rotated[:, i] = arr[lh_spins[:, i]]

    corr, pval = compare_images(
        arr_masked,
        bb_masked,
        nulls=rotated,
        metric="pearsonr",
    )
    results.append({"celltype": celltype, "corr": corr, "pval": pval})

df_results = pd.DataFrame(results)
print(df_results)


# =========================
# KDE correlation plots
# =========================
sns.set_style("white")
sns.set_context("talk")

for _, row in df_results.iterrows():
    celltype = row["celltype"]
    r = row["corr"]
    p_spin = row["pval"]

    celltype_img = celltype_imgs[celltype]
    arr = celltype_img.darrays[0].data

    arr_masked = np.zeros(len(arr))
    arr_masked[lh_rsn_networks_dmn] = arr[lh_rsn_networks_dmn]

    mask_valid = (arr_masked != 0) & (bb_masked != 0)
    x = zscore(arr_masked[mask_valid])
    y = bb_masked[mask_valid]

    plt.figure(figsize=(5, 5))
    sns.kdeplot(
        x=x,
        y=y,
        fill=True,
        cmap="Blues",
        thresh=0.05,
        levels=30,
        alpha=0.9,
    )
    sns.regplot(
        x=x,
        y=y,
        scatter=False,
        color="steelblue",
        line_kws={"linewidth": 1.5},
    )

    plt.text(
        0.05,
        0.95,
        f"$r$ = {r:.2f}\n$P$ = {p_spin:.3f}",
        transform=plt.gca().transAxes,
        fontsize=28,
        ha="left",
        va="top",
    )

    ax = plt.gca()
    sns.despine(ax=ax)
    plt.grid(False)
    ax.spines["left"].set_position(("outward", 20))
    ax.spines["bottom"].set_position(("outward", 20))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)

    plt.margins(x=0.05, y=0.05)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{celltype}_kdeplot.svg")
    plt.show()
    plt.close()

# =========================
# Load DMN subregion indices
# =========================
with open(f"{PATH}/materials/DMN/subregion_indices.pkl", "rb") as f:
    subregion_indices = pickle.load(f)

subregion_dict = {
    "precuneus": subregion_indices["precuneus"],
    "prefrontal": subregion_indices["prefrontal"],
    "inferior_frontal": subregion_indices["inferior_frontal"],
    "parahippocampus": subregion_indices["parahippocampus"],
    "inferior_parietal": subregion_indices["inferior_parietal"],
    "middle_temporal": subregion_indices["middle_temporal"],
}


# =========================
# RainCloud plots across Economo classes
# =========================
y_vars = ["bigbrain_G1", "L2_3_IT_value", "L4_IT_value", "L5_IT_value"]

econ_label_map = {
    1: "Kon",
    2: "Eu-III",
    3: "Eu-II",
    4: "Eu-I",
    5: "Dys",
    6: "Ag",
}
order = ["Kon", "Eu-III", "Eu-II", "Eu-I", "Dys", "Ag"]
palette = ["#7f8cac", "#8ba7b0", "#abbaa2", "#dac699", "#fdd3c8", "#fce5fc"]

for region_name, subregion_index in subregion_dict.items():
    df = pd.DataFrame(
        {
            "bigbrain_G1": bigbrain_g1.darrays[0].data[subregion_index],
            "Economo_type": lh_economo.darrays[0].data[subregion_index],
            "L2_3_IT_value": lh_gii_l2_3_it_img.darrays[0].data[subregion_index],
            "L4_IT_value": lh_gii_l4_it_img.darrays[0].data[subregion_index],
            "L5_IT_value": lh_gii_l5_it_img.darrays[0].data[subregion_index],
        }
    )

    df = df[df["Economo_type"] != 0]
    df["Economo_label"] = df["Economo_type"].map(econ_label_map)

    valid_labels = [label for label in order if label in df["Economo_label"].unique()]
    valid_palette = [palette[order.index(label)] for label in valid_labels]

    for y_var in y_vars:
        plt.figure(figsize=(5, 5))
        ax = pt.RainCloud(
            x="Economo_label",
            y=y_var,
            hue="Economo_label",
            hue_order=valid_labels,
            data=df,
            order=valid_labels,
            palette=valid_palette,
            width_viol=0.4,
            width_box=0.2,
            orient="v",
            alpha=0.9,
            offset=0.2,
            move=0,
            point_size=4,
            box_showfliers=False,
            box_linewidth=0.7,
            dodge=False,
        )

        if ax.legend_:
            ax.legend_.remove()

        medians = df.groupby("Economo_label")[y_var].median().reindex(valid_labels)
        x_positions = np.arange(len(valid_labels))

        plt.plot(
            x_positions,
            medians.values,
            color="gray",
            marker="o",
            markersize=6,
            linewidth=3,
            zorder=10,
        )

        sns.despine(left=True, top=True, right=True)
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(ticks=x_positions, labels=valid_labels, fontsize=20)
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"{region_name}_{y_var}.pdf")
        plt.savefig(out_path)
        plt.show()
        plt.close()