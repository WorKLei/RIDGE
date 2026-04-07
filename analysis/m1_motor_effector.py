import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from palettable.colorbrewer.sequential import YlGnBu_7, YlGn_9, YlOrBr_7


# =========================
# Configuration
# =========================
PATH = "project_whole_brain_mapping/final_code"
CELLTYPE_GII_SUMMARY_PATH = f"{PATH}/results/celltype_map"
LH_PARCELL_PATH = f"{PATH}/materials/ROI/fslr32k/glasser360_conte69_lh.label.gii"
FLAT_SURF_PATH = f"{PATH}/materials/template/fs_LR.32k.L.flat.surf.gii"

MOTOR_ROI = nib.load(f"{PATH}/materials/M1C/lh.Gordon17_with_SCAN_HCPMMP1_3a.label.gii")
ROI_LABELS = {
    "SCAN_top": 18,
    "SCAN_middle": 19,
    "SCAN_bottom": 20,
    "Foot": 17,
    "Hand": 10,
    "Mouth": 11,
}

os.makedirs(f"{PATH}/results/M1C", exist_ok=True)

NUM_BINS = 100
WINDOW_SIZE = 6
QUANTILE_CUTOFF = 0.5
BAR_HEIGHT = 0.05


# =========================
# Ventral-dorsal marker positions
# =========================
ventral_dorsal_all = nib.load(FLAT_SURF_PATH).darrays[0].data[:, 1]
ranges = []

for name, label_val in ROI_LABELS.items():
    indices = np.where(MOTOR_ROI.darrays[0].data == label_val)[0]
    vd_values = ventral_dorsal_all[indices]
    ranges.append(
        {
            "ROI": name,
            "VD_min": vd_values.min(),
            "VD_max": vd_values.max(),
        }
    )

ranges_df = pd.DataFrame(ranges)


def v(name, col):
    """Return the ventral-dorsal boundary value for a given ROI."""
    return ranges_df.loc[ranges_df["ROI"] == name, col].iloc[0]


foot = v("Foot", "VD_max")
inter_effector_top = (v("SCAN_top", "VD_max") + v("Foot", "VD_min")) / 2
hand = (v("Hand", "VD_max") + v("SCAN_top", "VD_min")) / 2
inter_effector_middle = (v("Hand", "VD_min") + v("SCAN_middle", "VD_max")) / 2
mouth = v("Mouth", "VD_max")
inter_effector_bottom = (v("Mouth", "VD_min") + v("SCAN_bottom", "VD_max")) / 2
inter_effector_bottom_2 = 0

marker_positions = {
    "Foot": foot,
    "inter_effector_top": inter_effector_top,
    "Hand": hand,
    "inter_effector_middle": inter_effector_middle,
    "Mouth": mouth,
    "inter_effector_bottom": inter_effector_bottom,
    "inter_effector_bottom_2": inter_effector_bottom_2,
}

label_pairs = [
    ("Foot", "inter_effector_top", "Foot"),
    ("inter_effector_top", "Hand", "inter-\neffector"),
    ("Hand", "inter_effector_middle", "Hand"),
    ("inter_effector_middle", "Mouth", "inter-\neffector"),
    ("Mouth", "inter_effector_bottom", "Mouth"),
    ("inter_effector_bottom", "inter_effector_bottom_2", "inter-\neffector"),
]


# =========================
# Utility functions
# =========================
def scale_to_minus1_1(arr):
    """Rescale an array to the range [-1, 1]."""
    arr = np.asarray(arr)
    vmin, vmax = np.min(arr), np.max(arr)
    if vmax == vmin:
        return np.zeros_like(arr)
    return 2 * (arr - vmin) / (vmax - vmin) - 1


def sliding_window_median(arr, window_size=5):
    """Apply a centered rolling median."""
    return pd.Series(arr).rolling(
        window=window_size,
        center=True,
        min_periods=1
    ).median().to_numpy()


def bin_and_trimmax(x_axis, values, n_bins=100, q=0.5):
    """Bin values along an axis and extract trimmed maxima within each bin."""
    bins = np.linspace(x_axis.min(), x_axis.max(), n_bins + 1)
    bin_centers, vmaxs = [], []

    for i in range(n_bins):
        idx = (x_axis >= bins[i]) & (x_axis < bins[i + 1])
        if np.any(idx):
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            v = values[idx]
            thr = np.quantile(v, q)
            vmaxs.append(np.max(v[v <= thr]))

    return np.array(bin_centers), np.array(vmaxs)


# =========================
# Core plotting functions
# =========================
def plot_celltype_bandfigure_pro_to_dis(
    category,
    ylabel,
    cmap_name,
    save_path,
    celltype_dir,
    sign_roi=None,
    top_cmap=None,
):
    """Plot a single cell-type profile along the ventral-dorsal motor axis."""
    lh_parcellation = nib.load(LH_PARCELL_PATH)
    label_data = lh_parcellation.darrays[0].data
    indices = (label_data == 8).nonzero()[0]

    ventral_dorsal = nib.load(FLAT_SURF_PATH).darrays[0].data[:, 1][indices]
    celltype_gii = f"{celltype_dir}/lh_{category}_fslr_cell2map.func.gii"
    celltype_data = nib.load(celltype_gii).darrays[0].data[indices]

    valid = celltype_data != 0
    ventral_dorsal = ventral_dorsal[valid]
    celltype_data = celltype_data[valid]

    x_bins, v_max = bin_and_trimmax(
        ventral_dorsal,
        celltype_data,
        n_bins=NUM_BINS,
        q=QUANTILE_CUTOFF,
    )

    y_scaled = sliding_window_median(
        scale_to_minus1_1(v_max),
        window_size=WINDOW_SIZE,
    )

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap(cmap_name)

    sns.lineplot(x=x_bins, y=y_scaled, color="gray", linewidth=1.5, zorder=1, ax=ax)
    ax.scatter(
        x_bins,
        y_scaled,
        c=y_scaled,
        cmap=cmap,
        s=80,
        alpha=0.95,
        edgecolor="gray",
        zorder=2,
    )

    ax.set_ylim(-1.2, 1.2)
    y_pos_top = 1.20
    y_pos_bot = y_pos_top - BAR_HEIGHT
    x0, x1 = ax.get_xlim()

    grad = np.linspace(0, 1, 256).reshape(1, -1)
    grad = np.vstack([grad, grad])

    cmap_top = top_cmap if top_cmap is not None else cmap
    ax.imshow(
        grad,
        extent=(x0, x1, y_pos_bot, y_pos_top),
        origin="lower",
        aspect="auto",
        cmap=cmap_top,
        zorder=1,
    )

    for left, right, label in label_pairs:
        if label.startswith("inter-"):
            continue
        x_left = marker_positions[left]
        x_right = marker_positions[right]
        x_mid = (x_left + x_right) / 2.0
        ax.text(
            x_mid,
            y_pos_top,
            label,
            ha="center",
            va="bottom",
            fontsize=25,
            color="black",
            zorder=2,
        )

    if sign_roi is not None:
        for left, right, label in label_pairs:
            if label.lower() == sign_roi.lower():
                xpos_left = marker_positions[left]
                xpos_right = marker_positions[right]
                ax.axvline(
                    x=xpos_left,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=3,
                )
                ax.axvline(
                    x=xpos_right,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=3,
                )
                break
        else:
            print(f"Warning: sign_roi='{sign_roi}' must be one of Foot, Hand, or Mouth.")

    ax.set_ylabel(ylabel, fontsize=25)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)
    ax.spines["left"].set_position(("outward", 20))
    ax.spines["bottom"].set_position(("outward", 20))
    sns.despine()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    plt.show()


def plot_multi_celltype_bandfigure_pro_to_dis(
    categories,
    ylabels,
    colors=None,
    cmap_name="YlOrBr",
    save_path=None,
    celltype_dir="",
    sign_roi=None,
    top_cmap=None,
):
    """Plot multiple cell-type profiles along the ventral-dorsal motor axis."""
    lh_parcellation = nib.load(LH_PARCELL_PATH)
    label_data = lh_parcellation.darrays[0].data
    indices = (label_data == 8).nonzero()[0]

    if isinstance(colors, str):
        cmap_obj = plt.get_cmap(colors)
        colors = [cmap_obj(i) for i in range(len(categories))]
    elif colors is None:
        color_cycle = plt.get_cmap("tab10").colors
        colors = [color_cycle[i % len(color_cycle)] for i in range(len(categories))]

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(12, 8))

    for category, ylabel, color in zip(categories, ylabels, colors):
        ventral_dorsal = nib.load(FLAT_SURF_PATH).darrays[0].data[:, 1][indices]
        celltype_gii = f"{celltype_dir}/lh_{category}_fslr_cell2map.func.gii"
        celltype_data = nib.load(celltype_gii).darrays[0].data[indices]

        valid = celltype_data != 0
        ventral_dorsal = ventral_dorsal[valid]
        celltype_data = celltype_data[valid]

        x_bins, v_max = bin_and_trimmax(
            ventral_dorsal,
            celltype_data,
            n_bins=NUM_BINS,
            q=QUANTILE_CUTOFF,
        )

        y_scaled = sliding_window_median(
            scale_to_minus1_1(v_max),
            window_size=WINDOW_SIZE,
        )

        sns.lineplot(
            x=x_bins,
            y=y_scaled,
            color=color,
            linewidth=1.5,
            zorder=1,
            ax=ax,
            label=ylabel,
        )
        ax.scatter(
            x_bins,
            y_scaled,
            color=color,
            s=40,
            alpha=0.8,
            edgecolor="gray",
            zorder=2,
        )

    ax.set_ylim(-1.2, 1.2)

    y_pos_top = 1.20
    y_pos_bot = y_pos_top - BAR_HEIGHT
    x0, x1 = ax.get_xlim()

    grad = np.linspace(0, 1, 256).reshape(1, -1)
    grad = np.vstack([grad, grad])

    cmap_top = top_cmap if top_cmap is not None else plt.get_cmap(cmap_name)
    ax.imshow(
        grad,
        extent=(x0, x1, y_pos_bot, y_pos_top),
        origin="lower",
        aspect="auto",
        cmap=cmap_top,
        zorder=1,
    )

    for left, right, label in label_pairs:
        if label.startswith("inter-"):
            continue
        x_left = marker_positions[left]
        x_right = marker_positions[right]
        x_mid = (x_left + x_right) / 2.0
        ax.text(
            x_mid,
            y_pos_top,
            label,
            ha="center",
            va="bottom",
            fontsize=25,
            color="black",
            zorder=2,
        )

    if sign_roi is not None:
        for left, right, label in label_pairs:
            if label.lower() == sign_roi.lower():
                xpos_left = marker_positions[left]
                xpos_right = marker_positions[right]
                ax.axvline(
                    x=xpos_left,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=3,
                )
                ax.axvline(
                    x=xpos_right,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=3,
                )
                break
        else:
            print(f"Warning: sign_roi='{sign_roi}' must be one of Foot, Hand, or Mouth.")

    ax.legend(
        fontsize=12,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.spines["left"].set_position(("outward", 20))
    ax.spines["bottom"].set_position(("outward", 20))
    sns.despine()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    plt.show()


# =========================
# Single cell-type plots
# =========================
plot_celltype_bandfigure_pro_to_dis(
    category="L5_ET",
    ylabel="L5 ET",
    cmap_name=YlOrBr_7.mpl_colormap.reversed(),
    save_path=f"{PATH}/results/M1C/lh_L5_ET_pro_to_dis.svg",
    celltype_dir=CELLTYPE_GII_SUMMARY_PATH,
    sign_roi="Foot",
)

plot_celltype_bandfigure_pro_to_dis(
    category="Oligo",
    ylabel="Oligo",
    cmap_name=YlGn_9.mpl_colormap.reversed(),
    save_path=f"{PATH}/results/M1C/lh_Oligo_pro_to_dis.svg",
    celltype_dir=CELLTYPE_GII_SUMMARY_PATH,
    sign_roi="Hand",
)

plot_celltype_bandfigure_pro_to_dis(
    category="Sst",
    ylabel="Sst",
    cmap_name=YlGnBu_7.mpl_colormap.reversed(),
    save_path=f"{PATH}/results/M1C/lh_Sst_pro_to_dis.svg",
    celltype_dir=CELLTYPE_GII_SUMMARY_PATH,
    sign_roi="Mouth",
)


# =========================
# Multi cell-type plots
# =========================
layer_ex_original_categories = [
    "L2/3 IT", "L4 IT", "L5 ET", "L5 IT", "L5/6 NP",
    "L6 CT", "L6 IT", "L6 IT Car3", "L6b"
]
layer_in_original_categories = [
    "Chandelier", "Lamp5", "Lamp5 Lhx6", "Pax6", "Pvalb",
    "Sncg", "Sst", "Sst Chodl", "Vip"
]
layer_non_neuronal_original_categories = [
    "Astro", "Micro/PVM", "OPC", "Oligo", "VLMC", "Endo"
]

layer_ex_selected_categories = [
    s.replace("/", "_").replace(" ", "_") for s in layer_ex_original_categories
]
layer_in_selected_categories = [
    s.replace("/", "_").replace(" ", "_") for s in layer_in_original_categories
]
layer_non_neuronal_selected_categories = [
    s.replace("/", "_").replace(" ", "_") for s in layer_non_neuronal_original_categories
]

plot_multi_celltype_bandfigure_pro_to_dis(
    categories=layer_ex_selected_categories,
    ylabels=layer_ex_original_categories,
    colors="Set3",
    cmap_name=YlOrBr_7.mpl_colormap.reversed(),
    celltype_dir=CELLTYPE_GII_SUMMARY_PATH,
    save_path=f"{PATH}/results/M1C/supp_lh_EX.svg",
)

plot_multi_celltype_bandfigure_pro_to_dis(
    categories=layer_in_selected_categories,
    ylabels=layer_in_original_categories,
    colors="Set3",
    cmap_name=YlGnBu_7.mpl_colormap.reversed(),
    celltype_dir=CELLTYPE_GII_SUMMARY_PATH,
    save_path=f"{PATH}/results/M1C/supp_lh_IN.svg",
)

plot_multi_celltype_bandfigure_pro_to_dis(
    categories=layer_non_neuronal_selected_categories,
    ylabels=layer_non_neuronal_original_categories,
    colors="Set3",
    cmap_name=YlGn_9.mpl_colormap.reversed(),
    celltype_dir=CELLTYPE_GII_SUMMARY_PATH,
    save_path=f"{PATH}/results/M1C/supp_lh_No_neuronal.svg",
)