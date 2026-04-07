import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from PIL import Image
from scipy.stats import zscore
from neuromaps.stats import compare_images
from netneurotools import plotting
from palettable.colorbrewer.sequential import YlGnBu_7, YlGn_9, YlOrBr_7


# =========================
# Configuration
# =========================
PARC = "Schaefer400"
BASE_PATH = "project_whole_brain_mapping/final_code/"
OUTPUT_ROOT = os.path.join(
    BASE_PATH, "results", f"parcellated_{PARC}", "celltype_brainmap_compare_multimap"
)
ATLAS_PATH = "analysis/ROI/gii_atlas/fslr32k/"
SPIN_PATH = os.path.join(
    BASE_PATH, "materials", "fslr32k_spins1000_Schaefer400.npy"
)

SCATTER_FIGSIZE = (6, 6)
FONT_SIZE = 23

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# =========================
# Helper functions
# =========================
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def create_and_save_pv_plot_surface(data, hemisphere_mesh, output_path, cmap="viridis"):
    data_array = np.zeros(len(hemisphere_mesh.darrays[0].data))
    for i, value in enumerate(data):
        data_array[hemisphere_mesh.darrays[0].data == i + 1] = value

    plotting.pv_plot_surface(
        data_array,
        template="fslr32k",
        surf="inflated",
        hemi="L",
        cmap=cmap,
        lighting_style="threelights",
        zoom_ratio=1.2,
        show_colorbar=True,
        plotter_kws={"window_size": [2000, 1500]},
        cbar_kws={
            "label_font_size": 60,
            "vertical": True,
            "width": 0.15,
            "height": 0.2,
            "position_x": 0.8,
            "position_y": 0.1,
        },
        mesh_kws={"opacity": 1, "diffuse": 0.9, "specular": 0.2},
        save_fig=output_path,
    )

    try:
        im = Image.open(output_path)
        im.save(output_path, dpi=(450, 450))
    except Exception as e:
        print(f"Warning: Image DPI adjustment failed for {output_path}: {e}")


def run_analysis_and_plot(
    data_a,
    data_b,
    name_a,
    name_b,
    spins,
    atlas_mesh,
    cmap_obj,
    color_hex,
    label_a=None,
    label_b=None,
):
    safe_name_a = name_a.replace("/", "_").replace(" ", "_")
    safe_name_b = name_b.replace("/", "_").replace(" ", "_")

    current_out_dir = os.path.join(OUTPUT_ROOT, f"{safe_name_a}_{safe_name_b}_corr")
    os.makedirs(current_out_dir, exist_ok=True)

    n_perm = spins.shape[1]
    rotated = np.zeros((len(data_a), n_perm), dtype=float)
    for i in range(n_perm):
        rotated[:, i] = data_a[spins[:, i]]

    corr, pval = compare_images(data_a, data_b, nulls=rotated, metric="pearsonr")
    print(f"{name_a} vs {name_b}: r = {corr:.4f}, p-spin = {pval:.4f}")

    data_a_z = zscore(data_a)
    data_b_z = zscore(data_b)

    plt.figure(figsize=SCATTER_FIGSIZE)
    mpl_cmap = cmap_obj.mpl_colormap.reversed() if hasattr(cmap_obj, "mpl_colormap") else cmap_obj

    plt.scatter(
        data_b_z,
        data_a_z,
        c=data_b_z,
        cmap=mpl_cmap,
        alpha=0.95,
        s=300,
    )
    sns.regplot(
        x=data_b_z,
        y=data_a_z,
        scatter=False,
        color=color_hex,
        line_kws={"linewidth": 5},
    )

    pval_text = r"$P$ < 0.001" if pval < 0.001 else rf"$P$ = {pval:.3f}"
    plt.text(
        0.05,
        1.05,
        f"$r$ = {corr:.3f}\n{pval_text}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=FONT_SIZE + 2,
    )

    plt.xlabel(label_b if label_b else name_b, fontsize=FONT_SIZE)
    plt.ylabel(label_a if label_a else name_a, fontsize=FONT_SIZE)

    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position(("outward", 20))
    ax.spines["bottom"].set_position(("outward", 20))
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(
        os.path.join(current_out_dir, f"{safe_name_a}_{safe_name_b}_corr.svg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    create_and_save_pv_plot_surface(
        data_a_z,
        atlas_mesh,
        os.path.join(current_out_dir, f"{safe_name_a}_pv_plot_surface.png"),
        cmap=mpl_cmap,
    )
    create_and_save_pv_plot_surface(
        data_b_z,
        atlas_mesh,
        os.path.join(current_out_dir, f"{safe_name_b}_pv_plot_surface.png"),
        cmap=mpl_cmap,
    )


# =========================
# Load data
# =========================
layer_path = os.path.join(BASE_PATH, f"results/parcellated_{PARC}")

idx_ex = load_pkl(os.path.join(layer_path, "lh_layer_EX_index.pkl"))
idx_in = load_pkl(os.path.join(layer_path, "lh_layer_IN_index.pkl"))
idx_no = load_pkl(os.path.join(layer_path, "lh_layer_NO_neuron_index.pkl"))
Layer_ALL_index = {**idx_ex, **idx_in, **idx_no}

with open(os.path.join(BASE_PATH, "materials", "filtered_compared_map_index.pkl"), "rb") as f:
    filtered_compared_map_index = pickle.load(f)

lh_parcellation_atlas = nib.load(f"{ATLAS_PATH}{PARC}_conte69_lh.label.gii")
spins = np.load(SPIN_PATH)[:200, :]

# =========================
# analyses
# =========================

# L2/3 IT + L5 IT vs evolutionary expansion
run_analysis_and_plot(
    data_a=Layer_ALL_index["L2/3 IT"] + Layer_ALL_index["L5 IT"],
    data_b=filtered_compared_map_index["lh_xu2020_desc_evoexp"],
    name_a="L2_3_IT_L5_IT",
    name_b="evolutionary_expansion",
    label_a="L2/3 IT L5 IT",
    label_b="Evolutionary expansion",
    spins=spins,
    atlas_mesh=lh_parcellation_atlas,
    cmap_obj=YlOrBr_7,
    color_hex="orange",
)

# L4 IT vs L4 thickness
run_analysis_and_plot(
    data_a=Layer_ALL_index["L4 IT"],
    data_b=filtered_compared_map_index["lh_L4_layer_thickness"],
    name_a="L4_IT",
    name_b="L4_layer_thickness",
    label_a="L4 IT",
    label_b="L4 layer thickness",
    spins=spins,
    atlas_mesh=lh_parcellation_atlas,
    cmap_obj=YlOrBr_7,
    color_hex="orange",
)

# OPC/Oligo vs functional connectivity gradient
run_analysis_and_plot(
    data_a=np.log2(Layer_ALL_index["OPC"] / Layer_ALL_index["Oligo"]),
    data_b=filtered_compared_map_index["lh_fcgradient01"],
    name_a="OPC_Oligo",
    name_b="FC_gradient",
    label_a="OPC/Oligo",
    label_b="FC gradient",
    spins=spins,
    atlas_mesh=lh_parcellation_atlas,
    cmap_obj=YlGn_9,
    color_hex="green",
)

# Pvalb vs high gamma
run_analysis_and_plot(
    data_a=Layer_ALL_index["Pvalb"],
    data_b=filtered_compared_map_index["lh_gamma2"],
    name_a="Pvalb",
    name_b="High_gamma",
    label_a="Pvalb",
    label_b="High gamma",
    spins=spins,
    atlas_mesh=lh_parcellation_atlas,
    cmap_obj=YlGnBu_7,
    color_hex="skyblue",
)