# This script integrates dtangle-derived cell-type proportion estimates
# Note
# This script converts dtangle-derived cell-type expression estimates
# into fsLR-32k cortical maps. Each regional deconvolution result is
# first restricted to its corresponding Brodmann ROI, then merged across
# ROIs to generate whole-cortex cell-type expression maps.
import os
import glob
import re
import gc
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from neuromaps.datasets import fetch_fslr
from surfplot import Plot

# ============================================================
# Set working directory
# ============================================================
os.chdir("/mnt/data0/home/wklei/neuro_work")

# ============================================================
# Set paths
# ============================================================
input_path = (
    "/mount_data_file/materials/deconvoluation_content/"
    "dtangle_result_wholecortex_MAGICC"
)

brodmann_gii_path = (
    "/mnt/data0/home/wklei/neuro_work/"
    "analysis/ROI/gii_atlas/fslr32k/Brodmann_conte69_lh.label.gii"
)

midthickness_surface_path = (
    "/mnt/data0/home/wklei/neuromaps-data/atlases/fsLR/"
    "tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii"
)

brodmann_roi_csv_dir = os.path.join(
    input_path,
    "brodmann_roi_csv"
)

origin_gii_dir = os.path.join(
    input_path,
    "cell2map_expr_gii"
)

smoothed_gii_dir = os.path.join(
    input_path,
    "cell2map_expr_sm6_gii"
)

smoothed_fig_dir = os.path.join(
    input_path,
    "cell2map_expr_sm6_brainmap"
)

proportion_csv_dir = os.path.join(
    input_path,
    "cell2map_sm6_proportion"
)

proportion_fig_dir = os.path.join(
    input_path,
    "cell2map_sm6_proportion_brainmap"
)

for folder in [
    brodmann_roi_csv_dir,
    origin_gii_dir,
    smoothed_gii_dir,
    smoothed_fig_dir,
    proportion_csv_dir,
    proportion_fig_dir,
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# Cell-type groups
# ============================================================
layer_EX_selected_categories = [
    "L2_3_IT", "L4_IT", "L5_ET", "L5_IT", "L5_6_NP",
    "L6_CT", "L6_IT", "L6_IT_Car3", "L6b"
]

layer_IN_selected_categories = [
    "Chandelier", "Lamp5", "Lamp5_Lhx6", "Pax6", "Pvalb",
    "Sncg", "Sst", "Sst_Chodl", "Vip"
]

layer_NO_neuron_selected_categories = [
    "Astro", "Micro_PVM", "OPC", "Oligo", "VLMC", "Endo"
]

categories = [
    layer_EX_selected_categories,
    layer_IN_selected_categories,
    layer_NO_neuron_selected_categories,
]

cell_types = (
    layer_EX_selected_categories
    + layer_IN_selected_categories
    + layer_NO_neuron_selected_categories
)


# ============================================================
# Brodmann mapping
# ============================================================
df_to_brodmann = {
    "BA9": [7, 8, 9],
    "Human_A5-A7_2": 6,
    "Human_A5-A7": 4,
    "Human_LEC": 28,
    "Human_A19": 15,
    "Human_A23": 19,
    "Human_A29-A30": 22,
    "Human_FI": 11,
    "Human_A14": 10,
    "Human_ITG": 16,
    "Human_V1C": 13,
    "Human_Idg": 35,
    "Human_MEC": 24,
    "Human_A46": 38,
    "Human_MTG": 17,
    "Human_A43": 35,
    "Human_A35-A36": 29,
    "Human_V2": 14,
    "Human_A13": 39,
    "Human_TH-TL": 23,
    "Human_M1C": [3, 5],
    "Human_A1C": 34,
    "Human_S1C_2": 2,
    "Human_S1C": 1,
    "Human_A32": 26,
    "Human_Ig": 12,
    "Human_A25": 21,
    "Human_STG": 18,
    "Human_A40": 33,
    "Human_TF": 30,
    "Human_A38": 31,
    "Human_A24": 20,
    "Human_A44-A45_2": 37,
    "Human_A44-A45": 36,
    "Human_A31": 25,
    "AnG": 32,
}


def clean_dtangle_basename(filename):
    suffix = "_wholecortex_MAGICC_deconvlution.csv"

    if filename.endswith(suffix):
        return filename[:-len(suffix)]

    return filename.replace(".csv", "")


# ============================================================
# Load Brodmann atlas and fsLR surface
# ============================================================
lh_brodmann = nib.load(brodmann_gii_path)
lh_brodmann_data = lh_brodmann.darrays[0].data

surfaces = fetch_fslr(density="32k")
lh_surface, rh_surface = surfaces["inflated"]

n_vertices = len(lh_brodmann_data)


# ============================================================
# Remove old intermediate ROI CSV files
# ============================================================
for old_file in glob.glob(os.path.join(brodmann_roi_csv_dir, "*.csv")):
    os.remove(old_file)


# ============================================================
# Split each dtangle result into corresponding Brodmann ROI CSV
# ============================================================
for csv_file in glob.glob(os.path.join(input_path, "*.csv")):

    filename = os.path.basename(csv_file)
    region_key = clean_dtangle_basename(filename)

    brodmann_value = df_to_brodmann.get(region_key)

    if brodmann_value is None:
        print(f"No Brodmann mapping found for: {filename}")
        continue

    df_dtangle = pd.read_csv(csv_file)

    if "Unnamed: 0" in df_dtangle.columns:
        df_dtangle = df_dtangle.rename(columns={"Unnamed: 0": "label"})

    if "label" not in df_dtangle.columns:
        print(f"No label column found in: {filename}")
        continue

    df_dtangle["label"] = df_dtangle["label"].astype(int)

    selected_columns = ["label"] + [
        ct for ct in cell_types
        if ct in df_dtangle.columns
    ]

    df_dtangle = df_dtangle[selected_columns].copy()

    if not isinstance(brodmann_value, list):
        brodmann_value = [brodmann_value]

    for value in brodmann_value:

        indices = np.where(lh_brodmann_data == value)[0]

        new_df = df_dtangle[
            df_dtangle["label"].isin(indices)
        ].copy()

        output_file = os.path.join(
            brodmann_roi_csv_dir,
            f"{region_key}_brodmann_{value}.csv"
        )

        new_df.to_csv(output_file, index=False)

        print(f"Saved Brodmann ROI CSV: {output_file}")


# ============================================================
# Merge ROI CSV files into cell-type-wise cortical expression maps
# ============================================================
roi_csv_files = glob.glob(
    os.path.join(brodmann_roi_csv_dir, "*.csv")
)

roi_dataframes = [
    pd.read_csv(file)
    for file in roi_csv_files
]

if len(roi_dataframes) == 0:
    raise ValueError("No Brodmann ROI CSV files were generated.")

for cell_type in cell_types:

    celltype_parts = []

    for df in roi_dataframes:

        if cell_type in df.columns:
            temp_df = df[["label", cell_type]].copy()
        else:
            temp_df = pd.DataFrame({
                "label": df["label"],
                cell_type: np.zeros(len(df))
            })

        celltype_parts.append(temp_df)

    cell_df = pd.concat(celltype_parts, axis=0)

    cell_df = (
        cell_df
        .groupby("label", as_index=True)[cell_type]
        .mean()
        .sort_index()
    )

    lh_data = np.zeros(n_vertices, dtype=np.float32)
    lh_data[cell_df.index.astype(int)] = cell_df.values.astype(np.float32)

    lh_gii_data_array = nib.gifti.GiftiDataArray(
        data=lh_data.astype(np.float32)
    )

    lh_gii_img_save = nib.gifti.GiftiImage(
        darrays=[lh_gii_data_array]
    )

    gii_filename = f"lh_{cell_type}_cell2map_expr.func.gii"

    nib.save(
        lh_gii_img_save,
        os.path.join(origin_gii_dir, gii_filename)
    )

    print(f"Saved expression GII: {gii_filename}")


# ============================================================
# Smooth cell-type expression maps with Workbench
# ============================================================
input_gii_files = glob.glob(
    os.path.join(origin_gii_dir, "lh_*_cell2map_expr.func.gii")
)

base_command = [
    "wb_command",
    "-metric-smoothing",
    "-fix-zeros",
    midthickness_surface_path,
]

for input_file in input_gii_files:

    match = re.search(
        r"lh_(.*)_cell2map_expr\.func\.gii",
        os.path.basename(input_file)
    )

    if match is None:
        print(f"Cannot parse cell type from: {input_file}")
        continue

    cell_type = match.group(1)

    output_file = os.path.join(
        smoothed_gii_dir,
        f"lh_{cell_type}_cell2map_expr_sm6.func.gii"
    )

    command = base_command + [
        input_file,
        "6",
        output_file
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if process.returncode != 0:
        print(f"Smoothing failed for {input_file}: {process.stderr}")
        continue

    smoothed_gii = nib.load(output_file)
    smoothed_data = smoothed_gii.darrays[0].data

    smoothed_data[lh_brodmann_data == 0] = 0
    smoothed_gii.darrays[0].data = smoothed_data

    nib.save(smoothed_gii, output_file)

    print(f"Saved smoothed expression GII: {output_file}")


# ============================================================
# Plot smoothed expression maps
# ============================================================
smoothed_files = glob.glob(
    os.path.join(smoothed_gii_dir, "lh_*_cell2map_expr_sm6.func.gii")
)

for smoothed_file in smoothed_files:

    match = re.search(
        r"lh_(.*)_cell2map_expr_sm6\.func\.gii",
        os.path.basename(smoothed_file)
    )

    if match is None:
        continue

    cell_type = match.group(1)

    celltype_sm6_gii = nib.load(smoothed_file)
    celltype_sm6_data = celltype_sm6_gii.darrays[0].data

    p = Plot(
        surf_lh=lh_surface,
        size=(400, 200),
        zoom=1.2
    )

    p.add_layer(
        {"left": celltype_sm6_data},
        cbar=True,
        cmap="inferno"
    )

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

    fig = p.build(cbar_kws=cbar_kws)

    plt.suptitle(
        f"{cell_type}-deconvolution expression",
        y=0.85
    )

    plt.savefig(
        os.path.join(
            smoothed_fig_dir,
            f"lh_{cell_type}_cell2map_expr_sm6_view.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    gc.collect()

    print(f"Saved expression brainmap: {cell_type}")


# ============================================================
# Build row-wise cell-type proportion table from smoothed maps
# ============================================================
data_dict = {}
all_non_zero_indices = set()

for cell_type in cell_types:

    gii_path = os.path.join(
        smoothed_gii_dir,
        f"lh_{cell_type}_cell2map_expr_sm6.func.gii"
    )

    if not os.path.isfile(gii_path):
        print(f"Missing smoothed GII file: {gii_path}")
        continue

    gii_img = nib.load(gii_path)
    data = gii_img.darrays[0].data

    non_zero_indices = np.where(data != 0)[0]
    non_zero_data = data[non_zero_indices]

    all_non_zero_indices.update(non_zero_indices)

    data_dict[cell_type] = {
        "indices": non_zero_indices,
        "data": non_zero_data,
    }

if not data_dict:
    raise ValueError("No valid smoothed cell-type maps found.")

all_non_zero_indices = sorted(all_non_zero_indices)

df = pd.DataFrame({
    "label": all_non_zero_indices
})

for cell_type, info in data_dict.items():

    temp_series = pd.Series(
        data=info["data"],
        index=info["indices"],
        name=cell_type
    )

    df = df.merge(
        temp_series,
        left_on="label",
        right_index=True,
        how="left"
    )

df = df.fillna(0)

value_columns = [
    col for col in df.columns
    if col != "label"
]

row_sum = df[value_columns].sum(axis=1)

df_proportion = df.copy()

valid_rows = row_sum > 0

df_proportion.loc[valid_rows, value_columns] = (
    df.loc[valid_rows, value_columns]
    .div(row_sum[valid_rows], axis=0)
)

df_proportion.loc[~valid_rows, value_columns] = 0

proportion_csv_file = os.path.join(
    proportion_csv_dir,
    "celltype_sm6_proportion.csv"
)

df_proportion.to_csv(
    proportion_csv_file,
    index=False
)

print(f"Saved proportion CSV: {proportion_csv_file}")

lh_data = np.zeros(n_vertices, dtype=np.float32)

for category in categories:

    existing_cell_types = [
        ct for ct in category
        if ct in df_proportion.columns
    ]

    if len(existing_cell_types) == 0:
        continue

    for cell_type in existing_cell_types:

        lh_data.fill(0)

        labels = df_proportion["label"].astype(int).values
        values = df_proportion[cell_type].values.astype(np.float32)

        lh_data[labels] = values

        p = Plot(
            surf_lh=lh_surface,
            size=(200, 100),
            zoom=1.2
        )

        p.add_layer(
            {"left": lh_data},
            cbar=True,
            cmap="inferno"
        )

        cbar_kws = {
            "location": "right",
            "label_direction": 45,
            "decimals": 2,
            "fontsize": 8,
            "n_ticks": 2,
            "shrink": 0.25,
            "aspect": 8,
            "draw_border": False,
        }

        fig = p.build(cbar_kws=cbar_kws)
        plt.suptitle(
            f"{cell_type}-deconvolution proportion",
            y=0.85
        )
        plt.savefig(
            os.path.join(
                proportion_fig_dir,
                f"lh_{cell_type}_cell2map_sm6_proportion_view.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        gc.collect()
        print(f"Saved proportion brainmap: {cell_type}")