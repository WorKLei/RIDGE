import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import shap
from matplotlib.patches import Rectangle, Ellipse
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from palettable.cmocean.sequential import Amp_10
from netneurotools import plotting


# =========================
# Labels
# =========================
celltype_realname = [
    "L2/3 IT", "L4 IT", "L5 ET", "L5 IT", "L5/6 NP", "L6 CT", "L6 IT", "L6 IT Car3", "L6b",
    "Chandelier", "Lamp5", "Lamp5-lhx6", "Pax6", "Pvalb", "Sncg", "Sst", "Sst Chodl", "Vip",
    "Astro", "Micro/PVM", "OPC", "Oligo", "Endo", "VLMC"
]

disease_total_names = [
    "22q11.2", "ADHD", "ASD", "epilepsy(IGE)",
    "epilepsy (RTLE)", "epilepsy (LTLE)", "MDD", "OCD", "schizophrenia",
    "bipolar disorder", "obesity", "schizotypy", "parkinson’s"
]


# =========================
# Helper functions
# =========================
def create_and_save_pv_plot_surface(data, hemisphere, output_path, cmap="viridis"):
    """Project parcel-level data to the surface and save the figure."""
    data_array = np.zeros(len(hemisphere.darrays[0].data))
    for i, value in enumerate(data):
        data_array[hemisphere.darrays[0].data == i + 1] = value

    plotting.pv_plot_surface(
        data_array,
        template="fslr32k",
        surf="inflated",
        hemi="L",
        show_colorbar=False,
        cmap=cmap,
        lighting_style="threelights",
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


# =========================
# Configuration
# =========================
parc = "Cammoun033_no_corpuscallosum"
path = "project_whole_brain_mapping/final_code/"
enigmapath = f"{path}materials/enigma_disease_exposure"
roi_path = f"{path}materials/ROI/fslr32k"

os.makedirs(f"{enigmapath}/figures", exist_ok=True)


# =========================
# Load data
# =========================
disease_profiles = zscore(np.genfromtxt(f"{enigmapath}/enigma_ct.csv", delimiter=","))
disease_names = np.load(f"{enigmapath}/disorders.npy")

with open(f"{path}results/parcellated_{parc}/lh_layer_EX_index.pkl", "rb") as f:
    lh_layer_ex_index = pickle.load(f)
with open(f"{path}results/parcellated_{parc}/lh_layer_IN_index.pkl", "rb") as f:
    lh_layer_in_index = pickle.load(f)
with open(f"{path}results/parcellated_{parc}/lh_layer_NO_neuron_index.pkl", "rb") as f:
    lh_layer_no_neuron_index = pickle.load(f)

lh_layer_all_index = {
    **lh_layer_ex_index,
    **lh_layer_in_index,
    **lh_layer_no_neuron_index,
}

lh_parcellation_atlas = nib.load(f"{roi_path}/{parc}_conte69_lh.label.gii")
lh_spins = np.load(f"{enigmapath}/lh_spins_Cammoun033.npy")


# =========================
# Prepare GLM data
# =========================
disease_dict = {name: disease_profiles[:, idx] for idx, name in enumerate(disease_names)}
first_value = disease_dict[next(iter(disease_dict))]
half_length = len(first_value) // 2
lh_disease_dict = {key: value[:half_length] for key, value in disease_dict.items()}

lh_layer_all_df = pd.DataFrame(lh_layer_all_index).apply(zscore)
lh_disease_dict_df = pd.DataFrame(lh_disease_dict).apply(zscore)

X = lh_layer_all_df
disease_columns = lh_disease_dict_df.columns


# =========================
# GLM model fit
# =========================
regressor = LinearRegression()
r_squared_dict = {}
adjusted_r_squared_dict = {}

for disease in disease_columns:
    y = lh_disease_dict_df[disease]
    regressor.fit(X, y)
    r_squared = regressor.score(X, y)

    n = X.shape[0]
    p = X.shape[1]
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    r_squared_dict[disease] = r_squared
    adjusted_r_squared_dict[disease] = adjusted_r_squared

r_squared_df = pd.DataFrame({
    "Disease": list(r_squared_dict.keys()),
    "R2": list(r_squared_dict.values()),
    "Adjusted_R2": list(adjusted_r_squared_dict.values()),
})
print(r_squared_df)


# =========================
# Spin-based significance for GLM
# =========================
nspins = 1000
model_pval = np.zeros(len(disease_names))
model_pval_df = pd.DataFrame(model_pval, index=disease_names, columns=["p_value"])

for disease in disease_columns:
    y = lh_disease_dict_df[disease]
    regressor = LinearRegression()
    regressor.fit(X, y)
    emp = regressor.score(X, y)

    null = np.zeros((nspins,))
    for s in range(nspins):
        regressor = LinearRegression()
        regressor.fit(X.iloc[lh_spins[:, s]], y)
        null[s] = regressor.score(X.iloc[lh_spins[:, s]], y)

    model_pval_df.loc[disease, "p_value"] = (1 + np.sum(null > emp)) / (nspins + 1)

model_pval_df["p_value_fdr"] = multipletests(
    model_pval_df["p_value"].values,
    method="fdr_bh"
)[1]

print(model_pval_df)


# =========================
# Plot adjusted R²
# =========================
merged_df = r_squared_df.merge(model_pval_df, left_on="Disease", right_index=True)

def get_significance_star(row):
    if row["p_value_fdr"] < 0.05:
        return "*"
    elif row["p_value"] < 0.05:
        return "*"
    return ""

merged_df["Significance"] = merged_df.apply(get_significance_star, axis=1)
merged_df["BarColor"] = merged_df["p_value_fdr"].apply(
    lambda p: "#f8caca" if p < 0.05 else "#e9e9e9"
)
merged_df["Disease"] = disease_total_names

fig, ax = plt.subplots(figsize=(6, 8))
sns.barplot(
    data=merged_df,
    x="Adjusted_R2",
    y="Disease",
    width=0.3,
    ax=ax,
    palette=list(merged_df["BarColor"]),
)

for i, (r2, pval) in enumerate(zip(merged_df["Adjusted_R2"], merged_df["p_value_fdr"])):
    dot_color = "#d61a1a" if pval < 0.05 else "#b5b5b5"
    ax.scatter(r2, i, s=150, color=dot_color, edgecolor="white", linewidth=2, zorder=3)

min_r2 = merged_df["Adjusted_R2"].min()
ax.set_xlim(min_r2 - 0.1, 1.0)
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20)
sns.despine(top=True, right=True, left=True)
ax.set_xlabel("")
ax.set_ylabel("")

ellipse = Ellipse(
    xy=(0.5, 0.92),
    width=0.022,
    height=0.018,
    transform=fig.transFigure,
    color="#d61a1a",
    clip_on=False,
)
fig.patches.append(ellipse)
fig.text(0.53, 0.92, "p-FDR < 0.05", va="center", fontsize=20, color="black")

fig.subplots_adjust(left=0.35, right=0.95, top=0.90)
plt.savefig(
    f"{enigmapath}/figures/lh_r2_celltype_disease_exposure.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# =========================
# SHAP values from empirical models
# =========================
r_squared_dict = {}
shap_values_dict = {}

for disease in disease_columns:
    y = lh_disease_dict_df[disease]
    model = LinearRegression().fit(X, y)
    r_squared_dict[disease] = model.score(X, y)

    masker = shap.maskers.Independent(X)
    explainer = shap.LinearExplainer(model=model, data=X, masker=masker)
    shap_vals = explainer.shap_values(X)
    shap_values_dict[disease] = shap_vals

disease_shap_dict = {}
for disease in disease_columns:
    shap_vals = shap_values_dict[disease]
    disease_shap_dict[disease] = np.abs(shap_vals).mean(axis=0)

shap_df = pd.DataFrame.from_dict(
    disease_shap_dict,
    orient="index",
    columns=X.columns
)

for disease in model_pval_df.index:
    if model_pval_df.loc[disease, "p_value"] > 0.05:
        shap_df.loc[disease] = np.nan


# =========================
# Preliminary SHAP heatmap
# =========================
plt.figure(figsize=(9, 6))
ax = sns.heatmap(shap_df, annot=False, cmap="viridis")
plt.title("SHAP")
plt.xlabel("")
plt.ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
plt.tight_layout()
plt.show()
plt.close()


# =========================
# Spin-based significance for SHAP
# =========================
null_shap_dict = {}
empirical_shap_dict = {}

for disease in disease_columns:
    y = lh_disease_dict_df[disease]

    model = LinearRegression().fit(X, y)
    masker = shap.maskers.Independent(X)
    explainer = shap.LinearExplainer(model=model, data=X, masker=masker)
    shap_vals = explainer.shap_values(X)
    empirical_shap = np.abs(shap_vals).mean(axis=0)
    empirical_shap_dict[disease] = empirical_shap

    null_shap = np.zeros((nspins, X.shape[1]))
    for s in range(nspins):
        y_perm = y[lh_spins[:, s]]
        null_model = LinearRegression().fit(X, y_perm)
        null_masker = shap.maskers.Independent(X)
        null_explainer = shap.LinearExplainer(model=null_model, data=X, masker=null_masker)
        shap_vals_null = null_explainer.shap_values(X)
        null_shap[s, :] = np.abs(shap_vals_null).mean(axis=0)

    null_shap_dict[disease] = null_shap

p_values_dict = {}
for disease in disease_columns:
    emp_shap = empirical_shap_dict[disease]
    null_shap = null_shap_dict[disease]

    p_vals = np.zeros(emp_shap.shape)
    for i in range(len(emp_shap)):
        p_vals[i] = (1 + np.sum(null_shap[:, i] > emp_shap[i])) / (nspins + 1)

    p_values_dict[disease] = p_vals

p_values_df = pd.DataFrame.from_dict(
    p_values_dict,
    orient="index",
    columns=X.columns
)


# =========================
# Final SHAP heatmap with significance boxes
# =========================
plt.figure(figsize=(9, 6))
ax = sns.heatmap(
    shap_df,
    fmt="",
    linewidths=2,
    cmap=Amp_10.mpl_colormap,
    cbar_kws={"shrink": 0.3, "aspect": 5},
)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=0, length=0)
cbar.ax.text(1.5, 0.0, "min", ha="left", va="center", fontsize=25)
cbar.ax.text(1.5, 2.2, "max", ha="left", va="center", fontsize=25)

for y in range(p_values_df.shape[0]):
    for x in range(p_values_df.shape[1]):
        pval = p_values_df.iloc[y, x]
        shap_val = shap_df.iloc[y, x]
        if not np.isnan(pval) and pval <= 0.05 and not np.isnan(shap_val):
            rect = Rectangle((x, y), 1, 1, fill=False, edgecolor="black", linewidth=2.2)
            ax.add_patch(rect)

plt.xlabel("")
plt.ylabel("")
ax.set_xticklabels(celltype_realname, fontsize=20, rotation=45, ha="right")
ax.set_yticklabels(disease_total_names, fontsize=20, rotation=0)
ax.set_yticklabels([])

plt.tight_layout()
plt.savefig(
    f"{enigmapath}/figures/lh_shap_celltype_disease_exposure.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close()
