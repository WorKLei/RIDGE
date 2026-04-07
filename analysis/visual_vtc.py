import os
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# =========================
# Configuration
# =========================
PATH = "project_whole_brain_mapping/final_code"
RESULTS_DIR = os.path.join(PATH, "results/visual")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figure")
os.makedirs(FIGURES_DIR, exist_ok=True)

PC_FILE = os.path.join(RESULTS_DIR, "Visual_stream_celltype_PC1_2.csv")
CYTO_LABEL = os.path.join(PATH, "materials/visual/lh.Rosenke_vcAtlas.32k.label.gii")
CATEGORY_LABEL = os.path.join(PATH, "materials/visual/lh.VTC_categories_fsLR32k.label.gii")
DIST_FILE = os.path.join(PATH, "materials/visual/lh_Calcarine_sulcus_gc_distance.func.gii")
ECCENTRICITY_FILE = os.path.join(
    PATH, "materials/visual/lh.S1200_7T_Retinotopy181.Fit1_Eccentricity_MSMAll.func.gii"
)

CYTO_MAPPING = {"FG1": 5, "FG2": 6, "FG3": 7, "FG4": 8}
CATEGORY_MAPPING = {
    "CoS places": 1,
    "OTS bodies": 2,
    "mFus faces": 3,
    "mOTS words": 4,
    "pFus faces": 5,
    "pOTS words": 6,
}


# =========================
# Helper functions
# =========================
def load_label_data(filepath):
    """Load a label or functional GIFTI file and return its data array."""
    return nib.load(filepath).darrays[0].data


def get_indices(data_array, mapping):
    """Return vertex indices for each label defined in a mapping dictionary."""
    return {key: np.where(data_array == val)[0] for key, val in mapping.items()}


def assign_label_from_indices(vertex_label, index_dict):
    """Assign a categorical label to a vertex based on membership in index groups."""
    for key, idx_array in index_dict.items():
        if vertex_label in idx_array:
            return key
    return "Other"


def perform_regression_analysis(X_vals, y_vals, degree=1):
    """Fit polynomial regression and return model-fit metrics."""
    X_reshaped = X_vals.reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_reshaped)

    model = LinearRegression().fit(X_poly, y_vals)
    y_pred = model.predict(X_poly)
    r2 = r2_score(y_vals, y_pred)

    ols = sm.OLS(y_vals, X_poly).fit()

    return {
        "degree": degree,
        "r2": r2,
        "adj_r2": ols.rsquared_adj,
        "aic": ols.aic,
        "bic": ols.bic,
    }


def plot_strip_reg(data, x_col, y_col, x_labels, y_label_text, palette, order=1, save_name=None):
    """Plot stripplot with polynomial regression overlay."""
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    sns.stripplot(
        data=data,
        x=x_col,
        y=y_col,
        palette=palette,
        jitter=True,
        zorder=1,
        size=8,
    )

    sns.regplot(
        data=data,
        x=x_col,
        y=y_col,
        scatter=False,
        truncate=False,
        order=order,
        color="black",
    )

    plt.xticks(ticks=range(len(x_labels)), labels=x_labels, rotation=45, fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("")
    plt.ylabel(y_label_text, fontsize=25)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_name:
        plt.savefig(os.path.join(FIGURES_DIR, save_name), dpi=300)
    plt.show()


# =========================
# Data processing
# =========================
visual_stream_pc = pd.read_csv(PC_FILE, index_col=0)
cyto_data = load_label_data(CYTO_LABEL)
category_data = load_label_data(CATEGORY_LABEL)

cyto_indices = get_indices(cyto_data, CYTO_MAPPING)
category_indices = get_indices(category_data, CATEGORY_MAPPING)

edited_indices_map = {
    "OTS bodies": np.intersect1d(category_indices["OTS bodies"], cyto_indices["FG4"]),
    "mOTS words": np.intersect1d(category_indices["mOTS words"], cyto_indices["FG4"]),
    "mFus faces": np.intersect1d(category_indices["mFus faces"], cyto_indices["FG4"]),
    "pFus faces": np.intersect1d(category_indices["pFus faces"], cyto_indices["FG2"]),
    "pOTS words": np.intersect1d(category_indices["pOTS words"], cyto_indices["FG2"]),
    "CoS places": np.intersect1d(category_indices["CoS places"], cyto_indices["FG3"]),
}
vtc_roi_indices = np.concatenate(list(edited_indices_map.values()))

df = visual_stream_pc[visual_stream_pc["label"].isin(vtc_roi_indices)].copy()

df["cyto"] = df["label"].apply(lambda x: assign_label_from_indices(x, cyto_indices))
df["category"] = df["label"].apply(lambda x: assign_label_from_indices(x, category_indices))

dist_data = load_label_data(DIST_FILE)
df["Fit_gc_distance"] = dist_data[df["label"]]

eccentricity_data = load_label_data(ECCENTRICITY_FILE)
df["Eccentricity"] = eccentricity_data[df["label"]]

cate_map = {
    "CoS places": "places",
    "OTS bodies": "bodies",
    "mFus faces": "faces",
    "mOTS words": "words",
    "pFus faces": "faces",
    "pOTS words": "words",
}

df["cyto_num"] = df["cyto"].map({"FG2": 1, "FG3": 2, "FG4": 3})
df["category_num"] = df["category"].map({
    "CoS places": 0,
    "mFus faces": 1,
    "pFus faces": 2,
    "OTS bodies": 3,
    "pOTS words": 4,
    "mOTS words": 5,
})

# =========================
# Analysis and visualization
# =========================
X_cyto = df["cyto_num"].values
y_dist = df["Fit_gc_distance"].values

stats_lin = perform_regression_analysis(X_cyto, y_dist, degree=1)
stats_quad = perform_regression_analysis(X_cyto, y_dist, degree=2)

print(f"Linear Fit: R² = {stats_lin['r2']:.3f}, AIC = {stats_lin['aic']:.2f}")
print(f"Quadratic Fit: R² = {stats_quad['r2']:.3f}, AIC = {stats_quad['aic']:.2f}")

plot_strip_reg(
    data=df,
    x_col="cyto_num",
    y_col="Fit_gc_distance",
    x_labels=["FG2", "FG3", "FG4"],
    y_label_text="Geodesic distance",
    palette=["#de4a96", "#3a59a5", "#a1534a"],
    order=1,
    save_name="VTC_cyto_regplot_geodesic_distance.svg",
)

plot_strip_reg(
    data=df,
    x_col="cyto_num",
    y_col="pca_component_1",
    x_labels=["FG2", "FG3", "FG4"],
    y_label_text="PC1 score",
    palette=["#de4a96", "#3a59a5", "#a1534a"],
    order=1,
    save_name="VTC_pc1_cyto_regplot.svg",
)

category_labels_ordered = [
    "CoS places",
    "mFus faces",
    "pFus faces",
    "OTS bodies",
    "pOTS words",
    "mOTS words",
]

plot_strip_reg(
    data=df,
    x_col="category_num",
    y_col="pca_component_2",
    x_labels=category_labels_ordered,
    y_label_text="PC2 score",
    palette="Set2",
    order=2,
    save_name="VTC_pc2_category_regplot.svg",
)

plot_strip_reg(
    data=df,
    x_col="category_num",
    y_col="Eccentricity",
    x_labels=category_labels_ordered,
    y_label_text="Eccentricity",
    palette="Set2",
    order=2,
    save_name="VTC_Eccentricity_category_regplot.svg",
)