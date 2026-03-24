from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold

# STEP 1 #

DATA_ROOT = "/home/tomerd/Documents/projects/MSc/lab/thesis_practice/relational_alignment/soto_data/ds003927/derivatives/sub-01/"
DATA_ROOT_02 = "/home/tomerd/Documents/projects/MSc/lab/thesis_practice/relational_alignment/soto_data/ds003927/derivatives/sub-02/"
OUT = "/home/tomerd/Documents/projects/MSc/lab/thesis_practice/relational_alignment/poc_output/"

bold_img = nib.load(DATA_ROOT + "wholebrain_conscious.nii.gz")
mask_img = nib.load(DATA_ROOT + "mask.nii.gz")
roi_img  = nib.load(DATA_ROOT + "func_masks/lateral_occipital_mask.nii.gz")

print("=== BOLD ===")
print(f"  shape  : {bold_img.shape}")
print(f"  affine :\n{bold_img.affine.round(3)}")

print("\n=== full_mask ===")
print(f"  shape  : {mask_img.shape}")
print(f"  affine :\n{mask_img.affine.round(3)}")

print("\n=== lateral_occipital ROI mask ===")
print(f"  shape  : {roi_img.shape}")
print(f"  affine :\n{roi_img.affine.round(3)}")

# Check if shapes match
shapes_ok = bold_img.shape[:3] == mask_img.shape == roi_img.shape[:3]
print(f"\nShapes consistent: {shapes_ok}")

# Check if affines match
affine_ok = np.allclose(bold_img.affine, roi_img.affine, atol=1e-3)
print(f"Affines match    : {affine_ok}")


# STEP 2 #
mean_bold = nib.load(DATA_ROOT + "wholebrain_conscious.nii.gz").get_fdata().mean(axis=-1)
full_mask = mask_img.get_fdata().astype(bool)
roi_mask  = roi_img.get_fdata().astype(bool)

roi_voxels        = roi_mask.sum()
overlap_voxels    = (roi_mask & full_mask).sum()
overlap_pct       = 100 * overlap_voxels / roi_voxels if roi_voxels > 0 else 0

print(f"ROI total voxels           : {roi_voxels}")
print(f"Overlap with full_mask     : {overlap_voxels}")
print(f"Overlap %                  : {overlap_pct:.1f}%")


# STEP 3 #

df = pd.read_csv(DATA_ROOT + "wholebrain_conscious.csv")

print("=== CSV structure ===")
print(df.columns.tolist())
print(df[["session", "run", "labels", "targets", "visibility"]].head(20).to_string())
print(f"\nSession values : {sorted(df['session'].unique())}")
print(f"Run values     : {sorted(df['run'].unique())}")
print(f"\nFirst 5 rows session/run combos:")
print(df.groupby(["session","run"]).size().head(10))

print("\n=== id column ===")
print(df["id"].head(20).to_string())
print(f"id range: {df['id'].min()} – {df['id'].max()}")
print(f"id is sequential from 0: {list(df['id'].values[:5])}")

# STEP 4 #
roi_files = sorted(Path(DATA_ROOT + "func_masks/").glob("*_mask.nii*"))

fig, axes = plt.subplots(len(roi_files), 3, figsize=(12, len(roi_files) * 2.5))

for row, roi_path in enumerate(roi_files):
    roi_mask = nib.load(roi_path).get_fdata().astype(bool)
    roi_name = roi_path.name.replace("_mask.nii.gz", "").replace("_mask.nii", "")

    # Find the slices where this ROI actually has voxels
    z_coords = np.where(roi_mask.any(axis=(0, 1)))[0]
    z_slices = [z_coords[len(z_coords) // 4], z_coords[len(z_coords) // 2], z_coords[3 * len(z_coords) // 4]]

    for col, z in enumerate(z_slices):
        ax = axes[row, col]
        bg = mean_bold[:, :, z].T
        ax.imshow(bg, cmap="gray", origin="lower",
                  vmin=np.percentile(bg[bg > 0], 2), vmax=np.percentile(bg[bg > 0], 98))
        red = np.zeros((*bg.shape, 4))
        red[roi_mask[:, :, z].T] = [1, 0, 0, 0.7]
        ax.imshow(red, origin="lower")
        ax.set_title(f"z={z}", fontsize=7)
        ax.axis("off")

    axes[row, 0].set_ylabel(roi_name, fontsize=7, rotation=0, labelpad=80, va="center")

plt.suptitle("All ROI masks — verify anatomy", fontsize=10)
plt.tight_layout()
plt.savefig(OUT + "all_rois_anatomy_check.png", dpi=120)
print("Saved.")

# STEP 5 #

df   = pd.read_csv(DATA_ROOT + "wholebrain_conscious.csv")
bold = nib.load(DATA_ROOT + "wholebrain_conscious.nii.gz").get_fdata()
mask = nib.load(DATA_ROOT + "mask.nii.gz").get_fdata().astype(bool)

# Lateral occipital ROI
roi  = nib.load(DATA_ROOT + "func_masks/lateral_occipital_mask.nii.gz").get_fdata().astype(bool)
roi_in_mask = roi & mask

# Extract patterns
X = bold[roi_in_mask].T          # (514, n_roi_voxels)
y = (df["targets"] == "Living_Things").astype(int).values

# Simple 50/50 train-test split — no CV, just sanity check
n = len(y)
split = n // 2
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# Z-score each voxel on training set, apply to test
mean = X_tr.mean(axis=0); std = X_tr.std(axis=0); std[std<1e-8]=1
X_tr = (X_tr - mean) / std
X_te = (X_te - mean) / std

clf = LinearSVC(penalty="l1", loss="squared_hinge", dual=False, C=1.0, max_iter=10000)
clf.fit(X_tr, y_tr)
auc = roc_auc_score(y_te, clf.decision_function(X_te))
print(f"Sanity-check AUC (simple split): {auc:.4f}")
print(f"Train size: {len(y_tr)}, Test size: {len(y_te)}")
print(f"Train class balance: {y_tr.mean():.2f}, Test: {y_te.mean():.2f}")

# STEP 6 #

# Are the binary labels actually meaningful?
print("Label distribution:")
print(df["targets"].value_counts())
print(f"\ny (binary) distribution: {y.mean():.3f} living")

# Does the label vector have any structure at all?
# Check if living/nonliving alternates, clusters, or is random
print("\nFirst 30 labels:", y[:30].tolist())

# Check the actual NIfTI values — are they z-scored already?
print(f"\nX stats (raw patterns):")
print(f"  mean: {X.mean():.4f}")
print(f"  std:  {X.std():.4f}")
print(f"  min:  {X.min():.4f}")
print(f"  max:  {X.max():.4f}")

# Most important: is there ANY univariate signal?
from scipy.stats import ttest_ind
living_mean    = X[y==1].mean(axis=0)
nonliving_mean = X[y==0].mean(axis=0)
diff = np.abs(living_mean - nonliving_mean)
print(f"\nMean |living - nonliving| per voxel: {diff.mean():.6f}")
print(f"Max  |living - nonliving| per voxel: {diff.max():.6f}")


# STEP 7 #
# Raw patterns — NO additional z-scoring
X_raw = bold[roi_in_mask].T   # (514, n_roi_voxels)
y     = (df["targets"] == "Living_Things").astype(int).values

aucs = []
skf  = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for tr, te in skf.split(X_raw, y):
    X_tr, X_te = X_raw[tr], X_raw[te]
    y_tr, y_te = y[tr],     y[te]
    # Only remove zero-variance features
    var_mask = X_tr.var(axis=0) > 1e-8
    clf = LinearSVC(penalty="l1", loss="squared_hinge", dual=False,
                    C=1.0, max_iter=10000, class_weight="balanced")
    clf.fit(X_tr[:, var_mask], y_tr)
    aucs.append(roc_auc_score(y_te, clf.decision_function(X_te[:, var_mask])))

print(f"No extra z-score  | mean AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# Now compare with run-wise z-scored patterns
run_ids = df.groupby(["session", "run"]).ngroup().values
X_zs    = X_raw.copy()
for r in np.unique(run_ids):
    m        = run_ids == r
    d        = X_zs[m]
    mu, sig  = d.mean(0), d.std(0)
    sig[sig < 1e-8] = 1.0
    X_zs[m] = (d - mu) / sig

aucs_zs = []
for tr, te in skf.split(X_zs, y):
    X_tr, X_te = X_zs[tr], X_zs[te]
    y_tr, y_te = y[tr],     y[te]
    var_mask = X_tr.var(axis=0) > 1e-8
    clf = LinearSVC(penalty="l1", loss="squared_hinge", dual=False,
                    C=1.0, max_iter=10000, class_weight="balanced")
    clf.fit(X_tr[:, var_mask], y_tr)
    aucs_zs.append(roc_auc_score(y_te, clf.decision_function(X_te[:, var_mask])))

print(f"Run-wise z-scored | mean AUC = {np.mean(aucs_zs):.4f} ± {np.std(aucs_zs):.4f}")

# STEP 8 #

# Check all subjects' conscious lateral occipital AUC from the paper's Figure 3
# The paper reports individual subject results — sub-01 may genuinely be lower

# Meanwhile, test on sub-02 to see if it's a subject-level effect
df2   = pd.read_csv(DATA_ROOT_02 + "wholebrain_conscious.csv")
bold2 = nib.load(DATA_ROOT_02 + "wholebrain_conscious.nii.gz").get_fdata()
mask2 = nib.load(DATA_ROOT_02 + "mask.nii.gz").get_fdata().astype(bool)
roi2  = nib.load(DATA_ROOT_02 + "func_masks/lateral_occipital_mask.nii.gz").get_fdata().astype(bool)

X2 = bold2[roi2 & mask2].T
y2 = (df2["targets"] == "Living_Things").astype(int).values

# Run-wise z-score
run_ids2 = df2.groupby(["session","run"]).ngroup().values
X2_zs = X2.copy()
for r in np.unique(run_ids2):
    m = run_ids2 == r
    d = X2_zs[m]; mu, sig = d.mean(0), d.std(0); sig[sig<1e-8]=1.0
    X2_zs[m] = (d - mu) / sig

aucs2 = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for tr, te in skf.split(X2_zs, y2):
    X_tr, X_te = X2_zs[tr], X2_zs[te]
    y_tr, y_te = y2[tr], y2[te]
    var_mask = X_tr.var(axis=0) > 1e-8
    clf = LinearSVC(penalty="l1", loss="squared_hinge", dual=False,
                    C=1.0, max_iter=10000, class_weight="balanced")
    clf.fit(X_tr[:, var_mask], y_tr)
    aucs2.append(roc_auc_score(y_te, clf.decision_function(X_te[:, var_mask])))

print(f"sub-01 lateral_occipital | mean AUC = {np.mean(aucs_zs):.4f}")
print(f"sub-02 lateral_occipital | mean AUC = {np.mean(aucs2):.4f}")

# STEP 9 #
import warnings
from sklearn.exceptions import ConvergenceWarning

results = []
for C in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
    aucs = []
    total_warns = 0
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for tr, te in skf.split(X_zs, y):
        X_tr, X_te = X_zs[tr], X_zs[te]
        y_tr, y_te = y[tr], y[te]
        var_mask = X_tr.var(axis=0) > 1e-8
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf = LinearSVC(penalty="l1", loss="squared_hinge", dual=False,
                            C=C, max_iter=50000, class_weight="balanced")
            clf.fit(X_tr[:, var_mask], y_tr)
        total_warns += sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
        aucs.append(roc_auc_score(y_te, clf.decision_function(X_te[:, var_mask])))
    results.append((C, np.mean(aucs), np.std(aucs), total_warns))
    print(f"C={C:<6} | AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f} | conv_warns={total_warns}/10")


