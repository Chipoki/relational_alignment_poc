"""
aroma_decoding_pipeline_v10.py

Adapted from v9-3 to run over the author_replication preprocessed data.

Path changes vs v9:
  - BOLD files:  author_replication/MRI/{subject}/func/session-{ses}/
                   {subject}_unfeat_run-{run}/outputs/func/ICAedfiltered/filtered.nii.gz
  - Brain mask:  same ICAedfiltered run dir -> ../mask.nii.gz  (per-run func mask;
                   first valid run's mask is used as the reference brain mask)
  - ROI masks:   author_replication/MRI/{subject}/anat/ROI_BOLD/
                   ctx-lh-{roi}_BOLD.nii.gz  +  ctx-rh-{roi}_BOLD.nii.gz  (bilateral union)
  - Events TSVs: still read from ds003927/{subject}/ses-{ses:02d}/func/  (unchanged)

New CLI args (replacing old aroma-root / deriv-root):
  --mri-root        points to author_replication/MRI          (contains sub-01/...)
  --events-root     points to ds003927                        (unchanged)
  --roi-name        FreeSurfer parcel name without hemi prefix (default: lateraloccipital)

Session/run folder naming in author_replication:
  session folder : session-{ses}          e.g. session-05
  run folder     : {subject}_unfeat_run-{run}
  BOLD file      : outputs/func/ICAedfiltered/filtered.nii.gz
  mask file      : outputs/func/mask.nii.gz
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning

from joblib import Parallel, delayed, parallel_backend
from nilearn.image import resample_to_img
from nilearn.signal import clean as clean_signal
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_mei_svm_pipeline(C=1.0):
    svm = LinearSVC(
        penalty='l1',
        dual=False,
        tol=1e-3,
        random_state=12345,
        max_iter=int(1e4),
        class_weight='balanced',
        C=C,
    )
    return make_pipeline(VarianceThreshold(), StandardScaler(), svm)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def get_resampled_mask(mask_path, ref_img_path):
    mask_orig = nib.load(mask_path)
    ref_img   = nib.load(ref_img_path)
    if mask_orig.shape == ref_img.shape and np.allclose(mask_orig.affine, ref_img.affine):
        return mask_orig.get_fdata().astype(bool)
    mask_resampled = resample_to_img(
        source_img=mask_orig, target_img=ref_img, interpolation="nearest"
    )
    return mask_resampled.get_fdata().astype(bool)


def get_bilateral_roi_mask(mri_root, subject, roi_name, ref_img_path):
    """
    Load left + right hemisphere ROI masks from author_replication/MRI/{subject}/anat/ROI_BOLD/
    and return their union, resampled to the BOLD reference image space.
    """
    roi_dir = os.path.join(mri_root, subject, "anat", "ROI_BOLD")
    lh_path = os.path.join(roi_dir, f"ctx-lh-{roi_name}_BOLD.nii.gz")
    rh_path = os.path.join(roi_dir, f"ctx-rh-{roi_name}_BOLD.nii.gz")

    missing = [p for p in (lh_path, rh_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"ROI mask(s) not found:\n" + "\n".join(missing)
        )

    lh_mask = get_resampled_mask(lh_path, ref_img_path)
    rh_mask = get_resampled_mask(rh_path, ref_img_path)
    return lh_mask | rh_mask  # bilateral union


# ---------------------------------------------------------------------------
# Path helpers (author_replication layout)
# ---------------------------------------------------------------------------

def get_run_bold_path(mri_root, subject, ses, run_idx):
    """
    author_replication/MRI/{subject}/func/session-{ses}/
        {subject}_unfeat_run-{run}/outputs/func/ICAedfiltered/filtered.nii.gz
    """
    return os.path.join(
        mri_root, subject, "func",
        f"session-{ses:02d}",
        f"{subject}_unfeat_run-{run_idx}",
        "outputs", "func", "ICAed_filtered", "filtered.nii.gz",
    )


def get_run_mask_path(mri_root, subject, ses, run_idx):
    """
    author_replication/MRI/{subject}/func/session-{ses}/
        {subject}_unfeat_run-{run}/outputs/func/mask.nii.gz
    """
    return os.path.join(
        mri_root, subject, "func",
        f"session-{ses:02d}",
        f"{subject}_unfeat_run-{run_idx}",
        "outputs", "func", "mask.nii.gz",
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_aroma_run(ses, run_num, bold_path, events_path, roi_in_brain_mask, visibility):
    """
    EXACT BILATERAL SIGNAL CLEANING PARITY (unchanged from v9):
    1. Detrend the full run
    2. Slice the volume_interest volumes
    3. Temporal Z-Score the sliced volumes
    """
    bold_img = nib.load(bold_path)
    ts = bold_img.get_fdata(dtype=np.float32)[roi_in_brain_mask].T

    # Step 1: Detrend the whole run before slicing
    # TODO - supposed to happen already in the preprocessing pipeline, no?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts_detrended = clean_signal(ts, t_r=0.85, detrend=True, standardize=False)

    events = pd.read_csv(events_path, sep="\t" if events_path.endswith(".tsv") else ",")
    events = events.dropna(subset=["targets"])
    vi_events = events[events["volume_interest"] == 1].copy()

    if vi_events.empty:
        return []

    tr_indices = (
        vi_events["time_indices"].astype(int).values
        if "time_indices" in vi_events.columns
        else vi_events.index.values
    )

    # Step 2: Pick the volumes
    raw_matrix = ts_detrended[tr_indices]

    # Step 3: Temporal Z-Score on the picked volumes
    means = raw_matrix.mean(axis=0, keepdims=True)
    stds  = raw_matrix.std(axis=0, ddof=1, keepdims=True)
    stds[stds == 0] = 1.0
    zscored_matrix = (raw_matrix - means) / stds

    trial_ids_arr = vi_events["trials"].values
    unique_trials = sorted(np.unique(trial_ids_arr))
    results = []

    for trial_num in unique_trials:
        row_indices   = np.where(trial_ids_arr == trial_num)[0]
        first_vi_idx  = vi_events.index[vi_events["trials"] == trial_num][0]
        row           = vi_events.loc[first_vi_idx]

        if str(row["visibility"]) != visibility:
            continue

        pattern  = zscored_matrix[row_indices].mean(axis=0)
        trial_id = ses * 10000 + run_num * 100 + float(trial_num)

        results.append((
            pattern,
            {
                "session":   ses,
                "run":       run_num,
                "trial":     float(trial_num),
                "id":        trial_id,
                "target_1d": 1 if str(row["targets"]) == "Nonliving_Things" else 0,
                "item_name": str(row["labels"]) if "labels" in row.index else f"item_{int(trial_num)}",
            },
        ))
    return results


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def loo_partition(y_1d, item_labels):
    living_items    = np.unique(item_labels[y_1d == 0])
    nonliving_items = np.unique(item_labels[y_1d == 1])
    splits = []
    for li in living_items:
        for nli in nonliving_items:
            test_mask = np.isin(item_labels, [li, nli])
            if (~test_mask).sum() == 0 or test_mask.sum() == 0:
                continue
            splits.append((np.where(~test_mask)[0], np.where(test_mask)[0]))
    return splits


def run_one_fold(train_idx, test_idx, X, y_1d, C):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y_1d[train_idx], y_1d[test_idx]

    pipeline = build_mei_svm_pipeline(C=C)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X_tr, y_tr)

    scores = pipeline.decision_function(X_te)
    try:
        fold_auc = roc_auc_score(y_te, scores)
    except ValueError:
        fold_auc = np.nan

    return fold_auc, 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Living/non-living SVM decoding over author_replication preprocessed data."
    )
    parser.add_argument("--subject",     default="sub-01")
    # --- NEW: point at author_replication/MRI ---
    parser.add_argument(
        "--mri-root",
        default="./../../../soto_data/author_replication/MRI",
        help="Path to author_replication/MRI  (contains {subject}/func and {subject}/anat)",
    )
    # --- UNCHANGED: events still live in ds003927 ---
    parser.add_argument(
        "--events-root",
        default="./../../../soto_data/ds003927",
        help="Path to ds003927 BIDS root  (contains {subject}/ses-XX/func/*_events.tsv)",
    )
    # --- NEW: bilateral ROI name (FreeSurfer parcel, without ctx-lh/rh prefix) ---
    parser.add_argument(
        "--roi-name",
        default="lateraloccipital",
        help=(
            "FreeSurfer parcel name used for ROI mask lookup under "
            "MRI/{subject}/anat/ROI_BOLD/ctx-{{lh,rh}}-{roi_name}_BOLD.nii.gz"
        ),
    )
    parser.add_argument("--out-dir",   default="./../../../poc_output/decoding")
    parser.add_argument("--sessions",  nargs="+", type=int, default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--C-values",  nargs="+", type=float, default=[1.0, 10.0])
    parser.add_argument("--n-jobs",    type=int, default=1)
    parser.add_argument("--no-cache",  action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    mri_root    = args.mri_root
    events_root = os.path.join(args.events_root, args.subject)
    os.makedirs(args.out_dir, exist_ok=True)

    CACHE_X    = os.path.join(args.out_dir, f"cache_{args.subject}_AROMA_Xbrain.npy")
    CACHE_META = os.path.join(args.out_dir, f"cache_{args.subject}_AROMA_meta.csv")

    # ------------------------------------------------------------------
    # Build run job list — discover runs via events TSVs (as before),
    # but resolve BOLD + mask paths from author_replication layout.
    # ------------------------------------------------------------------
    run_jobs        = []
    first_bold_path = None
    first_mask_path = None

    for ses in args.sessions:
        ses_str  = f"ses-{ses:02d}"
        func_dir = os.path.join(events_root, ses_str, "func")
        if not os.path.isdir(func_dir):
            continue

        events_files = sorted([
            f for f in os.listdir(func_dir)
            if f.endswith("_events.tsv") and "task-recog" in f
        ])

        for idx, fname in enumerate(events_files):
            run_idx   = idx + 1
            bold_path = get_run_bold_path(mri_root, args.subject, ses, run_idx)
            mask_path = get_run_mask_path(mri_root, args.subject, ses, run_idx)

            if not os.path.exists(bold_path):
                print(f"  [skip] BOLD not found: {bold_path}")
                continue

            if first_bold_path is None:
                first_bold_path = bold_path
            if first_mask_path is None and os.path.exists(mask_path):
                first_mask_path = mask_path

            run_jobs.append((ses, run_idx, bold_path, os.path.join(func_dir, fname)))

    if not run_jobs:
        raise RuntimeError("No valid runs found. Check --mri-root and --sessions.")

    print(f"Found {len(run_jobs)} run(s) to process.")

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    if first_mask_path is None:
        raise FileNotFoundError(
            "Could not locate any mask.nii.gz under the run directories. "
            "Check your --mri-root path."
        )

    print(f"Brain mask : {first_mask_path}")
    brain_mask_bool = get_resampled_mask(first_mask_path, first_bold_path)

    print(f"ROI        : bilateral ctx-{{lh,rh}}-{args.roi_name}_BOLD.nii.gz")
    roi_mask        = get_bilateral_roi_mask(mri_root, args.subject, args.roi_name, first_bold_path)
    roi_in_brain    = roi_mask & brain_mask_bool

    # ------------------------------------------------------------------
    # Feature extraction (cached)
    # ------------------------------------------------------------------
    if not args.no_cache and os.path.exists(CACHE_X):
        print("Loading cached features …")
        X_full  = np.load(CACHE_X)
        meta_df = pd.read_csv(CACHE_META)
    else:
        with parallel_backend("loky", n_jobs=args.n_jobs):
            raw_outputs = Parallel()(
                delayed(extract_aroma_run)(ses, run, bp, ep, brain_mask_bool, "conscious")
                for ses, run, bp, ep in tqdm(run_jobs, desc="Extracting")
            )

        all_patterns = [p for run in raw_outputs for p, m in run]
        all_meta     = [m for run in raw_outputs for p, m in run]
        X_full  = np.array(all_patterns)
        meta_df = pd.DataFrame(all_meta)
        np.save(CACHE_X, X_full)
        meta_df.to_csv(CACHE_META, index=False)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    y_1d        = meta_df["target_1d"].values
    X_clean     = X_full[:, roi_in_brain[brain_mask_bool]]
    item_labels = meta_df["item_name"].values

    print(
        f"\n conscious trials: {X_clean.shape[0]} "
        f"(Living: {(y_1d==0).sum()}  Non-living: {(y_1d==1).sum()})"
    )

    splits = loo_partition(y_1d, item_labels)

    for C in args.C_values:
        print(f"\n--- Decoding C={C} ---")
        with parallel_backend("loky", n_jobs=args.n_jobs):
            fold_results = Parallel()(
                delayed(run_one_fold)(tr, te, X_clean, y_1d, C)
                for tr, te in tqdm(splits, desc="SVM CV")
            )

        valid_aucs = [r[0] for r in fold_results if not np.isnan(r[0])]
        print(f"  Average Fold AUC = {np.mean(valid_aucs):.4f}")


if __name__ == "__main__":
    main()
