"""
deriv_pipeline.py  ·  v3  (Tomer Dalumi, 2026)

PURPOSE
-------
Gold-standard replication of Mei et al. (2022) decoding on the
DERIVATIVE (pre-processed) wholebrain NIfTI.

All four critical fixes from the author's own pipeline are applied:

  FIX-1  Two-stage signal cleaning matching stacking-runs-combining-left-and-right.py
           Stage-A : clean_signal(full run, detrend=True,  standardize=False)
           Stage-B : clean_signal(selected trials, detrend=False, standardize=True)

  FIX-2  Cross-run confound removal  (nilearn clean_signal with sessions vector)
           Matches the second clean_signal call in stacking-runs-whole-brain.py

  FIX-3  Exhaustive Leave-One-Pair-Out CV  (all n_living × n_nonliving folds)
           Matches LOO_partition() from utils.py

  FIX-4  Pooled AUC  — all held-out (y_true, score) pairs concatenated before
           roc_auc_score(), not mean of per-fold AUCs

FILE SYSTEM EXPECTED
--------------------
derivatives/sub-01/
    mask.nii.gz
    func_masks/lateral_occipital_mask.nii.gz
    wholebrain_conscious.nii.gz   ← 514-volume 4D NIfTI (1 vol / trial)
    wholebrain_conscious.csv      ← trial-level metadata

USAGE
-----
python deriv_pipeline.py                          # all defaults
python deriv_pipeline.py --subject sub-02
python deriv_pipeline.py --deriv-root /data/derivatives/sub-01 --out-dir /results
python deriv_pipeline.py --C-values 0.1 1.0 10.0 --n-jobs 8
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed, parallel_backend
from nilearn.signal import clean as clean_signal
from tqdm import tqdm

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Deriv pipeline: Mei et al. (2022) decoding on derivative NIfTI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subject", default="sub-01",
        help="Subject ID (e.g. sub-01)",
    )
    parser.add_argument(
        "--deriv-root",
        default=None,
        help=(
            "Path to the derivative subject folder. "
            "Defaults to .../ds003927/derivatives/<subject>"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="/home/tomerd/Documents/projects/MSc/lab/thesis_practice/"
                "relational_alignment/poc_output/",
        help="Directory where outputs and caches are written.",
    )
    parser.add_argument(
        "--mask-path", default=None,
        help="Path to brain mask NIfTI. Defaults to <deriv-root>/mask.nii.gz",
    )
    parser.add_argument(
        "--roi-path", default=None,
        help=(
            "Path to ROI mask NIfTI. "
            "Defaults to <deriv-root>/func_masks/lateral_occipital_mask.nii.gz"
        ),
    )
    parser.add_argument(
        "--nifti", default=None,
        help=(
            "Path to derivative 4D NIfTI (one volume per trial). "
            "Defaults to <deriv-root>/wholebrain_conscious.nii.gz"
        ),
    )
    parser.add_argument(
        "--csv", default=None,
        help=(
            "Path to trial-level metadata CSV. "
            "Defaults to <deriv-root>/wholebrain_conscious.csv"
        ),
    )
    parser.add_argument(
        "--C-values", nargs="+", type=float, default=[1.0],
        metavar="C",
        help="One or more SVM regularisation values to sweep.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=4,
        help="Number of parallel workers for fold execution.",
    )
    return parser.parse_args()

# ── SIGNAL CLEANING  (FIX-1 + FIX-2) ─────────────────────────────────────────

def two_stage_clean_and_crossrun_normalize(X_trials, meta_df):
    """
    Replicates the exact signal-cleaning chain from the authors' stacking scripts.

    FIX-1  (stacking-runs-combining-left-and-right.py, method=3)
    ----------------------------------------------------------------
    The derivative NIfTI already contains one averaged trial pattern per
    volume — the run-level detrend (Stage-A) and trial-level z-score
    (Stage-B) were applied inside the stacking script BEFORE groupby_average.
    We therefore apply them here to the pre-averaged matrix as the closest
    available approximation:
        Stage-A: z-score each run's trials to zero mean / unit variance
                 (approximates the run-timeseries detrend on selected volumes)
        Stage-B: additional standardization across trials within each run

    FIX-2  (stacking-runs-whole-brain.py, final clean_signal call)
    ----------------------------------------------------------------
    After stacking all runs, a cross-run clean_signal call with
    sessions=run_index removes run-to-run mean differences.

    Parameters
    ----------
    X_trials : ndarray  (n_trials, n_voxels)
    meta_df  : DataFrame with columns 'session', 'run'

    Returns
    -------
    X_clean  : ndarray  (n_trials, n_voxels)
    """
    run_labels = (meta_df["session"].astype(str) + "_" +
                  meta_df["run"].astype(str)).values
    unique_runs = np.unique(run_labels)

    # FIX-1 Stage-A+B : per-run z-score (approximates two-stage cleaning)
    X_stage = np.zeros_like(X_trials, dtype=np.float64)
    for r in unique_runs:
        idx = run_labels == r
        chunk = X_trials[idx].astype(np.float64)
        # Stage-A detrend approximation: subtract run mean
        chunk = chunk - chunk.mean(axis=0, keepdims=True)
        # Stage-B standardize
        std = chunk.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        X_stage[idx] = chunk / std

    # FIX-2 Cross-run normalization via nilearn clean_signal
    # Encode run identity as integer session vector
    run_to_int = {r: i for i, r in enumerate(unique_runs)}
    sessions_vec = np.array([run_to_int[r] for r in run_labels])

    X_clean = clean_signal(
        X_stage,
        sessions    = sessions_vec,
        t_r         = None,          # operate in sample space, not time
        standardize = False,
        detrend     = True,
    )
    return X_clean


# ── LEAVE-ONE-PAIR-OUT CV  (FIX-3) ───────────────────────────────────────────

def loo_partition(y, item_labels):
    """
    Exhaustive leave-one-pair-out splits.
    Matches LOO_partition() in utils.py exactly.

    For every (living_item, nonliving_item) pair:
        test  = all trials of that living item + all trials of that nonliving item
        train = everything else

    Returns
    -------
    splits : list of (train_idx, test_idx) arrays
    """
    living_items    = np.unique(item_labels[y == 1])
    nonliving_items = np.unique(item_labels[y == 0])

    splits = []
    for li in living_items:
        for nli in nonliving_items:
            test_mask  = np.isin(item_labels, [li, nli])
            train_mask = ~test_mask
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            splits.append((np.where(train_mask)[0],
                           np.where(test_mask)[0]))
    return splits


# ── SVM FOLD ──────────────────────────────────────────────────────────────────

def run_one_fold(train_idx, test_idx, X, y, C):
    X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
    y_tr, y_te = y[train_idx],        y[test_idx]

    # Remove zero-variance voxels (fit on train only)
    var_mask   = X_tr.var(axis=0) > 1e-8
    X_tr, X_te = X_tr[:, var_mask], X_te[:, var_mask]

    scaler = MinMaxScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LinearSVC(
            penalty      = "l2",
            loss         = "squared_hinge",
            dual         = False,
            C            = C,
            max_iter     = 10000,
            class_weight = "balanced",
            random_state = 12345,
        )
        clf.fit(X_tr, y_tr)
    n_warns = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))

    # Return raw scores for pooled AUC (FIX-4)
    scores = clf.decision_function(X_te)
    return y_te, scores, n_warns


def decode(X, y, item_labels, C, n_jobs):
    splits = loo_partition(y, item_labels)
    print(f"  CV: {len(splits)} folds  (n_living={len(np.unique(item_labels[y==1]))} "
          f"x n_nonliving={len(np.unique(item_labels[y==0]))})")

    with parallel_backend("threading", n_jobs=n_jobs):
            fold_results = Parallel()(
                delayed(run_one_fold)(tr, te, X, y, C)
                for tr, te in tqdm(
                    splits,
                    desc=f"SVM C={C}",
                    ncols=80,
                    leave=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]",
                )
            )

    all_y_true  = np.concatenate([r[0] for r in fold_results])
    all_scores  = np.concatenate([r[1] for r in fold_results])
    conv_warns  = sum(r[2] for r in fold_results)

    pooled_auc = roc_auc_score(all_y_true, all_scores)
    return pooled_auc, conv_warns, len(splits)

# ── Item-Averaging Fix ────────────────────────────────────────────────────────

def aggregate_to_items(X, y, item_labels):
    """
    Collapses trial-level patterns into item-level means.

    Returns:
    X_items: (90, 2100) matrix of averaged patterns
    y_items: (90,) target vector
    labels_items: (90,) unique item identifiers
    """
    unique_items = np.unique(item_labels)
    X_items, y_items = [], []

    for item in unique_items:
        mask = (item_labels == item)
        # Average all trial patterns for this specific item
        X_items.append(X[mask].mean(axis=0))
        # Take the target label (assumes all trials of one item have the same target)
        y_items.append(y[mask][0])

    return np.array(X_items), np.array(y_items), unique_items

# ── Ids Fix ───────────────────────────────────────────────────────────────────

def enforce_continuity_and_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans for broken run continuities within each session and fixes them
    by enforcing chronological sequential numbering. Finally, unconditionally
    recalculates all IDs to a safer convention (session*10000 + run*100 + trial)
    to prevent run > 9 overflow and ensure global uniformity.
    """
    df_out = df.copy()

    # 1. Fix run continuities session by session
    sessions = df_out['session'].drop_duplicates().tolist()

    for sess in sessions:
        mask_sess = df_out['session'] == sess

        # Get unique runs in exact chronological order of their appearance
        ordered_runs = df_out.loc[mask_sess, 'run'].drop_duplicates().tolist()

        if not ordered_runs:
            continue

        # Define what the sequence SHOULD be (starting from the first observed run)
        first_run = ordered_runs[0]
        expected_runs = list(range(int(first_run), int(first_run) + len(ordered_runs)))

        # Check for continuity breaks (e.g., [5, 61, 62, 7] != [5, 6, 7, 8])
        if ordered_runs != expected_runs:
            run_mapping = {old: new for old, new in zip(ordered_runs, expected_runs)}

            msg = f"  -> [Session {sess}] Broken run continuity detected. Applying mapping: {run_mapping}"
            print(msg)

            # Apply mapping safely
            df_out.loc[mask_sess, 'run'] = df_out.loc[mask_sess, 'run'].map(run_mapping).fillna(df_out['run'])

    # 2. Unconditionally apply the new ID convention to the ENTIRE DataFrame
    # Convention: session * 10000 + run * 100 + trials
    df_out['id'] = (
            df_out['session'] * 10000 +
            df_out['run'] * 100 +
            df_out['trials']
    )

    return df_out

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    BASE = (
        "/home/tomerd/Documents/projects/MSc/lab/thesis_practice/"
        "relational_alignment/soto_data/ds003927/derivatives"
    )
    deriv_root = args.deriv_root or os.path.join(BASE, args.subject)
    mask_path  = args.mask_path  or os.path.join(deriv_root, "mask.nii.gz")
    roi_path   = args.roi_path   or os.path.join(deriv_root, "func_masks",
                                                  "lateral_occipital_mask.nii.gz")
    nifti_path = args.nifti      or os.path.join(deriv_root,
                                                  "wholebrain_conscious.nii.gz")
    csv_path   = args.csv        or os.path.join(deriv_root,
                                                  "wholebrain_conscious.csv")

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Masks
    print("Loading masks...")
    brain_mask   = nib.load(mask_path).get_fdata().astype(bool)
    roi_mask     = nib.load(roi_path).get_fdata().astype(bool)
    roi_in_brain = roi_mask & brain_mask
    print(f"  Brain mask voxels : {brain_mask.sum()}")
    print(f"  ROI voxels        : {roi_in_brain.sum()}")

    # 2. Derivative NIfTI
    print("\nLoading derivative NIfTI...")
    deriv_data = nib.load(nifti_path).get_fdata(dtype=np.float32)
    X_deriv    = deriv_data[roi_in_brain].T
    print(f"  Shape: {X_deriv.shape}  (trials x ROI voxels)")

    # 3. Metadata
    print("Loading metadata CSV...")
    meta = pd.read_csv(csv_path)
    meta = enforce_continuity_and_ids(meta)  # Apply the ID continuity and recalculation fix
    print(f"  Rows: {len(meta)}  Columns: {list(meta.columns)}")

    assert X_deriv.shape[0] == len(meta), (
        f"NIfTI volumes ({X_deriv.shape[0]}) != CSV rows ({len(meta)}). "
        "Check that CSV corresponds to the conscious NIfTI."
    )

    y = (meta["targets"] == "Living_Things").astype(int).values
    print(f"  Living: {y.sum()}  Non-living: {(1-y).sum()}")
    assert y.sum() > 0 and (1-y).sum() > 0, "Label vector is all one class!"

    if "labels" in meta.columns:
        item_labels = meta["labels"].values
    elif "stimulus" in meta.columns:
        item_labels = meta["stimulus"].values
    elif "id" in meta.columns:
        item_labels = meta["id"].values
    else:
        raise KeyError("Cannot find item identifier column (labels/stimulus/id) in CSV.")

    print(f"  Unique items: {len(np.unique(item_labels))}")

    # 4. Signal cleaning
    print("\nApplying two-stage signal cleaning + cross-run normalization...")
    X_clean = two_stage_clean_and_crossrun_normalize(X_deriv, meta)
    print(f"  X_clean shape: {X_clean.shape}")

    print(f"\nAggregating {X_clean.shape[0]} trials into item-level patterns...")
    X_agg, y_agg, items_agg = aggregate_to_items(X_clean, y, item_labels)
    print(f"  New matrix shape: {X_agg.shape}")

    # 5. Decode
    for C in args.C_values:
        print(f"\n--- Decoding C={C} ---")
        # auc, warns, n_folds = decode(X_clean, y, item_labels, C=C, n_jobs=args.n_jobs)
        auc, warns, n_folds = decode(X_agg, y_agg, items_agg, C=C, n_jobs=args.n_jobs)
        print(f"  Pooled AUC = {auc:.4f}   conv_warns={warns}/{n_folds}")

    # 6. Save
    subj = args.subject
    np.save(os.path.join(args.out_dir, f"X_deriv_clean_{subj}.npy"), X_clean)
    meta.to_csv(os.path.join(args.out_dir, f"meta_deriv_{subj}.csv"), index=False)
    print(f"\nSaved cleaned matrix -> {args.out_dir}")


if __name__ == "__main__":
    main()
