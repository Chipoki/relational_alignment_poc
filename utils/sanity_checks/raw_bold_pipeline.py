"""
raw_bold_pipeline.py  ·  v4  (Tomer Dalumi, 2026)

PURPOSE
-------
Replicates Mei et al. (2022) preprocessing starting from the raw BOLD
runs in the ds003927 BIDS directory and evaluates decoding on the
lateral occipital ROI.

All four critical fixes are applied:

  FIX-1  Two-stage per-run signal cleaning
           Stage-A : nilearn clean_signal(full run, detrend=True, standardize=False)
           Stage-B : clean_signal(selected trial TRs, detrend=False, standardize=True)

  FIX-2  Cross-run confound removal via clean_signal(sessions=run_index)
           after all runs are stacked

  FIX-3  Exhaustive Leave-One-Pair-Out CV
           (all n_living x n_nonliving folds, matching utils.LOO_partition)

  FIX-4  Pooled AUC — concatenate all held-out scores before roc_auc_score()

FILE SYSTEM EXPECTED
--------------------
ds003927/sub-01/
    ses-02/func/sub-01_ses-02_task-recog_run-1_bold.nii.gz
                sub-01_ses-02_task-recog_run-1_events.tsv
    ses-03/func/ ...  (ses-02 through ses-07)
derivatives/sub-01/
    mask.nii.gz
    func_masks/lateral_occipital_mask.nii.gz
    wholebrain_conscious.csv   <- used only to resolve item labels for LOPO CV

USAGE
-----
python raw_bold_pipeline.py                          # all defaults
python raw_bold_pipeline.py --subject sub-02
python raw_bold_pipeline.py --raw-root /data/ds003927/sub-01 \
                             --deriv-root /data/derivatives/sub-01 \
                             --out-dir /results
python raw_bold_pipeline.py --sessions 2 3 4 --C-values 0.1 1.0 --n-jobs 8
python raw_bold_pipeline.py --no-cache
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
        description="Raw BOLD pipeline: Mei et al. (2022) decoding from raw NIfTIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subject", default="sub-01",
        help="Subject ID (e.g. sub-01)",
    )
    parser.add_argument(
        "--raw-root", default=None,
        help=(
            "Path to the raw BIDS subject folder. "
            "Defaults to .../temp/ds003927/<subject>"
        ),
    )
    parser.add_argument(
        "--deriv-root", default=None,
        help=(
            "Path to the derivative subject folder (masks + metadata CSV). "
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
        "--sessions", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7],
        metavar="SES",
        help="Session numbers to include (BIDS ses-XX indices).",
    )
    parser.add_argument(
        "--visibility", default="conscious",
        choices=["conscious", "glimpse", "unconscious"],
        help="Visibility condition to decode.",
    )
    parser.add_argument(
        "--C-values", nargs="+", type=float, default=[1.0],
        metavar="C",
        help="One or more SVM regularisation values to sweep.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore existing cache and re-extract patterns from scratch.",
    )
    return parser.parse_args()


# ── SINGLE-RUN EXTRACTION  (FIX-1 Stage-A + Stage-B) ─────────────────────────

def extract_run(ses, run_num, bold_path, events_path, brain_mask, tr, hp_freq,
                visibility):
    """
    Load one run, apply two-stage cleaning, extract trial patterns.

    volume_interest flags 3-4 consecutive HRF-peak TRs per trial.
    These are averaged (.mean(axis=0)) to give one robust pattern per trial.
    No cross-contamination: each trial block is separated by several
    non-flagged TRs (trial duration ~12-15s, flagged window ~2.5-3.4s).
    """
    bold_img  = nib.load(bold_path)
    bold_data = bold_img.get_fdata(dtype=np.float32)
    ts = bold_data[brain_mask].T   # (T, V)

    # Stage-A
    ts_a = clean_signal(
        ts,
        t_r         = tr,
        detrend     = True,
        standardize = False,
        high_pass   = hp_freq,
    )

    events    = pd.read_csv(events_path, sep="\t")
    events    = events.dropna(subset=["targets"])
    vi_events = events[events["volume_interest"] == 1].copy()

    if vi_events.empty:
        return []

    tr_to_pattern = {}
    for trial_num, group in vi_events.groupby("trials", sort=True):
        tr_indices = group.index.tolist()
        tr_to_pattern[trial_num] = (ts_a[tr_indices, :].mean(axis=0), group.iloc[0])

    if not tr_to_pattern:
        return []

    # Stage-B
    trial_matrix   = np.stack([v[0] for v in tr_to_pattern.values()])
    trial_matrix_b = clean_signal(
        trial_matrix,
        t_r         = None,
        detrend     = False,
        standardize = 'zscore_sample',
    )

    results = []
    for idx, (trial_num, (_, row)) in enumerate(tr_to_pattern.items()):
        if str(row["visibility"]) != visibility:
            continue

        # Calculate overflow-proof ID natively
        trial_id = ses * 10000 + run_num * 100 + float(trial_num)

        item_name = str(row["labels"]) if "labels" in row else (
            str(row["stimulus"]) if "stimulus" in row
            else f"item_{int(trial_num)}")

        results.append((
            trial_matrix_b[idx],
            {
                "session": ses,
                "run": run_num,  # Passed in as strictly continuous
                "trial": float(trial_num),
                "id": trial_id,
                "label": str(row["targets"]),
                "visibility": str(row["visibility"]),
                "item_name": item_name,
            }
        ))
    return results


# ── CROSS-RUN NORMALIZATION  (FIX-2) ─────────────────────────────────────────

def cross_run_normalize(X_trials, meta_df):
    """
    NOTE: session column in meta_df is always 1; real session from id // 1000.
    """
    meta_df = meta_df.copy()
    meta_df["_ses_idx"] = (meta_df["id"] // 1000).astype(int)
    run_labels   = (meta_df["_ses_idx"].astype(str) + "_" +
                    meta_df["run"].astype(str)).values
    unique_runs  = np.unique(run_labels)
    run_to_int   = {r: i for i, r in enumerate(unique_runs)}
    sessions_vec = np.array([run_to_int[r] for r in run_labels])

    X_clean = clean_signal(
        X_trials.astype(np.float64),
        sessions    = sessions_vec,
        t_r         = None,
        standardize = 'zscore_sample',
        detrend     = True,
    )
    return X_clean


# ── LEAVE-ONE-PAIR-OUT CV  (FIX-3) ───────────────────────────────────────────

def loo_partition(y, item_labels):
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
            max_iter     = 20000,
            class_weight = "balanced",
            random_state = 12345,
        )
        clf.fit(X_tr, y_tr)
    n_warns = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))

    scores = clf.decision_function(X_te)
    return y_te, scores, n_warns


def decode(X, y, item_labels, C, n_jobs):
    splits = loo_partition(y, item_labels)
    print(f"  CV: {len(splits)} folds  "
          f"(n_living={len(np.unique(item_labels[y==1]))} "
          f"x n_nonliving={len(np.unique(item_labels[y==0]))})")

    with parallel_backend("loky", n_jobs=n_jobs):
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

    all_y_true = np.concatenate([r[0] for r in fold_results])
    all_scores = np.concatenate([r[1] for r in fold_results])
    conv_warns = sum(r[2] for r in fold_results)

    pooled_auc = roc_auc_score(all_y_true, all_scores)
    return pooled_auc, conv_warns, len(splits)

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
            # extract last two digits (before decimal point)
            df_out['id'].mod(100).astype(int)
    )

    return df_out

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    BASE_RAW   = (
        "/home/tomerd/Documents/projects/MSc/lab/thesis_practice/"
        "relational_alignment/soto_data/temp/ds003927"
    )
    BASE_DERIV = (
        "/home/tomerd/Documents/projects/MSc/lab/thesis_practice/"
        "relational_alignment/soto_data/ds003927/derivatives"
    )

    raw_root   = args.raw_root   or os.path.join(BASE_RAW,   args.subject)
    deriv_root = args.deriv_root or os.path.join(BASE_DERIV, args.subject)
    mask_path  = args.mask_path  or os.path.join(deriv_root, "mask.nii.gz")
    roi_path   = args.roi_path   or os.path.join(deriv_root, "func_masks",
                                                  "lateral_occipital_mask.nii.gz")
    os.makedirs(args.out_dir, exist_ok=True)

    subj       = args.subject
    TR         = 0.85
    HP_FREQ    = 1 / 60.0
    CACHE_X    = os.path.join(args.out_dir, f"cache_{subj}_raw_Xbrain.npy")
    CACHE_META = os.path.join(args.out_dir, f"cache_{subj}_raw_meta.csv")

    # 1. Masks
    print("Loading masks...")
    brain_mask   = nib.load(mask_path).get_fdata().astype(bool)
    roi_mask     = nib.load(roi_path).get_fdata().astype(bool)
    roi_in_brain = roi_mask & brain_mask
    print(f"  Brain mask voxels : {brain_mask.sum()}")
    print(f"  ROI voxels        : {roi_in_brain.sum()}")

    # 2. Cache or extract
    use_cache = (not args.no_cache and
                 os.path.exists(CACHE_X) and os.path.exists(CACHE_META))

    if use_cache:
        print("\nCache found — loading extracted patterns...")
        X_full  = np.load(CACHE_X)
        meta_df = pd.read_csv(CACHE_META)
        print(f"  Loaded: {X_full.shape[0]} trials x {X_full.shape[1]} voxels")
    else:
        run_jobs = []
        for ses in args.sessions:
            ses_str = f"ses-{ses:02d}"
            func_dir = os.path.join(raw_root, ses_str, "func")
            if not os.path.isdir(func_dir):
                print(f"  [WARN] Not found: {func_dir}")
                continue

            # Sort files chronologically
            events_files = sorted([
                f for f in os.listdir(func_dir)
                if f.endswith("_events.tsv") and "task-recog" in f
            ])

            # Filter for valid pairs
            valid_runs = []
            for fname in events_files:
                ev_path = os.path.join(func_dir, fname)
                bold_path = os.path.join(func_dir, fname.replace("_events.tsv", "_bold.nii.gz"))
                if os.path.exists(bold_path):
                    valid_runs.append((bold_path, ev_path))

            # Enforce chronological continuity at the job dispatch level
            for idx, (bold_path, ev_path) in enumerate(valid_runs):
                continuous_run_num = idx + 1  # Forces 1, 2, 3... with no gaps
                run_jobs.append((ses, continuous_run_num, bold_path, ev_path))

        print(f"\nFound {len(run_jobs)} runs — extracting ({args.n_jobs} workers)...")
        with parallel_backend("loky", n_jobs=args.n_jobs):
            raw_outputs = Parallel()(
                delayed(extract_run)(ses, run, bp, ep, brain_mask,
                                     TR, HP_FREQ, args.visibility)
                for ses, run, bp, ep in tqdm(
                    run_jobs, desc="Extracting runs", ncols=80, leave=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} runs "
                               "[{elapsed}<{remaining}]",
                )
            )

        all_patterns, all_meta = [], []
        for run_result in raw_outputs:
            for pattern, m in run_result:
                all_patterns.append(pattern)
                all_meta.append(m)

        X_full  = np.array(all_patterns)
        meta_df = pd.DataFrame(all_meta)
        meta_df = enforce_continuity_and_ids(meta_df)
        np.save(CACHE_X, X_full)
        meta_df.to_csv(CACHE_META, index=False)
        print(f"  Cached -> {CACHE_X}")
        print(f"  Patterns: {X_full.shape[0]}  Brain voxels: {X_full.shape[1]}")

    # 3. Build ROI matrix
    X_roi = X_full[:, roi_in_brain[brain_mask]]
    y     = (meta_df["label"] == "Living_Things").astype(int).values
    print(f"\n  {args.visibility} trials: {X_roi.shape[0]}  "
          f"(Living: {y.sum()}  Non-living: {(1-y).sum()})")
    assert y.sum() > 0 and (1-y).sum() > 0, "Label vector is all one class!"

    item_labels = meta_df["item_name"].values
    print(f"  Unique items: {len(np.unique(item_labels))}")

    # 4. Cross-run normalization
    print("\nApplying cross-run normalization...")
    X_clean = cross_run_normalize(X_roi, meta_df)

    # 5. Decode
    for C in args.C_values:
        print(f"\n--- Decoding C={C} ---")
        auc, warns, n_folds = decode(X_clean, y, item_labels, C=C,
                                     n_jobs=args.n_jobs)
        print(f"  Pooled AUC = {auc:.4f}   conv_warns={warns}/{n_folds}")

    # 6. Save
    np.save(os.path.join(args.out_dir, f"X_raw_clean_{subj}.npy"), X_clean)
    meta_df.to_csv(os.path.join(args.out_dir, f"meta_raw_{subj}.csv"), index=False)
    print(f"\nSaved cleaned matrix -> {args.out_dir}")


if __name__ == "__main__":
    main()
