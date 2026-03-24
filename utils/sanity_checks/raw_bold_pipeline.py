"""
raw_bold_pipeline.py

Replicates Mei et al. (2022) preprocessing from raw BOLD runs and
immediately evaluates decoding performance on lateral occipital ROI.

File system layout expected:
  ds003927/
    sub-01/
      ses-02/func/sub-01_ses-02_task-recog_run-1_bold.nii.gz
                  sub-01_ses-02_task-recog_run-1_events.tsv
      ses-03/func/ ...
      ...
    derivatives/
      sub-01/
        mask.nii.gz
        func_masks/lateral_occipital_mask.nii.gz
        wholebrain_conscious.csv   ← used only for labels/metadata
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed, parallel_backend
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

# ── PATHS ────────────────────────────────────────────────────────────────────
SUBJECT        = "sub-01"
RAW_ROOT       = f"/home/tomerd/Documents/projects/MSc/lab/thesis_practice/relational_alignment/soto_data/temp/ds003927/{SUBJECT}"
DERIV_ROOT     = f"/home/tomerd/Documents/projects/MSc/lab/thesis_practice/relational_alignment/soto_data/ds003927/derivatives/{SUBJECT}"
OUT_DIR        = "/home/tomerd/Documents/projects/MSc/lab/thesis_practice/relational_alignment/poc_output/"

MASK_PATH      = os.path.join(DERIV_ROOT, "mask.nii.gz")
ROI_PATH       = os.path.join(DERIV_ROOT, "func_masks", "lateral_occipital_mask.nii.gz")
META_CSV       = os.path.join(DERIV_ROOT, "wholebrain_conscious.csv")

CACHE_X        = os.path.join(OUT_DIR, f"cache_{SUBJECT}_X_brain.npy")
CACHE_META     = os.path.join(OUT_DIR, f"cache_{SUBJECT}_meta.csv")

TR             = 0.85          # seconds
HRF_WINDOW_S   = (4.0, 7.0)   # seconds post-onset to average (paper: 4–7 s)
VISIBILITY     = "conscious"
SESSIONS       = [2, 3, 4, 5, 6, 7]
N_JOBS         = 4
os.makedirs(OUT_DIR, exist_ok=True)


# ── HELPER: single run extraction ────────────────────────────────────────────

def extract_run(ses, run_num, bold_path, events_path, brain_mask, meta):
    bold_img  = nib.load(bold_path)
    bold_data = bold_img.get_fdata(dtype=np.float32)
    n_vols    = bold_data.shape[-1]
    ts        = bold_data[brain_mask].T  # (T, V)

    # block-wise z-score per voxel
    mu  = ts.mean(axis=0)
    sig = ts.std(axis=0)
    sig[sig < 1e-8] = 1.0
    ts  = (ts - mu) / sig

    # linear detrend per voxel
    t_axis = np.arange(n_vols, dtype=np.float32)
    t_axis = t_axis - t_axis.mean()
    slope  = (ts * t_axis[:, None]).sum(0) / (t_axis ** 2).sum()
    ts     = ts - np.outer(t_axis, slope)

    events = pd.read_csv(events_path, sep="\t")

    # drop fixation/blank TRs, then take first TR per trial as trial onset
    events = events.dropna(subset=["targets"])
    trial_events = events.groupby("trials", sort=True).first().reset_index()

    # filter to requested visibility condition
    if "visibility" in trial_events.columns:
        trial_events = trial_events[trial_events["visibility"] == VISIBILITY]

    results = []
    for _, ev_row in trial_events.iterrows():
        onset_s   = float(ev_row["onset"])
        onset_vol = int(round(onset_s / TR))
        hrf_start = onset_vol + int(round(HRF_WINDOW_S[0] / TR))
        hrf_end   = min(onset_vol + int(round(HRF_WINDOW_S[1] / TR)), n_vols)

        if hrf_start >= n_vols or (hrf_end - hrf_start) < 1:
            continue

        pattern = ts[hrf_start:hrf_end, :].mean(axis=0)
        label   = str(ev_row.get("targets", "unknown"))

        results.append((pattern, {
            "session": ses,
            "run"    : run_num,
            "onset_s": onset_s,
            "label"  : label,
        }))

    return results



# ── HELPER: single SVM fold ───────────────────────────────────────────────────

def run_one_fold(train_idx, test_idx, X, y, C):
    X_tr, X_te = X[train_idx].copy(), X[test_idx].copy()
    y_tr, y_te = y[train_idx], y[test_idx]

    var_mask   = X_tr.var(axis=0) > 1e-8
    X_tr, X_te = X_tr[:, var_mask], X_te[:, var_mask]

    scaler = MinMaxScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LinearSVC(penalty="l1", loss="squared_hinge", dual=False,
                        C=C, max_iter=50000, class_weight="balanced")
        clf.fit(X_tr, y_tr)
    n_warns = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))

    return roc_auc_score(y_te, clf.decision_function(X_te)), n_warns


def decode(X, y, C=1.0, n_splits=10, seed=42):
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))

    with parallel_backend("loky", n_jobs=N_JOBS):
        fold_results = Parallel()(
            delayed(run_one_fold)(tr, te, X, y, C)
            for tr, te in tqdm(
                splits, desc=f"SVM C={C}", ncols=80, leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} folds [{elapsed}<{remaining}]"
            )
        )

    aucs       = [r[0] for r in fold_results]
    conv_warns = sum(r[1] for r in fold_results)
    return np.array(aucs), conv_warns


# ── STEP 1: load masks ────────────────────────────────────────────────────────
print("Loading masks...")
brain_mask = nib.load(MASK_PATH).get_fdata().astype(bool)
roi_mask   = nib.load(ROI_PATH).get_fdata().astype(bool)
roi_in_brain = roi_mask & brain_mask
print(f"  Brain mask voxels : {brain_mask.sum()}")
print(f"  ROI voxels        : {roi_in_brain.sum()}")


# ── STEP 2: load metadata ─────────────────────────────────────────────────────
print("Loading metadata CSV...")
meta = pd.read_csv(META_CSV)
meta = meta[meta["visibility"] == VISIBILITY].sort_values("id").reset_index(drop=True)
print(f"  Conscious trials  : {len(meta)}")


# ── STEP 3: extract patterns (cached) ────────────────────────────────────────

if os.path.exists(CACHE_X) and os.path.exists(CACHE_META):
    print("Cache found — loading extracted patterns...")
    X_full  = np.load(CACHE_X)
    meta_df = pd.read_csv(CACHE_META)
    print(f"  Loaded: {X_full.shape[0]} trials × {X_full.shape[1]} voxels")

else:
    # collect run jobs
    run_jobs = []
    for ses in SESSIONS:
        ses_str  = f"ses-{ses:02d}"
        func_dir = os.path.join(RAW_ROOT, ses_str, "func")
        if not os.path.isdir(func_dir):
            continue
        run_files = sorted([
            f for f in os.listdir(func_dir)
            if f.endswith("_bold.nii.gz") and "task-recog" in f
        ])
        for bold_fname in run_files:
            run_num     = int(bold_fname.split("_run-")[1].split("_")[0])
            bold_path   = os.path.join(func_dir, bold_fname)
            events_path = bold_path.replace("_bold.nii.gz", "_events.tsv")
            if not os.path.exists(events_path):
                continue
            run_jobs.append((ses, run_num, bold_path, events_path))

    print(f"  Found {len(run_jobs)} runs — extracting in parallel ({N_JOBS} workers)...")

    with parallel_backend("loky", n_jobs=N_JOBS):
        raw_outputs = Parallel()(
            delayed(extract_run)(ses, run, bp, ep, brain_mask, meta)
            for ses, run, bp, ep in tqdm(
                run_jobs, desc="Extracting runs", ncols=80, leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} runs [{elapsed}<{remaining}]"
            )
        )

    all_patterns, all_meta = [], []
    for run_result in raw_outputs:
        for pattern, m in run_result:
            all_patterns.append(pattern)
            all_meta.append(m)

    X_full  = np.array(all_patterns)
    meta_df = pd.DataFrame(all_meta)

    np.save(CACHE_X, X_full)
    meta_df.to_csv(CACHE_META, index=False)
    print(f"  Cached → {CACHE_X}")
    print(f"  Total patterns    : {X_full.shape[0]}")
    print(f"  Brain voxels      : {X_full.shape[1]}")

print(meta_df["label"].value_counts())
print(f"Rows with unknown label: {(meta_df['label']=='unknown').sum()}")

# ── STEP 4: assemble pattern matrix ──────────────────────────────────────────
print("\nAssembling pattern matrix...")
X_roi = X_full[:, roi_in_brain[brain_mask]]
y_all = (meta_df["label"] == "Living_Things").astype(int).values

print(f"  Total patterns    : {X_full.shape[0]}")
print(f"  Brain voxels      : {X_full.shape[1]}")
print(f"  ROI voxels        : {X_roi.shape[1]}")
print(f"  After dedup       : {len(meta_df)} unique trials")

# ── filter to conscious trials only ──────────────────────────────────────────

# The events TSV doesn't carry visibility, but meta (from wholebrain_conscious.csv)
# does. Match by session + run + nearest onset:
meta_conscious = pd.read_csv(META_CSV)
meta_conscious = meta_conscious[meta_conscious["visibility"] == VISIBILITY] \
                               .sort_values("id").reset_index(drop=True)

# match extracted rows to conscious rows via session + run + onset
meta_df["conscious"] = False
for _, row in meta_conscious.iterrows():
    match = (
        (meta_df["session"] == row["session"]) &
        (meta_df["run"] == row["run"]) &
        (np.abs(meta_df["onset_s"] - row["onset"]) < TR * 1.5) &
        (meta_df["label"] == row["targets"])
    )
    meta_df.loc[match, "conscious"] = True

X = X_roi[meta_df["conscious"].values]
y = y_all[meta_df["conscious"].values]

print(f"  Conscious trials  : {X.shape[0]}  (expected ~514)")
print(f"  Living            : {y.sum()}  Non-living: {(1-y).sum()}")

if y.sum() == 0 or (1 - y).sum() == 0:
    print("\n[ERROR] Label vector is all one class — check label matching above.")
    raise SystemExit



# ── STEP 5: SVM decoding ─────────────────────────────────────────────────────
print("\nRunning SVM cross-validation (10-fold stratified)...")

aucs_C1, warns_C1 = decode(X, y, C=1.0)
aucs_C5, warns_C5 = decode(X, y, C=5.0)

print(f"\n{'='*50}")
print(f"  C=1  | AUC = {aucs_C1.mean():.4f} ± {aucs_C1.std():.4f}  convWarns={warns_C1}/10")
print(f"  C=5  | AUC = {aucs_C5.mean():.4f} ± {aucs_C5.std():.4f}  convWarns={warns_C5}/10")
print(f"{'='*50}")
print("\nComparison against derivative-based result (~0.557):")
print(f"  Raw BOLD pipeline (C=1): {aucs_C1.mean():.4f}")
print(f"  Derivative pipeline    :  0.5570  (from earlier run)")
delta = aucs_C1.mean() - 0.5570
print(f"  Delta                  : {delta:+.4f}  {'↑ signal recovered' if delta > 0.03 else '≈ same, derivative was sufficient' if abs(delta) < 0.03 else '↓ worse — further investigation needed'}")

# ── STEP 6: save pattern matrix for reuse ────────────────────────────────────
np.save(os.path.join(OUT_DIR, "X_raw_pipeline_lateral_occipital.npy"), X)
meta_df.to_csv(os.path.join(OUT_DIR, "meta_raw_pipeline.csv"), index=False)
print(f"\nSaved pattern matrix → {OUT_DIR}")

