"""
Detrend validation script for ds003927 derivatives.
Run from the project root:  python check_detrend.py
"""
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

# ── Paths (from config.yaml) ──────────────────────────────────────────────────
DATA_ROOT = Path("/home/tomerd/Documents/projects/MSc/lab/thesis_practice/"
                 "relational_alignment/soto_data/ds003927/derivatives/")
OUT_DIR   = Path("/home/tomerd/Documents/projects/MSc/lab/thesis_practice/"
                 "relational_alignment/poc_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Test on sub-01, conscious (representative — swap state/subject as needed)
SUBJECT   = "sub-01"
STATE     = "conscious"

sub_dir   = DATA_ROOT / SUBJECT
nifti_path = sub_dir / f"wholebrain_{STATE}.nii.gz"
csv_path   = sub_dir / f"wholebrain_{STATE}.csv"
mask_path  = sub_dir / "mask.nii.gz"

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {nifti_path.name} …")
img  = nib.load(nifti_path)
data = img.get_fdata()
mask = nib.load(mask_path).get_fdata().astype(bool)

print(f"  NIfTI shape : {data.shape}")        # expect (X, Y, Z, n_trials)
print(f"  Mask voxels : {mask.sum()}")
print(f"  Value range : [{data.min():.4f}, {data.max():.4f}]")
print(f"  Global mean : {data.mean():.4f}  std: {data.std():.4f}")

flat = data[mask].T   # (n_trials, n_voxels)

# ── Load CSV and assign run indices ──────────────────────────────────────────
df = pd.read_csv(csv_path)
df["run_idx"] = df.groupby(["session", "run"]).ngroup()

assert len(df) == flat.shape[0], (
    f"CSV rows ({len(df)}) ≠ NIfTI volumes ({flat.shape[0]}) — mismatch!"
)

print(f"\nCSV rows    : {len(df)}")
print(f"Unique runs : {df['run_idx'].nunique()}")
print(df.groupby(["session", "run", "run_idx"]).size()
        .rename("n_trials").reset_index().to_string(index=False))

# ── Linear regression slope test per run ─────────────────────────────────────
print("\n── Per-run linear slope test ────────────────────────────────────────────")
slopes = []
for run_id in sorted(df["run_idx"].unique()):
    idx      = df[df["run_idx"] == run_id].index.values
    run_mean = flat[idx].mean(axis=1)   # mean across voxels per trial
    x        = np.arange(len(run_mean))
    slope, intercept, r, p, _ = linregress(x, run_mean)
    sess = df.loc[idx[0], "session"]
    run  = df.loc[idx[0], "run"]
    slopes.append({"run_idx": run_id, "session": sess, "run": run,
                   "n_trials": len(idx), "slope": slope, "r": r, "p": p})

slope_df = pd.DataFrame(slopes)
print(slope_df.round(6).to_string(index=False))
print(f"\nMean |slope|            : {slope_df['slope'].abs().mean():.6f}")
print(f"Runs with p < 0.05      : {(slope_df['p'] < 0.05).sum()} / {len(slope_df)}")
print(f"Runs with |r| > 0.4     : {(slope_df['r'].abs() > 0.4).sum()} / {len(slope_df)}")

# ── Interpretation ────────────────────────────────────────────────────────────
sig_frac = (slope_df['p'] < 0.05).mean()
if sig_frac < 0.1:
    verdict = "✅ Likely already detrended — skip scipy.signal.detrend in subject_builder.py"
elif sig_frac < 0.35:
    verdict = "⚠️  Ambiguous — check the plot carefully before deciding"
else:
    verdict = "❌ Residual trend detected — ADD detrending in subject_builder.py"
print(f"\nVERDICT: {verdict}")

# ── Plot: voxel-mean time series per run ─────────────────────────────────────
from scipy.signal import detrend as scipy_detrend

n_runs  = df["run_idx"].nunique()
n_cols  = 6
n_rows  = (n_runs + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 3.5, n_rows * 2.5),
                         sharey=False)
axes = axes.flatten()

for run_id in sorted(df["run_idx"].unique()):
    idx       = df[df["run_idx"] == run_id].index.values
    raw       = flat[idx].mean(axis=1)
    detrended = scipy_detrend(raw, type="linear")
    ax        = axes[run_id]
    ax.plot(raw,       lw=1.2, label="raw derivative")
    ax.plot(detrended, lw=1.2, linestyle="--", label="after detrend")
    sess = df.loc[idx[0], "session"]
    run  = df.loc[idx[0], "run"]
    row  = slope_df[slope_df["run_idx"] == run_id].iloc[0]
    ax.set_title(f"ses-{sess} run-{run}\n"
                 f"slope={row['slope']:.4f}  p={row['p']:.3f}",
                 fontsize=7)
    ax.set_xlabel("trial", fontsize=7)
    ax.tick_params(labelsize=6)

axes[0].legend(fontsize=6)
for ax in axes[n_runs:]:
    ax.set_visible(False)

plt.suptitle(f"{SUBJECT} | {STATE} — voxel-mean activation per run\n"
             f"(raw vs. what detrending would do)", fontsize=9)
plt.tight_layout()
out_path = OUT_DIR / f"detrend_check_{SUBJECT}_{STATE}.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved → {out_path}")
