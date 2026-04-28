# Soto et al. (2022) — fMRI Preprocessing Replication

Replication of the preprocessing pipeline from:
> Mei, Santana & Soto (2022). *Informative neural representations of unseen
> contents during higher-order processing in human brains and deep artificial
> networks.* Nature Human Behaviour, 6, 720–731.

---

## Directory layout (your system)

```
soto_data/
├── ds003927/                       ← OpenNeuro BIDS dataset (read-only)
│   ├── sub-01/ … sub-07/
│   └── derivatives/sub-01/ …
├── author_replication/             ← ALL outputs go here (created by scripts)
│   ├── MRI/
│   │   └── sub-01/
│   │       ├── anat/               ← symlinks + FreeSurfer output + ROI masks
│   │       └── func/
│   │           └── session-02/
│   │               └── sub-01_unfeat_run-1/
│   │                   ├── sub-01_unfeat_run-1_bold.nii.gz  (symlink)
│   │                   └── outputs/
│   │                       ├── func/
│   │                       │   ├── prefiltered_func.nii.gz   ← step 01
│   │                       │   ├── mask.nii.gz
│   │                       │   ├── example_func.nii.gz
│   │                       │   ├── mean_func.nii.gz
│   │                       │   ├── MC/MCflirt.par
│   │                       │   ├── ICA_AROMA/
│   │                       │   │   └── denoised_func_data_nonaggr.nii.gz  ← step 02
│   │                       │   └── ICAed_filtered/
│   │                       │       └── filtered.nii.gz       ← step 06
│   │                       └── reg/   (reference run only)   ← step 01
│   │                           ├── example_func2highres.mat
│   │                           └── highres2standard_warp.nii.gz
│   ├── work/                       ← nipype temp dirs (auto-cleaned)
│   └── manifests/sub-01_manifest.json
├── data/
│   └── standard_brain/             ← MNI152 T1 2mm files (or use $FSLDIR)
│       ├── MNI152_T1_2mm_brain.nii.gz
│       ├── MNI152_T1_2mm.nii.gz
│       └── MNI152_T1_2mm_brain_mask_dil.nii.gz
└── utils.py                        ← author's nipype workflow factory (required)
```

---

## Dependencies

| Package | Version tested by authors |
|---------|--------------------------|
| FSL     | 5.0.10 (≥ 5.0.9, ≠ 6.0.0 — known compatibility issue in utils.py) |
| FreeSurfer | 6.0.0 |
| ICA-AROMA | any (called via nipype's FSL interface) |
| Python  | 3.x |
| nipype  | ≥ 1.6 |
| nilearn | ≥ 0.7 |
| numpy, pandas, tqdm | standard |

Set environment variables before running:
```bash
export FSLDIR=/usr/local/fsl          # adjust to your installation
source $FSLDIR/etc/fslconf/fsl.sh
export FREESURFER_HOME=/usr/local/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

Copy `utils.py` (the author's shared library) to the same directory as
these scripts **or** to the project root (`soto_data/utils.py`).

---

## Important notes on sub-01

- **Sessions**: ses-02 through ses-07 (no ses-01). First session = `ses-02`.
- **Session 04**: contains run-61 and run-62 instead of run-6 (a split scan).
  These are included as separate runs.
- **BET brain**: `derivatives/sub-01/anat/sub-01_T1w_0.39_brain.nii.gz`
  already exists — step 00 symlinks it into the author tree.
- **FreeSurfer**: `derivatives/sub-01/fs_workspace/sub-01/` is partial
  (no `aparc+aseg.mgz`). Step 03 runs full `recon-all` (~8–12 h).
- **ROI masks**: `derivatives/sub-01/func_masks/` contains the authors'
  *final* BOLD-space masks. After running the full pipeline you can diff
  these against your step 04 outputs to verify correctness.

---

## Author scripts — what was used

| Author file | Your replacement | Used? |
|---|---|---|
| `0.preprocess fmri.py` | `01_preprocess_fmri.py` | ✅ |
| `1.ICA_AROMA.py` | `02_ica_aroma.py` | ✅ |
| `2.freesurfer reconall.py` | `03_freesurfer_reconall.py` | ✅ |
| `3.extract ROI … BOLD space.py` | `04_extract_roi_bold.py` | ✅ |
| `3.2.extract ROI … standard space.py` | `05_extract_roi_standard.py` | ✅ |
| `4.highpass filter.py` | `06_highpass_filter.py` | ✅ |
| `stacking runs*.py` | — | ❌ Post-preprocessing (SVM inputs, not raw fMRI) |
| `create event file.py` | — | ❌ Behavioural only |
| `modify and save_*.py` | — | ❌ Cluster-submission helpers |
| `MRIconvert.py` | — | ❌ DICOM→NIfTI (already BIDS NIfTI) |

---

## Step-by-step execution

Place all scripts + `utils.py` in the same working directory, then:

### Step 0 — Build working tree  (~5 seconds)
```bash
python 00_prepare_author_tree.py --subject sub-01
```
Creates author-style directory tree with symlinks into ds003927.

### Step 1 — FEAT-like preprocessing  (~4–6 h for all 54 runs)
```bash
python 01_preprocess_fmri.py --subject sub-01 --workers 4
```
- Reference run (ses-02, run-1) processed first and sequentially.
- Remaining 53 runs processed in parallel with `--workers 4`.
- Increase to `--workers 8` if RAM permits (monitor with `htop`).

**Key parameters** (from paper & utils.py):
| Parameter | Value |
|---|---|
| Drop first N volumes | 8 (`n_vol_remove=8`) |
| Total volumes kept | 495 (`total_vol=495`) |
| Reference volume | middle volume of run-1 (`whichvol='middle'`) |
| MCFLIRT interpolation | spline |
| Spatial smoothing FWHM | 3 mm |
| Intensity normalisation | median → 10 000 |

### Step 2 — ICA-AROMA denoising  (~30–60 min total)
```bash
python 02_ica_aroma.py --subject sub-01 --workers 4
```
Non-aggressive denoising (`denoise_type='nonaggr'`).

### Step 3 — FreeSurfer recon-all  (~8–12 h)
```bash
python 03_freesurfer_reconall.py --subject sub-01 --openmp 8
```
Runs `recon-all -all` to produce `aparc+aseg.mgz` for ROI extraction.

### Step 4 — ROI masks → BOLD space  (~15–30 min)
```bash
python 04_extract_roi_bold.py --subject sub-01
```
Extracts 28 cortical ROIs + bilateral ventrolateralPFC. Uses
`create_simple_struc2BOLD` (FLIRT, linear registration).

### Step 5 — ROI masks → MNI space  (~15–30 min)
```bash
python 05_extract_roi_standard.py --subject sub-01
```
Same ROIs warped to MNI space via FNIRT warp field.

### Step 6 — High-pass filter  (~1–2 h total)
```bash
python 06_highpass_filter.py --subject sub-01 --workers 4
```
**Key parameters**:
| Parameter | Value |
|---|---|
| Cutoff period | 60 s |
| TR | 0.85 s |
| FSL sigma (volumes) | 60 / 2 / 0.85 = 35.29 |
| FSL flag | `-bptf 35.2941176471 -1` |

---

## Output verification

Compare your step 04 ROI BOLD masks against the authors' final masks:
```bash
fslcc author_replication/MRI/sub-01/anat/ROI_BOLD/ctx-lh-fusiform_fsl_BOLD.nii.gz \
      ds003927/derivatives/sub-01/func_masks/fusiform_mask.nii.gz
```
Expect spatial correlation > 0.95 (minor differences from registration
may give < 1.0).

Final preprocessed data per run:
```
author_replication/MRI/sub-01/func/session-*/sub-01_unfeat_run-*/
    outputs/func/ICAed_filtered/filtered.nii.gz
```
These are the files the authors fed into the MVPA (SVM) decoding analyses.

---

## Common issues

| Problem | Fix |
|---|---|
| `FileNotFoundError: MNI152 standard files` | Copy FSL standard files to `soto_data/data/standard_brain/` or ensure `$FSLDIR` is set |
| `FSL version 6.0.0` registration errors | Use FSL 5.0.9–5.0.11 (known utils.py incompatibility with 6.0) |
| `ICA_AROMA` not found | Install via `pip install ICA_AROMA` or ensure it's in PATH |
| Memory errors with workers > 4 | Reduce `--workers` |
| recon-all fails | Check `FREESURFER_HOME` and `SUBJECTS_DIR` |
