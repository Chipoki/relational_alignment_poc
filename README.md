# Neuro-RSA POC

> **Representational Similarity Analysis + Gromov-Wasserstein Optimal Transport**  
> Exploratory analysis of human fMRI activity patterns vs. deep feedforward neural networks  
> following [Mei, Santana & Soto (2022), *Nature Human Behaviour*](https://doi.org/10.1038/s41562-021-01274-7)

---

## Overview

This project implements the six-phase Proof-of-Concept (POC) described in the attached design document.
It is structured as a **modular, OOP Python project** that can run in full pipeline mode or phase-by-phase.

```
Phase 1  →  Data loading & FCNN embedding extraction
Phase 2  →  Dual-state intra-modality RDM construction
Phase 3  →  Balanced inter-subject representational analysis  (RSA + GWOT)
Phase 4  →  2×2 cross-modality alignment
Phase 5  →  Structural invariance metric
Phase 6  →  Relational visualisations
```

---

## Project Layout

```
poc_rsa_project/
├── config/                     # Typed Settings class + YAML config
│   ├── config.yaml
│   └── settings.py
├── data/                       # fMRI data loading, Subject containers, preprocessors
│   ├── loaders/
│   │   ├── subject.py          # Subject & VisibilityData dataclasses
│   │   ├── fmri_loader.py      # NIfTI BOLD + mask loading, HRF windowing
│   │   └── behavioral_loader.py
│   └── preprocessors/
│       ├── roi_extractor.py    # Per-ROI pattern extraction
│       └── subject_builder.py  # End-to-end subject assembly
├── embeddings/                 # MobileNetV2 hidden-layer extractor + fMRI organiser
│   ├── fcnn_embedder.py
│   ├── fmri_embedder.py
│   └── embedding_store.py
├── analysis/                   # RDM construction, RSA, Gromov-Wasserstein
│   ├── rsa/
│   │   ├── rdm.py
│   │   ├── rsa_analyzer.py
│   │   └── noise_ceiling.py
│   └── gromov_wasserstein/
│       ├── gw_aligner.py
│       └── gw_result.py
├── visualization/              # All figure generation
│   ├── rdm_plotter.py
│   ├── meta_mds_plotter.py
│   ├── transport_plotter.py
│   └── summary_plotter.py
├── utils/
│   ├── logging_utils.py
│   └── io_utils.py
├── scripts/
│   └── run_pipeline.py         # CLI entry point / full orchestrator
└── environment.yml             # Conda environment specification
```

---

## Environment Setup

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate poc_rsa

# Verify
python -c "import torch; print(torch.__version__)"
python -c "import ot; print('POT OK')"
python -c "import nilearn, nibabel; print('Neuro libs OK')"
```

> **System note**: Tested on Ubuntu 24.04, Intel Core i7-13700H, 16 GB RAM.  
> All PyTorch operations are CPU-only (`cpuonly` build) — no discrete GPU required.

---

## Data

Public fMRI data (OpenNeuro `ds003927`):  
https://openneuro.org/datasets/ds003927

Original analysis code:  
https://github.com/nmningmei/unconfeats

Locally acquire the data by running the following sequence in your IDE terminal, while the
relevant conda env is activated (navigate to your desired local directory first):

```bash
conda install -c conda-forge datalad git-annex
datalad clone https://github.com/OpenNeuroDatasets/ds003927.git
cd ds003927
datalad get derivatives/
```

### Expected per-subject layout

The pipeline expects the **flat ds003927 derivative layout** where each visibility state
has its own pre-separated NIfTI and CSV, all at the subject root:

```
data/raw/
└── sub-01/
    ├── wholebrain_conscious.nii.gz    # 4D BOLD – conscious trials only
    ├── wholebrain_conscious.csv       # Trial events – conscious trials
    ├── wholebrain_unconscious.nii.gz  # 4D BOLD – unconscious trials only
    ├── wholebrain_unconscious.csv     # Trial events – unconscious trials
    ├── wholebrain_glimpse.nii.gz      # 4D BOLD – glimpse trials (unused by default)
    ├── wholebrain_glimpse.csv         # Trial events – glimpse trials
    ├── mask.nii.gz                    # Single shared whole-brain binary mask
    ├── example_func.nii.gz            # Reference functional image
    └── anat/                          # Anatomical directory (not used by pipeline)
```

**Optional — ROI masks:** if you have FreeSurfer-derived ROI masks in native BOLD space,
place them in `sub-01/rois/` as `<roi_name>_mask.nii.gz`. Without them the pipeline
runs in whole-brain mode and ROI extraction is skipped with a warning.

The event CSVs must follow the PsychoPy schema:
`onset, duration, labels, targets, visibility, volume_interest, session, run, id, ...`

---

## Running the Pipeline

### Full pipeline (all 6 phases)

```bash
python scripts/run_pipeline.py \
    --config  config/config.yaml \
    --subjects sub-01 sub-02 sub-03 \
    --stimulus-dir /path/to/stimulus/images \
    --log-level INFO
```

### Single phase (re-run without reloading everything)

```bash
python scripts/run_pipeline.py --phase 3   # Phase 3 only (RSA + GWOT)
python scripts/run_pipeline.py --phase 6   # Phase 6 only (visualisations)
```

### Full CLI help

```bash
python scripts/run_pipeline.py --help
```

---

## Programmatic API

```python
from config.settings import Settings
from scripts.run_pipeline import POCPipeline

cfg      = Settings("config/config.yaml")
pipeline = POCPipeline(cfg)

# Load subjects and run all phases
pipeline.load_subjects(subject_ids=["sub-01", "sub-02"])
pipeline.phase1_extract_embeddings(stimulus_dir="path/to/images")
pipeline.phase2_build_rdms()
pipeline.phase3_inter_subject_rsa()
pipeline.phase4_cross_modality_alignment()
pipeline.phase5_structural_invariance()
pipeline.phase6_visualize()

# Access artefacts
print(pipeline.rdms)          # dict: subject_id → state → roi → RDM
print(pipeline.rsa_results)   # dict: roi → RSAResult
print(pipeline.gw_results)    # dict: roi → GWResult
```

---

## Key Design Decisions

| Concern | Approach |
|---|---|
| **Data layout** | Flat ds003927 derivative format; per-state NIfTI + CSV at subject root |
| **Config management** | Single `Settings` dataclass loaded from YAML; all paths/hyperparameters centralised |
| **Modularity** | Each phase is an independent method; all modules are importable standalone |
| **Statistical rigor** | 10,000-iteration permutation tests throughout; Bonferroni/FDR correction across 12 ROIs |
| **GWOT** | Entropic GW via POT library; Top-k matching rate; noise ceiling upper/lower bounds |
| **FCNN** | MobileNetV2 backbone (frozen conv layers); clear-noise (σ²=0) and chance-noise (σ²=300) |
| **Visualisation** | All figures saved to disk via non-interactive Agg backend |

---

## Reference

Mei N, Santana R, Soto D (2022).  
**Informative neural representations of unseen contents during higher-order processing in human brains and deep artificial networks.**  
*Nature Human Behaviour* 6, 720–731.  
https://doi.org/10.1038/s41562-021-01274-7
