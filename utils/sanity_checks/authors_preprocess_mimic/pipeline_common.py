#!/usr/bin/env python3
"""
pipeline_common.py
==================
Shared helpers for the Soto et al. (2022) fMRI preprocessing replication.

Directory conventions
---------------------
All outputs go under:
    <root>/author_replication/
        MRI/
            <subject>/
                anat/           ← symlinks to T1w; FreeSurfer SUBJECTS_DIR
                func/
                    session-<SS>/
                        <subject>_unfeat_run-<R>/
                            <subject>_unfeat_run-<R>_bold.nii.gz  ← symlink
                            outputs/
                                func/
                                    prefiltered_func.nii.gz
                                    mask.nii.gz
                                    example_func.nii.gz
                                    mean_func.nii.gz
                                    MC/
                                        MCflirt.par
                                    ICA_AROMA/
                                        denoised_func_data_nonaggr.nii.gz
                                    ICAed_filtered/
                                        filtered.nii.gz
                                reg/          ← only for session-<first>, run-1
                                    example_func2highres.mat
                                    highres2standard_warp.nii.gz
                                    ...
                anat/
                    ROIs_anat/
                    ROI_BOLD/
                    ROI_standard/
        work/                   ← nipype working dirs (cleaned up after each step)

Notes on run numbering
----------------------
BIDS uses run-1 … run-9 (bare integers, no zero-padding).
sub-01 ses-04 has runs 61 & 62 (split run; authors likely concatenated or treated
individually — we include all BIDS runs verbatim).
The author working-tree also uses bare integers: _unfeat_run-1, _unfeat_run-2, …
"""
from __future__ import annotations

import os
import re
import json
import argparse
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# ── top-level paths ──────────────────────────────────────────────────────────
DEFAULT_ROOT = Path(
    '/home/tomerd/Documents/projects/MSc/lab/thesis_practice/'
    'relational_alignment/soto_data'
)

# ── ROI label → aparc+aseg integer index (hard-coded, replaces FreesurferLTU.csv)
# These exactly reproduce the author's substring-matching logic:
#   name in label_name AND 'ctx' in label_name
#   AND str(idx)[1] == '0' AND idx < 3000 AND 'caudal' not in label_name
# The pars* labels are later combined into a bilateral ventrolateralPFC mask.
ROI_LABELS = {
    'ctx-lh-fusiform':           1007,
    'ctx-rh-fusiform':           2007,
    'ctx-lh-inferiorparietal':   1008,
    'ctx-rh-inferiorparietal':   2008,
    'ctx-lh-inferiortemporal':   1009,
    'ctx-rh-inferiortemporal':   2009,
    'ctx-lh-lateraloccipital':   1011,
    'ctx-rh-lateraloccipital':   2011,
    'ctx-lh-lingual':            1013,
    'ctx-rh-lingual':            2013,
    'ctx-lh-parahippocampal':    1016,
    'ctx-rh-parahippocampal':    2016,
    # pars* → combined into inferior frontal (ventrolateralPFC) later
    'ctx-lh-parsopercularis':    1018,
    'ctx-rh-parsopercularis':    2018,
    'ctx-lh-parsorbitalis':      1019,
    'ctx-rh-parsorbitalis':      2019,
    'ctx-lh-parstriangularis':   1020,
    'ctx-rh-parstriangularis':   2020,
    'ctx-lh-pericalcarine':      1021,
    'ctx-rh-pericalcarine':      2021,
    'ctx-lh-precuneus':          1025,
    'ctx-rh-precuneus':          2025,
    # rostralmiddlefrontal = middlefrontal in the paper (same region, FS name)
    'ctx-lh-rostralmiddlefrontal': 1027,
    'ctx-rh-rostralmiddlefrontal': 2027,
    'ctx-lh-superiorfrontal':    1028,
    'ctx-rh-superiorfrontal':    2028,
    'ctx-lh-superiorparietal':   1029,
    'ctx-rh-superiorparietal':   2029,
}

# ── regex helpers ─────────────────────────────────────────────────────────────
_SESSION_RE  = re.compile(r'^session-(\d+)$')
_BIDS_SES_RE = re.compile(r'^ses-(\d+)$')
_BIDS_RUN_RE = re.compile(r'run-(\d+)')
_RUN_DIR_RE  = re.compile(r'^(?P<subject>sub-\d+)_unfeat_run-(?P<run>\d+)$')


@dataclass(frozen=True)
class RunSpec:
    subject: str
    ses: str   # zero-padded, e.g. "02"
    run: str   # bare integer string, e.g. "1"  (matches BIDS and author tree)
    bold: Path


# ── normalisation helpers ─────────────────────────────────────────────────────

def _norm_ses(x: str | int) -> str:
    """Return zero-padded 2-digit session id, e.g. '02'."""
    m = re.search(r'(\d+)', str(x))
    if not m:
        raise ValueError(f'Cannot parse session id from {x!r}')
    return f'{int(m.group(1)):02d}'


def _norm_run(x: str | int) -> str:
    """Return bare integer run id, e.g. '1' or '61' (preserves multi-digit)."""
    m = re.search(r'(\d+)', str(x))
    if not m:
        raise ValueError(f'Cannot parse run id from {x!r}')
    return str(int(m.group(1)))


def _abs(p: Path | str) -> str:
    return str(Path(p).resolve())


# ── path layout ───────────────────────────────────────────────────────────────

def ds_root(root: Path) -> Path:
    return root / 'ds003927'


def replica_root(root: Path) -> Path:
    return root / 'author_replication'


def subject_bids_dir(root: Path, subject: str) -> Path:
    return ds_root(root) / subject


def subject_deriv_dir(root: Path, subject: str) -> Path:
    return ds_root(root) / 'derivatives' / subject


def subject_replica_dir(root: Path, subject: str) -> Path:
    return replica_root(root) / 'MRI' / subject


def subject_anat_dir(root: Path, subject: str) -> Path:
    """Author-style anat dir: author_replication/MRI/<sub>/anat/"""
    return subject_replica_dir(root, subject) / 'anat'


def subject_func_dir(root: Path, subject: str) -> Path:
    """Author-style func dir: author_replication/MRI/<sub>/func/"""
    return subject_replica_dir(root, subject) / 'func'


def work_dir(root: Path) -> Path:
    return replica_root(root) / 'work'


# ── finding anatomical files ──────────────────────────────────────────────────

def find_anat_head(root: Path, subject: str) -> Path:
    """
    Find the full T1w (skull-on) NIfTI.
    Checks in order: BIDS anat, derivatives anat.
    """
    for base in [
        subject_bids_dir(root, subject) / 'anat',
        subject_deriv_dir(root, subject) / 'anat',
        subject_anat_dir(root, subject),  # author tree symlink
    ]:
        if not base.exists():
            continue
        for pat in ['*T1w.nii.gz', '*t1*.nii*', 'T1.nii.gz']:
            for p in sorted(base.glob(pat)):
                if 'brain' not in p.name.lower():
                    return p.resolve()
    raise FileNotFoundError(f'No skull-on T1w found for {subject}')


def find_anat_brain(root: Path, subject: str) -> Path:
    """
    Find the BET brain-extracted T1w.
    Prefers derivatives (already computed for all subjects in ds003927).
    """
    for base in [
        subject_deriv_dir(root, subject) / 'anat',
        subject_anat_dir(root, subject),
    ]:
        if not base.exists():
            continue
        for p in sorted(base.glob('*brain*.nii*')):
            return p.resolve()
    raise FileNotFoundError(f'No brain-extracted T1w found for {subject}')


# ── author working-tree paths ─────────────────────────────────────────────────

def author_run_dir(root: Path, subject: str, ses: str, run: str) -> Path:
    """
    Returns the per-run working directory, creating it if needed.
    Does NOT raise if missing — callers decide when to error.
    """
    ses = _norm_ses(ses)
    run = _norm_run(run)
    return subject_func_dir(root, subject) / f'session-{ses}' / f'{subject}_unfeat_run-{run}'


def author_bold_file(root: Path, subject: str, ses: str, run: str) -> Path:
    run_dir = author_run_dir(root, subject, ses, run)
    run_id  = _norm_run(run)
    expected = run_dir / f'{subject}_unfeat_run-{run_id}_bold.nii.gz'
    if expected.exists():
        return expected.resolve()
    # fallback: any bold nifti
    for p in sorted(run_dir.glob('*_bold.nii*')):
        return p.resolve()
    raise FileNotFoundError(f'No BOLD nifti found in {run_dir}')


def author_output_dir(root: Path, subject: str, ses: str, run: str) -> Path:
    return author_run_dir(root, subject, ses, run) / 'outputs'


def author_func_output_dir(root: Path, subject: str, ses: str, run: str) -> Path:
    return author_output_dir(root, subject, ses, run) / 'func'


def author_reg_dir(root: Path, subject: str, ses: str, run: str = '1') -> Path:
    """Registration outputs (only produced for the very first run)."""
    return author_output_dir(root, subject, ses, run) / 'reg'


# ── session / run discovery ───────────────────────────────────────────────────

def first_session(root: Path, subject: str) -> str:
    """
    Return the lowest session id (zero-padded) for this subject.
    Searches the author working tree first, falls back to BIDS.
    """
    func_dir = subject_func_dir(root, subject)
    sessions = []

    if func_dir.exists():
        for p in sorted(func_dir.glob('session-*')):
            m = _SESSION_RE.fullmatch(p.name)
            if m:
                sessions.append(int(m.group(1)))

    if not sessions:
        bids_dir = subject_bids_dir(root, subject)
        for p in sorted(bids_dir.glob('ses-*')):
            m = _BIDS_SES_RE.fullmatch(p.name)
            if m:
                sessions.append(int(m.group(1)))

    if not sessions:
        raise RuntimeError(f'No sessions found for {subject}')
    return _norm_ses(min(sessions))


def list_runs(root: Path, subject: str) -> List[RunSpec]:
    """
    Return all (session, run) pairs, sorted by (session, run).
    Reads from the author working tree if it exists; otherwise from BIDS.
    Excludes runs whose directory name contains 'wrong' (author convention).
    """
    func_dir = subject_func_dir(root, subject)
    runs: List[RunSpec] = []

    # ── author working tree ───────────────────────────────────────────────────
    if func_dir.exists():
        for ses_dir in sorted(func_dir.glob('session-*')):
            sm = _SESSION_RE.fullmatch(ses_dir.name)
            if not sm:
                continue
            ses = _norm_ses(sm.group(1))
            for run_dir in sorted(p for p in ses_dir.iterdir() if p.is_dir()):
                if 'wrong' in run_dir.name:
                    continue
                rm = _RUN_DIR_RE.fullmatch(run_dir.name)
                if not rm:
                    continue
                run = _norm_run(rm.group('run'))
                bold = run_dir / f'{subject}_unfeat_run-{run}_bold.nii.gz'
                if not bold.exists():
                    cands = sorted(run_dir.glob('*_bold.nii*'))
                    if not cands:
                        continue
                    bold = cands[0]
                runs.append(RunSpec(subject=subject, ses=ses, run=run, bold=bold.resolve()))
        if runs:
            return sorted(runs, key=lambda x: (int(x.ses), int(x.run)))

    # ── BIDS fallback ─────────────────────────────────────────────────────────
    bids_dir = subject_bids_dir(root, subject)
    for ses_dir in sorted(bids_dir.glob('ses-*')):
        sm = _BIDS_SES_RE.fullmatch(ses_dir.name)
        if not sm:
            continue
        ses = _norm_ses(sm.group(1))
        func_bids = ses_dir / 'func'
        if not func_bids.exists():
            continue
        for bold in sorted(func_bids.glob(f'{subject}_ses-*_*run-*_bold.nii.gz')):
            rm = _BIDS_RUN_RE.search(bold.name)
            if not rm:
                continue
            run = _norm_run(rm.group(1))
            runs.append(RunSpec(subject=subject, ses=ses, run=run, bold=bold.resolve()))

    if not runs:
        raise RuntimeError(f'No runs found for {subject}')
    return sorted(runs, key=lambda x: (int(x.ses), int(x.run)))


# ── FSL standard files ────────────────────────────────────────────────────────

def standard_files(root: Path) -> Tuple[str, str, str]:
    """
    Return (brain, head, mask) paths for the MNI152 2mm standard brain.
    Checks: local data/standard_brain/, then $FSLDIR/data/standard/.
    """
    candidates = [
        root / 'data' / 'standard_brain',
        DEFAULT_ROOT / 'data' / 'standard_brain',
    ]
    fsldir = os.environ.get('FSLDIR')
    if fsldir:
        candidates.append(Path(fsldir) / 'data' / 'standard')

    for std in candidates:
        brain = std / 'MNI152_T1_2mm_brain.nii.gz'
        head  = std / 'MNI152_T1_2mm.nii.gz'
        mask  = std / 'MNI152_T1_2mm_brain_mask_dil.nii.gz'
        if brain.exists() and head.exists() and mask.exists():
            return (_abs(brain), _abs(head), _abs(mask))

    raise FileNotFoundError(
        'MNI152 standard files not found. Checked: '
        + ', '.join(str(c) for c in candidates)
        + '\nEnsure $FSLDIR is set or place files in data/standard_brain/.'
    )


# ── utils.py loader ───────────────────────────────────────────────────────────

def find_project_utils(root: Path) -> Path:
    """Locate the authors' utils.py."""
    candidates = [
        root / 'utils.py',
        Path(__file__).resolve().parent / 'utils.py',
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        'utils.py not found. Place it in the project root alongside these scripts.'
    )


def load_utils(root: Path):
    """Dynamically import the authors' utils module."""
    utils_path = find_project_utils(root)
    spec = importlib.util.spec_from_file_location('author_utils', str(utils_path))
    mod  = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ── filesystem helpers ────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def symlink_force(src: Path, dst: Path) -> None:
    """Create (or replace) a symlink at dst pointing to src."""
    ensure_dir(dst.parent)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


# ── CLI factory ───────────────────────────────────────────────────────────────

def make_parser(desc: str) -> argparse.ArgumentParser:
    """
    Standard argument parser shared by all pipeline scripts.

    --workers default: 4 parallel processes.
    With 20 cores and 16 GB RAM, 4 is conservative; each FSL job can use
    several GB. Raise to 8 if memory permits (monitor with `htop`).
    Use --workers 1 for sequential (safest for debugging).
    """
    p = argparse.ArgumentParser(description=desc)
    p.add_argument('--root',    default=str(DEFAULT_ROOT),
                   help='Project root (parent of ds003927/ and author_replication/)')
    p.add_argument('--subject', default='sub-01',
                   help='Subject label, e.g. sub-01')
    p.add_argument('--workers', type=int, default=4,
                   help='Parallel workers (default 4; each FSL job ~2–4 GB RAM)')
    p.add_argument('--overwrite', action='store_true',
                   help='Re-run even if outputs already exist')
    p.add_argument('--dry-run', action='store_true',
                   help='Print what would be done without running anything')
    return p


# ── manifest ─────────────────────────────────────────────────────────────────

def write_manifest(root: Path, subject: str) -> Path:
    runs = list_runs(root, subject)
    manifest = {
        'subject':       subject,
        'root':          str(root),
        'first_session': first_session(root, subject),
        'runs': [
            {'ses': r.ses, 'run': r.run, 'bold': str(r.bold)}
            for r in runs
        ],
    }
    out = ensure_dir(replica_root(root) / 'manifests') / f'{subject}_manifest.json'
    out.write_text(json.dumps(manifest, indent=2))
    return out
