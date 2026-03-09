"""data/preprocessors/subject_builder.py – Orchestrate per-subject data assembly.

Adapted for the flat ds003927 derivative layout::

    <subject_root>/
        wholebrain_conscious.nii.gz    – 4D BOLD (conscious trials already separated)
        wholebrain_unconscious.nii.gz  – 4D BOLD (unconscious trials)
        wholebrain_glimpse.nii.gz      – 4D BOLD (glimpse trials, unused by default)
        wholebrain_conscious.csv       – trial events for conscious trials
        wholebrain_unconscious.csv     – trial events for unconscious trials
        wholebrain_glimpse.csv         – trial events for glimpse trials
        mask.nii.gz                    – single shared whole-brain binary mask
        example_func.nii.gz            – reference functional image
        anat/                          – anatomical directory (not used here)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import Settings
from data.loaders.behavioral_loader import BehavioralLoader
from data.loaders.fmri_loader import FMRILoader
from data.loaders.subject import Subject, VisibilityData
from data.preprocessors.roi_extractor import ROIExtractor

logger = logging.getLogger(__name__)


class SubjectBuilder:
    """
    High-level orchestrator that, given a subject's root directory,
    produces a fully populated :class:`Subject` object.

    The subject directory is expected to follow the ds003927 flat layout
    where each visibility state has its own pre-separated NIfTI and CSV:

        <subject_root>/
            wholebrain_conscious.nii.gz
            wholebrain_conscious.csv
            wholebrain_unconscious.nii.gz
            wholebrain_unconscious.csv
            mask.nii.gz
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._behavioral = BehavioralLoader(settings)
        self._fmri = FMRILoader(settings)
        self._roi_extractor = ROIExtractor(settings)
        self._prefix: str = settings.nifti_prefix          # "wholebrain"
        self._mask_fn: str = settings.mask_filename        # "mask.nii.gz"
        self._states: list[str] = settings.visibility_states  # ["conscious","unconscious"]

    # ── Public API ───────────────────────────────────────────────────────────

    def build(self, subject_root: str | Path, subject_id: str) -> Subject:
        """
        Build and return a :class:`Subject` with both visibility states populated.

        Expects the flat ds003927 derivative layout (see module docstring).
        """
        subject_root = Path(subject_root)
        subject = Subject(subject_id=subject_id)

        # Single shared mask for all states
        mask_path = subject_root / self._mask_fn
        if not mask_path.exists():
            # Fallback: search for any mask file
            candidates = list(subject_root.glob("mask*.nii*"))
            if not candidates:
                raise FileNotFoundError(
                    f"No mask file '{self._mask_fn}' found in {subject_root}"
                )
            mask_path = candidates[0]
            logger.warning("Using fallback mask: %s", mask_path)

        full_mask = self._fmri.load_mask(mask_path)
        subject.mask_paths = [mask_path]

        for state in self._states:
            nifti_path, csv_path = self._resolve_state_paths(subject_root, state)

            if nifti_path is None:
                logger.warning("NIfTI not found for state='%s' in %s – skipping", state, subject_root)
                continue
            if csv_path is None:
                logger.warning("CSV not found for state='%s' in %s – skipping", state, subject_root)
                continue

            logger.info("[%s] Loading state='%s'  nifti=%s", subject_id, state, nifti_path.name)

            # Track NIfTI paths on the Subject object
            subject.fmri_paths.append(nifti_path)

            # Load events
            events_df = self._behavioral.load(csv_path)

            # Extract volume indices from CSV (volume_interest column)
            vol_col = self._settings.data["stimuli_csv_col"]["volume"]
            volumes = events_df[vol_col].dropna().astype(int).tolist()
            # Convert 1-based → 0-based if necessary
            if volumes and min(volumes) >= 1:
                volumes = [v - 1 for v in volumes]

            # Load BOLD and extract whole-brain patterns
            bold = self._fmri.load_bold(nifti_path)
            patterns = self._fmri.extract_trial_patterns(bold, full_mask, volumes)
            patterns = self._fmri.zscore_patterns(patterns)  # (n_trials, n_voxels)

            # Extract per-ROI patterns from whole-brain patterns.
            # ROI masks are looked up inside subject_root (flat layout has no rois/ subdir)
            # or a sibling rois/ directory; ROIExtractor handles both gracefully.
            roi_dir = self._resolve_roi_dir(subject_root)
            roi_patterns = self._roi_extractor.extract_all_rois(
                patterns, full_mask, roi_dir
            )

            # Build labels and stimulus names from behavioural CSV
            labels = self._behavioral.extract_binary_labels(events_df)
            label_strings = events_df[
                self._settings.data["stimuli_csv_col"]["category"]
            ].values
            stim_names = self._behavioral.extract_stimulus_names(events_df)

            visibility_data = VisibilityData(
                state=state,
                bold_patterns=roi_patterns,
                labels=labels,
                label_strings=label_strings,
                stimulus_names=stim_names,
                events=events_df,
            )
            setattr(subject, state, visibility_data)

        logger.info("Built %s", subject)
        return subject

    # ── Private helpers ──────────────────────────────────────────────────────

    def _resolve_state_paths(
        self,
        subject_root: Path,
        state: str,
    ) -> tuple[Path | None, Path | None]:
        """
        Resolve NIfTI and CSV paths for a given visibility state.

        Tries the canonical naming convention first::
            wholebrain_<state>.nii.gz  /  wholebrain_<state>.csv

        Falls back to glob-based matching if that fails.
        """
        prefix = self._prefix  # "wholebrain"

        # ── NIfTI ────────────────────────────────────────────────────────────
        nifti_path: Path | None = None
        for suffix in (".nii.gz", ".nii"):
            candidate = subject_root / f"{prefix}_{state}{suffix}"
            if candidate.exists():
                nifti_path = candidate
                break
        if nifti_path is None:
            # Glob fallback
            hits = list(subject_root.glob(f"*{state}*.nii*"))
            nifti_path = hits[0] if hits else None

        # ── CSV ──────────────────────────────────────────────────────────────
        csv_path: Path | None = None
        candidate_csv = subject_root / f"{prefix}_{state}.csv"
        if candidate_csv.exists():
            csv_path = candidate_csv
        else:
            hits = list(subject_root.glob(f"*{state}*.csv"))
            csv_path = hits[0] if hits else None

        return nifti_path, csv_path

    def _resolve_roi_dir(self, subject_root: Path) -> Path:
        """
        Return the directory where ROI masks are stored.

        Preference order:
        1. <subject_root>/rois/
        2. <subject_root>/   (flat layout — ROI masks alongside data files)
        3. <data_root>/rois/ (dataset-level shared masks)
        """
        rois_subdir = subject_root / "rois"
        if rois_subdir.exists():
            return rois_subdir
        # In the flat ds003927 layout there are no individual ROI masks;
        # ROI extraction will be skipped gracefully by ROIExtractor (warns per ROI).
        # Return subject_root so the extractor at least attempts a search there.
        return subject_root
