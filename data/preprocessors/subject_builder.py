"""data/preprocessors/subject_builder.py – Orchestrate per-subject data assembly.

Adapted for the flat ds003927 derivative layout::

    <subject_root>/
        wholebrain_conscious.nii.gz
        wholebrain_unconscious.nii.gz
        wholebrain_conscious.csv
        wholebrain_unconscious.csv
        mask.nii.gz
        example_func.nii.gz
        anat/
        func_masks/          ← optional: <roi_name>_mask.nii.gz per region
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

    After all subjects are built, call ``register_rois_with_settings()``
    so that downstream phases operate on the exact ROI set present in the
    data rather than the static config list.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings       = settings
        self._behavioral     = BehavioralLoader(settings)
        self._fmri           = FMRILoader(settings)
        self._roi_extractor  = ROIExtractor(settings)
        self._prefix: str    = settings.nifti_prefix
        self._mask_fn: str   = settings.mask_filename
        self._states: list[str] = settings.visibility_states

        # Accumulate all ROI keys seen across every subject × state
        self._discovered_rois: set[str] = set()

    # ── Public API ───────────────────────────────────────────────────────────

    def build(self, subject_root: str | Path, subject_id: str) -> Subject:
        """Build and return a :class:`Subject` with both visibility states populated."""
        subject_root = Path(subject_root)
        subject      = Subject(subject_id=subject_id)

        mask_path = subject_root / self._mask_fn
        if not mask_path.exists():
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
                logger.warning(
                    "[%s] NIfTI not found for state='%s' – skipping", subject_id, state
                )
                continue
            if csv_path is None:
                logger.warning(
                    "[%s] CSV not found for state='%s' – skipping", subject_id, state
                )
                continue

            logger.info(
                "[%s] Loading state='%s'  nifti=%s", subject_id, state, nifti_path.name
            )
            subject.fmri_paths.append(nifti_path)

            events_df = self._behavioral.load(csv_path)

            vol_col = self._settings.data["stimuli_csv_col"]["volume"]
            volumes = events_df[vol_col].dropna().astype(int).tolist()
            volumes = list(range(len(events_df)))

            logger.info(
                "[%s] state='%s': Mapping %d CSV rows to %d BOLD volumes.",
                subject_id, state, len(events_df), len(volumes),
            )

            bold     = self._fmri.load_bold(nifti_path)
            patterns = self._fmri.extract_trial_patterns(bold, full_mask, volumes)
            patterns = self._fmri.zscore_patterns(patterns)

            roi_dir = self._resolve_roi_dir(subject_root)

            # Discover what mask files actually exist for this subject
            available_masks = [
                f.name.replace("_mask.nii.gz", "").replace("_mask.nii", "")
                for f in roi_dir.glob("*_mask.nii*")
            ]
            self._roi_extractor._roi_names = available_masks

            roi_patterns = self._roi_extractor.extract_all_rois(
                patterns, full_mask, roi_dir
            )

            # ── Whole-brain is always included ────────────────────────────
            roi_patterns["wholebrain"] = patterns
            self._discovered_rois.add("wholebrain")

            if not available_masks:
                logger.info(
                    "[%s] No ROI mask files found in %s – running whole-brain only.",
                    subject_id, roi_dir,
                )
            else:
                logger.info(
                    "[%s] Extracted %d region ROIs + wholebrain: %s",
                    subject_id, len(roi_patterns) - 1, sorted(available_masks),
                )
                self._discovered_rois.update(roi_patterns.keys())

            labels        = self._behavioral.extract_binary_labels(events_df)
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

    def register_rois_with_settings(self) -> None:
        """
        Push all discovered ROI names into settings so that phases 3–6
        iterate over the real ROI set rather than the static config list.

        Call this once after all subjects have been built (i.e. from
        POCPipeline.load_subjects()).
        """
        self._settings.register_active_rois(sorted(self._discovered_rois))
        logger.info(
            "Registered %d active ROIs with settings: %s",
            len(self._settings.active_roi_names),
            self._settings.active_roi_names,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _resolve_state_paths(
        self,
        subject_root: Path,
        state: str,
    ) -> tuple[Path | None, Path | None]:
        prefix = self._prefix

        nifti_path: Path | None = None
        for suffix in (".nii.gz", ".nii"):
            candidate = subject_root / f"{prefix}_{state}{suffix}"
            if candidate.exists():
                nifti_path = candidate
                break
        if nifti_path is None:
            hits       = list(subject_root.glob(f"*{state}*.nii*"))
            nifti_path = hits[0] if hits else None

        csv_path: Path | None = None
        candidate_csv = subject_root / f"{prefix}_{state}.csv"
        if candidate_csv.exists():
            csv_path = candidate_csv
        else:
            hits     = list(subject_root.glob(f"*{state}*.csv"))
            csv_path = hits[0] if hits else None

        return nifti_path, csv_path

    def _resolve_roi_dir(self, subject_root: Path) -> Path:
        rois_subdir = subject_root / "func_masks"
        if rois_subdir.exists():
            return rois_subdir
        return subject_root
