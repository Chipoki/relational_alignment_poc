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
import gc
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

            # Load and immediately heal continuity/IDs
            events_df = self._behavioral.load(csv_path)
            events_df = self._enforce_continuity_and_ids(events_df)

            # since the volume & behavioral CSVs are of the same size and are presumably correspondingly pre-ordered
            volumes = list(range(len(events_df)))

            logger.info(
                "[%s] state='%s': Mapping %d CSV rows to %d BOLD volumes.",
                subject_id, state, len(events_df), len(volumes),
            )

            bold     = self._fmri.load_bold(nifti_path)
            patterns = self._fmri.extract_trial_patterns(bold, full_mask, volumes)

            # Run-wise z-scoring: normalize each voxel within each run independently
            run_ids = events_df.groupby(['session', 'run']).ngroup().values  # integer run index per trial

            patterns_normed = patterns.copy()
            for run_idx in np.unique(run_ids):
                mask_r = run_ids == run_idx
                run_data = patterns[mask_r]  # shape (n_trials_in_run, n_voxels)
                mean_r = run_data.mean(axis=0)
                std_r = run_data.std(axis=0)
                std_r[std_r < 1e-8] = 1.0  # avoid division by zero
                patterns_normed[mask_r] = (run_data - mean_r) / std_r

            patterns = patterns_normed

            # ── IMMEDIATE MEMORY FLUSH ──
            # Destroy the disk-backed object to free file handles and RAM
            del bold
            gc.collect()
            # ────────────────────────────

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

    @staticmethod
    def _enforce_continuity_and_ids(df: pd.DataFrame) -> pd.DataFrame:
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
                logger.info(msg)
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
