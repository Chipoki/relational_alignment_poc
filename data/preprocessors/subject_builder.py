"""data/preprocessors/subject_builder.py – Orchestrate per-subject data assembly.

Supports two modes, controlled by ``settings.data_source``:

derivatives mode
----------------
Flat ds003927 derivative layout::

    <derivatives_root>/<sub-XX>/
        wholebrain_conscious.nii.gz  / .csv
        wholebrain_unconscious.nii.gz / .csv
        mask.nii.gz
        func_masks/<roi>_mask.nii.gz   (optional)

replication mode
----------------
Author replication layout — matches aroma_decoding_pipeline_v11.py exactly::

    <replication_root>/MRI/<sub-XX>/
        func/session-<SS>/<sub-XX>_unfeat_run-<R>/
            outputs/func/ICAed_filtered/filtered.nii.gz
            outputs/func/mask.nii.gz           (per-run mask; first valid used)
        anat/ROI_BOLD/
            ctx-lh-<roi>_BOLD.nii.gz
            ctx-rh-<roi>_BOLD.nii.gz   (bilateral union masks)

    Event .tsv files are read from the ds003927 BIDS root::

    <ds003927_root>/<sub-XX>/ses-<SS>/func/
        <sub-XX>_ses-<SS>_task-recog_run-<R>_events.tsv
"""
from __future__ import annotations

import gc
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
    High-level orchestrator that produces a fully populated :class:`Subject`
    object, regardless of the active data_source mode.

    After all subjects are built, call ``register_rois_with_settings()`` so
    that downstream phases operate on the actual ROI set rather than the
    static config list.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings      = settings
        self._behavioral    = BehavioralLoader(settings)
        self._fmri          = FMRILoader(settings)
        self._roi_extractor = ROIExtractor(settings)
        self._prefix: str   = settings.nifti_prefix
        self._mask_fn: str  = settings.mask_filename
        self._states: list[str] = settings.visibility_states

        # Accumulate all ROI keys seen across every subject × state
        self._discovered_rois: set[str] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, subject_root: str | Path, subject_id: str) -> Subject:
        """Build and return a :class:`Subject` with all visibility states loaded."""
        if self._settings.data_source == "replication":
            return self._build_replication(subject_id)
        else:
            return self._build_derivatives(Path(subject_root), subject_id)

    def register_rois_with_settings(self) -> None:
        """
        Push all discovered ROI names into settings so that phases 3–6
        iterate over the real ROI set rather than the static config list.

        Call once after all subjects have been built.
        """
        self._settings.register_active_rois(sorted(self._discovered_rois))
        logger.info(
            "Registered %d active ROIs with settings: %s",
            len(self._settings.active_roi_names),
            self._settings.active_roi_names,
        )

    # ── Derivatives mode ──────────────────────────────────────────────────────

    def _build_derivatives(self, subject_root: Path, subject_id: str) -> Subject:
        """Build Subject from flat ds003927 derivatives layout."""
        subject = Subject(subject_id=subject_id)

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
            nifti_path, csv_path = self._resolve_deriv_state_paths(subject_root, state)

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
            events_df = self._enforce_continuity_and_ids(events_df)

            # In derivatives mode each CSV row maps 1:1 to a stacked BOLD volume
            volumes = list(range(len(events_df)))

            logger.info(
                "[%s] state='%s': Mapping %d CSV rows to %d BOLD volumes.",
                subject_id, state, len(events_df), len(volumes),
            )

            bold     = self._fmri.load_bold(nifti_path)
            patterns = self._fmri.extract_trial_patterns(bold, full_mask, volumes)

            # Run-wise z-scoring: normalise each voxel within each run
            patterns = self._runwise_zscore(patterns, events_df)

            del bold
            gc.collect()

            roi_dir     = self._resolve_deriv_roi_dir(subject_root)
            roi_patterns = self._extract_roi_patterns(
                patterns, full_mask, roi_dir, subject_id
            )

            labels        = self._behavioral.extract_binary_labels(events_df)
            label_strings = events_df[
                self._settings.data["stimuli_csv_col"]["category"]
            ].values
            stim_names    = self._behavioral.extract_stimulus_names(events_df)

            visibility_data = VisibilityData(
                state=state,
                bold_patterns=roi_patterns,
                labels=labels,
                label_strings=label_strings,
                stimulus_names=stim_names,
                events=events_df,
            )
            setattr(subject, state, visibility_data)

        logger.info("Built %s (derivatives mode)", subject)
        return subject

    # ── Replication mode ──────────────────────────────────────────────────────

    def _build_replication(self, subject_id: str) -> Subject:
        """
        Build Subject from author_replication layout, exactly replicating
        aroma_decoding_pipeline_v11.py's data loading strategy:

            • BOLD:       author_replication/MRI/<sub>/func/session-<SS>/
                              <sub>_unfeat_run-<R>/outputs/func/ICAed_filtered/filtered.nii.gz
            • Brain mask: first valid per-run mask.nii.gz
            • ROI masks:  ctx-lh-<roi>_BOLD.nii.gz + ctx-rh-<roi>_BOLD.nii.gz  (bilateral union)
            • Events:     ds003927/<sub>/ses-<SS>/func/*_events.tsv
        """
        subject = Subject(subject_id=subject_id)
        cfg     = self._settings

        # ── Discover run-level BOLD jobs ──────────────────────────────────────
        run_jobs: list[tuple[int, int, Path, Path]] = []  # (session, run, bold_path, tsv_path)
        first_bold_path: Path | None = None
        first_mask_path: Path | None = None

        n_sessions = cfg.n_sessions
        n_runs     = cfg.n_runs_per_session

        for ses in range(1, n_sessions + 1):
            ses_func_dir = cfg.ds003927_session_func_dir(subject_id, ses)
            if not ses_func_dir.exists():
                logger.debug("[%s] ds003927 session dir missing: %s", subject_id, ses_func_dir)
                continue

            # Discover event TSVs in sorted order (run index = position+1)
            tsv_files = sorted(
                f for f in ses_func_dir.iterdir()
                if f.name.endswith("_events.tsv") and "task-recog" in f.name
            )

            for run_idx, tsv_path in enumerate(tsv_files, start=1):
                run_dir   = cfg.replic_run_dir(subject_id, ses, run_idx)
                bold_path = cfg.replic_filtered_bold(run_dir)
                mask_path = cfg.replic_run_mask(run_dir)

                if not bold_path.exists():
                    logger.debug(
                        "[%s] BOLD not found, skipping ses=%d run=%d: %s",
                        subject_id, ses, run_idx, bold_path
                    )
                    continue

                if first_bold_path is None:
                    first_bold_path = bold_path
                if first_mask_path is None and mask_path.exists():
                    first_mask_path = mask_path

                run_jobs.append((ses, run_idx, bold_path, tsv_path))

        if not run_jobs:
            raise RuntimeError(
                f"[{subject_id}] No valid BOLD runs found under "
                f"{cfg.replic_func_dir(subject_id)}"
            )
        logger.info("[%s] Found %d run(s) to process.", subject_id, len(run_jobs))

        # ── Brain mask: first valid per-run func mask ──────────────────────────
        if first_mask_path is None:
            raise FileNotFoundError(
                f"[{subject_id}] No per-run mask.nii.gz found. "
                f"Check {cfg.replic_func_dir(subject_id)}."
            )
        full_mask = self._fmri.load_mask(first_mask_path)
        subject.mask_paths = [first_mask_path]

        # ── Bilateral ROI masks ────────────────────────────────────────────────
        roi_bold_dir = cfg.replic_roi_bold_dir(subject_id)
        bilateral_masks: dict[str, np.ndarray] = (
            self._load_bilateral_roi_masks(subject_id, roi_bold_dir, full_mask, first_bold_path)
        )

        # ── Per-state accumulation buffers ────────────────────────────────────
        # state → { roi_name → list[np.ndarray] (one per trial) }
        roi_buffers:  dict[str, dict[str, list[np.ndarray]]] = {
            s: {"wholebrain": []} for s in self._states
        }
        for s in self._states:
            for roi_name in bilateral_masks:
                roi_buffers[s][roi_name] = []

        meta_buffers: dict[str, list[dict]] = {s: [] for s in self._states}

        # ── Process each run ──────────────────────────────────────────────────
        for ses, run_idx, bold_path, tsv_path in run_jobs:
            logger.info("[%s] ses=%d run=%d → %s", subject_id, ses, run_idx, bold_path.name)

            # Load TSV and filter to rows with volume_interest == 1 (matching v11)
            try:
                tsv_df = self._behavioral.load_tsv(tsv_path)
            except Exception as exc:
                logger.warning("[%s] ses=%d run=%d TSV load failed: %s", subject_id, ses, run_idx, exc)
                continue

            tsv_df = tsv_df.dropna(subset=["targets"])
            vi_df  = tsv_df[tsv_df["volume_interest"] == 1].copy()
            if vi_df.empty:
                logger.debug("[%s] ses=%d run=%d: no volume_interest==1 rows.", subject_id, ses, run_idx)
                continue

            # Resolve TR indices (matches v11: "time_indices" column if present, else index)
            if "time_indices" in vi_df.columns:
                tr_indices = vi_df["time_indices"].astype(int).values
            else:
                tr_indices = vi_df.index.values.astype(int)

            # Extract whole-brain patterns for this run (detrend + z-score inside)
            run_patterns = self._fmri.extract_replication_run_patterns(
                bold_path   = bold_path,
                brain_mask  = full_mask,
                trial_vol_indices = tr_indices,
            )
            if run_patterns.shape[0] == 0:
                continue

            # Build per-trial metadata (matches v11 trial_id convention)
            for row_i, (_, row) in enumerate(vi_df.iterrows()):
                trial_num = float(row["trials"])
                vis_state = str(row["visibility"]).strip().lower()
                if vis_state not in self._states:
                    continue

                # Compute compound ID (ses * 10000 + run * 100 + trial)
                trial_id = ses * 10000 + run_idx * 100 + trial_num

                meta_buffers[vis_state].append({
                    "session":    ses,
                    "run":        run_idx,
                    "trials":     trial_num,
                    "id":         trial_id,
                    "targets":    str(row.get("targets", "")),
                    "labels":     str(row.get("labels", f"item_{int(trial_num)}")),
                    "visibility": vis_state,
                    "volume_interest": 1,
                    "onset":      float(row.get("onset", 0)),
                })

                wb_pattern = run_patterns[row_i]
                roi_buffers[vis_state]["wholebrain"].append(wb_pattern)

                for roi_name, roi_bitmask in bilateral_masks.items():
                    # roi_bitmask is already intersected with full_mask in load step
                    # Map brain-space voxel indices to pattern columns
                    roi_pattern = self._apply_precomputed_roi_mask(
                        wb_pattern, full_mask, roi_bitmask
                    )
                    roi_buffers[vis_state][roi_name].append(roi_pattern)

        # ── Assemble Subject visibility states ────────────────────────────────
        for state in self._states:
            meta_list = meta_buffers[state]
            if not meta_list:
                logger.warning("[%s] No trials found for state='%s'.", subject_id, state)
                continue

            events_df = pd.DataFrame(meta_list).reset_index(drop=True)

            # Compile roi_patterns dict
            roi_patterns: dict[str, np.ndarray] = {}
            for roi_name, trial_list in roi_buffers[state].items():
                if not trial_list:
                    continue
                roi_patterns[roi_name] = np.stack(trial_list, axis=0)
                self._discovered_rois.add(roi_name)

            labels        = (events_df["targets"] == "Living_Things").astype(int).values
            label_strings = events_df["targets"].values
            stim_names    = events_df["labels"].values.astype(str)

            visibility_data = VisibilityData(
                state=state,
                bold_patterns=roi_patterns,
                labels=labels,
                label_strings=label_strings,
                stimulus_names=stim_names,
                events=events_df,
            )
            setattr(subject, state, visibility_data)
            logger.info(
                "[%s] state='%s': %d trials, %d ROIs",
                subject_id, state, len(events_df), len(roi_patterns)
            )

        logger.info("Built %s (replication mode)", subject)
        return subject

    # ── Bilateral ROI mask loading (replication mode) ─────────────────────────

    def _load_bilateral_roi_masks(
        self,
        subject_id:      str,
        roi_bold_dir:    Path,
        full_mask:       np.ndarray,
        ref_bold_path:   Path,
    ) -> dict[str, np.ndarray]:
        """
        Discover all FreeSurfer ROI labels available in ROI_BOLD/, load the
        bilateral (lh + rh) union mask for each, resampled to BOLD space, and
        return a dict: roi_name → boolean (X, Y, Z) array intersected with full_mask.

        Mirrors get_bilateral_roi_mask() in aroma_decoding_pipeline_v11.py.
        """
        try:
            from nilearn.image import resample_to_img as _resample
        except ImportError:
            _resample = None

        import nibabel as nib
        import re as _re

        ref_img = nib.load(str(ref_bold_path))

        # Discover all unique ROI names from lh files
        lh_pattern = _re.compile(r"^ctx-lh-(.+)_BOLD\.nii(?:\.gz)?$")
        roi_names: set[str] = set()
        for f in roi_bold_dir.glob("ctx-lh-*_BOLD.nii*"):
            m = lh_pattern.match(f.name)
            if m:
                roi_names.add(m.group(1))

        bilateral: dict[str, np.ndarray] = {}
        for roi_name in sorted(roi_names):
            lh_path = roi_bold_dir / f"ctx-lh-{roi_name}_BOLD.nii.gz"
            rh_path = roi_bold_dir / f"ctx-rh-{roi_name}_BOLD.nii.gz"

            if not lh_path.exists() or not rh_path.exists():
                logger.warning(
                    "[%s] Missing lh or rh mask for ROI '%s' – skipping.",
                    subject_id, roi_name
                )
                continue

            lh_mask = self._load_and_resample_mask(lh_path, ref_img, _resample)
            rh_mask = self._load_and_resample_mask(rh_path, ref_img, _resample)
            combined = (lh_mask | rh_mask) & full_mask

            if combined.sum() == 0:
                logger.warning(
                    "[%s] ROI '%s' has 0 voxels after intersection with brain mask.",
                    subject_id, roi_name
                )
                continue

            bilateral[roi_name] = combined
            logger.debug("[%s] ROI '%s': %d voxels", subject_id, roi_name, combined.sum())

        logger.info(
            "[%s] Loaded %d bilateral ROI masks from %s",
            subject_id, len(bilateral), roi_bold_dir
        )
        return bilateral

    @staticmethod
    def _load_and_resample_mask(
        mask_path: Path,
        ref_img,
        resample_fn,
    ) -> np.ndarray:
        """Load mask; resample to ref_img space if shapes/affines differ."""
        import nibabel as nib
        import numpy as np

        mask_img = nib.load(str(mask_path))
        if (mask_img.shape == ref_img.shape[:3]
                and np.allclose(mask_img.affine, ref_img.affine)):
            return mask_img.get_fdata().astype(bool)

        if resample_fn is not None:
            resampled = resample_fn(
                source_img=mask_img,
                target_img=ref_img,
                interpolation="nearest",
            )
            return resampled.get_fdata().astype(bool)

        # Fallback: just load raw (may cause shape mismatch later)
        logger.warning(
            "Mask %s shape %s differs from ref %s; nilearn unavailable for resampling.",
            mask_path, mask_img.shape, ref_img.shape
        )
        return mask_img.get_fdata().astype(bool)

    @staticmethod
    def _apply_precomputed_roi_mask(
        wb_pattern: np.ndarray,  # (n_brain_voxels,) – whole-brain flat pattern
        full_mask:  np.ndarray,  # (X, Y, Z) bool
        roi_mask:   np.ndarray,  # (X, Y, Z) bool (already intersected with full_mask)
    ) -> np.ndarray:
        """
        Subset a flat whole-brain pattern to ROI voxels.

        Parameters
        ----------
        wb_pattern : 1-D array of length = full_mask.sum()
        full_mask  : the mask used to create wb_pattern
        roi_mask   : ROI mask already intersected with full_mask
        """
        full_flat = np.flatnonzero(full_mask)
        roi_flat  = np.flatnonzero(roi_mask)
        index_map = {v: k for k, v in enumerate(full_flat)}
        col_idx   = np.array([index_map[v] for v in roi_flat if v in index_map])
        if col_idx.size == 0:
            return np.zeros(0, dtype=wb_pattern.dtype)
        return wb_pattern[col_idx]

    # ── Derivatives-mode helpers ──────────────────────────────────────────────

    def _resolve_deriv_state_paths(
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

    def _resolve_deriv_roi_dir(self, subject_root: Path) -> Path:
        rois_subdir = subject_root / "func_masks"
        if rois_subdir.exists():
            return rois_subdir
        return subject_root

    def _extract_roi_patterns(
        self,
        patterns:  np.ndarray,
        full_mask: np.ndarray,
        roi_dir:   Path,
        subject_id: str,
    ) -> dict[str, np.ndarray]:
        """Extract all ROI sub-patterns from whole-brain patterns (derivatives mode)."""
        available_masks = [
            f.name.replace("_mask.nii.gz", "").replace("_mask.nii", "")
            for f in roi_dir.glob("*_mask.nii*")
        ]
        self._roi_extractor._roi_names = available_masks

        roi_patterns = self._roi_extractor.extract_all_rois(patterns, full_mask, roi_dir)

        # Whole-brain is always included
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

        return roi_patterns

    @staticmethod
    def _runwise_zscore(patterns: np.ndarray, events_df: pd.DataFrame) -> np.ndarray:
        """Normalise each voxel within each run independently (derivatives mode)."""
        if "session" not in events_df.columns or "run" not in events_df.columns:
            return patterns
        run_ids       = events_df.groupby(["session", "run"]).ngroup().values
        patterns_norm = patterns.copy()
        for run_idx in np.unique(run_ids):
            mask_r    = run_ids == run_idx
            run_data  = patterns[mask_r]
            mean_r    = run_data.mean(axis=0)
            std_r     = run_data.std(axis=0)
            std_r[std_r < 1e-8] = 1.0
            patterns_norm[mask_r] = (run_data - mean_r) / std_r
        return patterns_norm

    @staticmethod
    def _enforce_continuity_and_ids(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix broken run-number continuities within each session and
        recompute the compound trial ID (session*10000 + run*100 + trials).
        """
        df_out = df.copy()
        sessions = df_out["session"].drop_duplicates().tolist()

        for sess in sessions:
            mask_sess    = df_out["session"] == sess
            ordered_runs = df_out.loc[mask_sess, "run"].drop_duplicates().tolist()
            if not ordered_runs:
                continue
            first_run     = ordered_runs[0]
            expected_runs = list(range(int(first_run), int(first_run) + len(ordered_runs)))
            if ordered_runs != expected_runs:
                run_mapping = {old: new for old, new in zip(ordered_runs, expected_runs)}
                logger.info(
                    "  -> [Session %s] Broken run continuity. Mapping: %s", sess, run_mapping
                )
                df_out.loc[mask_sess, "run"] = (
                    df_out.loc[mask_sess, "run"].map(run_mapping).fillna(df_out["run"])
                )

        df_out["id"] = (
            df_out["session"] * 10000
            + df_out["run"] * 100
            + df_out["trials"]
        )
        return df_out
