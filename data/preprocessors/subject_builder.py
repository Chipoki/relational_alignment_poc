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
        subject = Subject(subject_id=subject_id)
        cfg = self._settings

        # ── Discover run jobs ──────────────────────────────────────────────
        #
        # The pairing problem
        # -------------------
        # We need to match TSV event files (in ds003927) to BOLD run folders
        # (in author_replication).  In an ideal world the run number embedded
        # in the TSV filename (e.g. _run-6_events.tsv) equals the run-folder
        # suffix (e.g. sub-01_unfeat_run-6/).
        #
        # In practice two classes of mismatch exist:
        #
        #   Class A – BOLD folder absent entirely (defective / missing run):
        #       sub-03 ses-01 run-4: TSV says run-4, BOLD folder run-4 exists
        #       but filtered.nii.gz inside it is missing.  → Correct action:
        #       skip that run.  This is handled by the bold_path.exists() guard.
        #
        #   Class B – BOLD folder numbered differently from the TSV:
        #       sub-01 ses-04 runs 61 & 62: TSV says run-61 and run-62, but
        #       the BOLD folders are named unfeat_run-6 and unfeat_run-7.
        #       There is NO folder called unfeat_run-61.
        #
        # Strategy
        # --------
        # 1. Sort TSV files by their embedded run number (temporal order).
        # 2. Discover all BOLD run folders that actually have a filtered.nii.gz
        #    and sort them by their folder run number (also temporal order).
        # 3. Try a direct number match first (TSV run N → folder run N).
        #    If the folder exists and has a BOLD → use it.
        # 4. If no direct match is found, fall back to POSITIONAL pairing:
        #    sort both remaining unmatched lists by number and pair them 1:1.
        #    This recovers Class-B mismatches (sub-01 ses-04 runs 61/62 → 6/7).
        # 5. Any TSV that ends up with no BOLD counterpart is logged and
        #    skipped (Class A).
        #
        # The positional fallback is safe because both sides are sorted by
        # run number, which reflects chronological acquisition order, and
        # the FEAT preprocessing pipeline preserves that order in the folder
        # names regardless of what number is written on them.
        import re as _re

        run_jobs: list[tuple[int, int, Path, Path]] = []
        first_bold_path: Path | None = None
        first_mask_path: Path | None = None

        for ses in range(cfg.session_start, cfg.session_start + cfg.n_sessions):
            ses_func_dir = cfg.ds003927_session_func_dir(subject_id, ses)
            if not ses_func_dir.exists():
                continue

            # --- Collect and sort TSV files by embedded run number ---
            tsv_entries: list[tuple[int, Path]] = []
            for f in ses_func_dir.iterdir():
                if not (f.name.endswith("_events.tsv") and "task-recog" in f.name):
                    continue
                m = _re.search(r"_run-(\d+)_events\.tsv$", f.name)
                if not m:
                    logger.warning(
                        "[%s] ses=%d: cannot parse run# from %s – skipping.",
                        subject_id, ses, f.name,
                    )
                    continue
                tsv_entries.append((int(m.group(1)), f))
            tsv_entries.sort(key=lambda x: x[0])

            if not tsv_entries:
                continue

            # --- Collect BOLD run folders that contain a valid filtered.nii.gz ---
            ses_bold_dir = cfg.replic_func_dir(subject_id) / f"session-{ses:02d}"
            bold_entries: list[tuple[int, Path, Path]] = []  # (folder_run_num, bold, mask)
            if ses_bold_dir.exists():
                folder_pat = _re.compile(rf"^{_re.escape(subject_id)}_unfeat_run-(\d+)$")
                for run_dir in ses_bold_dir.iterdir():
                    fm = folder_pat.match(run_dir.name)
                    if not fm:
                        continue
                    bold_path = cfg.replic_filtered_bold(run_dir)
                    mask_path = cfg.replic_run_mask(run_dir)
                    if not bold_path.exists():
                        continue
                    bold_entries.append((int(fm.group(1)), bold_path, mask_path))
                bold_entries.sort(key=lambda x: x[0])

            if not bold_entries:
                logger.warning(
                    "[%s] ses=%d: no valid BOLD folders found – skipping session.",
                    subject_id, ses,
                )
                continue

            bold_by_num = {num: (bp, mp) for num, bp, mp in bold_entries}
            tsv_by_num = {num: tsv_path for num, tsv_path in tsv_entries}

            # --- Step 3: direct number match ---
            all_ses_matches: list[tuple[int, int, Path, Path]] = []
            if len(tsv_entries) == len(bold_entries):
                tsv_to_bold = self.pair_runs([entry[0] for entry in tsv_entries], [entry[0] for entry in bold_entries])
                for tsv_run, bold_run in tsv_to_bold:
                    if bold_run in bold_by_num:
                        bp, mp = bold_by_num[bold_run]
                        tsv_path = tsv_by_num[tsv_run]
                        all_ses_matches.append((tsv_run, bold_run, bp, tsv_path))
            else:
                matched_tsv_nums: set[int] = set()
                matched_bold_nums: set[int] = set()
                direct_matches: list[tuple[int, int, Path, Path]] = []  # (tsv_run, folder_run, bold, tsv)
                for tsv_run, tsv_path in tsv_entries:
                    if tsv_run in bold_by_num:
                        bp, mp = bold_by_num[tsv_run]
                        direct_matches.append((tsv_run, tsv_run, bp, tsv_path))
                        matched_tsv_nums.add(tsv_run)
                        matched_bold_nums.add(tsv_run)

                # --- Step 4: positional fallback for unmatched pairs ---
                unmatched_tsvs  = [(n, p) for n, p in tsv_entries
                                   if n not in matched_tsv_nums]
                unmatched_bolds = [(n, bp, mp) for n, bp, mp in bold_entries
                                   if n not in matched_bold_nums]

                positional_matches: list[tuple[int, int, Path, Path]] = []
                if unmatched_tsvs and unmatched_bolds:
                    if len(unmatched_tsvs) != len(unmatched_bolds):
                        logger.warning(
                            "[%s] ses=%d: %d unmatched TSV(s) vs %d unmatched BOLD(s) "
                            "after direct matching – positional pairing will be "
                            "truncated to the shorter side.",
                            subject_id, ses, len(unmatched_tsvs), len(unmatched_bolds),
                        )
                    for (tsv_run, tsv_path), (bold_num, bp, _) in zip(
                        unmatched_tsvs, unmatched_bolds
                    ):
                        logger.info(
                            "[%s] ses=%d: positional fallback – TSV run-%d paired "
                            "with BOLD folder run-%d (filtered.nii.gz exists).",
                            subject_id, ses, tsv_run, bold_num,
                        )
                        positional_matches.append((tsv_run, bold_num, bp, tsv_path))
                elif unmatched_tsvs:
                    for tsv_run, tsv_path in unmatched_tsvs:
                        logger.warning(
                            "[%s] ses=%d: TSV run-%d has no matching BOLD folder – skipped.",
                            subject_id, ses, tsv_run,
                        )

                # --- Merge and append to run_jobs ---
                all_ses_matches = sorted(
                    direct_matches + positional_matches, key=lambda x: x[0]
                )
            # Build a quick lookup of mask paths by bold path
            bold_to_mask: dict[Path, Path] = {bp: mp for _, bp, mp in bold_entries}

            for tsv_run, folder_run, bp, tsv_path in all_ses_matches:
                run_jobs.append((ses, tsv_run, bp, tsv_path))
                if first_bold_path is None:
                    first_bold_path = bp
                mp = bold_to_mask.get(bp)
                if first_mask_path is None and mp is not None and mp.exists():
                    first_mask_path = mp

        if not run_jobs:
            raise RuntimeError(
                f"[{subject_id}] No valid BOLD runs found under "
                f"{cfg.replic_func_dir(subject_id)}"
            )
        logger.info("[%s] Found %d run(s) to process.", subject_id, len(run_jobs))

        if first_mask_path is None:
            raise FileNotFoundError(
                f"[{subject_id}] No per-run mask.nii.gz found."
            )
        full_mask = self._fmri.load_mask(first_mask_path)
        subject.mask_paths = [first_mask_path]

        roi_bold_dir = cfg.replic_roi_bold_dir(subject_id)
        bilateral_masks = self._load_bilateral_roi_masks(
            subject_id, roi_bold_dir, full_mask, first_bold_path
        )
        roi_names_ordered = list(bilateral_masks.keys())

        # ── Pre-compute ROI column-index arrays ────────────────────────────
        full_flat = np.flatnonzero(full_mask)
        full_flat_inv = {v: k for k, v in enumerate(full_flat)}
        roi_col_indices: dict[str, np.ndarray] = {}
        for roi_name, roi_bitmask in bilateral_masks.items():
            roi_flat = np.flatnonzero(roi_bitmask)
            roi_col_indices[roi_name] = np.array(
                [full_flat_inv[v] for v in roi_flat if v in full_flat_inv], dtype=np.intp
            )

        # ── Compact accumulators (Bug-B fix) ───────────────────────────────
        # state → np.ndarray | None  (None = no trials yet)
        wb_accum: dict[str, np.ndarray | None] = {s: None for s in self._states}
        roi_accum: dict[str, dict[str, np.ndarray | None]] = {
            s: {r: None for r in roi_names_ordered} for s in self._states
        }
        meta_buffers: dict[str, list[dict]] = {s: [] for s in self._states}

        # ── Per-run processing ─────────────────────────────────────────────
        for ses, run_idx, bold_path, tsv_path in run_jobs:
            logger.info("[%s] ses=%d run=%d → %s", subject_id, ses, run_idx, bold_path.name)

            try:
                tsv_df = self._behavioral.load_tsv(tsv_path)
            except Exception as exc:
                logger.warning("[%s] ses=%d run=%d TSV load failed: %s",
                               subject_id, ses, run_idx, exc)
                continue

            tsv_df = tsv_df.dropna(subset=["targets"])
            vi_df = tsv_df[tsv_df["volume_interest"] == 1].copy()
            if vi_df.empty:
                continue

            tr_indices = (
                vi_df["time_indices"].astype(int).values
                if "time_indices" in vi_df.columns
                else vi_df.index.values.astype(int)
            )

            # Bug-A fix: bad runs return empty array; exception → skip run
            try:
                run_patterns = self._fmri.extract_replication_run_patterns(
                    bold_path=bold_path,
                    brain_mask=full_mask,
                    trial_vol_indices=tr_indices,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] ses=%d run=%d pattern extraction failed (%s) – skipping run.",
                    subject_id, ses, run_idx, exc,
                )
                continue

            if run_patterns.shape[0] == 0:
                logger.warning(
                    "[%s] ses=%d run=%d returned 0 patterns – skipping run.",
                    subject_id, ses, run_idx,
                )
                continue

            n_run_trials = run_patterns.shape[0]

            # ── Trial-level averaging (v11 parity) ────────────────────────
            # v11 groups all volume_interest rows sharing the same trial_num
            # and averages their z-scored volumes into one pattern per trial.
            # We replicate that here: collect row indices per (state, trial),
            # then average the corresponding run_patterns rows.

            # Build a mapping: trial_num → list of row positions in vi_df
            trial_ids_arr = vi_df["trials"].values
            unique_trials = sorted(np.unique(trial_ids_arr))

            # Per-state accumulators for THIS run (one averaged row per trial)
            run_meta: dict[str, list[dict]] = {s: [] for s in self._states}
            # averaged patterns per state: list of 1-D arrays
            run_patterns_by_state: dict[str, list[np.ndarray]] = {
                s: [] for s in self._states
            }

            vi_df_reset = vi_df.reset_index(drop=True)

            for trial_num in unique_trials:
                row_positions = np.where(trial_ids_arr == trial_num)[0]
                # Guard: only include positions that fall within extracted patterns
                row_positions = row_positions[row_positions < n_run_trials]
                if row_positions.size == 0:
                    continue

                # Determine visibility from the first matching row (as v11 does)
                first_row = vi_df_reset.iloc[row_positions[0]]
                vis_state = str(first_row.get("visibility", "")).strip().lower()
                if vis_state not in self._states:
                    continue

                # Average all vi volumes for this trial (v11: zscored_matrix[row_indices].mean)
                averaged_pattern = run_patterns[row_positions].mean(axis=0)

                run_patterns_by_state[vis_state].append(averaged_pattern)
                run_meta[vis_state].append({
                    "session": ses,
                    "run": run_idx,
                    "trials": trial_num,
                    "id": ses * 10000 + run_idx * 100 + trial_num,
                    "targets": str(first_row.get("targets", "")),
                    "labels": str(first_row.get("labels", f"item_{int(trial_num)}")),
                    "visibility": vis_state,
                    "volume_interest": 1,
                    "onset": float(first_row.get("onset", 0)),
                })

            for state in self._states:
                if not run_patterns_by_state[state]:
                    continue

                # Stack averaged trial patterns: (n_trials_this_run, n_voxels)
                wb_chunk = np.stack(run_patterns_by_state[state], axis=0).astype(np.float32)

                wb_accum[state] = (
                    wb_chunk.copy() if wb_accum[state] is None
                    else np.concatenate([wb_accum[state], wb_chunk], axis=0)
                )

                for roi_name, col_idx in roi_col_indices.items():
                    roi_chunk = (
                        wb_chunk[:, col_idx] if col_idx.size > 0
                        else np.zeros((wb_chunk.shape[0], 0), dtype=np.float32)
                    )
                    roi_accum[state][roi_name] = (
                        roi_chunk.copy() if roi_accum[state][roi_name] is None
                        else np.concatenate([roi_accum[state][roi_name], roi_chunk], axis=0)
                    )

                meta_buffers[state].extend(run_meta[state])

            del run_patterns, run_patterns_by_state, run_meta
            gc.collect()

        # ── Assemble Subject visibility states ─────────────────────────────
        for state in self._states:
            meta_list = meta_buffers[state]
            if not meta_list:
                logger.warning("[%s] No trials found for state='%s'.", subject_id, state)
                continue

            events_df = pd.DataFrame(meta_list).reset_index(drop=True)
            roi_patterns: dict[str, np.ndarray] = {}

            if wb_accum[state] is not None:
                roi_patterns["wholebrain"] = wb_accum[state]
                self._discovered_rois.add("wholebrain")

            for roi_name in roi_names_ordered:
                arr = roi_accum[state][roi_name]
                if arr is not None and arr.shape[0] > 0:
                    roi_patterns[roi_name] = arr
                    self._discovered_rois.add(roi_name)

            labels = (events_df["targets"] == "Living_Things").astype(int).values
            label_strings = events_df["targets"].values
            stim_names = events_df["labels"].values.astype(str)

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

    # ── Helper: pair runs ─────────────────────────────────────────────────────

    @staticmethod
    def pair_runs(tsv_run_names: list, unfeat_run_names: list) -> list[tuple]:
        """
        Pair TSV run numbers with BOLD folder run numbers.
        
        Strategy:
        1. Direct match: pair runs where TSV[i] == unfeat[i] for initial sequence
           (stop at first mismatch to preserve chronological order)
        2. For remaining TSV runs:
           - Separate into those that CAN'T match (don't exist in unfeat list)
             and those that CAN match (exist in unfeat but weren't used)
           - Pair cannot-match runs first with remaining unfeat
           - Then pair can-match runs with remaining unfeat
        
        Example:
            tsv_run_names = [1, 2, 3, 4, 5, 7, 8, 9, 61, 62]
            unfeat_run_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            Returns: [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), 
                      (61, 6), (62, 7), (7, 8), (8, 9), (9, 10)]
        """
        paired_runs = []
        used_unfeat = set()
        unfeat_list = list(unfeat_run_names)
        unfeat_set = set(unfeat_run_names)
        
        # Step 1: Direct match for initial sequence where TSV[i] == unfeat[i]
        direct_match_count = 0
        for i in range(min(len(tsv_run_names), len(unfeat_list))):
            if tsv_run_names[i] == unfeat_list[i]:
                paired_runs.append((tsv_run_names[i], unfeat_list[i]))
                used_unfeat.add(unfeat_list[i])
                direct_match_count += 1
            else:
                break  # Stop at first mismatch
        
        # Step 2: Separate remaining TSV into cannot-match and can-match
        remaining_tsv = tsv_run_names[direct_match_count:]
        cannot_match = [t for t in remaining_tsv if t not in unfeat_set]
        can_match = [t for t in remaining_tsv if t in unfeat_set]
        
        # Step 3: Get remaining unfeat
        remaining_unfeat = [u for u in unfeat_list if u not in used_unfeat]
        
        # Step 4: Pair cannot-match first, then can-match
        all_remaining_tsv = cannot_match + can_match
        for tsv_run, unfeat_run in zip(all_remaining_tsv, remaining_unfeat):
            paired_runs.append((tsv_run, unfeat_run))
        
        return paired_runs

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
