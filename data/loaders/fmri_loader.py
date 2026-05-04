"""data/loaders/fmri_loader.py – Load NIfTI fMRI volumes and extract ROI patterns.

Handles two data flow modes:

derivatives mode
    • A single whole-brain 4-D NIfTI (wholebrain_<state>.nii.gz) is loaded.
    • Trial volumes are extracted by direct indexing into the pre-stacked
      volume (1 volume per trial, ``n_hrf_volumes`` averaged if > 1).
    • No detrending: the derivatives are already preprocessed.

replication mode
    • Per-run ICA-AROMA filtered 4-D NIfTI files are loaded one at a time.
    • Each run's time series is optionally detrended (nilearn clean_signal),
      then the volume(s) of interest are sliced and temporally z-scored,
      exactly matching aroma_decoding_pipeline_v11.py.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import nibabel as nib

from config.settings import Settings

logger = logging.getLogger(__name__)


class FMRILoader:
    """
    Loads 4-D NIfTI BOLD data and extracts trial-level multi-voxel
    patterns from a binary brain mask.
    """

    def __init__(self, settings: Settings) -> None:
        self._tr:         float = settings.data["tr"]
        self._hrf_delay:  float = settings.data.get("hrf_delay_seconds", 5.0)
        self._n_hrf_vols: int   = settings.data.get("n_hrf_volumes", 1)
        self._detrend:    bool  = settings.data.get("replication_detrend", True)
        self._zscore_run: bool  = settings.data.get("replication_zscore_per_run", True)

    # ── Public API ───────────────────────────────────────────────────────────

    def load_bold(self, nifti_path: str | Path) -> np.ndarray:
        """Load 4-D NIfTI entirely into RAM. Returns float32 (X, Y, Z, T)."""
        img  = nib.load(str(nifti_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
        logger.info("Loaded BOLD %s → shape %s", nifti_path, data.shape)
        return data

    def load_mask(self, mask_path: str | Path) -> np.ndarray:
        """Load a binary mask NIfTI and return a bool (X, Y, Z) array."""
        img  = nib.load(str(mask_path))
        mask = np.asarray(img.dataobj, dtype=bool)
        logger.info("Loaded mask %s → %d voxels", mask_path, mask.sum())
        return mask

    def extract_trial_patterns(
        self,
        bold:          np.ndarray,    # (X, Y, Z, T) in-memory array
        mask:          np.ndarray,    # (X, Y, Z) bool
        trial_volumes: Sequence[int], # 0-based volume index per trial
    ) -> np.ndarray:
        """
        Extract trial-level patterns from a pre-loaded whole-brain BOLD array.

        Used in **derivatives mode** where bold is the pre-stacked
        wholebrain_<state>.nii.gz and each trial maps directly to one volume.

        Returns (n_trials, n_voxels).
        """
        n_timepoints = bold.shape[-1]
        n_trials     = len(trial_volumes)
        n_voxels     = int(mask.sum())

        patterns = np.zeros((n_trials, n_voxels), dtype=np.float32)

        for i, vol_idx in enumerate(trial_volumes):
            start = int(vol_idx)
            end   = min(start + self._n_hrf_vols, n_timepoints)
            if start >= n_timepoints:
                continue
            window      = bold[..., start:end]
            avg_vol     = window.mean(axis=-1)
            patterns[i] = avg_vol[mask]

        return patterns

    def extract_replication_run_patterns(
        self,
        bold_path:          str | Path,
        brain_mask:         np.ndarray,  # (X, Y, Z) bool  – whole-brain mask
        trial_vol_indices:  Sequence[int],  # 0-based TR indices for selected trials
    ) -> np.ndarray:
        """
        Extract trial patterns from one ICA-AROMA filtered per-run BOLD file,
        following the three-step pipeline in aroma_decoding_pipeline_v11.py:

            Step 1 – Detrend the full run (nilearn.signal.clean).
            Step 2 – Slice the trial-of-interest volumes.
            Step 3 – Temporal z-score on the sliced volumes (per-run, ddof=1).

        Used in **replication mode**.

        The brain_mask is matched to the current run's spatial shape:
        if there is a mismatch (different runs may have been preprocessed
        with slightly different bounding boxes) the mask is resampled via
        nilearn or, when nilearn is unavailable, the run is skipped by
        returning an empty matrix so that the rest of the subject is
        preserved.

        Returns (n_selected_trials, n_brain_voxels) where n_brain_voxels
        equals brain_mask.sum() for the *reference* mask (voxels that exist
        in both mask and run).  On any unrecoverable error an empty matrix
        with shape (0, brain_mask.sum()) is returned so callers can skip
        gracefully.
        """
        try:
            from nilearn.signal import clean as _clean_signal
        except ImportError:
            _clean_signal = None

        n_ref_voxels = int(brain_mask.sum())

        bold_img      = nib.load(str(bold_path))
        bold_arr      = np.asarray(bold_img.dataobj, dtype=np.float32)  # (X, Y, Z, T)
        bold_shape_3d = bold_arr.shape[:3]
        n_timepoints  = bold_arr.shape[-1]

        # ── Bug-A fix: reconcile mask spatial shape with this run's BOLD shape ──
        run_mask = self._match_mask_to_bold(
            brain_mask   = brain_mask,
            bold_shape   = bold_shape_3d,
            bold_img_obj = bold_img,
            bold_path    = bold_path,
        )
        if run_mask is None:
            logger.warning(
                "Skipping run %s: could not reconcile mask shape %s with BOLD shape %s.",
                bold_path, brain_mask.shape, bold_shape_3d,
            )
            return np.zeros((0, n_ref_voxels), dtype=np.float32)

        # Flatten to (T, n_brain_voxels) in one shot – avoids Python loops
        brain_flat = bold_arr[run_mask]   # (n_run_voxels, T)
        ts         = brain_flat.T         # (T, n_run_voxels)
        del bold_arr, brain_flat

        # Step 1: Detrend (matches v11's nilearn.signal.clean usage)
        if self._detrend and _clean_signal is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ts = _clean_signal(ts, t_r=self._tr, detrend=True, standardize=False)
        elif self._detrend:
            x    = np.arange(ts.shape[0], dtype=np.float64)
            coef = np.polyfit(x, ts, 1)
            trend = np.outer(x, coef[0]) + coef[1]
            ts   = ts - trend.astype(np.float32)

        # Step 2: Slice trial volumes
        valid_idx = [v for v in trial_vol_indices if 0 <= v < n_timepoints]
        if not valid_idx:
            return np.zeros((0, n_ref_voxels), dtype=np.float32)
        raw_matrix = ts[np.array(valid_idx, dtype=int)]  # (n_trials, n_run_voxels)
        del ts

        # Step 3: Temporal z-score on the sliced volumes (ddof=1, matching v11)
        if self._zscore_run:
            means = raw_matrix.mean(axis=0, keepdims=True)
            stds  = raw_matrix.std(axis=0, ddof=1, keepdims=True)
            stds[stds == 0] = 1.0
            raw_matrix = (raw_matrix - means) / stds

        # If the run mask differs from the reference mask, project back into the
        # reference voxel space so all runs share the same column layout.
        if run_mask.shape != brain_mask.shape or not np.array_equal(run_mask, brain_mask):
            raw_matrix = self._project_to_ref_voxels(
                patterns = raw_matrix,
                run_mask = run_mask,
                ref_mask = brain_mask,
            )

        return raw_matrix.astype(np.float32)

    # ── Mask reconciliation helpers (Bug-A fix) ───────────────────────────────

    @staticmethod
    def _match_mask_to_bold(
        brain_mask:   np.ndarray,  # (X, Y, Z) bool  – reference mask
        bold_shape:   tuple,       # (X', Y', Z')     – this run's spatial shape
        bold_img_obj,              # nibabel image (for affine / resampling)
        bold_path:    Path | str,
    ) -> np.ndarray | None:
        """
        Return a boolean mask with the same spatial shape as `bold_shape`.

        Strategy (in order of preference):
        1. Shapes already match – return `brain_mask` as-is.
        2. Try nilearn.image.resample_img to warp the reference mask NIfTI
           into the run's voxel grid.
        3. Simple crop / zero-pad along each axis – shape-only fix, no
           affine awareness.  Acceptable because the spatial extent is
           typically identical and differences are due to slight bounding-box
           choices during preprocessing.
        4. Cannot reconcile → return None so the caller can skip the run.
        """
        if brain_mask.shape == bold_shape:
            return brain_mask

        logger.debug(
            "Mask shape %s ≠ BOLD shape %s for run %s – attempting reconciliation.",
            brain_mask.shape, bold_shape, bold_path,
        )

        # ── Strategy 2: nilearn resample ──────────────────────────────────────
        try:
            import nibabel as nib
            from nilearn.image import resample_img as _resample_img

            ref_affine = bold_img_obj.affine
            mask_nii   = nib.Nifti1Image(brain_mask.astype(np.float32), ref_affine)
            target_nii = nib.Nifti1Image(
                np.zeros(bold_shape, dtype=np.float32), ref_affine
            )
            resampled  = _resample_img(
                mask_nii,
                target_affine = target_nii.affine,
                target_shape  = bold_shape,
                interpolation = "nearest",
            )
            result = resampled.get_fdata().astype(bool)
            logger.debug(
                "Resampled mask %s → %s via nilearn (%d voxels).",
                brain_mask.shape, result.shape, result.sum(),
            )
            return result
        except Exception as exc:
            logger.debug("nilearn resample failed (%s); falling back to crop/pad.", exc)

        # ── Strategy 3: axis-wise crop / zero-pad ─────────────────────────────
        try:
            reconciled = np.zeros(bold_shape, dtype=bool)
            slices_src = []
            slices_dst = []
            for dim_src, dim_dst in zip(brain_mask.shape, bold_shape):
                min_dim = min(dim_src, dim_dst)
                slices_src.append(slice(0, min_dim))
                slices_dst.append(slice(0, min_dim))
            reconciled[tuple(slices_dst)] = brain_mask[tuple(slices_src)]
            logger.debug(
                "Crop/padded mask %s → %s (%d voxels).",
                brain_mask.shape, reconciled.shape, reconciled.sum(),
            )
            return reconciled
        except Exception as exc:
            logger.debug("Crop/pad failed (%s).", exc)

        return None

    @staticmethod
    def _project_to_ref_voxels(
        patterns: np.ndarray,   # (n_trials, n_run_voxels)
        run_mask: np.ndarray,   # (X', Y', Z') bool – mask used to extract patterns
        ref_mask: np.ndarray,   # (X,  Y,  Z)  bool – reference whole-subject mask
    ) -> np.ndarray:
        """
        Map columns of `patterns` (indexed by `run_mask`) into the column
        space defined by `ref_mask`, filling missing voxels with 0.
        """
        n_trials     = patterns.shape[0]
        n_ref_voxels = int(ref_mask.sum())
        out          = np.zeros((n_trials, n_ref_voxels), dtype=patterns.dtype)

        min_shape = tuple(min(a, b) for a, b in zip(run_mask.shape, ref_mask.shape))
        overlap   = (run_mask[:min_shape[0], :min_shape[1], :min_shape[2]] &
                     ref_mask[:min_shape[0], :min_shape[1], :min_shape[2]])

        run_col_map = {v: k for k, v in enumerate(np.flatnonzero(run_mask.ravel()))}
        ref_col_map = {v: k for k, v in enumerate(np.flatnonzero(ref_mask.ravel()))}

        overlap_flat = np.flatnonzero(
            np.pad(
                overlap,
                [(0, run_mask.shape[i] - min_shape[i]) for i in range(3)],
                mode="constant",
            ).ravel()
        )

        run_cols = np.array(
            [run_col_map[v] for v in overlap_flat if v in run_col_map], dtype=np.intp
        )
        ref_cols = np.array(
            [ref_col_map[v] for v in overlap_flat if v in ref_col_map], dtype=np.intp
        )

        n_common = min(run_cols.size, ref_cols.size)
        if n_common > 0:
            out[:, ref_cols[:n_common]] = patterns[:, run_cols[:n_common]]

        return out

    def zscore_patterns(self, patterns: np.ndarray) -> np.ndarray:
        """
        Z-score each voxel column across trials (used in derivatives mode for
        optional post-hoc normalisation).
        """
        mean = patterns.mean(axis=0, keepdims=True)
        std  = patterns.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (patterns - mean) / std

    def apply_roi_mask(
        self,
        patterns:  np.ndarray,   # (n_trials, n_voxels_full_brain)
        full_mask: np.ndarray,   # (X, Y, Z) – full-brain binary mask
        roi_mask:  np.ndarray,   # (X, Y, Z) – ROI binary mask
    ) -> np.ndarray:
        """
        Subset a full-brain voxel pattern array to a specific ROI.

        Parameters
        ----------
        patterns     : (n_trials, N_full_voxels) extracted with ``full_mask``
        full_mask    : boolean (X, Y, Z) mask used to create ``patterns``
        roi_mask     : boolean (X, Y, Z) mask for the desired ROI

        Returns
        -------
        (n_trials, N_roi_voxels)
        """
        full_indices = np.flatnonzero(full_mask)
        roi_indices  = np.flatnonzero(roi_mask & full_mask)
        index_map    = {v: k for k, v in enumerate(full_indices)}
        col_indices  = np.array([index_map[idx] for idx in roi_indices
                                  if idx in index_map])
        if col_indices.size == 0:
            return np.zeros((patterns.shape[0], 0), dtype=patterns.dtype)
        return patterns[:, col_indices]

    def load_session(
        self,
        nifti_path:    str | Path,
        mask_path:     str | Path,
        trial_volumes: Sequence[int],
    ) -> np.ndarray:
        """
        Convenience wrapper (derivatives mode): load BOLD + mask, extract patterns.
        Returns (n_trials, n_voxels).
        """
        bold    = self.load_bold(nifti_path)
        mask    = self.load_mask(mask_path)
        return self.extract_trial_patterns(bold, mask, trial_volumes)

    # ── Mask creation helpers ─────────────────────────────────────────────────

    @staticmethod
    def make_mask_from_bold(bold_arr: np.ndarray, percentile: float = 5.0) -> np.ndarray:
        """
        Create a brain mask from a 4-D BOLD array by thresholding the mean
        image at ``percentile``% of its maximum.  Used when no explicit mask
        file is available.

        Returns bool (X, Y, Z).
        """
        mean_img   = bold_arr.mean(axis=-1)
        threshold  = np.percentile(mean_img[mean_img > 0], percentile)
        return (mean_img > threshold).astype(bool)