"""data/loaders/fmri_loader.py – Load NIfTI fMRI volumes and extract ROI patterns."""
from __future__ import annotations

import logging
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

    The loader replicates the paper's HRF windowing:
        - Average volumes in [hrf_delay_s, hrf_delay_s + n_hrf_volumes * TR]
          after stimulus onset (i.e. ``volume_interest`` column from events).
    """

    def __init__(self, settings: Settings) -> None:
        self._tr: float = settings.data["tr"]
        self._hrf_delay: float = settings.data["hrf_delay_seconds"]
        self._n_hrf_vols: int = settings.data["n_hrf_volumes"]

    # ── Public API ───────────────────────────────────────────────────────────

    def load_bold(self, nifti_path: str | Path) -> np.ndarray:
        """Load 4-D NIfTI entirely into RAM for fast unzipping and slicing."""
        img = nib.load(str(nifti_path))
        # This unzips and loads the ~1GB array into RAM quickly
        data = np.asarray(img.dataobj, dtype=np.float32)
        logger.info("Loaded BOLD %s → shape %s", nifti_path, data.shape)
        return data

    def load_mask(self, mask_path: str | Path) -> np.ndarray:
        """Load a binary mask NIfTI and return a bool (X, Y, Z) array."""
        img = nib.load(str(mask_path))
        mask = np.asarray(img.dataobj, dtype=bool)
        logger.info("Loaded mask %s → %d voxels", mask_path, mask.sum())
        return mask

    def extract_trial_patterns(
            self,
            bold: np.ndarray,  # Back to expecting an in-memory numpy array
            mask: np.ndarray,
            trial_volumes: Sequence[int],
    ) -> np.ndarray:
        """
        Average the in-memory BOLD signal across the HRF window.
        """
        n_timepoints = bold.shape[-1]
        n_trials = len(trial_volumes)
        n_voxels = int(mask.sum())

        patterns = np.zeros((n_trials, n_voxels), dtype=np.float32)

        for i, vol_idx in enumerate(trial_volumes):
            start = int(vol_idx)
            end = min(start + self._n_hrf_vols, n_timepoints)
            if start >= n_timepoints:
                continue

            # Fast in-memory slicing
            window = bold[..., start:end]
            avg_vol = window.mean(axis=-1)
            patterns[i] = avg_vol[mask]

        return patterns

    def zscore_patterns(self, patterns: np.ndarray) -> np.ndarray:
        """
        Z-score each voxel column across trials (replicates block-wise
        normalisation described in the Methods section).
        """
        mean = patterns.mean(axis=0, keepdims=True)
        std = patterns.std(axis=0, keepdims=True)
        std[std == 0] = 1.0   # avoid division-by-zero for flat voxels
        return (patterns - mean) / std

    def apply_roi_mask(
        self,
        patterns: np.ndarray,   # (n_trials, n_voxels_full_brain)
        full_mask: np.ndarray,  # (X, Y, Z) – full-brain binary mask
        roi_mask: np.ndarray,   # (X, Y, Z) – ROI binary mask
    ) -> np.ndarray:
        """
        Subset a full-brain voxel pattern array to a specific ROI.

        Parameters
        ----------
        patterns     : (n_trials, N_full_voxels) array extracted with ``full_mask``
        full_mask    : boolean (X, Y, Z) mask used to create ``patterns``
        roi_mask     : boolean (X, Y, Z) mask for the desired ROI

        Returns
        -------
        (n_trials, N_roi_voxels) array
        """
        # Find which full-brain voxel indices correspond to the ROI
        full_indices = np.flatnonzero(full_mask)
        roi_indices = np.flatnonzero(roi_mask & full_mask)
        # Map global flat-indices to the column indices in ``patterns``
        index_map = {v: k for k, v in enumerate(full_indices)}
        col_indices = np.array([index_map[idx] for idx in roi_indices])
        return patterns[:, col_indices]

    # ── Session-level helpers ────────────────────────────────────────────────

    def load_session(
        self,
        nifti_path: str | Path,
        mask_path: str | Path,
        trial_volumes: Sequence[int],
    ) -> np.ndarray:
        """
        Convenience wrapper: load BOLD + mask, extract and z-score patterns.

        Returns (n_trials, n_voxels)
        """
        bold = self.load_bold(nifti_path)
        mask = self.load_mask(mask_path)
        patterns = self.extract_trial_patterns(bold, mask, trial_volumes)
        return patterns
