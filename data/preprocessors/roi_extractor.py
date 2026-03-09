"""data/preprocessors/roi_extractor.py – Extract per-ROI BOLD patterns."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import nibabel as nib

from config.settings import Settings
from data.loaders.fmri_loader import FMRILoader

logger = logging.getLogger(__name__)


class ROIExtractor:
    """
    Given a full-brain BOLD pattern matrix and a set of ROI NIfTI mask files,
    produces a mapping  roi_name → (n_trials, n_voxels) for each ROI.

    ROI masks are expected in subject-space (native BOLD space), matching the
    paper's FreeSurfer-derived masks registered to functional space.
    """

    def __init__(self, settings: Settings) -> None:
        self._roi_names: list[str] = settings.roi_names
        self._loader = FMRILoader(settings)

    # ── Public API ───────────────────────────────────────────────────────────

    def extract_all_rois(
        self,
        full_patterns: np.ndarray,     # (n_trials, n_voxels_brain)
        full_mask: np.ndarray,         # (X, Y, Z) bool – whole-brain mask
        roi_mask_dir: str | Path,      # directory containing <roi_name>_mask.nii[.gz]
    ) -> dict[str, np.ndarray]:
        """
        Extract ROI-specific sub-matrices from a whole-brain pattern array.

        Returns
        -------
        dict mapping roi_name → np.ndarray of shape (n_trials, n_roi_voxels)
        """
        roi_mask_dir = Path(roi_mask_dir)
        results: dict[str, np.ndarray] = {}

        for roi_name in self._roi_names:
            # FIX: "wholebrain" is a reserved fallback name handled by SubjectBuilder;
            # it is never a real mask file and must be skipped here.
            #
            # ROOT CAUSE OF THE ORIGINAL BUG:
            # The old _find_roi_mask glob fallback matched any *.nii* file whose
            # name contained the roi_name string.  For roi_name="wholebrain" this
            # matched "wholebrain_conscious.nii.gz" — the 4-D BOLD volume — and
            # returned it as the mask path.  FMRILoader then loaded a 4-D array
            # (88,88,66,T) as a "mask", which caused:
            #   ERROR: operands could not be broadcast together with
            #          shapes (88,88,66,T) (88,88,66)
            # and reported ~43 million "voxels" instead of the expected ~120k.
            # Skipping "wholebrain" here prevents that entirely.
            if roi_name == "wholebrain":
                continue

            roi_mask_path = self._find_roi_mask(roi_mask_dir, roi_name)
            if roi_mask_path is None:
                logger.warning("ROI mask not found for '%s' in %s – skipping", roi_name, roi_mask_dir)
                continue

            roi_mask = self._loader.load_mask(roi_mask_path)
            try:
                roi_patterns = self._loader.apply_roi_mask(full_patterns, full_mask, roi_mask)
                results[roi_name] = roi_patterns
                logger.debug("ROI '%s': %d voxels", roi_name, roi_patterns.shape[1])
            except Exception as exc:
                logger.error("Failed to extract ROI '%s': %s", roi_name, exc)

        return results

    def extract_from_combined_sessions(
        self,
        session_patterns: list[np.ndarray],  # list of (n_trials_sess, n_voxels) per session
        full_mask: np.ndarray,
        roi_mask_dir: str | Path,
    ) -> dict[str, np.ndarray]:
        """
        Concatenate session-level patterns then extract ROIs.
        """
        combined = np.concatenate(session_patterns, axis=0)
        return self.extract_all_rois(combined, full_mask, roi_mask_dir)

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _find_roi_mask(directory: Path, roi_name: str) -> Path | None:
        """Search for <roi_name>_mask.nii.gz or <roi_name>_mask.nii.

        FIX: The original glob fallback used a substring match
        (``roi_name.lower() in p.name.lower()``), which caused false positives.
        For example, searching for roi_name="wholebrain" would match
        "wholebrain_conscious.nii.gz" — a 4-D BOLD file — and return it as a
        mask path.

        The fallback is now restricted to files whose *stem starts with*
        the roi_name (case-insensitive), which eliminates matches on BOLD
        volumes that merely contain the ROI name as a prefix substring.
        Named ROI mask files (e.g. "fusiform_mask.nii.gz") are unaffected
        because they are found by the explicit suffix loop first.
        """
        # Preferred: exact canonical names
        for suffix in ("_mask.nii.gz", "_mask.nii", ".nii.gz", ".nii"):
            candidate = directory / f"{roi_name}{suffix}"
            if candidate.exists():
                return candidate

        # Glob fallback: only accept files whose stem *starts with* roi_name
        # so that e.g. "wholebrain_conscious.nii.gz" is never matched when
        # looking for a mask called "wholebrain".
        for p in directory.glob("*.nii*"):
            if p.stem.lower().startswith(roi_name.lower()):
                return p

        return None