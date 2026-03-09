"""embeddings/fmri_embedder.py – Organise per-ROI fMRI patterns as embeddings."""
from __future__ import annotations

import logging

import numpy as np

from data.loaders.subject import Subject, VisibilityData
from config.settings import Settings

logger = logging.getLogger(__name__)


class FMRIEmbedder:
    """
    Thin wrapper that presents the per-ROI BOLD pattern matrices from a
    :class:`Subject` as a standardised embedding dictionary.

    For each ROI the embedding shape is (n_trials, n_voxels).  When
    multiple subjects are aggregated, a common-subset stimulus ordering
    is used so RDMs are comparable.
    """

    def __init__(self, settings: Settings) -> None:
        self._roi_names = settings.roi_names

    # ── Public API ───────────────────────────────────────────────────────────

    def get_roi_embeddings(
        self,
        visibility_data: VisibilityData,
        sort_by_category: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Return a mapping roi_name → (n_trials, n_voxels) embedding.

        Parameters
        ----------
        visibility_data  : A VisibilityData object (conscious or unconscious)
        sort_by_category : If True, sort trials so Living precedes Non-Living
                           (maximises cluster visibility in RDMs as per POC)
        """
        patterns = visibility_data.bold_patterns
        if sort_by_category:
            sort_idx = np.argsort(1 - visibility_data.labels)  # Living first (label=1)
            return {roi: patterns[roi][sort_idx] for roi in patterns}
        return dict(patterns)

    def get_sorted_labels(
        self,
        visibility_data: VisibilityData,
    ) -> np.ndarray:
        """Return labels sorted Living-first to match ``get_roi_embeddings`` order."""
        sort_idx = np.argsort(1 - visibility_data.labels)
        return visibility_data.labels[sort_idx]

    def get_sorted_stimulus_names(
        self,
        visibility_data: VisibilityData,
    ) -> np.ndarray:
        """Return stimulus names sorted Living-first."""
        sort_idx = np.argsort(1 - visibility_data.labels)
        return visibility_data.stimulus_names[sort_idx]

    def align_stimuli_across_subjects(
        self,
        subjects: list[Subject],
        state: str,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        """
        Find the intersection of stimulus names across subjects and
        return aligned embedding dictionaries and shared stimulus order.

        Returns
        -------
        aligned_embeddings : list of dict[roi_name → (n_common, n_voxels)]
        common_stimuli     : array of shared stimulus name strings
        """
        # Collect stimulus name sets
        name_sets = []
        for subj in subjects:
            vis = getattr(subj, state, None)
            if vis is None:
                continue
            name_sets.append(set(vis.stimulus_names))

        if not name_sets:
            raise ValueError(f"No subjects have '{state}' data loaded.")

        common_names = sorted(name_sets[0].intersection(*name_sets[1:]))
        common_stimuli = np.array(common_names)

        aligned: list[dict[str, np.ndarray]] = []
        for subj in subjects:
            vis = getattr(subj, state, None)
            if vis is None:
                continue
            stim_names = vis.stimulus_names
            idx = np.array([np.where(stim_names == n)[0][0] for n in common_names
                            if n in stim_names])
            roi_embs = {
                roi: vis.bold_patterns[roi][idx]
                for roi in vis.bold_patterns
            }
            aligned.append(roi_embs)

        return aligned, common_stimuli
