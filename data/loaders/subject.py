"""data/loaders/subject.py – Subject data container."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class VisibilityData:
    """Holds BOLD patterns + labels for one visibility state (conscious / unconscious)."""

    state: str                            # "conscious" | "unconscious"
    bold_patterns: dict[str, np.ndarray]  # roi_name → (n_trials, n_voxels)
    labels: np.ndarray                    # (n_trials,)  1=Living, 0=NonLiving
    label_strings: np.ndarray             # (n_trials,)  "Living_Things" / "Nonliving_Things"
    stimulus_names: np.ndarray            # (n_trials,)  image filename stems
    events: pd.DataFrame                  # raw behavioural events

    @property
    def n_trials(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:
        rois = list(self.bold_patterns.keys())
        return (
            f"VisibilityData(state={self.state!r}, "
            f"n_trials={self.n_trials}, rois={rois})"
        )


@dataclass
class Subject:
    """All data associated with one participant.

    Under the flat ds003927 derivative layout each visibility state has its own
    pre-separated NIfTI file, so ``fmri_paths`` stores one path per loaded state
    (e.g. wholebrain_conscious.nii.gz, wholebrain_unconscious.nii.gz) rather than
    one path per session.  ``mask_paths`` stores the single shared mask.
    """

    subject_id: str
    fmri_paths: list[Path] = field(default_factory=list)  # per-state NIfTI bold files
    mask_paths: list[Path] = field(default_factory=list)  # single shared mask (list of 1)
    conscious: VisibilityData | None = None
    unconscious: VisibilityData | None = None

    def __repr__(self) -> str:
        states = [
            s for s, d in [("conscious", self.conscious), ("unconscious", self.unconscious)]
            if d is not None
        ]
        return f"Subject(id={self.subject_id!r}, states_loaded={states})"
