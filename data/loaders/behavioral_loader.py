"""data/loaders/behavioral_loader.py – Load and validate behavioural CSV files."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from config.settings import Settings


class BehavioralLoader:
    """
    Loads behavioural event CSV files produced by PsychoPy.

    Expected columns (configurable via config):
        onset, duration, labels, targets, visibility, volume_interest, session, run, id
    """

    LIVING_LABEL = "Living_Things"
    NONLIVING_LABEL = "Nonliving_Things"

    def __init__(self, settings: Settings) -> None:
        self._cfg = settings.data["stimuli_csv_col"]

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self, csv_path: str | Path) -> pd.DataFrame:
        """Load a single CSV and return a cleaned DataFrame."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Behavioural CSV not found: {path}")
        df = pd.read_csv(path)
        self._validate(df)
        return self._clean(df)

    def load_visibility_state(
        self,
        csv_path: str | Path,
        state: Literal["conscious", "unconscious"],
    ) -> pd.DataFrame:
        """Load CSV and filter to the requested visibility state."""
        df = self.load(csv_path)
        mask = df[self._cfg["visibility"]].str.lower() == state.lower()
        subset = df[mask].reset_index(drop=True)
        if subset.empty:
            raise ValueError(
                f"No trials with visibility={state!r} found in {csv_path}"
            )
        return subset

    def extract_binary_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary array: 1 = Living_Things, 0 = Nonliving_Things."""
        col = self._cfg["category"]
        labels = (df[col] == self.LIVING_LABEL).astype(int).values
        return labels

    def extract_stimulus_names(self, df: pd.DataFrame) -> np.ndarray:
        """Return array of stimulus label strings (image filename stem)."""
        return df[self._cfg["label"]].values.astype(str)

    def extract_volume_indices(self, df: pd.DataFrame) -> np.ndarray:
        """Return the fMRI volume index of interest for each trial."""
        return df[self._cfg["volume"]].values.astype(int)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        required = list(self._cfg.values())
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Behavioural CSV missing columns: {missing}")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Normalise category strings
        df[self._cfg["category"]] = df[self._cfg["category"]].str.strip()
        df[self._cfg["visibility"]] = df[self._cfg["visibility"]].str.strip().str.lower()
        return df
