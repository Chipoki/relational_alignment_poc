"""data/loaders/behavioral_loader.py – Load and validate behavioural event files.

Supports two file formats:
  • CSV  (derivatives mode)  – wholebrain_<state>.csv produced by the authors'
                               analysis pipeline; one row per trial already
                               filtered to the relevant visibility state.
  • TSV  (replication mode) – raw BIDS events files
                               <sub>_ses-<SS>_task-recog_run-<R>_events.tsv;
                               all trials mixed; visibility filter applied in
                               SubjectBuilder per-run extraction.

Expected columns (configurable via config):
    onset, labels, targets, visibility, volume_interest, session, run, trials
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from config.settings import Settings


class BehavioralLoader:
    """
    Loads behavioural event CSV/TSV files produced by the experiment software.
    """

    LIVING_LABEL    = "Living_Things"
    NONLIVING_LABEL = "Nonliving_Things"

    def __init__(self, settings: Settings) -> None:
        self._cfg = settings.data["stimuli_csv_col"]

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self, csv_path: str | Path) -> pd.DataFrame:
        """Load a single CSV (derivatives mode) and return a cleaned DataFrame."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Behavioural CSV not found: {path}")
        df = pd.read_csv(path)
        self._validate(df)
        return self._clean(df)

    def load_tsv(self, tsv_path: str | Path) -> pd.DataFrame:
        """
        Load a single BIDS events TSV (replication mode) and return a cleaned
        DataFrame.  The TSV contains ALL trials for one run (all visibility
        states mixed); filtering to a specific state is done by the caller.
        """
        path = Path(tsv_path)
        if not path.exists():
            raise FileNotFoundError(f"Events TSV not found: {path}")
        df = pd.read_csv(path, sep="\t")
        # Drop damaged entries (n.a. of RT_response or target columns, but potentially other corrupted rows)
        df = df.dropna()
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
            raise ValueError(f"Behavioural file missing columns: {missing}")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Normalise category and visibility strings
        if self._cfg["category"] in df.columns:
            df[self._cfg["category"]] = df[self._cfg["category"]].str.strip()
        if self._cfg["visibility"] in df.columns:
            df[self._cfg["visibility"]] = (
                df[self._cfg["visibility"]].str.strip().str.lower()
            )
        return df
