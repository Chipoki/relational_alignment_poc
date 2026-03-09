"""analysis/gromov_wasserstein/gw_result.py – Result containers for GW alignment."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GWResult:
    """Result of a single Gromov-Wasserstein alignment."""

    source_id: str          # e.g. "sub-01_conscious" or "fcnn_clear"
    target_id: str
    roi_or_layer: str
    gw_distance: float      # Scalar GW distance
    transport_plan: np.ndarray   # Optimal transport matrix (n_s × n_t)
    top_k_matching_rate: float   # Fraction of Top-k matches that are correct category
    p_value: float | None = None
    significant: bool = False

    def __repr__(self) -> str:
        return (
            f"GWResult({self.source_id} → {self.target_id}, "
            f"roi={self.roi_or_layer}, GWD={self.gw_distance:.4f})"
        )


@dataclass
class GWDistanceMatrix:
    """Pairwise GW distances for all subject/model combinations."""

    labels: list[str]                   # Row/column identifiers
    matrix: np.ndarray                  # (N, N) symmetric distance matrix
    state: str
    roi_or_layer: str
    results: list[GWResult] = field(default_factory=list)

    @property
    def n_entities(self) -> int:
        return len(self.labels)

    def __repr__(self) -> str:
        return (
            f"GWDistanceMatrix(n={self.n_entities}, "
            f"state={self.state!r}, roi={self.roi_or_layer!r})"
        )
