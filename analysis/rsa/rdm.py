"""analysis/rsa/rdm.py – Construct Representational Dissimilarity Matrices.

Implements Phase 2: Dual-State Intra-Modality RDM Construction.
Uses 1 − Spearman correlation as the dissimilarity metric, matching the
POC specification and established RSA practice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class RDM:
    """Container for a single Representational Dissimilarity Matrix."""

    matrix: np.ndarray          # (n_stimuli, n_stimuli) symmetric, zero-diagonal
    stimulus_names: np.ndarray  # (n_stimuli,) string labels
    labels: np.ndarray          # (n_stimuli,) binary category labels (1=Living, 0=NonLiving)
    roi_or_layer: str           # e.g. "fusiform" or "fcnn_hidden_clear"
    subject_id: str
    state: str                  # "conscious" | "unconscious" | "clear" | "chance"

    @property
    def n_stimuli(self) -> int:
        return self.matrix.shape[0]

    def upper_triangle(self) -> np.ndarray:
        """Return the upper-triangular values (excluding diagonal) as a flat vector."""
        idx = np.triu_indices(self.n_stimuli, k=1)
        return self.matrix[idx]

    def __repr__(self) -> str:
        return (
            f"RDM(subject={self.subject_id!r}, state={self.state!r}, "
            f"roi={self.roi_or_layer!r}, n={self.n_stimuli})"
        )


# ── RDM Builder ─────────────────────────────────────────────────────────────

class RDMBuilder:
    """
    Constructs RDMs from multi-voxel or hidden-unit pattern arrays.

    The dissimilarity between stimulus i and stimulus j is computed as:
        d(i, j) = 1 − Spearman_ρ(pattern_i, pattern_j)

    When the number of voxels / units is large, Spearman rank correlation
    is more robust to outlier voxels than Pearson.
    """

    DISTANCE: Literal["spearman"] = "spearman"

    # ── Public API ──────────────────────────────────────────────────────────

    def build(
        self,
        patterns: np.ndarray,           # (n_stimuli, n_features)
        stimulus_names: np.ndarray,
        labels: np.ndarray,
        roi_or_layer: str,
        subject_id: str,
        state: str,
    ) -> RDM:
        """
        Build an RDM from a pattern matrix.

        Parameters
        ----------
        patterns        : (n_stimuli, n_features)  — voxels or hidden units
        stimulus_names  : (n_stimuli,)
        labels          : (n_stimuli,) binary category labels
        roi_or_layer    : name tag for the region / layer
        subject_id      : participant identifier
        state           : visibility or noise state

        Returns
        -------
        :class:`RDM`
        """
        n = patterns.shape[0]
        dist_matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                rho, _ = spearmanr(patterns[i], patterns[j])
                # Guard against NaN (e.g. constant voxel patterns)
                d = 1.0 - (rho if np.isfinite(rho) else 0.0)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return RDM(
            matrix=dist_matrix,
            stimulus_names=stimulus_names,
            labels=labels,
            roi_or_layer=roi_or_layer,
            subject_id=subject_id,
            state=state,
        )

    def build_vectorised(
        self,
        patterns: np.ndarray,
        stimulus_names: np.ndarray,
        labels: np.ndarray,
        roi_or_layer: str,
        subject_id: str,
        state: str,
    ) -> RDM:
        """
        Faster RDM construction using rank transformation + correlation matrix.
        Equivalent to the loop version but ~10× faster for large n.
        """
        n = patterns.shape[0]
        # Rank transform each row (stimulus pattern)
        ranked = np.apply_along_axis(
            lambda x: np.argsort(np.argsort(x)).astype(float), axis=1, arr=patterns
        )
        # Centre ranks
        ranked -= ranked.mean(axis=1, keepdims=True)

        # Compute Pearson correlation on rank-transformed patterns ≡ Spearman
        norms = np.linalg.norm(ranked, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        ranked_norm = ranked / norms
        corr_matrix = ranked_norm @ ranked_norm.T
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        np.fill_diagonal(corr_matrix, 1.0)

        dist_matrix = 1.0 - corr_matrix
        np.fill_diagonal(dist_matrix, 0.0)

        return RDM(
            matrix=dist_matrix,
            stimulus_names=stimulus_names,
            labels=labels,
            roi_or_layer=roi_or_layer,
            subject_id=subject_id,
            state=state,
        )

    def build_from_embeddings(
        self,
        embeddings: dict[str, np.ndarray],   # roi_name → (n_stimuli, n_features)
        stimulus_names: np.ndarray,
        labels: np.ndarray,
        subject_id: str,
        state: str,
        vectorised: bool = True,
    ) -> dict[str, RDM]:
        """
        Build one RDM per ROI/layer entry.

        Returns
        -------
        dict mapping roi_name → :class:`RDM`
        """
        rdms: dict[str, RDM] = {}
        build_fn = self.build_vectorised if vectorised else self.build

        for roi_name, patterns in embeddings.items():
            if patterns.ndim != 2 or patterns.shape[0] < 2:
                logger.warning("Skipping ROI '%s': insufficient pattern data", roi_name)
                continue
            rdm = build_fn(
                patterns=patterns,
                stimulus_names=stimulus_names,
                labels=labels,
                roi_or_layer=roi_name,
                subject_id=subject_id,
                state=state,
            )
            rdms[roi_name] = rdm
            logger.debug("Built RDM for %s / %s / %s", subject_id, state, roi_name)

        return rdms

    # ── Persistence helpers ─────────────────────────────────────────────────

    @staticmethod
    def save(rdm: RDM, path: str) -> None:
        np.save(path, {
            "matrix": rdm.matrix,
            "stimulus_names": rdm.stimulus_names,
            "labels": rdm.labels,
            "roi_or_layer": rdm.roi_or_layer,
            "subject_id": rdm.subject_id,
            "state": rdm.state,
        }, allow_pickle=True)

    @staticmethod
    def load(path: str) -> RDM:
        data = np.load(path, allow_pickle=True).item()
        return RDM(**data)
