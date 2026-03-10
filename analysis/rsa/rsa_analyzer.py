"""analysis/rsa/rsa_analyzer.py – RSA correlations and statistical testing.

Covers:
  Phase 3 – Inter-subject RSA (supervised noise ceiling)
  Phase 4 – Cross-modality RSA (C-C and U-U alignments)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from scipy.stats import spearmanr

from config.settings import Settings
from .rdm import RDM

logger = logging.getLogger(__name__)


@dataclass
class RSAResult:
    """One RSA pairwise correlation result."""

    subject_a: str
    subject_b: str
    roi_or_layer: str
    state_a: str
    state_b: str
    rho: float
    p_value: float
    significant: bool

    def __repr__(self) -> str:
        return (
            f"RSAResult({self.subject_a}×{self.subject_b}, "
            f"roi={self.roi_or_layer}, ρ={self.rho:.3f}, p={self.p_value:.4f})"
        )


class RSAAnalyzer:
    """
    Performs RSA between pairs of RDMs using permutation testing.

    Phase 3 – Inter-subject RSA: correlate human–human RDM pairs
              for Conscious (noise ceiling) and Unconscious (conserved geometry).

    Phase 4 – Cross-modality RSA: correlate FCNN–human RDMs in
              matched (C-C, U-U) configurations.
    """

    def __init__(self, settings: Settings) -> None:
        cfg = settings.rsa
        self._n_perms: int = cfg["n_permutations"]
        self._alpha: float = cfg["alpha"]
        self._correction: str = cfg.get("correction", "bonferroni")
        self._rng = np.random.default_rng(cfg.get("seed", 42))

    # ── Public API ───────────────────────────────────────────────────────────

    def correlate(self, rdm_a: RDM, rdm_b: RDM) -> tuple[float, float]:
        """
        Compute Spearman ρ between the upper triangles of two RDMs.
        Uses permutation testing to derive p-value.

        Returns (rho, p_value)
        """
        vec_a = rdm_a.upper_triangle()
        vec_b = rdm_b.upper_triangle()
        return self._permutation_test(vec_a, vec_b)

    def inter_subject_rsa(
        self,
        rdms: list[RDM],
        correct: bool = True,
    ) -> list[RSAResult]:
        """
        Compute all pairwise inter-subject RSA correlations for a set of RDMs
        (same ROI, same state, different subjects).

        Parameters
        ----------
        rdms    : list of RDMs (one per subject, same roi/state)
        correct : apply multiple-comparison correction

        Returns
        -------
        List of :class:`RSAResult`
        """
        results: list[RSAResult] = []
        for rdm_i, rdm_j in combinations(rdms, 2):
            rho, p = self.correlate(rdm_i, rdm_j)
            results.append(RSAResult(
                subject_a=rdm_i.subject_id,
                subject_b=rdm_j.subject_id,
                roi_or_layer=rdm_i.roi_or_layer,
                state_a=rdm_i.state,
                state_b=rdm_j.state,
                rho=rho,
                p_value=p,
                significant=False,  # filled after correction
            ))

        if correct:
            results = self._apply_correction(results)
        else:
            for r in results:
                r.significant = r.p_value < self._alpha

        return results

    def cross_modality_rsa(
        self,
        human_rdms: list[RDM],         # one per subject
        model_rdm: RDM,
        correct: bool = True,
    ) -> list[RSAResult]:
        """
        Correlate a model RDM against each subject's RDM (C-C or U-U alignment).
        """
        results: list[RSAResult] = []
        for h_rdm in human_rdms:
            rho, p = self.correlate(h_rdm, model_rdm)
            results.append(RSAResult(
                subject_a=h_rdm.subject_id,
                subject_b=model_rdm.subject_id,
                roi_or_layer=h_rdm.roi_or_layer,
                state_a=h_rdm.state,
                state_b=model_rdm.state,
                rho=rho,
                p_value=p,
                significant=False,
            ))

        if correct:
            results = self._apply_correction(results)
        else:
            for r in results:
                r.significant = r.p_value < self._alpha

        return results

    def mean_rho(self, results: list[RSAResult]) -> float:
        """Return mean Spearman ρ across a list of RSAResults."""
        return float(np.mean([r.rho for r in results]))

    # ── Private helpers ──────────────────────────────────────────────────────

    def _permutation_test(
        self, vec_a: np.ndarray, vec_b: np.ndarray
    ) -> tuple[float, float]:
        """
        Permute vec_b ``n_perms`` times and compute empirical p-value
        (proportion of permuted ρ ≥ observed ρ).
        """
        obs_rho, _ = spearmanr(vec_a, vec_b)
        if not np.isfinite(obs_rho):
            return 0.0, 1.0

        null_dist = np.zeros(self._n_perms)
        for k in range(self._n_perms):
            perm_b = self._rng.permutation(vec_b)
            rho_k, _ = spearmanr(vec_a, perm_b)
            null_dist[k] = rho_k if np.isfinite(rho_k) else 0.0

        p_value = float((np.abs(null_dist) >= np.abs(obs_rho)).mean())  # two-tailed

        return float(obs_rho), p_value

    def _apply_correction(self, results: list[RSAResult]) -> list[RSAResult]:
        """Apply Bonferroni or FDR correction to a list of results."""
        n = len(results)
        if n == 0:
            return results
        if self._correction == "bonferroni":
            threshold = self._alpha / n
            for r in results:
                r.significant = r.p_value < threshold
        elif self._correction == "fdr_bh":
            sorted_idx = np.argsort([r.p_value for r in results])
            for rank, idx in enumerate(sorted_idx):
                corrected = results[idx].p_value * n / (rank + 1)
                results[idx].significant = corrected < self._alpha
        return results
