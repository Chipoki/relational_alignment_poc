"""analysis/gromov_wasserstein/gw_aligner.py – Gromov-Wasserstein optimal transport.

Implements:
  Phase 3 – Inter-subject GW alignment (baseline taxonomy)
  Phase 4 – Cross-modality GW alignment (FCNN ↔ human)
  Phase 5 – Structural Invariance Metric
              (ΔGW conscious→unconscious vs. ΔGW clear→noisy)
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import ot  # Python Optimal Transport library

from config.settings import Settings
from .gw_result import GWResult, GWDistanceMatrix
from analysis.rsa.rdm import RDM

logger = logging.getLogger(__name__)


class GromovWassersteinAligner:
    """
    Unsupervised structural alignment between representational geometries
    using the Gromov-Wasserstein (GW) distance.

    The GW distance measures how well the pairwise distance structure of
    one space can be matched to another, without requiring point
    correspondences.

    Uses the ``ot.gromov_wasserstein`` solver from the POT library.
    """

    def __init__(self, settings: Settings) -> None:
        cfg = settings.gromov_wasserstein
        self._loss_fun: str = cfg.get("loss_fun", "square_loss")
        self._log: bool = cfg.get("log", True)
        self._top_k: int = cfg.get("top_k", 5)
        self._n_perms: int = cfg.get("n_permutations", 10000)
        self._alpha: float = settings.rsa.get("alpha", 0.05)

    # ── Public API ───────────────────────────────────────────────────────────

    def align(
        self,
        rdm_source: RDM,
        rdm_target: RDM,
    ) -> GWResult:
        """
        Compute the GW distance between two RDMs and return the transport plan.

        Parameters
        ----------
        rdm_source : RDM from source modality/subject
        rdm_target : RDM from target modality/subject

        Returns
        -------
        :class:`GWResult`
        """
        C1 = rdm_source.matrix.astype(np.float64)
        C2 = rdm_target.matrix.astype(np.float64)

        n1, n2 = C1.shape[0], C2.shape[0]
        p = np.ones(n1) / n1    # uniform source distribution
        q = np.ones(n2) / n2    # uniform target distribution

        # Normalise cost matrices to [0, 1]
        C1 = C1 / (C1.max() + 1e-8)
        C2 = C2 / (C2.max() + 1e-8)

        T, log = ot.gromov_wasserstein(
            C1, C2, p, q,
            loss_fun=self._loss_fun,
            log=True,
            verbose=False,
        )
        gw_dist = float(log["gw_dist"])

        top_k_rate = self._compute_top_k_matching_rate(
            T, rdm_source.labels, rdm_target.labels
        )

        result = GWResult(
            source_id=f"{rdm_source.subject_id}_{rdm_source.state}",
            target_id=f"{rdm_target.subject_id}_{rdm_target.state}",
            roi_or_layer=rdm_source.roi_or_layer,
            gw_distance=gw_dist,
            transport_plan=T,
            top_k_matching_rate=top_k_rate,
        )
        logger.debug("GW align: %s → %s, GWD=%.4f", result.source_id, result.target_id, gw_dist)
        return result

    def align_with_permutation_test(
        self,
        rdm_source: RDM,
        rdm_target: RDM,
    ) -> GWResult:
        """
        As :meth:`align` but also computes an empirical p-value via
        permutation of the target's row/column ordering.
        """
        result = self.align(rdm_source, rdm_target)
        p_val = self._permutation_test(rdm_source, rdm_target, result.gw_distance)
        result.p_value = p_val
        result.significant = p_val < self._alpha
        return result

    def build_pairwise_distance_matrix(
        self,
        rdms: list[RDM],
        ids: list[str] | None = None,
        with_permutation_test: bool = False,
    ) -> GWDistanceMatrix:
        """
        Compute GW distances for all pairs in a list of RDMs.
        Returns a symmetric :class:`GWDistanceMatrix`.

        Parameters
        ----------
        rdms                  : list of RDMs (subjects + optionally model)
        ids                   : display labels for each RDM
        with_permutation_test : run permutation tests (slow but thorough)
        """
        n = len(rdms)
        if ids is None:
            ids = [f"{r.subject_id}_{r.state}" for r in rdms]

        dist_mat = np.zeros((n, n), dtype=np.float64)
        results: list[GWResult] = []

        for i in range(n):
            for j in range(i + 1, n):
                fn = self.align_with_permutation_test if with_permutation_test else self.align
                res = fn(rdms[i], rdms[j])
                dist_mat[i, j] = res.gw_distance
                dist_mat[j, i] = res.gw_distance
                results.append(res)

        return GWDistanceMatrix(
            labels=ids,
            matrix=dist_mat,
            state=rdms[0].state,
            roi_or_layer=rdms[0].roi_or_layer,
            results=results,
        )

    def structural_invariance_metric(
        self,
        human_rdms_conscious: list[RDM],
        human_rdms_unconscious: list[RDM],
        fcnn_rdm_clear: RDM,
        fcnn_rdm_noisy: RDM,
    ) -> dict[str, float]:
        """
        Phase 5 – Structural Invariance Metric.

        Computes:
            ΔGW_human   = mean GWD(conscious_i, unconscious_i) across subjects
            ΔGW_fcnn    = GWD(clear_fcnn, noisy_fcnn)

        Returns dict with keys: delta_gw_human, delta_gw_fcnn,
                                 human_variance, bioplausibility_check
        """
        human_deltas = []
        for c_rdm, u_rdm in zip(human_rdms_conscious, human_rdms_unconscious):
            res = self.align(c_rdm, u_rdm)
            human_deltas.append(res.gw_distance)

        fcnn_result = self.align(fcnn_rdm_clear, fcnn_rdm_noisy)

        delta_human_mean = float(np.mean(human_deltas))
        delta_human_std = float(np.std(human_deltas))
        delta_fcnn = fcnn_result.gw_distance

        # Bioplausibility: FCNN ΔGW should be within ±2 SD of human ΔGW
        bioplausible = abs(delta_fcnn - delta_human_mean) <= 2 * delta_human_std

        return {
            "delta_gw_human_mean": delta_human_mean,
            "delta_gw_human_std": delta_human_std,
            "delta_gw_fcnn": delta_fcnn,
            "human_variance": float(np.var(human_deltas)),
            "bioplausibility_check": bioplausible,
            "individual_human_deltas": human_deltas,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _compute_top_k_matching_rate(
        self,
        transport_plan: np.ndarray,   # (n_source, n_target)
        source_labels: np.ndarray,
        target_labels: np.ndarray,
    ) -> float:
        """
        For each source stimulus, find the top-k matched target stimuli.
        The Top-k Matching Rate is the fraction of source stimuli whose
        top-k matches share the same category label.
        """
        k = min(self._top_k, transport_plan.shape[1])
        n_source = transport_plan.shape[0]
        matches = 0

        for i in range(n_source):
            top_k_idx = np.argsort(transport_plan[i])[::-1][:k]
            top_k_labels = target_labels[top_k_idx]
            # Count as match if majority of top-k agree with source label
            majority = (top_k_labels == source_labels[i]).sum() > k / 2
            if majority:
                matches += 1

        return matches / n_source

    def _permutation_test(
        self, rdm_source: RDM, rdm_target: RDM, observed_gwd: float
    ) -> float:
        """
        Permute the rows/columns of the target RDM and recompute GWD.
        Returns empirical p-value (fraction of permuted GWD ≤ observed GWD).
        Note: smaller GWD = better alignment, so we test the left tail.
        """
        n_perms = min(self._n_perms, 500)   # cap at 500 for practical runtime
        C1 = rdm_source.matrix.astype(np.float64)
        C2 = rdm_target.matrix.astype(np.float64)
        n1, n2 = C1.shape[0], C2.shape[0]
        p = np.ones(n1) / n1
        q = np.ones(n2) / n2
        C1 = C1 / (C1.max() + 1e-8)
        C2_base = C2 / (C2.max() + 1e-8)

        null_gwds = []
        rng = np.random.default_rng()

        for _ in range(n_perms):
            perm = rng.permutation(n2)
            C2_perm = C2_base[np.ix_(perm, perm)]
            _, log = ot.gromov_wasserstein(
                C1, C2_perm, p, q, loss_fun=self._loss_fun, log=True, verbose=False
            )
            null_gwds.append(float(log["gw_dist"]))

        null_array = np.array(null_gwds)
        p_value = float((null_array <= observed_gwd).mean())
        return p_value
