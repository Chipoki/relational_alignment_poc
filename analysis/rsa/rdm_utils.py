"""analysis/rsa/rdm_utils.py – Cross-subject aggregate RDMs and optimal-sort utilities.

Provides:
  * aggregate_rdm()  – collapse a list of RDMs with a chosen estimator
                       (mean | median).  mean_rdm() is a thin alias for
                       backward compatibility.
  * sorted_order()   – Ward + silhouette optimal reordering of a matrix
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from analysis.rsa.rdm import RDM, RDMBuilder

_AggMethod = Literal["mean", "median"]


# ── Aggregate RDM ────────────────────────────────────────────────────────────

def aggregate_rdm(
    rdms: list[RDM],
    roi_or_layer: str,
    state: str,
    method: _AggMethod = "mean",
) -> RDM:
    """
    Collapse a list of subject-level RDMs into a single group-level RDM.

    All RDMs must share the same stimulus ordering (guaranteed upstream by
    the common-stimuli alignment in Phase 2).

    Parameters
    ----------
    rdms        : subject-level RDMs, all (n, n) with identical stimulus order
    roi_or_layer: tag for the resulting RDM
    state       : visibility state tag
    method      : ``"mean"``   – element-wise arithmetic mean (sensitive to
                                 outlier subjects but preserves magnitude).
                  ``"median"`` – element-wise median (robust to outlier
                                 subjects; preferred when N_subjects is small
                                 or subject variance is high).

    Returns
    -------
    RDM with subject_id=<method>, e.g. ``'mean'`` or ``'median'``.
    """
    if not rdms:
        raise ValueError("Cannot aggregate an empty list of RDMs.")

    stacked = np.stack([r.matrix for r in rdms], axis=0)   # (n_subj, n, n)
    agg = stacked.mean(axis=0) if method == "mean" else np.median(stacked, axis=0)
    np.fill_diagonal(agg, 0.0)

    return RDMBuilder().build_from_matrix(
        matrix=agg,
        stimulus_names=rdms[0].stimulus_names,
        labels=rdms[0].labels,
        roi_or_layer=roi_or_layer,
        subject_id=method,           # 'mean' or 'median'
        state=state,
    )


# ── Backward-compat alias ────────────────────────────────────────────────────

def mean_rdm(rdms: list[RDM], roi_or_layer: str, state: str) -> RDM:
    """Thin alias → aggregate_rdm(..., method='mean').  Kept for compatibility."""
    return aggregate_rdm(rdms, roi_or_layer, state, method="mean")


# ── Ward / Silhouette optimal sort ───────────────────────────────────────────

def sorted_order(
    matrix: np.ndarray,
    k_min: int = 3,
    k_max: int = 40,
) -> tuple[np.ndarray, int, float]:
    """
    Determine the optimal row/column reordering that maximises silhouette score.

    Algorithm (Mei et al. 2022 / standard RSA practice):
      1. Ward linkage on the dissimilarity matrix rows.
      2. For k in [k_min, k_max]: cut dendrogram → compute silhouette score.
      3. Select k* with highest silhouette score.
      4. Return the index permutation that groups cluster members contiguously.

    Parameters
    ----------
    matrix  : (n, n) symmetric dissimilarity matrix (zero diagonal)
    k_min   : minimum number of clusters to test
    k_max   : maximum number of clusters to test

    Returns
    -------
    order       : (n,) index array — apply as matrix[np.ix_(order, order)]
    best_k      : optimal number of clusters
    best_score  : silhouette score at best_k
    """
    n = matrix.shape[0]
    k_max = min(k_max, n - 1)
    k_min = max(k_min, 2)

    condensed = squareform(matrix, checks=False)
    condensed = np.clip(condensed, 0.0, None)   # guard numerical negatives

    Z = linkage(condensed, method="ward")

    best_score  = -np.inf
    best_labels = np.zeros(n, dtype=int)
    best_k      = k_min

    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, k, criterion="maxclust") - 1   # 0-indexed
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels, metric="precomputed")
        if score > best_score:
            best_score  = score
            best_labels = labels
            best_k      = k

    order = np.argsort(best_labels, kind="stable")
    return order, best_k, float(best_score)
