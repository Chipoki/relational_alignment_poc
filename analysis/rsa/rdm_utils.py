"""analysis/rsa/rdm_utils.py – Cross-subject mean RDM and optimal-sort utilities.

Provides:
  * mean_rdm()       – average a list of RDMs into a single group-level RDM
  * sorted_order()   – Ward + silhouette optimal reordering of a matrix
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from analysis.rsa.rdm import RDM, RDMBuilder


# ── Mean RDM ────────────────────────────────────────────────────────────────

def mean_rdm(rdms: list[RDM], roi_or_layer: str, state: str) -> RDM:
    """
    Average a list of subject-level RDMs into a single group-level RDM.

    All RDMs must share the same stimulus ordering (enforce this upstream
    by aligning on common_stims before calling).

    Returns
    -------
    RDM with subject_id='mean', containing the element-wise average matrix.
    """
    if not rdms:
        raise ValueError("Cannot compute mean RDM from an empty list.")
    stacked = np.stack([r.matrix for r in rdms], axis=0)  # (n_subj, n, n)
    avg = stacked.mean(axis=0)
    np.fill_diagonal(avg, 0.0)
    return RDMBuilder().build_from_matrix(
        matrix=avg,
        stimulus_names=rdms[0].stimulus_names,
        labels=rdms[0].labels,
        roi_or_layer=roi_or_layer,
        subject_id="mean",
        state=state,
    )


# ── Ward / Silhouette optimal sort ──────────────────────────────────────────

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

    # Condense the symmetric matrix for scipy linkage
    condensed = squareform(matrix, checks=False)
    condensed = np.clip(condensed, 0.0, None)   # guard numerical negatives

    Z = linkage(condensed, method="ward")

    best_score = -np.inf
    best_labels = np.zeros(n, dtype=int)
    best_k = k_min

    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, k, criterion="maxclust") - 1  # 0-indexed
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels, metric="precomputed")
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    # Sort: group cluster members together, preserve within-cluster order
    order = np.argsort(best_labels, kind="stable")
    return order, best_k, float(best_score)
