"""analysis/rsa/rdm_utils.py – Cross-subject aggregate RDMs and optimal-sort utilities.

Provides
--------
aggregate_rdm(rdms, roi, state, method)
    Element-wise mean or median across subjects.

mean_rdm(...)   Thin backward-compat alias for aggregate_rdm(..., method='mean').

gw_consensus_matrix(rdms)
    Fréchet mean in GW space (ot.gromov_barycenters).
    Returns the (n, n) barycenter dissimilarity matrix whose Ward sort is used
    as the *consensus* ordering shared across all sorted-consensus figures.

sorted_order(matrix)
    Ward + silhouette optimal reordering; returns (order, best_k, best_score).
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
    method : ``'mean'``   element-wise arithmetic mean.
             ``'median'`` element-wise median (robust to outlier subjects).

    The resulting RDM carries ``subject_id = method`` so that plot titles
    can be derived without any extra parameter.
    """
    if not rdms:
        raise ValueError("Cannot aggregate an empty list of RDMs.")
    stacked = np.stack([r.matrix for r in rdms], axis=0)  # (N, n, n)
    agg = stacked.mean(axis=0) if method == "mean" else np.median(stacked, axis=0)
    np.fill_diagonal(agg, 0.0)
    return RDMBuilder().build_from_matrix(
        matrix=agg,
        stimulus_names=rdms[0].stimulus_names,
        labels=rdms[0].labels,
        roi_or_layer=roi_or_layer,
        subject_id=method,
        state=state,
    )


def mean_rdm(rdms: list[RDM], roi_or_layer: str, state: str) -> RDM:
    """Backward-compat alias → aggregate_rdm(..., method='mean')."""
    return aggregate_rdm(rdms, roi_or_layer, state, method="mean")


# ── GW-barycenter consensus matrix ──────────────────────────────────────────

def gw_consensus_matrix(
    rdms: list[RDM],
    loss_fun: str = "square_loss",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Compute the Fréchet mean in GW space via the POT gromov_barycenters solver.

    This is the (n, n) dissimilarity matrix that minimises the sum of squared
    GW distances to all subject RDMs.  It is the geometrically correct group
    representative because it accounts for the fact that subjects may have
    rotated / permuted representations internally.

    Used exclusively to derive the *consensus* Ward sort shared across all
    ``sorted_consensus`` figures; it is NOT used as an aggregate RDM for RSA.

    Parameters
    ----------
    rdms     : subject-level RDMs, all (n, n)  (common-stim aligned)
    loss_fun : GW loss function passed to POT ('square_loss' or 'kl_loss')
    max_iter : barycenter iteration limit
    tol      : convergence tolerance

    Returns
    -------
    (n, n) numpy array  – the GW Fréchet mean dissimilarity matrix,
    normalised to [0, 1] and with zero diagonal.
    """
    import ot  # Python Optimal Transport (already in environment.yml)

    n = rdms[0].matrix.shape[0]
    N = len(rdms)

    # Normalise each subject matrix to [0,1] (required by POT)
    Cs = [
        (m := r.matrix.astype(np.float64)) / (m.max() + 1e-8)
        for r in rdms
    ]
    # Uniform distributions over stimuli
    ps = [np.ones(n) / n for _ in range(N)]
    weights = np.ones(N) / N            # equal weight per subject
    C_init  = np.mean(Cs, axis=0)       # warm-start from arithmetic mean

    try:
        C_bary = ot.gromov.gromov_barycenters(
            N=n,
            Cs=Cs,
            ps=ps,
            p=np.ones(n) / n,
            lambdas=weights,
            loss_fun=loss_fun,
            max_iter=max_iter,
            tol=tol,
            init_C=C_init,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001  – fallback to arithmetic mean
        import logging
        logging.getLogger(__name__).warning(
            "gw_consensus_matrix: POT barycenter failed (%s); "
            "falling back to arithmetic mean.", exc
        )
        C_bary = C_init

    bary = np.array(C_bary, dtype=np.float64)
    np.fill_diagonal(bary, 0.0)
    # Re-normalise to [0,1]
    mx = bary.max()
    if mx > 1e-8:
        bary /= mx
    return bary


# ── Ward / Silhouette optimal sort ───────────────────────────────────────────

def sorted_order(
    matrix: np.ndarray,
    k_min: int = 3,
    k_max: int = 40,
) -> tuple[np.ndarray, int, float]:
    """
    Determine the optimal row/column reordering that maximises silhouette score.

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
    condensed = np.clip(condensed, 0.0, None)
    Z = linkage(condensed, method="ward")

    best_score  = -np.inf
    best_labels = np.zeros(n, dtype=int)
    best_k      = k_min

    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, k, criterion="maxclust") - 1
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels, metric="precomputed")
        if score > best_score:
            best_score  = score
            best_labels = labels
            best_k      = k

    order = np.argsort(best_labels, kind="stable")
    return order, best_k, float(best_score)
