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
    import logging
    import ot  # Python Optimal Transport (already in environment.yml)

    _log = logging.getLogger(__name__)
    n = rdms[0].matrix.shape[0]
    N = len(rdms)

    def _sanitise(mat: np.ndarray) -> np.ndarray | None:
        """
        Return a clean, normalised copy of *mat* suitable for POT, or None
        if the matrix is too degenerate to use.

        Real RDMs (1 − Spearman ρ) can contain:
          - NaNs  (degenerate trials / stimuli with a single repetition)
          - small negative values from floating-point error
          - zero rows/columns (stimulus never co-occurring with others)
        Any of these cause POT's internal barycenter update to produce NaN
        transport plans, which then triggers the 'numpy.ndarray has no
        attribute append' crash when POT tries to log convergence info on
        the NaN array.
        """
        m = mat.astype(np.float64)
        # Replace NaNs with column means, then with 0 if whole column is NaN
        col_means = np.nanmean(m, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask  = np.isnan(m)
        m[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        # Ensure symmetry (numerical drift can break this)
        m = (m + m.T) / 2.0
        # Clip negatives introduced by floating-point error
        np.clip(m, 0.0, None, out=m)
        np.fill_diagonal(m, 0.0)
        # Normalise to [0, 1]
        mx = m.max()
        if mx < 1e-8:
            return None   # zero matrix — degenerate, skip this subject
        m /= mx
        # Guard: any remaining non-finite value → give up on this matrix
        if not np.isfinite(m).all():
            return None
        return m

    Cs_raw = [_sanitise(r.matrix) for r in rdms]
    Cs     = [c for c in Cs_raw if c is not None]
    n_dropped = N - len(Cs)
    if n_dropped:
        _log.warning(
            "gw_consensus_matrix: dropped %d/%d degenerate subject RDM(s) "
            "before calling POT.", n_dropped, N,
        )
    if len(Cs) < 2:
        _log.warning(
            "gw_consensus_matrix: fewer than 2 usable RDMs after sanitisation; "
            "falling back to arithmetic mean."
        )
        safe = [c for c in Cs_raw if c is not None] or \
               [np.zeros((n, n))]
        bary = np.mean(safe, axis=0)
        np.fill_diagonal(bary, 0.0)
        return bary

    n_use = Cs[0].shape[0]   # may differ from n if a subject was dropped mid-shape
    N_use = len(Cs)

    ps      = [np.ones(n_use) / n_use for _ in range(N_use)]
    weights = [1.0 / N_use] * N_use
    C_init  = np.mean(Cs, axis=0)

    gw_kwargs: dict = dict(
        N=n_use,
        Cs=Cs,
        ps=ps,
        p=np.ones(n_use) / n_use,
        lambdas=weights,
        loss_fun=loss_fun,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )
    try:
        import inspect as _inspect
        if "init_C" in _inspect.signature(ot.gromov.gromov_barycenters).parameters:
            gw_kwargs["init_C"] = C_init
    except Exception:
        pass

    try:
        C_bary = ot.gromov.gromov_barycenters(**gw_kwargs)
        bary   = np.array(C_bary, dtype=np.float64)
        if not np.isfinite(bary).all():
            raise ValueError("POT returned non-finite barycenter matrix.")
    except Exception as exc:
        _log.warning(
            "gw_consensus_matrix: POT barycenter failed (%s); "
            "falling back to arithmetic mean.", exc,
        )
        bary = C_init

    np.fill_diagonal(bary, 0.0)
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
