"""analysis/rsa/rdm_utils.py – Cross-subject aggregate RDMs and optimal-sort utilities.

Provides
--------
aggregate_rdm(rdms, roi, state, method)
    Element-wise mean or median across subjects.

mean_rdm(...)   Thin backward-compat alias for aggregate_rdm(..., method='mean').

gw_consensus_matrix(rdms)
    Fréchet mean in GW space (ot.gromov_barycenters).

sorted_order(matrix)
    Linkage optimal reordering; returns (order, best_k, best_score).

sorted_order_within_category(matrix, labels)
    Performs clustering independently within animacy boundaries.
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
    return aggregate_rdm(rdms, roi_or_layer, state, method="mean")


# ── GW-barycenter consensus matrix ──────────────────────────────────────────

def gw_consensus_matrix(
    rdms: list[RDM],
    loss_fun: str = "square_loss",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    import logging
    import ot

    _log = logging.getLogger(__name__)
    n = rdms[0].matrix.shape[0]
    N = len(rdms)

    def _sanitise(mat: np.ndarray) -> np.ndarray | None:
        m = mat.astype(np.float64)
        col_means = np.nanmean(m, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask  = np.isnan(m)
        m[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        m = (m + m.T) / 2.0
        np.clip(m, 0.0, None, out=m)
        np.fill_diagonal(m, 0.0)
        mx = m.max()
        if mx < 1e-8:
            return None
        m /= mx
        if not np.isfinite(m).all():
            return None
        return m

    Cs_raw = [_sanitise(r.matrix) for r in rdms]
    Cs     = [c for c in Cs_raw if c is not None]
    n_dropped = N - len(Cs)
    if n_dropped:
        _log.warning(
            "gw_consensus_matrix: dropped %d degenerate subject RDM(s)", n_dropped
        )
    if len(Cs) < 2:
        safe = [c for c in Cs_raw if c is not None] or [np.zeros((n, n))]
        bary = np.mean(safe, axis=0)
        np.fill_diagonal(bary, 0.0)
        return bary

    n_use = Cs[0].shape[0]
    N_use = len(Cs)

    ps      = [np.ones(n_use) / n_use for _ in range(N_use)]
    weights = [1.0 / N_use] * N_use
    C_init  = np.mean(Cs, axis=0)

    gw_kwargs: dict = dict(
        N=n_use, Cs=Cs, ps=ps, p=np.ones(n_use) / n_use, lambdas=weights,
        loss_fun=loss_fun, max_iter=max_iter, tol=tol, verbose=False,
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
        _log.warning("gw_consensus_matrix: POT barycenter failed (%s); falling back to mean.", exc)
        bary = C_init

    np.fill_diagonal(bary, 0.0)
    mx = bary.max()
    if mx > 1e-8:
        bary /= mx
    return bary


# ── Optimal sorts ───────────────────────────────────────────────────────────

def sorted_order(
    matrix: np.ndarray,
    k_min: int = 3,
    k_max: int = 40,
    method: str = "ward",
) -> tuple[np.ndarray, int, float]:
    """Determine optimal row/col reordering using specified linkage method."""
    n = matrix.shape[0]
    k_max = min(k_max, n - 1)
    k_min = max(k_min, 2)

    if n <= 2:
        return np.arange(n), 1, 0.0

    condensed = squareform(matrix, checks=False)
    condensed = np.clip(condensed, 0.0, None)
    Z = linkage(condensed, method=method)

    best_score  = -np.inf
    best_labels = np.zeros(n, dtype=int)
    best_k      = k_min

    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, k, criterion="maxclust") - 1
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = silhouette_score(matrix, labels, metric="precomputed")
        except ValueError:
            score = -1.0

        if score > best_score:
            best_score  = score
            best_labels = labels
            best_k      = k

    order = np.argsort(best_labels, kind="stable")
    return order, best_k, float(best_score)


def sorted_order_within_category(
    matrix: np.ndarray,
    labels: np.ndarray,
    k_min: int = 2,
    k_max: int = 40,
    method: str = "ward",
) -> tuple[np.ndarray, int, float]:
    """Perform clustering independently within animacy categories."""
    anim_idx = np.where(labels == 1)[0]
    inan_idx = np.where(labels == 0)[0]

    best_k_total = 0
    score_total = 0.0
    combined_order = []

    for idx in (anim_idx, inan_idx):
        if len(idx) > 2:
            sub_mat = matrix[np.ix_(idx, idx)]
            sub_order, sub_k, sub_score = sorted_order(sub_mat, k_min=k_min, k_max=k_max, method=method)
            combined_order.extend(idx[sub_order])
            best_k_total += sub_k
            score_total += sub_score * len(idx)
        else:
            combined_order.extend(idx)
            best_k_total += 1

    n = len(labels)
    best_score = score_total / n if n > 0 else 0.0
    return np.array(combined_order), best_k_total, best_score