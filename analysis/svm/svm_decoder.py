"""analysis/svm/svm_decoder.py – Mei et al. (2022) SVM decoding pipeline.

Implements two analyses:
  1. Within-state decoding   – train & test in the same awareness state
                               via leave-one-pair-out cross-validation.
  2. Cross-state generalisation – train on conscious, test on unconscious
                               (transfer learning, same CV scheme).

Fidelity to the paper
---------------------
* Linear SVM with L1 regularisation (C=1 default, controllable).
* Pre-processing pipeline: remove zero-variance features → MinMaxScaler [0,1].
  Both steps are fitted on the training fold only and applied to the test fold.
* Metric: ROC-AUC (robust to class imbalance, matches paper).
* Empirical chance via DummyClassifier (same CV folds).
* Permutation test: 10 000 iterations, one-tailed, Bonferroni-corrected
  across ROIs (matches paper exactly).
* Leave-one-pair-out CV: each fold holds out one living + one non-living item.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

_RNG_SEED = 42


@dataclass
class SVMResult:
    """Result container for one ROI × state (or generalisation) decoding."""

    roi: str
    state: str                    # "conscious" | "unconscious" | "c_to_u"
    subject_id: str
    auc_scores: list[float]       # per-fold AUC (true classifier)
    chance_scores: list[float]    # per-fold AUC (dummy classifier)
    mean_auc: float
    mean_chance: float
    p_value: float
    significant: bool
    n_folds: int

    @property
    def delta_auc(self) -> float:
        return self.mean_auc - self.mean_chance


class SVMDecoder:
    """
    Decode living/non-living category from multi-voxel BOLD patterns,
    following Mei, Santana & Soto (2022) Nat. Hum. Behav.

    Parameters
    ----------
    C           : SVM regularisation (L1 penalty, default 1.0)
    n_perms     : permutation iterations for p-value estimation
    alpha       : significance threshold (before Bonferroni)
    rng_seed    : reproducibility seed
    """

    def __init__(
        self,
        C: float = 1.0,
        n_perms: int = 10_000,
        alpha: float = 0.05,
        rng_seed: int = _RNG_SEED,
    ) -> None:
        self._C = C
        self._n_perms = n_perms
        self._alpha = alpha
        self._rng = np.random.default_rng(rng_seed)

    # ── Public API ──────────────────────────────────────────────────────────

    def decode_within_state(
        self,
        patterns: np.ndarray,   # (n_trials, n_voxels)
        labels: np.ndarray,     # (n_trials,) binary — 1=living, 0=non-living
        item_ids: np.ndarray,   # (n_trials,) unique stimulus identifier
        roi: str,
        state: str,
        subject_id: str,
    ) -> SVMResult:
        """
        Within-state leave-one-pair-out decoding (Step 1 of paper's analysis).
        """
        folds = self._leave_one_pair_out_folds(labels, item_ids)
        if not folds:
            return self._empty_result(roi, state, subject_id)

        true_aucs, chance_aucs = self._run_cv(patterns, labels, folds)
        p, sig = self._permutation_test(true_aucs, chance_aucs)

        return SVMResult(
            roi=roi, state=state, subject_id=subject_id,
            auc_scores=true_aucs, chance_scores=chance_aucs,
            mean_auc=float(np.mean(true_aucs)),
            mean_chance=float(np.mean(chance_aucs)),
            p_value=p, significant=sig,
            n_folds=len(folds),
        )

    def decode_generalisation(
        self,
        train_patterns: np.ndarray,  # (n_conscious, n_voxels)
        train_labels: np.ndarray,
        train_item_ids: np.ndarray,
        test_patterns: np.ndarray,   # (n_unconscious, n_voxels)
        test_labels: np.ndarray,
        test_item_ids: np.ndarray,
        roi: str,
        subject_id: str,
    ) -> SVMResult:
        """
        Cross-state generalisation: train on conscious, test on unconscious.
        Mirrors the paper's transfer-learning analysis.
        """
        folds = self._leave_one_pair_out_folds_cross(
            train_labels, train_item_ids,
            test_labels,  test_item_ids,
        )
        if not folds:
            return self._empty_result(roi, "c_to_u", subject_id)

        true_aucs, chance_aucs = self._run_cv_cross(
            train_patterns, train_labels,
            test_patterns, test_labels,
            folds,
        )
        p, sig = self._permutation_test(true_aucs, chance_aucs)

        return SVMResult(
            roi=roi, state="c_to_u", subject_id=subject_id,
            auc_scores=true_aucs, chance_scores=chance_aucs,
            mean_auc=float(np.mean(true_aucs)),
            mean_chance=float(np.mean(chance_aucs)),
            p_value=p, significant=sig,
            n_folds=len(folds),
        )

    def apply_bonferroni(self, results: list[SVMResult], n_rois: int) -> list[SVMResult]:
        """Re-apply Bonferroni correction across ROIs (as in the paper)."""
        threshold = self._alpha / max(n_rois, 1)
        for r in results:
            r.significant = r.p_value < threshold
        return results

    # ── Cross-validation helpers ────────────────────────────────────────────

    def _leave_one_pair_out_folds(
        self, labels: np.ndarray, item_ids: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Each fold: hold out one unique living item + one unique non-living item.
        Returns list of (train_idx, test_idx) tuples.
        """
        living_items    = np.unique(item_ids[labels == 1])
        nonliving_items = np.unique(item_ids[labels == 0])
        folds = []
        for liv, nonliv in product(living_items, nonliving_items):
            test_mask  = (item_ids == liv) | (item_ids == nonliv)
            train_mask = ~test_mask
            if train_mask.sum() < 2 or test_mask.sum() < 2:
                continue
            # Ensure both classes present in test
            if len(np.unique(labels[test_mask])) < 2:
                continue
            folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
        return folds

    def _leave_one_pair_out_folds_cross(
        self,
        train_labels: np.ndarray, train_item_ids: np.ndarray,
        test_labels: np.ndarray,  test_item_ids: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Cross-state: train on all conscious EXCEPT held-out pair,
        test on the unconscious counterparts of that held-out pair.
        """
        living_items    = np.unique(test_item_ids[test_labels == 1])
        nonliving_items = np.unique(test_item_ids[test_labels == 0])
        folds = []
        all_train = np.arange(len(train_labels))
        for liv, nonliv in product(living_items, nonliving_items):
            test_mask = (test_item_ids == liv) | (test_item_ids == nonliv)
            if test_mask.sum() < 2 or len(np.unique(test_labels[test_mask])) < 2:
                continue
            # Remove corresponding items from training set too
            train_exclude = (train_item_ids == liv) | (train_item_ids == nonliv)
            train_idx = all_train[~train_exclude]
            test_idx  = np.where(test_mask)[0]
            if len(train_idx) < 2:
                continue
            folds.append((train_idx, test_idx))
        return folds

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", MinMaxScaler()),
            ("svm",    LinearSVC(C=self._C, max_iter=5000, random_state=_RNG_SEED)),
        ])

    def _run_cv(
        self,
        X: np.ndarray, y: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[list[float], list[float]]:
        true_aucs, chance_aucs = [], []
        for train_idx, test_idx in folds:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # Remove zero-variance features (fitted on train only)
            var_mask = X_tr.var(axis=0) > 1e-8
            if var_mask.sum() == 0:
                continue
            X_tr, X_te = X_tr[:, var_mask], X_te[:, var_mask]

            clf  = self._build_pipeline()
            clf.fit(X_tr, y_tr)
            try:
                scores = clf.decision_function(X_te)
            except AttributeError:
                scores = clf.predict_proba(X_te)[:, 1]
            true_aucs.append(roc_auc_score(y_te, scores))

            dummy = DummyClassifier(strategy="stratified", random_state=_RNG_SEED)
            dummy.fit(X_tr, y_tr)
            chance_aucs.append(roc_auc_score(y_te, dummy.predict_proba(X_te)[:, 1]))

        return true_aucs, chance_aucs

    def _run_cv_cross(
        self,
        X_train_all: np.ndarray, y_train_all: np.ndarray,
        X_test_all:  np.ndarray, y_test_all:  np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[list[float], list[float]]:
        true_aucs, chance_aucs = [], []
        for train_idx, test_idx in folds:
            X_tr = X_train_all[train_idx]
            y_tr = y_train_all[train_idx]
            X_te = X_test_all[test_idx]
            y_te = y_test_all[test_idx]

            var_mask = X_tr.var(axis=0) > 1e-8
            if var_mask.sum() == 0:
                continue
            X_tr, X_te = X_tr[:, var_mask], X_te[:, var_mask]

            clf = self._build_pipeline()
            clf.fit(X_tr, y_tr)
            try:
                scores = clf.decision_function(X_te)
            except AttributeError:
                scores = clf.predict_proba(X_te)[:, 1]
            true_aucs.append(roc_auc_score(y_te, scores))

            dummy = DummyClassifier(strategy="stratified", random_state=_RNG_SEED)
            dummy.fit(X_tr, y_tr)
            chance_aucs.append(roc_auc_score(y_te, dummy.predict_proba(X_te)[:, 1]))

        return true_aucs, chance_aucs

    def _permutation_test(
        self,
        true_aucs: list[float],
        chance_aucs: list[float],
    ) -> tuple[float, bool]:
        """
        Paper method: concatenate true & chance, shuffle, recompute mean diff.
        10 000 iterations, one-tailed.
        """
        if not true_aucs:
            return 1.0, False
        obs = np.mean(true_aucs) - np.mean(chance_aucs)
        combined = np.array(true_aucs + chance_aucs)
        n = len(true_aucs)
        null = np.zeros(self._n_perms)
        for i in range(self._n_perms):
            shuffled = self._rng.permutation(combined)
            null[i]  = shuffled[:n].mean() - shuffled[n:].mean()
        p = float((null >= obs).mean())
        return p, p < self._alpha

    @staticmethod
    def _empty_result(roi: str, state: str, subject_id: str) -> SVMResult:
        return SVMResult(
            roi=roi, state=state, subject_id=subject_id,
            auc_scores=[], chance_scores=[],
            mean_auc=0.5, mean_chance=0.5,
            p_value=1.0, significant=False, n_folds=0,
        )
