"""analysis/svm/svm_decoder.py – Mei et al. (2022) SVM decoding pipeline.

Implements two analyses:
  1. Within-state decoding   – train & test in the same awareness state
                               via leave-one-pair-out cross-validation.
  2. Cross-state generalisation – train on conscious, test on unconscious
                               (transfer learning, same CV scheme).

Fidelity to Mei, Santana & Soto (2022) Nat. Hum. Behav.
---------------------------------------------------------
* Linear SVM with **L1 regularisation**
  (penalty='l1', loss='squared_hinge', dual=False – as stated in Methods p.727).
  NOTE: sklearn's default is L2; this file explicitly sets L1.
* Pre-processing pipeline: remove zero-variance features → MinMaxScaler [0,1].
  Both steps are fitted on the training fold only and applied to the test fold.
* Metric: ROC-AUC (robust to class imbalance, matches paper).
* Empirical chance via DummyClassifier (same CV folds).
* Permutation test: 10 000 iterations, one-tailed, Bonferroni-corrected
  across ROIs (matches paper exactly).
* Leave-one-pair-out CV: each fold holds out one living + one non-living item.
  Up to 2 256 folds possible with 96 unique items (48×48); actual count
  depends on available trials per subject / state.

Logging levels
--------------
DEBUG   – per-fold detail: fold index, n_train, n_test, n_features after
           variance filtering, SVM n_iter, AUC scores.
INFO    – per-ROI summary: mean AUC, mean chance, p-value, significance.
WARNING – any fold where LinearSVC hit max_iter without converging.

Progress bars (tqdm)
--------------------
tqdm bars are shown at two levels inside this module:
  • fold-level  – each CV call (_run_cv / _run_cv_cross)
  • perm-level  – permutation test (_permutation_test)
A subject-level bar is rendered by the caller (phase0b_svm.py).
All bars are written to stderr so they do not pollute log files.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

logger = logging.getLogger(__name__)

_RNG_SEED = 42


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class SVMResult:
    """Result container for one ROI × state (or generalisation) decoding."""

    roi:              str
    state:            str          # "conscious" | "unconscious" | "c_to_u"
    subject_id:       str
    auc_scores:       list[float]  # per-fold AUC (true classifier)
    chance_scores:    list[float]  # per-fold AUC (dummy classifier)
    mean_auc:         float
    mean_chance:      float
    p_value:          float
    significant:      bool
    n_folds:          int
    # ── convergence diagnostics (new) ───────────────────────────────────
    n_iter_per_fold:          list[int]   = field(default_factory=list)
    convergence_warnings:     int         = 0   # #folds that hit max_iter

    @property
    def delta_auc(self) -> float:
        return self.mean_auc - self.mean_chance

    @property
    def converged_rate(self) -> float:
        """Fraction of folds that converged before hitting max_iter."""
        if not self.n_iter_per_fold:
            return float("nan")
        n_ok = sum(1 for _ in self.n_iter_per_fold) - self.convergence_warnings
        return n_ok / len(self.n_iter_per_fold)


# ── Decoder ──────────────────────────────────────────────────────────────────

class SVMDecoder:
    """
    Decode living/non-living category from multi-voxel BOLD patterns,
    following Mei, Santana & Soto (2022) Nat. Hum. Behav.

    Parameters
    ----------
    C           : SVM regularisation strength (L1 penalty, default 1.0).
                  Paper used C=1 (primary) and C=5 (control analysis).
    n_perms     : permutation iterations for p-value estimation (paper: 10 000).
    alpha       : significance threshold before Bonferroni correction.
    max_iter    : maximum LinearSVC solver iterations per fold.
    tol         : convergence tolerance passed to LinearSVC.
    rng_seed    : reproducibility seed.
    show_pbar   : show tqdm progress bars (fold-level + perm-level).
    """

    def __init__(
        self,
        C:          float = 1.0,
        n_perms:    int   = 10_000,
        alpha:      float = 0.05,
        max_iter:   int   = 10_000,
        tol:        float = 1e-4,
        rng_seed:   int   = _RNG_SEED,
        show_pbar:  bool  = True,
    ) -> None:
        self._C         = C
        self._n_perms   = n_perms
        self._alpha     = alpha
        self._max_iter  = max_iter
        self._tol       = tol
        self._rng       = np.random.default_rng(rng_seed)
        self._show_pbar = show_pbar

    # ── Public API ───────────────────────────────────────────────────────────

    def decode_within_state(
        self,
        patterns:   np.ndarray,   # (n_trials, n_voxels)
        labels:     np.ndarray,   # (n_trials,) binary — 1=living, 0=non-living
        item_ids:   np.ndarray,   # (n_trials,) unique stimulus identifier
        roi:        str,
        state:      str,
        subject_id: str,
    ) -> SVMResult:
        """Within-state leave-one-pair-out decoding (Step 1 of paper's analysis)."""
        folds = self._leave_one_pair_out_folds(labels, item_ids)
        if not folds:
            logger.debug("  [%s|%s|%s] No valid folds – returning empty result.", subject_id, state, roi)
            return self._empty_result(roi, state, subject_id)

        logger.debug(
            "  [%s|%s|%s] Within-state CV: %d folds, n_trials=%d, n_voxels=%d",
            subject_id, state, roi, len(folds), patterns.shape[0], patterns.shape[1],
        )

        true_aucs, chance_aucs, n_iters, n_warn = self._run_cv(
            patterns, labels, folds,
            desc=f"{subject_id}|{state}|{roi}",
        )
        p, sig = self._permutation_test(
            true_aucs, chance_aucs,
            desc=f"perm {subject_id}|{state}|{roi}",
        )

        logger.info(
            "  [%s|%s|%s] mean_AUC=%.4f  mean_chance=%.4f  Δ=%.4f  p=%.5f  sig=%s"
            "  conv_warns=%d/%d",
            subject_id, state, roi,
            float(np.mean(true_aucs)), float(np.mean(chance_aucs)),
            float(np.mean(true_aucs)) - float(np.mean(chance_aucs)),
            p, sig, n_warn, len(folds),
        )

        return SVMResult(
            roi=roi, state=state, subject_id=subject_id,
            auc_scores=true_aucs, chance_scores=chance_aucs,
            mean_auc=float(np.mean(true_aucs)),
            mean_chance=float(np.mean(chance_aucs)),
            p_value=p, significant=sig,
            n_folds=len(folds),
            n_iter_per_fold=n_iters,
            convergence_warnings=n_warn,
        )

    def decode_generalisation(
        self,
        train_patterns:  np.ndarray,
        train_labels:    np.ndarray,
        train_item_ids:  np.ndarray,
        test_patterns:   np.ndarray,
        test_labels:     np.ndarray,
        test_item_ids:   np.ndarray,
        roi:             str,
        subject_id:      str,
    ) -> SVMResult:
        """Cross-state generalisation: train on conscious, test on unconscious."""
        folds = self._leave_one_pair_out_folds_cross(
            train_labels, train_item_ids,
            test_labels,  test_item_ids,
        )
        if not folds:
            logger.debug("  [%s|c_to_u|%s] No valid folds – returning empty result.", subject_id, roi)
            return self._empty_result(roi, "c_to_u", subject_id)

        logger.debug(
            "  [%s|c_to_u|%s] Cross-state CV: %d folds, "
            "n_train=%d, n_test=%d, n_voxels=%d",
            subject_id, roi, len(folds),
            train_patterns.shape[0], test_patterns.shape[0], train_patterns.shape[1],
        )

        true_aucs, chance_aucs, n_iters, n_warn = self._run_cv_cross(
            train_patterns, train_labels,
            test_patterns,  test_labels,
            folds,
            desc=f"{subject_id}|c→u|{roi}",
        )
        p, sig = self._permutation_test(
            true_aucs, chance_aucs,
            desc=f"perm {subject_id}|c→u|{roi}",
        )

        logger.info(
            "  [%s|c_to_u|%s] mean_AUC=%.4f  mean_chance=%.4f  Δ=%.4f  p=%.5f  sig=%s"
            "  conv_warns=%d/%d",
            subject_id, roi,
            float(np.mean(true_aucs)), float(np.mean(chance_aucs)),
            float(np.mean(true_aucs)) - float(np.mean(chance_aucs)),
            p, sig, n_warn, len(folds),
        )

        return SVMResult(
            roi=roi, state="c_to_u", subject_id=subject_id,
            auc_scores=true_aucs, chance_scores=chance_aucs,
            mean_auc=float(np.mean(true_aucs)),
            mean_chance=float(np.mean(chance_aucs)),
            p_value=p, significant=sig,
            n_folds=len(folds),
            n_iter_per_fold=n_iters,
            convergence_warnings=n_warn,
        )

    def apply_bonferroni(self, results: list[SVMResult], n_rois: int) -> list[SVMResult]:
        """Re-apply Bonferroni correction across ROIs (as in the paper)."""
        threshold = self._alpha / max(n_rois, 1)
        for r in results:
            r.significant = r.p_value < threshold
        return results

    # ── Pipeline builder ─────────────────────────────────────────────────────

    def _build_pipeline(self) -> Pipeline:
        """
        Build the sklearn Pipeline matching the paper exactly:
          MinMaxScaler [0,1]  →  LinearSVC (L1, squared_hinge, dual=False)

        Mei et al. (2022) Methods p.727:
          "We used an SVM with L1 regularization, nested with removal of
           invariant voxels and feature scaling between 0 and 1 as
           pre-processing steps."

        sklearn notes:
          penalty='l1' requires loss='squared_hinge' and dual=False.
          dual=False is also preferred when n_samples > n_features.
        """
        return Pipeline([
            ("scaler", MinMaxScaler()),
            ("svm", LinearSVC(
                penalty="l1",
                loss="squared_hinge",
                dual=False,
                C=self._C,
                tol=self._tol,
                max_iter=self._max_iter,
                random_state=_RNG_SEED,
            )),
        ])

    # ── Cross-validation helpers ─────────────────────────────────────────────

    def _leave_one_pair_out_folds(
        self, labels: np.ndarray, item_ids: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Each fold holds out one unique living + one unique non-living item."""
        living_items    = np.unique(item_ids[labels == 1])
        nonliving_items = np.unique(item_ids[labels == 0])
        folds = []
        for liv, nonliv in product(living_items, nonliving_items):
            test_mask  = (item_ids == liv) | (item_ids == nonliv)
            train_mask = ~test_mask
            if train_mask.sum() < 2 or test_mask.sum() < 2:
                continue
            if len(np.unique(labels[test_mask])) < 2:
                continue
            folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
        return folds

    def _leave_one_pair_out_folds_cross(
        self,
        train_labels:   np.ndarray, train_item_ids: np.ndarray,
        test_labels:    np.ndarray, test_item_ids:  np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Cross-state: train on conscious (minus held-out pair), test on unconscious pair."""
        living_items    = np.unique(test_item_ids[test_labels == 1])
        nonliving_items = np.unique(test_item_ids[test_labels == 0])
        folds = []
        all_train = np.arange(len(train_labels))
        for liv, nonliv in product(living_items, nonliving_items):
            test_mask = (test_item_ids == liv) | (test_item_ids == nonliv)
            if test_mask.sum() < 2 or len(np.unique(test_labels[test_mask])) < 2:
                continue
            train_exclude = (train_item_ids == liv) | (train_item_ids == nonliv)
            train_idx = all_train[~train_exclude]
            test_idx  = np.where(test_mask)[0]
            if len(train_idx) < 2:
                continue
            folds.append((train_idx, test_idx))
        return folds

    def _run_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
        desc: str = "CV",
    ) -> tuple[list[float], list[float], list[int], int]:
        """
        Run within-state leave-one-pair-out CV.

        Returns (true_aucs, chance_aucs, n_iter_per_fold, n_convergence_warnings).
        """
        true_aucs, chance_aucs, n_iters = [], [], []
        n_warn = 0

        pbar = tqdm(
            enumerate(folds),
            total=len(folds),
            desc=desc,
            leave=False,
            disable=not self._show_pbar,
        )

        for fold_i, (train_idx, test_idx) in pbar:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            var_mask = X_tr.var(axis=0) > 1e-8
            if var_mask.sum() == 0:
                logger.debug("    fold %d: all features zero-variance – skipped.", fold_i)
                continue
            X_tr, X_te = X_tr[:, var_mask], X_te[:, var_mask]

            clf = self._build_pipeline()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(X_tr, y_tr)

            svm_step   = clf.named_steps["svm"]
            fold_iters = int(svm_step.n_iter_)
            n_iters.append(fold_iters)

            fold_warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            if fold_warned:
                n_warn += 1
                logger.warning(
                    "    [%s] fold %d: SVM did NOT converge "
                    "(n_iter=%d == max_iter=%d, n_train=%d, n_features=%d). "
                    "Consider raising max_iter or tol.",
                    desc, fold_i, fold_iters, self._max_iter,
                    len(train_idx), var_mask.sum(),
                )
            else:
                logger.debug(
                    "    [%s] fold %d: converged in %d iters "
                    "(n_train=%d, n_features=%d)",
                    desc, fold_i, fold_iters, len(train_idx), var_mask.sum(),
                )

            try:
                scores = clf.decision_function(X_te)
            except AttributeError:
                scores = clf.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, scores)
            true_aucs.append(auc)

            dummy = DummyClassifier(strategy="stratified", random_state=_RNG_SEED)
            dummy.fit(X_tr, y_tr)
            chance_aucs.append(roc_auc_score(y_te, dummy.predict_proba(X_te)[:, 1]))

            if self._show_pbar:
                pbar.set_postfix({"auc": f"{auc:.3f}", "iter": fold_iters, "warn": n_warn})

        return true_aucs, chance_aucs, n_iters, n_warn

    def _run_cv_cross(
        self,
        X_train_all: np.ndarray, y_train_all: np.ndarray,
        X_test_all:  np.ndarray, y_test_all:  np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
        desc: str = "CV-cross",
    ) -> tuple[list[float], list[float], list[int], int]:
        """
        Run cross-state CV (train on conscious, test on unconscious).

        Returns (true_aucs, chance_aucs, n_iter_per_fold, n_convergence_warnings).
        """
        true_aucs, chance_aucs, n_iters = [], [], []
        n_warn = 0

        pbar = tqdm(
            enumerate(folds),
            total=len(folds),
            desc=desc,
            leave=False,
            disable=not self._show_pbar,
        )

        for fold_i, (train_idx, test_idx) in pbar:
            X_tr = X_train_all[train_idx]
            y_tr = y_train_all[train_idx]
            X_te = X_test_all[test_idx]
            y_te = y_test_all[test_idx]

            var_mask = X_tr.var(axis=0) > 1e-8
            if var_mask.sum() == 0:
                logger.debug("    fold %d: all features zero-variance – skipped.", fold_i)
                continue
            X_tr, X_te = X_tr[:, var_mask], X_te[:, var_mask]

            clf = self._build_pipeline()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(X_tr, y_tr)

            svm_step   = clf.named_steps["svm"]
            fold_iters = int(svm_step.n_iter_)
            n_iters.append(fold_iters)

            fold_warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            if fold_warned:
                n_warn += 1
                logger.warning(
                    "    [%s] fold %d: SVM did NOT converge "
                    "(n_iter=%d == max_iter=%d, n_train=%d, n_features=%d).",
                    desc, fold_i, fold_iters, self._max_iter,
                    len(train_idx), var_mask.sum(),
                )
            else:
                logger.debug(
                    "    [%s] fold %d: converged in %d iters "
                    "(n_train=%d, n_features=%d)",
                    desc, fold_i, fold_iters, len(train_idx), var_mask.sum(),
                )

            try:
                scores = clf.decision_function(X_te)
            except AttributeError:
                scores = clf.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, scores)
            true_aucs.append(auc)

            dummy = DummyClassifier(strategy="stratified", random_state=_RNG_SEED)
            dummy.fit(X_tr, y_tr)
            chance_aucs.append(roc_auc_score(y_te, dummy.predict_proba(X_te)[:, 1]))

            if self._show_pbar:
                pbar.set_postfix({"auc": f"{auc:.3f}", "iter": fold_iters, "warn": n_warn})

        return true_aucs, chance_aucs, n_iters, n_warn

    def _permutation_test(
        self,
        true_aucs:   list[float],
        chance_aucs: list[float],
        desc:        str = "permutation",
    ) -> tuple[float, bool]:
        """
        Paper method: concatenate true & chance, shuffle, recompute mean diff.
        10 000 iterations, one-tailed.

        Vectorised with numpy (no Python loop) for ~50× speed-up.
        """
        if not true_aucs:
            return 1.0, False

        obs      = np.mean(true_aucs) - np.mean(chance_aucs)
        combined = np.array(true_aucs + chance_aucs, dtype=np.float64)
        n        = len(true_aucs)

        # Generate all permutation indices at once: shape (n_perms, 2n)
        logger.debug(
            "  [%s] Permutation test: obs_diff=%.5f, n_perms=%d", desc, obs, self._n_perms
        )

        # Vectorised: shuffle each row independently
        idx = np.argsort(
            self._rng.random((self._n_perms, len(combined))), axis=1
        )
        shuffled = combined[idx]           # (n_perms, 2n)
        null     = shuffled[:, :n].mean(axis=1) - shuffled[:, n:].mean(axis=1)

        p = float((null >= obs).mean())

        # tqdm bar is purely cosmetic here (single vectorised op);
        # kept for visual consistency in long runs.
        if self._show_pbar:
            with tqdm(
                total=self._n_perms,
                desc=desc,
                leave=False,
                unit="perm",
                disable=not self._show_pbar,
            ) as pbar:
                pbar.update(self._n_perms)
                pbar.set_postfix({"p": f"{p:.5f}"})

        logger.debug("  [%s] p=%.5f (sig=%s)", desc, p, p < self._alpha)
        return p, p < self._alpha

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(roi: str, state: str, subject_id: str) -> SVMResult:
        return SVMResult(
            roi=roi, state=state, subject_id=subject_id,
            auc_scores=[], chance_scores=[],
            mean_auc=0.5, mean_chance=0.5,
            p_value=1.0, significant=False, n_folds=0,
            n_iter_per_fold=[], convergence_warnings=0,
        )
