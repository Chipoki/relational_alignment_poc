"""analysis/svm/svm_decoder.py – Mei et al. (2022) SVM decoding pipeline.

Implements two analyses:
  1. Within-state decoding   – train & test in the same awareness state
                               via leave-one-pair-out cross-validation.
  2. Cross-state generalisation – train on conscious, test on unconscious
                               (transfer learning, same CV scheme).

Fidelity to Mei, Santana & Soto (2022) Nat. Hum. Behav.
---------------------------------------------------------
* Linear SVM with L1 regularisation.
* Pre-processing pipeline: remove zero-variance features → MinMaxScaler [0,1].
* Metric: ROC-AUC.
* Empirical chance via DummyClassifier.
* Permutation test: 10 000 iterations, one-tailed, Bonferroni-corrected.
* Leave-one-pair-out CV.
"""
from __future__ import annotations

import logging
import warnings
import contextlib
import sys
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np
import joblib
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

logger = logging.getLogger(__name__)

_RNG_SEED = 42

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback


@dataclass
class SVMResult:
    roi:              str
    state:            str
    subject_id:       str
    auc_scores:       list[float]
    chance_scores:    list[float]
    mean_auc:         float
    mean_chance:      float
    p_value:          float
    significant:      bool
    n_folds:          int
    n_iter_per_fold:      list[int] = field(default_factory=list)
    convergence_warnings: int       = 0

    @property
    def delta_auc(self) -> float:
        return self.mean_auc - self.mean_chance

    @property
    def converged_rate(self) -> float:
        if not self.n_iter_per_fold:
            return float("nan")
        n_ok = sum(1 for _ in self.n_iter_per_fold) - self.convergence_warnings
        return n_ok / len(self.n_iter_per_fold)


class SVMDecoder:
    def __init__(
        self,
        C:          float = 1.0,
        n_perms:    int   = 10_000,
        alpha:      float = 0.05,
        max_iter:   int   = 10_000,
        tol:        float = 1e-4,
        rng_seed:   int   = _RNG_SEED,
        show_pbar:  bool  = True,
        n_jobs:     int   = -1,   # Added n_jobs for parallel execution
    ) -> None:
        self._C         = C
        self._n_perms   = n_perms
        self._alpha     = alpha
        self._max_iter  = max_iter
        self._tol       = tol
        self._rng       = np.random.default_rng(rng_seed)
        self._show_pbar = show_pbar
        self._n_jobs    = n_jobs

    def decode_within_state(self, patterns, labels, item_ids, roi, state, subject_id) -> SVMResult:
        folds = self._leave_one_pair_out_folds(labels, item_ids)
        if not folds:
            logger.debug("  [%s|%s|%s] No valid folds – returning empty result.", subject_id, state, roi)
            return self._empty_result(roi, state, subject_id)

        logger.debug(
            "  [%s|%s|%s] Within-state CV: %d folds, n_trials=%d, n_voxels=%d",
            subject_id, state, roi, len(folds), patterns.shape[0], patterns.shape[1],
        )

        true_aucs, chance_aucs, n_iters, n_warn = self._run_parallel_cv(
            patterns, labels, patterns, labels, folds, desc=f"{subject_id}|{state}|{roi}"
        )
        p, sig = self._permutation_test(true_aucs, chance_aucs, desc=f"perm {subject_id}|{state}|{roi}")

        logger.info(
            "  [%s|%s|%s] mean_AUC=%.4f  mean_chance=%.4f  Δ=%.4f  p=%.5f  sig=%s  conv_warns=%d/%d",
            subject_id, state, roi, np.mean(true_aucs), np.mean(chance_aucs),
            np.mean(true_aucs) - np.mean(chance_aucs), p, sig, n_warn, len(folds),
        )

        return SVMResult(
            roi=roi, state=state, subject_id=subject_id,
            auc_scores=true_aucs, chance_scores=chance_aucs,
            mean_auc=float(np.mean(true_aucs)), mean_chance=float(np.mean(chance_aucs)),
            p_value=p, significant=sig, n_folds=len(folds),
            n_iter_per_fold=n_iters, convergence_warnings=n_warn,
        )

    def decode_generalisation(self, train_patterns, train_labels, train_item_ids,
                              test_patterns, test_labels, test_item_ids, roi, subject_id) -> SVMResult:
        folds = self._leave_one_pair_out_folds_cross(train_labels, train_item_ids, test_labels, test_item_ids)
        if not folds:
            logger.debug("  [%s|c_to_u|%s] No valid folds – returning empty result.", subject_id, roi)
            return self._empty_result(roi, "c_to_u", subject_id)

        logger.debug(
            "  [%s|c_to_u|%s] Cross-state CV: %d folds, n_train=%d, n_test=%d, n_voxels=%d",
            subject_id, roi, len(folds), train_patterns.shape[0], test_patterns.shape[0], train_patterns.shape[1],
        )

        true_aucs, chance_aucs, n_iters, n_warn = self._run_parallel_cv(
            train_patterns, train_labels, test_patterns, test_labels, folds, desc=f"{subject_id}|c→u|{roi}"
        )
        p, sig = self._permutation_test(true_aucs, chance_aucs, desc=f"perm {subject_id}|c→u|{roi}")

        logger.info(
            "  [%s|c_to_u|%s] mean_AUC=%.4f  mean_chance=%.4f  Δ=%.4f  p=%.5f  sig=%s  conv_warns=%d/%d",
            subject_id, roi, np.mean(true_aucs), np.mean(chance_aucs),
            np.mean(true_aucs) - np.mean(chance_aucs), p, sig, n_warn, len(folds),
        )

        return SVMResult(
            roi=roi, state="c_to_u", subject_id=subject_id,
            auc_scores=true_aucs, chance_scores=chance_aucs,
            mean_auc=float(np.mean(true_aucs)), mean_chance=float(np.mean(chance_aucs)),
            p_value=p, significant=sig, n_folds=len(folds),
            n_iter_per_fold=n_iters, convergence_warnings=n_warn,
        )

    def apply_bonferroni(self, results: list[SVMResult], n_rois: int) -> list[SVMResult]:
        threshold = self._alpha / max(n_rois, 1)
        for r in results:
            r.significant = r.p_value < threshold
        return results

    def _leave_one_pair_out_folds(self, labels, item_ids):
        living_items = np.unique(item_ids[labels == 1])
        nonliving_items = np.unique(item_ids[labels == 0])
        folds = []
        for liv, nonliv in product(living_items, nonliving_items):
            test_mask = (item_ids == liv) | (item_ids == nonliv)
            train_mask = ~test_mask
            if train_mask.sum() < 2 or test_mask.sum() < 2 or len(np.unique(labels[test_mask])) < 2:
                continue
            folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
        return folds

    def _leave_one_pair_out_folds_cross(self, train_labels, train_item_ids, test_labels, test_item_ids):
        living_items = np.unique(test_item_ids[test_labels == 1])
        nonliving_items = np.unique(test_item_ids[test_labels == 0])
        folds = []
        all_train = np.arange(len(train_labels))
        for liv, nonliv in product(living_items, nonliving_items):
            test_mask = (test_item_ids == liv) | (test_item_ids == nonliv)
            if test_mask.sum() < 2 or len(np.unique(test_labels[test_mask])) < 2:
                continue
            train_exclude = (train_item_ids == liv) | (train_item_ids == nonliv)
            train_idx = all_train[~train_exclude]
            if len(train_idx) < 2:
                continue
            folds.append((train_idx, np.where(test_mask)[0]))
        return folds

    @staticmethod
    def _eval_fold(X_train_all, y_train_all, X_test_all, y_test_all, tr, te, C, tol, max_iter):
        """Stateless worker function designed to be efficiently executed in parallel by Joblib."""
        # Slice the arrays inside the worker to utilize joblib's shared memory
        X_tr, X_te = X_train_all[tr], X_test_all[te]
        y_tr, y_te = y_train_all[tr], y_test_all[te]

        var_mask = X_tr.var(axis=0) > 1e-8
        if var_mask.sum() == 0:
            return None

        X_tr, X_te = X_tr[:, var_mask], X_te[:, var_mask]

        clf = Pipeline([
            ("scaler", MinMaxScaler()),
            ("svm", LinearSVC(penalty="l1", loss="squared_hinge", dual=False,
                              C=C, tol=tol, max_iter=max_iter, random_state=_RNG_SEED)),
        ])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(X_tr, y_tr)

        svm_step = clf.named_steps["svm"]
        fold_iters = int(svm_step.n_iter_)
        fold_warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)

        try:
            scores = clf.decision_function(X_te)
        except AttributeError:
            scores = clf.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, scores)

        dummy = DummyClassifier(strategy="stratified", random_state=_RNG_SEED)
        dummy.fit(X_tr, y_tr)
        chance_auc = roc_auc_score(y_te, dummy.predict_proba(X_te)[:, 1])

        return {
            "true_auc": auc,
            "chance_auc": chance_auc,
            "n_iter": fold_iters,
            "warned": fold_warned,
            "n_features": var_mask.sum(),
            "train_size": len(y_tr)
        }

    def _run_parallel_cv(self, X_train_all, y_train_all, X_test_all, y_test_all, folds, desc="CV"):
        """Deduplicated, parallel execution of K-folds ensuring realtime TQDM mapping."""
        true_aucs, chance_aucs, n_iters = [], [], []
        n_warn = 0

        # Run folds in parallel
        with tqdm_joblib(tqdm(total=len(folds), desc=desc, leave=False,
                              disable=not self._show_pbar,
                              file=sys.stdout, ncols=100,
                              mininterval=0.5)) as pbar:
            results = Parallel(n_jobs=self._n_jobs, backend="threading")(
                delayed(self._eval_fold)(
                    X_train_all, y_train_all, X_test_all, y_test_all, tr, te,
                    self._C, self._tol, self._max_iter
                ) for tr, te in folds
            )

        # Reconstruct standard logging sequentially
        for fold_i, res in enumerate(results):
            if res is None:
                logger.debug("    [%s] fold %d: all features zero-variance – skipped.", desc, fold_i)
                continue

            true_aucs.append(res["true_auc"])
            chance_aucs.append(res["chance_auc"])
            n_iters.append(res["n_iter"])

            if res["warned"]:
                n_warn += 1
                # logger.warning(
                #     "    [%s] fold %d: SVM did NOT converge (n_iter=%d == max_iter=%d, n_train=%d, n_features=%d).",
                #     desc, fold_i, res["n_iter"], self._max_iter, res["train_size"], res["n_features"],
                # )
            else:
                logger.debug(
                    "    [%s] fold %d: converged in %d iters (n_train=%d, n_features=%d)",
                    desc, fold_i, res["n_iter"], res["train_size"], res["n_features"],
                )

        return true_aucs, chance_aucs, n_iters, n_warn

    def _permutation_test(self, true_aucs, chance_aucs, desc="permutation"):
        if not true_aucs:
            return 1.0, False

        obs = np.mean(true_aucs) - np.mean(chance_aucs)
        combined = np.array(true_aucs + chance_aucs, dtype=np.float64)
        n = len(true_aucs)

        # ── MEMORY-SAFE CHUNKED PERMUTATION ──
        null_dist = np.zeros(self._n_perms)
        chunk_size = 1000

        for i in range(0, self._n_perms, chunk_size):
            end = min(i + chunk_size, self._n_perms)
            size = end - i

            idx = np.argsort(self._rng.random((size, len(combined))), axis=1)
            shuffled = combined[idx]
            null_dist[i:end] = shuffled[:, :n].mean(axis=1) - shuffled[:, n:].mean(axis=1)
        # ─────────────────────────────────────

        p = float((null_dist >= obs).mean())

        if self._show_pbar:
            with tqdm(total=self._n_perms, desc=desc, leave=False, unit="perm",
                      disable=not self._show_pbar,
                      file=sys.stdout, ncols=100,
                      mininterval=0.5) as pbar:
                pbar.update(self._n_perms)

        logger.debug("  [%s] p=%.5f (sig=%s)", desc, p, p < self._alpha)
        return p, p < self._alpha

    @staticmethod
    def _empty_result(roi: str, state: str, subject_id: str) -> SVMResult:
        return SVMResult(
            roi=roi, state=state, subject_id=subject_id, auc_scores=[], chance_scores=[],
            mean_auc=0.5, mean_chance=0.5, p_value=1.0, significant=False, n_folds=0,
            n_iter_per_fold=[], convergence_warnings=0,
        )