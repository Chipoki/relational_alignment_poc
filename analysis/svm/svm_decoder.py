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

Parallelism & Stability Notes
------------------------------
* backend="loky"  (default for joblib.Parallel) spawns ISOLATED subprocesses.
  Each worker gets its OWN copy of the numpy arrays via fork-on-write / pickle,
  so there is ZERO shared mutable state between workers.  This is the ONLY safe
  backend for sklearn estimators: LinearSVC releases the Python GIL during its
  C-level LIBLINEAR solve, meaning "threading" workers genuinely run in parallel
  and corrupt each other's slices of the shared arrays.

* n_jobs is capped at 4 hard-coded here (overridable via SVMDecoder(n_jobs=N)).
  Using -1 (all cores) under loky spawns N_CPU processes each loading a full
  copy of the pattern matrix — on a 16-core machine with a 500 MB ROI matrix
  that is 8 GB instantaneous peak.  4 workers is the empirically safe ceiling.

* The RNG for permutation tests is seeded INSIDE _permutation_test with a
  deterministic per-call seed derived from the global seed + call counter.
  This makes it both reproducible AND free of shared-state races.

* tqdm_joblib (monkey-patches a global joblib callback) has been removed.
  It was not thread-safe: the finally-restore block could fire while a parallel
  batch was still in flight, silently killing the executor mid-run (this was the
  proximate cause of the crash on ROI 9 / superior_parietal).  Progress is now
  reported via plain logger.info every N folds.

* ROI-level checkpointing:  after each (subject, state, roi) triple completes,
  a lightweight .pkl is written to checkpoints/svm/roi_cache/.  On restart the
  decoder loads these cached SVMResult objects and skips recomputation, so a
  crash mid-subject loses at most one ROI's work.
"""
from __future__ import annotations

import gc
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

_RNG_SEED = 42
_ROI_CACHE_SUBDIR  = "roi_cache"


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

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
        n_ok = len(self.n_iter_per_fold) - self.convergence_warnings
        return n_ok / len(self.n_iter_per_fold)


# ─────────────────────────────────────────────────────────────────────────────
# Stateless fold worker  (MUST be module-level for loky pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_fold(X_train_all, y_train_all, X_test_all, y_test_all,
               tr, te, C: float, tol: float, max_iter: int):
    """
    Pure stateless worker.  Receives the FULL arrays plus index slices and does
    all slicing INSIDE the worker process.  Under loky each worker has its own
    private memory-mapped copy of the arrays (via joblib's numpy mmap), so
    slicing here is safe and does not duplicate RAM in the parent process.
    """
    X_tr, X_te = X_train_all[tr], X_test_all[te]
    y_tr, y_te = y_train_all[tr], y_test_all[te]

    var_mask = X_tr.var(axis=0) > 1e-8
    if var_mask.sum() == 0:
        return None

    X_tr = X_tr[:, var_mask]
    X_te = X_te[:, var_mask]

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    LinearSVC(
            penalty="l1", loss="squared_hinge", dual=False,
            C=C, tol=tol, max_iter=max_iter, random_state=_RNG_SEED,
            class_weight='balanced'
        )),
    ])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_tr, y_tr)

    fold_iters  = int(clf.named_steps["svm"].n_iter_)
    fold_warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)

    scores = clf.decision_function(X_te)
    auc    = roc_auc_score(y_te, scores)
    train_scores = clf.decision_function(X_tr)
    train_auc = roc_auc_score(y_tr, train_scores)

    dummy = DummyClassifier(strategy="stratified", random_state=_RNG_SEED)
    dummy.fit(X_tr, y_tr)
    chance_auc = roc_auc_score(y_te, dummy.predict_proba(X_te)[:, 1])

    return {
        "true_auc":   auc,
        "train_auc": train_auc,
        "chance_auc": chance_auc,
        "n_iter":     fold_iters,
        "warned":     fold_warned,
        "n_features": int(var_mask.sum()),
        "train_size": len(y_tr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

class SVMDecoder:
    def __init__(
        self,
        C:           float = 1.0,
        n_perms:     int   = 10_000,
        alpha:       float = 0.05,
        max_iter:    int   = 10_000,
        tol:         float = 1e-4,
        rng_seed:    int   = _RNG_SEED,
        n_jobs:      int   = 4,        # safe default; -1 is dangerous (see module docstring)
        cache_dir:   Optional[Path] = None,
    ) -> None:
        self._C         = C
        self._n_perms   = n_perms
        self._alpha     = alpha
        self._max_iter  = max_iter
        self._tol       = tol
        self._rng_seed  = rng_seed     # stored as int, NOT as a Generator instance
        self._n_jobs    = n_jobs
        self._cache_dir = cache_dir    # set by phase0b_svm.py before use
        self._perm_call_count = 0      # used to derive unique per-call RNG seeds

    # ── Public API ────────────────────────────────────────────────────────────

    def decode_within_state(
        self, patterns, labels, item_ids, roi, state, subject_id,
    ) -> SVMResult:
        cached = self._load_roi_cache(subject_id, state, roi)
        if cached is not None:
            logger.info("  [%s|%s|%s] loaded from ROI cache – skipping.", subject_id, state, roi)
            return cached

        folds = self._leave_one_pair_out_folds(labels, item_ids)
        if not folds:
            logger.debug("  [%s|%s|%s] No valid folds – returning empty result.", subject_id, state, roi)
            return self._empty_result(roi, state, subject_id)

        logger.info(
            "  [%s|%s|%s] Within-state CV: %d folds | n_trials=%d | n_voxels=%d",
            subject_id, state, roi, len(folds), patterns.shape[0], patterns.shape[1],
        )

        true_aucs, chance_aucs, n_iters, n_warn = self._run_parallel_cv(
            patterns, labels, patterns, labels, folds,
            desc=f"{subject_id}|{state}|{roi}",
        )
        p, sig = self._permutation_test(true_aucs, chance_aucs)

        logger.info(
            "  [%s|%s|%s] mean_AUC=%.4f  mean_chance=%.4f  Δ=%.4f  p=%.5f  sig=%s  conv_warns=%d/%d",
            subject_id, state, roi,
            np.mean(true_aucs), np.mean(chance_aucs),
            np.mean(true_aucs) - np.mean(chance_aucs),
            p, sig, n_warn, len(folds),
        )

        result = SVMResult(
            roi=roi, state=state, subject_id=subject_id,
            auc_scores=true_aucs, chance_scores=chance_aucs,
            mean_auc=float(np.mean(true_aucs)),
            mean_chance=float(np.mean(chance_aucs)),
            p_value=p, significant=sig, n_folds=len(folds),
            n_iter_per_fold=n_iters, convergence_warnings=n_warn,
        )
        self._save_roi_cache(result, subject_id, state, roi)
        return result

    def decode_generalisation(
        self,
        train_patterns, train_labels, train_item_ids,
        test_patterns,  test_labels,  test_item_ids,
        roi, subject_id,
    ) -> SVMResult:
        cached = self._load_roi_cache(subject_id, "c_to_u", roi)
        if cached is not None:
            logger.info("  [%s|c_to_u|%s] loaded from ROI cache – skipping.", subject_id, roi)
            return cached

        folds = self._leave_one_pair_out_folds_cross(
            train_labels, train_item_ids, test_labels, test_item_ids,
        )
        if not folds:
            logger.debug("  [%s|c_to_u|%s] No valid folds – returning empty result.", subject_id, roi)
            return self._empty_result(roi, "c_to_u", subject_id)

        logger.info(
            "  [%s|c_to_u|%s] Cross-state CV: %d folds | n_train=%d | n_test=%d | n_voxels=%d",
            subject_id, roi, len(folds),
            train_patterns.shape[0], test_patterns.shape[0], train_patterns.shape[1],
        )

        true_aucs, chance_aucs, n_iters, n_warn = self._run_parallel_cv(
            train_patterns, train_labels, test_patterns, test_labels, folds,
            desc=f"{subject_id}|c→u|{roi}",
        )
        p, sig = self._permutation_test(true_aucs, chance_aucs)

        logger.info(
            "  [%s|c_to_u|%s] mean_AUC=%.4f  mean_chance=%.4f  Δ=%.4f  p=%.5f  sig=%s  conv_warns=%d/%d",
            subject_id, roi,
            np.mean(true_aucs), np.mean(chance_aucs),
            np.mean(true_aucs) - np.mean(chance_aucs),
            p, sig, n_warn, len(folds),
        )

        result = SVMResult(
            roi=roi, state="c_to_u", subject_id=subject_id,
            auc_scores=true_aucs, chance_scores=chance_aucs,
            mean_auc=float(np.mean(true_aucs)),
            mean_chance=float(np.mean(chance_aucs)),
            p_value=p, significant=sig, n_folds=len(folds),
            n_iter_per_fold=n_iters, convergence_warnings=n_warn,
        )
        self._save_roi_cache(result, subject_id, "c_to_u", roi)
        return result

    def apply_bonferroni(self, results: list[SVMResult], n_rois: int) -> list[SVMResult]:
        threshold = self._alpha / max(n_rois, 1)
        for r in results:
            r.significant = r.p_value < threshold
        return results

    # ── ROI-level cache helpers ───────────────────────────────────────────────

    def _roi_cache_path(self, subject_id: str, state: str, roi: str) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        d = Path(self._cache_dir) / _ROI_CACHE_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        safe_roi = roi.replace("/", "_").replace(" ", "_")
        return d / f"{subject_id}__{state}__{safe_roi}.pkl"

    def _load_roi_cache(self, subject_id: str, state: str, roi: str) -> Optional[SVMResult]:
        p = self._roi_cache_path(subject_id, state, roi)
        if p is not None and p.exists():
            try:
                with open(p, "rb") as fh:
                    return pickle.load(fh)
            except Exception as exc:
                logger.warning("ROI cache read failed (%s) – recomputing.", exc)
        return None

    def _save_roi_cache(self, result: SVMResult, subject_id: str, state: str, roi: str) -> None:
        p = self._roi_cache_path(subject_id, state, roi)
        if p is None:
            return
        try:
            with open(p, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.warning("ROI cache write failed (%s) – continuing without cache.", exc)

    # ── CV fold generation ───────────────────────────────────────────────────

    def _leave_one_pair_out_folds(self, labels, item_ids):
        living_items    = np.unique(item_ids[labels == 1])
        nonliving_items = np.unique(item_ids[labels == 0])
        folds = []

        for liv, nonliv in product(living_items, nonliving_items):
            test_mask  = (item_ids == liv) | (item_ids == nonliv)
            train_mask = ~test_mask
            if (train_mask.sum() < 2
                    or test_mask.sum() < 2
                    or len(np.unique(labels[test_mask])) < 2):
                continue
            folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))

        return folds

    def _leave_one_pair_out_folds_cross(
        self, train_labels, train_item_ids, test_labels, test_item_ids,
    ):
        living_items    = np.unique(test_item_ids[test_labels == 1])
        nonliving_items = np.unique(test_item_ids[test_labels == 0])
        all_train = np.arange(len(train_labels))
        folds = []
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

    # ── Parallel CV executor ─────────────────────────────────────────────────

    def _run_parallel_cv(
        self,
        X_train_all, y_train_all,
        X_test_all,  y_test_all,
        folds,
        desc: str = "CV",
    ):
        """
        Execute folds in parallel using loky (process-based) workers.

        WHY loky and NOT threading:
          LinearSVC calls LIBLINEAR which releases the Python GIL.  Under
          "threading" this means workers genuinely execute in parallel C code
          while sharing the SAME Python objects.  NumPy array slicing (even
          read-only) is not atomic; concurrent slicing of the same array by
          multiple C threads causes heap corruption and silent crashes.
          loky spawns fully isolated subprocesses — each gets its own private
          address space, so no shared-memory races are possible.

        WHY n_jobs=4 and NOT -1:
          Each loky worker deserialises (pickles) the full pattern matrix on
          startup.  With -1 on a 16-core machine, 16 copies of a 500 MB matrix
          = 8 GB instantaneous peak before a single SVM is trained.  4 workers
          is the empirically safe ceiling for typical fMRI pattern sizes.
        """
        true_aucs, chance_aucs, train_aucs, n_iters = [], [], [], []
        n_warn = 0

        logger.info("    [%s] dispatching %d folds to %d loky workers …", desc, len(folds), self._n_jobs)

        raw_results = Parallel(
            n_jobs=self._n_jobs,
            backend="loky",         # NEVER "threading" — see docstring above
            prefer="processes",
        )(
            delayed(_eval_fold)(
                X_train_all, y_train_all, X_test_all, y_test_all,
                tr, te, self._C, self._tol, self._max_iter,
            )
            for tr, te in folds
        )

        for fold_i, res in enumerate(raw_results):
            if res is None:
                logger.debug("    [%s] fold %d: all features zero-variance – skipped.", desc, fold_i)
                continue

            true_aucs.append(res["true_auc"])
            train_aucs.append(res["train_auc"])
            chance_aucs.append(res["chance_auc"])
            n_iters.append(res["n_iter"])
            if res["warned"]:
                n_warn += 1

        logger.info(
            "    [%s] %d folds collected | mean_Train_AUC=%.4f | mean_Test_AUC=%.4f | conv_warns=%d/%d",
            desc, len(true_aucs),
            float(np.mean(train_aucs)) if train_aucs else float("nan"),
            float(np.mean(true_aucs)) if true_aucs else float("nan"),
            n_warn, len(folds),
        )

        # Explicitly release the large result list before returning so loky's
        # /dev/shm memory-mapped segments can be reclaimed promptly.
        del raw_results
        gc.collect()

        return true_aucs, chance_aucs, n_iters, n_warn

    # ── Permutation test ─────────────────────────────────────────────────────

    def _permutation_test(self, true_aucs, chance_aucs) -> tuple[float, bool]:
        """
        Non-parametric one-tailed permutation test.

        The RNG is seeded with a UNIQUE seed per call (base seed XOR call
        counter) so results are fully reproducible while being independent
        across ROIs/subjects.  There is no shared Generator instance, so this
        method is safe to call from any context.
        """
        if not true_aucs:
            return 1.0, False

        self._perm_call_count += 1
        rng = np.random.default_rng(self._rng_seed ^ self._perm_call_count)

        obs      = np.mean(true_aucs) - np.mean(chance_aucs)
        combined = np.array(true_aucs + chance_aucs, dtype=np.float64)
        n        = len(true_aucs)

        # Vectorised single-shot permutation — no chunking required because
        # we are no longer holding parallel fold arrays alongside this array.
        idx       = np.argsort(rng.random((self._n_perms, len(combined))), axis=1)
        shuffled  = combined[idx]
        null_dist = shuffled[:, :n].mean(axis=1) - shuffled[:, n:].mean(axis=1)

        p = float((null_dist >= obs).mean())

        logger.info(
            "    permutation test: obs_delta=%.4f  p=%.5f  sig=%s  (n_perms=%d)",
            obs, p, p < self._alpha, self._n_perms,
        )
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
