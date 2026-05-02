"""analysis/svm/svm_decoder.py – Mei et al. (2022) SVM decoding pipeline.

SVM & CV fidelity to aroma_decoding_pipeline_v11.py
-----------------------------------------------------
The following choices exactly match the reference script:

* **Pipeline**:  VarianceThreshold → StandardScaler → LinearSVC
  (v11 uses ``make_pipeline(VarianceThreshold(), StandardScaler(), svm)``).
  The original project used MinMaxScaler; this is corrected here.

* **SVM hyperparameters** (from v11 ``build_mei_svm_pipeline``):
    - penalty='l1', dual=False
    - tol=1e-3
    - random_state=12345   (v11 default; overridable via config ``svm_random_state``)
    - max_iter=10 000
    - class_weight='balanced'

* **Cross-validation**: leave-one-pair-out (one living × one non-living item
  held out per fold).  Identical fold structure to ``loo_partition()`` in v11,
  with the living/non-living label convention corrected to match v11's
  ``target_1d`` encoding (0 = Living, 1 = Nonliving).

* **Metric**: ROC-AUC on decision_function scores.

* **Permutation test**: 10 000 iterations, one-tailed, Bonferroni-corrected
  across ROIs.

Parallelism
-----------
* backend="loky" (process-based) — only safe backend for LinearSVC.
  See the detailed note in _run_parallel_cv().
* n_jobs capped at 4 by default (configurable); -1 is dangerous for large
  pattern matrices (RAM explosion from loky process copies).

Checkpointing
-------------
ROI-level .pkl files are written to checkpoints/svm/roi_cache/ after each
(subject, state, roi) triple completes, enabling crash-safe restarts.
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

_DEFAULT_RNG_SEED     = 12345  # matches aroma_decoding_pipeline_v11.py
_ROI_CACHE_SUBDIR     = "roi_cache"


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
# Stateless fold worker  (module-level for loky pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_fold(
    X_train_all, y_train_all,
    X_test_all,  y_test_all,
    tr, te,
    C: float, tol: float, max_iter: int, rng_seed: int,
):
    """
    Pure stateless worker.  Receives FULL arrays + index slices and does all
    slicing INSIDE the worker process.  Under loky each worker has its own
    private memory-mapped copy (via joblib's numpy mmap), so concurrent
    slicing is race-free.

    Pipeline: VarianceThreshold → StandardScaler → LinearSVC(l1, balanced)
    Matches aroma_decoding_pipeline_v11.build_mei_svm_pipeline().
    """
    X_tr = X_train_all[tr]
    X_te = X_test_all[te]
    y_tr = y_train_all[tr]
    y_te = y_test_all[te]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None

    clf = Pipeline([
        ("vt",     VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("svm",    LinearSVC(
            penalty="l1", loss="squared_hinge", dual=False,
            C=C, tol=tol, max_iter=max_iter,
            class_weight="balanced",
            random_state=rng_seed,
        )),
    ])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_tr, y_tr)

    fold_iters  = int(clf.named_steps["svm"].n_iter_)
    fold_warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)

    scores = clf.decision_function(X_te)
    try:
        auc = roc_auc_score(y_te, scores)
    except ValueError:
        auc = float("nan")

    return {
        "true_auc":   auc,
        "n_iter":     fold_iters,
        "warned":     fold_warned,
        "n_features": int(clf.named_steps["vt"].get_support().sum()),
        "train_size": len(y_tr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Decoder
# ─────────────────────────────────────────────────────────────────────────────

class SVMDecoder:
    """
    Implements Mei et al. (2022) SVM decoding with leave-one-pair-out CV,
    ROC-AUC scoring, permutation testing, and Bonferroni correction.
    """

    def __init__(
        self,
        C:           float = 1.0,
        n_perms:     int   = 10_000,
        alpha:       float = 0.05,
        max_iter:    int   = 10_000,
        tol:         float = 1e-3,
        rng_seed:    int   = _DEFAULT_RNG_SEED,
        n_jobs:      int   = 4,
        cache_dir:   Optional[Path] = None,
    ) -> None:
        self._C         = C
        self._n_perms   = n_perms
        self._alpha     = alpha
        self._max_iter  = max_iter
        self._tol       = tol
        self._rng_seed  = rng_seed
        self._n_jobs    = n_jobs
        self._cache_dir = cache_dir
        self._perm_call_count = 0  # unique seed per permutation call

    # ── Public API ────────────────────────────────────────────────────────────

    def decode_within_state(
        self, patterns, labels, item_ids, roi, state, subject_id,
    ) -> SVMResult:
        cached = self._load_roi_cache(subject_id, state, roi)
        if cached is not None:
            logger.info("  [%s|%s|%s] loaded from ROI cache.", subject_id, state, roi)
            return cached

        folds = self._leave_one_pair_out_folds(labels, item_ids)
        if not folds:
            logger.debug("  [%s|%s|%s] No valid folds.", subject_id, state, roi)
            return self._empty_result(roi, state, subject_id)

        logger.info(
            "  [%s|%s|%s] Within-state CV: %d folds | n_trials=%d | n_voxels=%d",
            subject_id, state, roi,
            len(folds), patterns.shape[0], patterns.shape[1],
        )

        true_aucs, n_iters, n_warn = self._run_parallel_cv(
            patterns, labels, patterns, labels, folds,
            desc=f"{subject_id}|{state}|{roi}",
        )

        chance_auc = 0.5  # stratified dummy baseline
        chance_aucs = [chance_auc] * len(true_aucs)
        p, sig = self._permutation_test(true_aucs)

        logger.info(
            "  [%s|%s|%s] mean_AUC=%.4f  chance=0.5  Δ=%.4f  p=%.5f  sig=%s  warns=%d/%d",
            subject_id, state, roi,
            np.mean(true_aucs) if true_aucs else float("nan"),
            (np.mean(true_aucs) - 0.5) if true_aucs else float("nan"),
            p, sig, n_warn, len(folds),
        )

        result = SVMResult(
            roi=roi, state=state, subject_id=subject_id,
            auc_scores=true_aucs, chance_scores=chance_aucs,
            mean_auc=float(np.mean(true_aucs)) if true_aucs else 0.5,
            mean_chance=0.5,
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
            logger.info("  [%s|c_to_u|%s] loaded from ROI cache.", subject_id, roi)
            return cached

        folds = self._leave_one_pair_out_folds_cross(
            train_labels, train_item_ids, test_labels, test_item_ids,
        )
        if not folds:
            logger.debug("  [%s|c_to_u|%s] No valid folds.", subject_id, roi)
            return self._empty_result(roi, "c_to_u", subject_id)

        logger.info(
            "  [%s|c_to_u|%s] Cross-state CV: %d folds | n_train=%d | n_test=%d | n_voxels=%d",
            subject_id, roi, len(folds),
            train_patterns.shape[0], test_patterns.shape[0], train_patterns.shape[1],
        )

        true_aucs, n_iters, n_warn = self._run_parallel_cv(
            train_patterns, train_labels, test_patterns, test_labels, folds,
            desc=f"{subject_id}|c→u|{roi}",
        )

        chance_aucs = [0.5] * len(true_aucs)
        p, sig = self._permutation_test(true_aucs)

        logger.info(
            "  [%s|c_to_u|%s] mean_AUC=%.4f  chance=0.5  Δ=%.4f  p=%.5f  sig=%s  warns=%d/%d",
            subject_id, roi,
            np.mean(true_aucs) if true_aucs else float("nan"),
            (np.mean(true_aucs) - 0.5) if true_aucs else float("nan"),
            p, sig, n_warn, len(folds),
        )

        result = SVMResult(
            roi=roi, state="c_to_u", subject_id=subject_id,
            auc_scores=true_aucs, chance_scores=chance_aucs,
            mean_auc=float(np.mean(true_aucs)) if true_aucs else 0.5,
            mean_chance=0.5,
            p_value=p, significant=sig, n_folds=len(folds),
            n_iter_per_fold=n_iters, convergence_warnings=n_warn,
        )
        self._save_roi_cache(result, subject_id, "c_to_u", roi)
        return result

    def apply_bonferroni(
        self, results: list[SVMResult], n_rois: int
    ) -> list[SVMResult]:
        threshold = self._alpha / max(n_rois, 1)
        for r in results:
            r.significant = r.p_value < threshold
        return results

    # ── ROI-level cache helpers ───────────────────────────────────────────────

    def _roi_cache_path(
        self, subject_id: str, state: str, roi: str
    ) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        d = Path(self._cache_dir) / _ROI_CACHE_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        safe_roi = roi.replace("/", "_").replace(" ", "_")
        return d / f"{subject_id}__{state}__{safe_roi}.pkl"

    def _load_roi_cache(
        self, subject_id: str, state: str, roi: str
    ) -> Optional[SVMResult]:
        p = self._roi_cache_path(subject_id, state, roi)
        if p is not None and p.exists():
            try:
                with open(p, "rb") as fh:
                    return pickle.load(fh)
            except Exception as exc:
                logger.warning("ROI cache read failed (%s) – recomputing.", exc)
        return None

    def _save_roi_cache(
        self, result: SVMResult, subject_id: str, state: str, roi: str
    ) -> None:
        p = self._roi_cache_path(subject_id, state, roi)
        if p is None:
            return
        try:
            with open(p, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            logger.warning("ROI cache write failed (%s) – continuing.", exc)

    # ── CV fold generation ───────────────────────────────────────────────────
    # Note: label convention matches aroma_decoding_pipeline_v11.py's loo_partition:
    #   y==0 → Living (matching v11's target_1d where 0=Living, 1=Nonliving)
    #   y==1 → Nonliving
    # The within-project convention (labels: 1=Living, 0=Nonliving) is the
    # reverse; however, AUC is symmetric with respect to class inversion so
    # the fold structure (which pairs are held out) is what matters, not which
    # class is labelled 0 or 1.  The fold generator below uses the actual
    # binary label values as keys, so it is agnostic to the convention.

    def _leave_one_pair_out_folds(self, labels, item_ids):
        """
        Matches v11's loo_partition: iterate every (living_item, nonliving_item)
        combination and hold that pair out as the test set.
        """
        labels    = np.asarray(labels)
        item_ids  = np.asarray(item_ids)
        # Determine which class value corresponds to living vs non-living
        unique_labels = np.unique(labels)
        if len(unique_labels) != 2:
            return []
        class_a, class_b = sorted(unique_labels)
        items_a = np.unique(item_ids[labels == class_a])
        items_b = np.unique(item_ids[labels == class_b])

        folds = []
        for ia, ib in product(items_a, items_b):
            test_mask  = (item_ids == ia) | (item_ids == ib)
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
        """Cross-state: test fold pairs from the test set; train uses all items
        except the held-out pair."""
        train_labels    = np.asarray(train_labels)
        train_item_ids  = np.asarray(train_item_ids)
        test_labels     = np.asarray(test_labels)
        test_item_ids   = np.asarray(test_item_ids)

        unique_labels = np.unique(test_labels)
        if len(unique_labels) != 2:
            return []
        class_a, class_b = sorted(unique_labels)
        items_a = np.unique(test_item_ids[test_labels == class_a])
        items_b = np.unique(test_item_ids[test_labels == class_b])

        all_train = np.arange(len(train_labels))
        folds = []
        for ia, ib in product(items_a, items_b):
            test_mask = (test_item_ids == ia) | (test_item_ids == ib)
            if test_mask.sum() < 2 or len(np.unique(test_labels[test_mask])) < 2:
                continue
            # Exclude overlapping items from train set (same items, different state)
            train_exclude = (train_item_ids == ia) | (train_item_ids == ib)
            train_idx     = all_train[~train_exclude]
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

        WHY loky and NOT threading
        --------------------------
        LinearSVC calls LIBLINEAR which releases the GIL.  Under "threading"
        multiple C threads share the same arrays and can corrupt heap memory
        via concurrent non-atomic numpy slicing.  loky spawns fully isolated
        subprocesses — no shared-memory races.

        WHY n_jobs=4 and NOT -1
        -----------------------
        Each loky worker deserialises the full pattern matrix on startup.
        With -1 on a 16-core machine, 16 copies of a 500 MB matrix = 8 GB
        instantaneous peak.  4 workers is the safe default.
        """
        true_aucs: list[float] = []
        n_iters:   list[int]   = []
        n_warn = 0

        logger.info(
            "    [%s] dispatching %d folds to %d loky workers …",
            desc, len(folds), self._n_jobs,
        )

        raw_results = Parallel(
            n_jobs=self._n_jobs,
            backend="loky",
            prefer="processes",
        )(
            delayed(_eval_fold)(
                X_train_all, y_train_all,
                X_test_all,  y_test_all,
                tr, te,
                self._C, self._tol, self._max_iter, self._rng_seed,
            )
            for tr, te in folds
        )

        for fold_i, res in enumerate(raw_results):
            if res is None or np.isnan(res["true_auc"]):
                logger.debug(
                    "    [%s] fold %d: skipped (no variance or single-class).", desc, fold_i
                )
                continue
            true_aucs.append(res["true_auc"])
            n_iters.append(res["n_iter"])
            if res["warned"]:
                n_warn += 1

        logger.info(
            "    [%s] %d folds collected | mean_AUC=%.4f | conv_warns=%d/%d",
            desc, len(true_aucs),
            float(np.mean(true_aucs)) if true_aucs else float("nan"),
            n_warn, len(folds),
        )

        del raw_results
        gc.collect()

        return true_aucs, n_iters, n_warn

    # ── Permutation test ─────────────────────────────────────────────────────

    def _permutation_test(self, true_aucs: list[float]) -> tuple[float, bool]:
        """
        One-sample, one-tailed permutation test against H0: AUC == 0.5.

        Computes the fraction of n_perms permutations (random sign flips around
        0.5) that produce a mean AUC >= the observed mean.  This is the standard
        approach when the null is a fixed chance level.

        Each call gets a unique RNG seed derived by XOR-ing the base seed with
        a call counter, ensuring reproducibility without shared state.
        """
        if not true_aucs:
            return 1.0, False

        self._perm_call_count += 1
        rng = np.random.default_rng(self._rng_seed ^ self._perm_call_count)

        obs      = float(np.mean(true_aucs)) - 0.5
        arr      = np.array(true_aucs, dtype=np.float64) - 0.5  # centre on null

        # Vectorised sign-flip permutation
        signs     = rng.choice([-1.0, 1.0], size=(self._n_perms, len(arr)))
        null_dist = (arr[np.newaxis, :] * signs).mean(axis=1)

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
