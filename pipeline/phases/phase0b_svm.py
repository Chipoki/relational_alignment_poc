"""Phase 0.2 – SVM Decoding (Mei et al. 2022).

Executes immediately after FCNN fine-tuning (Phase 0.1) so that the SVM
baseline is established before any embedding / RDM work begins.

Runs two analyses per subject per ROI:
  1. Within-state decoding  (conscious, unconscious separately)
  2. Cross-state generalisation  (train=conscious → test=unconscious)

Produces:
  • Per-subject bar charts of AUC by ROI (within-state)
  • Cross-subject generalisation heatmap (conscious → unconscious)
  • Group summary bar chart (all states)
  • stats/phase0b_svm.json with all numbers

SVM fidelity to aroma_decoding_pipeline_v11.py
----------------------------------------------
The SVMDecoder constructed here uses the same hyperparameters as
build_mei_svm_pipeline() in v11:
    LinearSVC(penalty='l1', dual=False, tol=1e-3, max_iter=10000,
              class_weight='balanced', random_state=12345)
  preceded by VarianceThreshold() and StandardScaler().

All three hyperparameters (C, tol, random_state) are read from the
``rsa`` section of config.yaml so they can be changed in one place
without touching Python code:

    rsa:
      svm_C:            1.0
      svm_n_perms:      10000
      svm_tol:          1.0e-3
      svm_max_iter:     10000
      svm_random_state: 12345

Idempotency
-----------
If ``checkpoints/svm/phase0b_svm_results.pkl`` already exists the phase
loads it and returns immediately, skipping all SVM training.

Checkpointing
-------------
After the per-subject loop the complete ``all_results`` dict is serialised
to ``checkpoints/svm/phase0b_svm_results.pkl`` via pickle.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from config.settings import Settings
from analysis.svm.svm_decoder import SVMDecoder, SVMResult
from visualization.svm_plotter import SVMPlotter
from utils.io_utils import save_json

logger = logging.getLogger(__name__)

_CHECKPOINT_SUBDIR = "svm"
_CHECKPOINT_FILE   = "phase0b_svm_results.pkl"


def _checkpoint_path(settings: Settings) -> Path:
    ckpt_dir = settings.checkpoints_dir / _CHECKPOINT_SUBDIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / _CHECKPOINT_FILE


def run(
    settings: Settings,
    subjects: list,
    human_rdms: dict,   # passed for potential stimulus metadata; raw patterns via subjects
) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 0.2 – SVM Decoding  (Mei et al. 2022)")
    logger.info("=" * 60)

    # ── Idempotency guard ──────────────────────────────────────────────────
    ckpt = _checkpoint_path(settings)
    if ckpt.exists():
        logger.info(
            "SVM checkpoint found at %s – Phase 0.2 skipped (loading cached results).",
            ckpt,
        )
        with open(ckpt, "rb") as fh:
            return pickle.load(fh)

    # ── Build SVMDecoder with v11-matching hyperparameters ─────────────────
    rsa_cfg = settings.rsa  # dict from config.yaml rsa: section
    decoder = SVMDecoder(
        C=float(rsa_cfg.get("svm_C", 1.0)),
        n_perms=int(rsa_cfg.get("svm_n_perms", 10_000)),
        alpha=float(rsa_cfg.get("alpha", 0.05)),
        tol=float(rsa_cfg.get("svm_tol", 1e-3)),
        max_iter=int(rsa_cfg.get("svm_max_iter", 10_000)),
        rng_seed=int(rsa_cfg.get("svm_random_state", 12345)),
        n_jobs=4,           # safe default; see SVMDecoder docstring
        cache_dir=ckpt.parent,  # checkpoints/svm/
    )

    plotter   = SVMPlotter(settings)
    # Exclude the massive wholebrain array from per-ROI SVM decoding
    roi_names = [roi for roi in settings.active_roi_names if roi != "wholebrain"]

    all_results: dict[str, dict[str, list[SVMResult]]] = {}
    # structure: subject_id → state → list[SVMResult]

    for subj in subjects:
        sid = subj.subject_id
        all_results[sid] = {"conscious": [], "unconscious": [], "c_to_u": []}

        c_data = getattr(subj, "conscious",   None)
        u_data = getattr(subj, "unconscious", None)

        # ── Within-state decoding ───────────────────────────────────────────
        for state, vis_data in [("conscious", c_data), ("unconscious", u_data)]:
            if vis_data is None:
                continue
            for roi in roi_names:
                patterns = vis_data.bold_patterns.get(roi)
                if patterns is None or patterns.shape[0] < 4:
                    continue
                labels   = vis_data.labels
                item_ids = vis_data.stimulus_names

                result = decoder.decode_within_state(
                    patterns=patterns,
                    labels=labels,
                    item_ids=item_ids,
                    roi=roi,
                    state=state,
                    subject_id=sid,
                )
                all_results[sid][state].append(result)

            # Bonferroni correction across ROIs for this subject × state
            decoder.apply_bonferroni(
                all_results[sid][state], n_rois=len(roi_names)
            )
            logger.info(
                "  %s | %s | %d/%d ROIs significant (within-state)",
                sid, state,
                sum(r.significant for r in all_results[sid][state]),
                len(all_results[sid][state]),
            )

        # ── Cross-state generalisation ──────────────────────────────────────
        if c_data is not None and u_data is not None:
            for roi in roi_names:
                c_patterns = c_data.bold_patterns.get(roi)
                u_patterns = u_data.bold_patterns.get(roi)
                if c_patterns is None or u_patterns is None:
                    continue
                if c_patterns.shape[0] < 4 or u_patterns.shape[0] < 4:
                    continue

                result = decoder.decode_generalisation(
                    train_patterns=c_patterns,
                    train_labels=c_data.labels,
                    train_item_ids=c_data.stimulus_names,
                    test_patterns=u_patterns,
                    test_labels=u_data.labels,
                    test_item_ids=u_data.stimulus_names,
                    roi=roi,
                    subject_id=sid,
                )
                all_results[sid]["c_to_u"].append(result)

            decoder.apply_bonferroni(
                all_results[sid]["c_to_u"], n_rois=len(roi_names)
            )
            logger.info(
                "  %s | c→u | %d/%d ROIs significant (generalisation)",
                sid,
                sum(r.significant for r in all_results[sid]["c_to_u"]),
                len(all_results[sid]["c_to_u"]),
            )

        # ── Per-subject figures ─────────────────────────────────────────────
        for state in ("conscious", "unconscious"):
            plotter.plot_decoding_by_roi(
                results=all_results[sid][state] + all_results[sid].get("c_to_u", []),
                state=state,
                roi_names=roi_names,
                subject_id=sid,
                save_name=f"phase0b_svm_{sid}_{state}.png",
            )

    # ── Cross-subject generalisation heatmap ──────────────────────────────
    gen_by_subject = {
        sid: all_results[sid]["c_to_u"]
        for sid in all_results
        if all_results[sid]["c_to_u"]
    }
    if gen_by_subject:
        plotter.plot_generalisation_heatmap(
            results_by_subject=gen_by_subject,
            roi_names=roi_names,
        )
        logger.info("Saved generalisation heatmap.")

    # ── Group summary ─────────────────────────────────────────────────────
    plotter.plot_group_summary(
        all_results=all_results,
        roi_names=roi_names,
    )
    logger.info("Saved group summary bar chart.")

    # ── Save stats JSON ───────────────────────────────────────────────────
    json_out: dict = {}
    for sid, state_dict in all_results.items():
        json_out[sid] = {}
        for state, results in state_dict.items():
            json_out[sid][state] = [
                {
                    "roi":         r.roi,
                    "mean_auc":    round(r.mean_auc,    4),
                    "mean_chance": round(r.mean_chance, 4),
                    "delta_auc":   round(r.delta_auc,   4),
                    "p_value":     round(r.p_value,     5),
                    "significant": r.significant,
                    "n_folds":     r.n_folds,
                }
                for r in results
            ]
    save_json(json_out, settings.stats_dir / "phase0b_svm.json")

    # ── Persist checkpoint ────────────────────────────────────────────────
    with open(ckpt, "wb") as fh:
        pickle.dump(all_results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("SVM results checkpointed to %s", ckpt)

    logger.info("Phase 0.2 complete.\n")
    return all_results
