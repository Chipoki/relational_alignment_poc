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
from typing import List

import numpy as np

from config.settings import Settings
from analysis.svm.svm_decoder import SVMDecoder, SVMResult
from data import Subject
from visualization.svm_plotter import SVMPlotter
from utils.io_utils import save_json
from utils.rdm_io import dump_svm_patterns, load_svm_patterns, list_dumped_subjects

logger = logging.getLogger(__name__)

_CHECKPOINT_SUBDIR = "svm"
_CHECKPOINT_FILE   = "phase0b_svm_results.pkl"


def _checkpoint_path(settings: Settings, subjects: List = None) -> Path:
    ckpt_dir = settings.checkpoints_dir / _CHECKPOINT_SUBDIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if subjects is not None:
        ckpt_file_name = _CHECKPOINT_FILE.split(".")[0]
        ckpt_file_suffix = _CHECKPOINT_FILE.split(".")[-1]
        ckpt_recon_file_name = f"{ckpt_file_name}_{'_'.join([sub.subject_id for sub in subjects])}.{ckpt_file_suffix}"
        ckpt_dir_path = ckpt_dir / ckpt_recon_file_name
    else:
        ckpt_dir_path = ckpt_dir / _CHECKPOINT_FILE
    return ckpt_dir_path


def run(
    settings: Settings,
    subjects: list,
    human_rdms: dict,   # passed for potential stimulus metadata; raw patterns via subjects
) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 0.2 – SVM Decoding  (Mei et al. 2022)")
    logger.info("=" * 60)

    # ── Idempotency guard ──────────────────────────────────────────────────
    ckpt = _checkpoint_path(settings, subjects)
    # add subject names to checkpoint filename to allow per-subject caching (optional)

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

        # ── Dump minimal SVM patterns for this subject (idempotent) ───────
        # Allows later runs with --from-subject-rdms to still run SVM
        # without loading BOLD again.
        rdm_dump_dir = settings.subject_rdm_dir
        _try_dump_svm_patterns(sid, c_data, u_data, rdm_dump_dir, roi_names)

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_dump_svm_patterns(
    subject_id: str,
    c_data,
    u_data,
    rdm_dump_dir: Path,
    roi_names: list[str],
) -> None:
    """Dump per-subject SVM patterns to disk (silently skip on any error)."""
    try:
        roi_patterns_by_state: dict = {}
        labels_by_state:       dict = {}
        stim_names_by_state:   dict = {}

        for state, vis_data in [("conscious", c_data), ("unconscious", u_data)]:
            if vis_data is None:
                continue
            roi_patterns_by_state[state] = {
                roi: vis_data.bold_patterns[roi]
                for roi in roi_names
                if roi in vis_data.bold_patterns
            }
            labels_by_state[state]     = vis_data.labels
            stim_names_by_state[state] = vis_data.stimulus_names

        if roi_patterns_by_state:
            dump_svm_patterns(
                subject_id            = subject_id,
                roi_patterns_by_state = roi_patterns_by_state,
                labels_by_state       = labels_by_state,
                stim_names_by_state   = stim_names_by_state,
                rdm_dump_dir          = rdm_dump_dir,
            )
    except Exception as exc:
        logger.warning("Could not dump SVM patterns for %s: %s", subject_id, exc)


def _build_decoder(settings: Settings, ckpt_dir: Path) -> SVMDecoder:
    """Construct an SVMDecoder with config-yaml hyperparameters."""
    rsa_cfg = settings.rsa
    return SVMDecoder(
        C=float(rsa_cfg.get("svm_C", 1.0)),
        n_perms=int(rsa_cfg.get("svm_n_perms", 10_000)),
        alpha=float(rsa_cfg.get("alpha", 0.05)),
        tol=float(rsa_cfg.get("svm_tol", 1e-3)),
        max_iter=int(rsa_cfg.get("svm_max_iter", 10_000)),
        rng_seed=int(rsa_cfg.get("svm_random_state", 12345)),
        n_jobs=4,
        cache_dir=ckpt_dir,
    )


def run_from_disk(
    settings: Settings,
    subject_ids: list[str] | None = None,
) -> dict:
    """
    Run SVM decoding (Phase 0.2) by loading per-subject patterns from the
    on-disk archives written by a previous ``--dump-subject-rdms`` run.

    No Subject objects, no BOLD loading.  Supports the same idempotency
    checkpoint as ``run()``.

    Parameters
    ----------
    settings    : active Settings
    subject_ids : subjects to process; defaults to all with dumped archives

    Returns the same ``all_results`` dict as ``run()``.
    """
    logger.info("=" * 60)
    logger.info("PHASE 0.2 (disk-load mode) – SVM Decoding from archived patterns")
    logger.info("=" * 60)

    rdm_dump_dir = settings.subject_rdm_dir
    available    = list_dumped_subjects(rdm_dump_dir)

    # Filter to subjects that also have SVM pattern archives
    from utils.rdm_io import _svm_patterns_path
    svm_available = [
        sid for sid in available
        if _svm_patterns_path(rdm_dump_dir, sid).exists()
    ]

    ids_to_run = [
        sid for sid in (subject_ids or svm_available)
        if sid in svm_available
    ]
    if not ids_to_run:
        logger.warning(
            "No SVM pattern archives found under %s.\n"
            "Run with --dump-subject-rdms (which includes phase 0b) first.",
            rdm_dump_dir,
        )
        return {}

    # Idempotency: use a distinct checkpoint file for disk-load mode
    ckpt_dir  = settings.checkpoints_dir / _CHECKPOINT_SUBDIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = "phase0b_svm_from_disk_" + "_".join(sorted(ids_to_run)) + ".pkl"
    ckpt      = ckpt_dir / ckpt_name
    if ckpt.exists():
        logger.info("SVM (disk-load) checkpoint found – loading cached results.")
        with open(ckpt, "rb") as fh:
            return pickle.load(fh)

    decoder   = _build_decoder(settings, ckpt_dir)
    plotter   = SVMPlotter(settings)
    roi_names = [roi for roi in settings.active_roi_names if roi != "wholebrain"]

    all_results: dict = {}

    for sid in ids_to_run:
        try:
            roi_patterns_by_state, labels_by_state, stim_names_by_state = \
                load_svm_patterns(sid, rdm_dump_dir)
        except Exception as exc:
            logger.error("Skipping %s – could not load SVM patterns: %s", sid, exc)
            continue

        all_results[sid] = {"conscious": [], "unconscious": [], "c_to_u": []}

        c_patterns_map = roi_patterns_by_state.get("conscious",   {})
        u_patterns_map = roi_patterns_by_state.get("unconscious", {})
        c_labels       = labels_by_state.get("conscious")
        u_labels       = labels_by_state.get("unconscious")
        c_stims        = stim_names_by_state.get("conscious")
        u_stims        = stim_names_by_state.get("unconscious")

        # ── Within-state ─────────────────────────────────────────────────
        for state, pat_map, labs, stims in [
            ("conscious",   c_patterns_map, c_labels, c_stims),
            ("unconscious", u_patterns_map, u_labels, u_stims),
        ]:
            if labs is None:
                continue
            for roi in roi_names:
                patterns = pat_map.get(roi)
                if patterns is None or patterns.shape[0] < 4:
                    continue
                result = decoder.decode_within_state(
                    patterns=patterns, labels=labs,
                    item_ids=stims, roi=roi,
                    state=state, subject_id=sid,
                )
                all_results[sid][state].append(result)
            decoder.apply_bonferroni(all_results[sid][state], n_rois=len(roi_names))
            logger.info(
                "  %s | %s | %d/%d ROIs significant (within-state)",
                sid, state,
                sum(r.significant for r in all_results[sid][state]),
                len(all_results[sid][state]),
            )

        # ── Cross-state generalisation ────────────────────────────────────
        if c_labels is not None and u_labels is not None:
            for roi in roi_names:
                c_pat = c_patterns_map.get(roi)
                u_pat = u_patterns_map.get(roi)
                if c_pat is None or u_pat is None:
                    continue
                if c_pat.shape[0] < 4 or u_pat.shape[0] < 4:
                    continue
                result = decoder.decode_generalisation(
                    train_patterns=c_pat, train_labels=c_labels,
                    train_item_ids=c_stims,
                    test_patterns=u_pat,  test_labels=u_labels,
                    test_item_ids=u_stims,
                    roi=roi, subject_id=sid,
                )
                all_results[sid]["c_to_u"].append(result)
            decoder.apply_bonferroni(all_results[sid]["c_to_u"], n_rois=len(roi_names))

        # Per-subject figures
        for state in ("conscious", "unconscious"):
            plotter.plot_decoding_by_roi(
                results=all_results[sid][state] + all_results[sid].get("c_to_u", []),
                state=state, roi_names=roi_names,
                subject_id=sid,
                save_name=f"phase0b_svm_{sid}_{state}.png",
            )

    # Group figures + stats (same as run())
    gen_by_subject = {
        sid: all_results[sid]["c_to_u"]
        for sid in all_results if all_results[sid]["c_to_u"]
    }
    if gen_by_subject:
        plotter.plot_generalisation_heatmap(
            results_by_subject=gen_by_subject, roi_names=roi_names,
        )
    plotter.plot_group_summary(all_results=all_results, roi_names=roi_names)

    json_out: dict = {}
    for sid, state_dict in all_results.items():
        json_out[sid] = {}
        for state, results in state_dict.items():
            json_out[sid][state] = [
                {
                    "roi": r.roi, "mean_auc": round(r.mean_auc, 4),
                    "mean_chance": round(r.mean_chance, 4),
                    "delta_auc": round(r.delta_auc, 4),
                    "p_value": round(r.p_value, 5),
                    "significant": r.significant, "n_folds": r.n_folds,
                }
                for r in results
            ]
    save_json(json_out, settings.stats_dir / "phase0b_svm.json")

    with open(ckpt, "wb") as fh:
        pickle.dump(all_results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("SVM (disk-load) results checkpointed to %s", ckpt)

    logger.info("Phase 0.2 (disk-load mode) complete.\n")
    return all_results

