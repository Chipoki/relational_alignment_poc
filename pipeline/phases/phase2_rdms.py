"""Phase 2 – Dual-State Intra-Modality RDM Construction."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np

from analysis.rsa.rdm import RDMBuilder, RDM
from analysis.rsa.rdm_utils import aggregate_rdm, gw_consensus_matrix, sorted_order
from config.settings import Settings
from embeddings.embedding_store import EmbeddingStore
from utils.io_utils import ensure_dir
from utils.rdm_io import (
    dump_subject_rdms,
    dump_aggregate_rdms,
    dump_fcnn_rdms,
    load_subject_rdms,
    load_aggregate_rdms,
    load_fcnn_rdms,
    list_dumped_subjects,
    dump_intersected_subject_rdms,
    dump_intersected_aggregate_rdms,
    list_intersected_subjects,
    load_intersected_subject_rdms,
)
from visualization.rdm_plotter import RDMPlotter

logger = logging.getLogger(__name__)

_AGG_METHODS: tuple[Literal["mean", "median"], ...] = ("mean", "median")


def _normalize_name(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', Path(str(s)).stem.lower())


def _rdm_cache_dir(settings: Settings) -> Path:
    d = Path(getattr(settings, "checkpoints_dir", "checkpoints")) / "rdms"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(
    settings: Settings,
    subjects: list,
    embedding_store: EmbeddingStore,
    rdm_builder: RDMBuilder,
    rdm_plotter: RDMPlotter,
) -> tuple[dict, dict]:
    """
    Build all human and FCNN RDMs.

    Returns
    -------
    human_rdms : dict  – subject_id → state → roi → RDM
                         '_agg_rdms' → method → state → roi → RDM
                         '_mean_rdms' → state → roi → RDM  (legacy alias)
    fcnn_rdms  : dict  – noise_state → roi → RDM
    """
    logger.info("=" * 60)
    logger.info("PHASE 2 – Dual-State Intra-Modality RDM Construction")
    logger.info("=" * 60)

    rdm_dir   = Path(settings.rdm_dir)
    cache_dir = _rdm_cache_dir(settings)
    ensure_dir(rdm_dir)

    human_rdms: dict = {}
    fcnn_rdms:  dict = {}

    # ── Common stimuli per state ──────────────────────────────────────────
    common_stims_by_state: dict[str, list] = {}
    for state in ("conscious", "unconscious"):
        name_sets = [
            set(getattr(subj, state).stimulus_names)
            for subj in subjects
            if getattr(subj, state, None) is not None
        ]
        common_stims_by_state[state] = (
            list(set.intersection(*name_sets)) if name_sets else []
        )

    # ── Human RDMs ────────────────────────────────────────────────────────
    for subj in subjects:
        human_rdms[subj.subject_id] = {}
        for state in ("conscious", "unconscious"):
            vis_data     = getattr(subj, state, None)
            common_stims = common_stims_by_state.get(state, [])

            if vis_data is None or not common_stims:
                logger.warning(
                    "Subject %s: no valid '%s' data or common stimuli – skipping.",
                    subj.subject_id, state,
                )
                continue

            stim_to_label  = dict(zip(vis_data.stimulus_names, vis_data.labels))
            sorted_stims   = sorted(
                common_stims, key=lambda x: (-stim_to_label.get(x, 0), x)
            )
            aligned_labels = np.array([stim_to_label[s] for s in sorted_stims])
            aligned_names  = np.array(sorted_stims)

            roi_averaged: dict = {roi: [] for roi in vis_data.bold_patterns}
            for stim in sorted_stims:
                idx = np.where(vis_data.stimulus_names == stim)[0]
                for roi, patterns in vis_data.bold_patterns.items():
                    roi_averaged[roi].append(patterns[idx].mean(axis=0))

            for roi in roi_averaged:
                arr   = np.nan_to_num(np.array(roi_averaged[roi]), nan=0.0)
                valid = np.var(arr, axis=0) > 1e-8
                roi_averaged[roi] = arr[:, valid] if valid.sum() > 0 else arr

            state_rdms = rdm_builder.build_from_embeddings(
                embeddings=roi_averaged,
                stimulus_names=aligned_names,
                labels=aligned_labels,
                subject_id=subj.subject_id,
                state=state,
            )
            human_rdms[subj.subject_id][state] = state_rdms
            logger.info(
                "Built %d RDMs for subject %s, state=%s (n_stimuli=%d)",
                len(state_rdms), subj.subject_id, state, len(sorted_stims),
            )

    # ── Dump per-subject RDMs immediately after building (idempotent) ─────
    # This happens regardless of run mode so that future per-subject runs
    # automatically build up the on-disk archive used by --from-subject-rdms.
    rdm_dump_dir = settings.subject_rdm_dir
    for subj in subjects:
        sid = subj.subject_id
        subj_state_rdms = human_rdms.get(sid, {})
        if subj_state_rdms:
            dump_subject_rdms(sid, subj_state_rdms, rdm_dump_dir)

    # (c) After all subjects in this run are dumped, reload ALL cached
    # per-subject archives (including previously computed subjects not in
    # the current `subjects` list) and regenerate aggregate RDMs + figures
    # so they always reflect the full set of data accumulated so far.
    _update_aggregate_figures(settings, rdm_dump_dir, rdm_builder, rdm_plotter)

    # Update intersected-stimulus RDMs and figures (global intersection across
    # all accumulated subjects, foundation for phases 3-6 in per-subject mode).
    _update_intersected_figures(settings, rdm_plotter)

    # ── FCNN RDMs ────────────────────────────────────────────────────────
    human_state_map = {"clear": "conscious", "chance": "unconscious"}

    for noise_state, store_key in [("clear", "fcnn_clear"), ("chance", "fcnn_chance")]:
        names_key      = f"{store_key}_names"
        rdm_cache_path = rdm_dir / f"fcnn_rdm_{noise_state}.npy"

        if rdm_cache_path.exists():
            logger.info(
                "Cached FCNN RDM found for '%s' – loading from disk.", noise_state
            )
            fcnn_rdm = RDMBuilder.load(str(rdm_cache_path))
            fcnn_rdms[noise_state] = {"fcnn_hidden": fcnn_rdm}
            continue

        if not embedding_store.exists(store_key) or not embedding_store.exists(names_key):
            logger.info(
                "FCNN embeddings not found for '%s' – skipping.", noise_state
            )
            continue

        fcnn_emb   = embedding_store.load(store_key)
        fcnn_names = embedding_store.load(names_key)

        human_state = human_state_map[noise_state]
        ref_stims   = common_stims_by_state.get(human_state, [])
        if not ref_stims:
            logger.warning("No common stimuli for human state '%s'.", human_state)
            continue

        vis_data = next(
            (getattr(s, human_state) for s in subjects
             if getattr(s, human_state, None)),
            None,
        )
        if vis_data is None:
            continue

        stim_to_label    = dict(zip(vis_data.stimulus_names, vis_data.labels))
        sorted_ref_stims = sorted(
            ref_stims, key=lambda x: (-stim_to_label.get(x, 0), x)
        )
        stim_to_fcnn_idx = {
            _normalize_name(n): i for i, n in enumerate(fcnn_names)
        }

        aligned_fcnn_emb, valid_labels, valid_stims = [], [], []
        for stim in sorted_ref_stims:
            idx = stim_to_fcnn_idx.get(_normalize_name(stim))
            if idx is not None:
                aligned_fcnn_emb.append(fcnn_emb[idx])
                valid_labels.append(stim_to_label[stim])
                valid_stims.append(stim)

        if not aligned_fcnn_emb:
            logger.warning("Failed to align FCNN stimuli for %s.", noise_state)
            continue

        fcnn_rdm = rdm_builder.build_vectorised(
            patterns=np.array(aligned_fcnn_emb),
            stimulus_names=np.array(valid_stims),
            labels=np.array(valid_labels),
            roi_or_layer="fcnn_hidden",
            subject_id=f"fcnn_{noise_state}",
            state=noise_state,
        )
        fcnn_rdms[noise_state] = {"fcnn_hidden": fcnn_rdm}
        RDMBuilder.save(fcnn_rdm, str(rdm_cache_path))
        logger.info(
            "Built and saved FCNN RDM for noise_state=%s (n_stimuli=%d) → %s",
            noise_state, len(valid_stims), rdm_cache_path,
        )

    # ── FCNN Dual-State RDM Figure ────────────────────────────────────────
    fcnn_clear_rdm  = fcnn_rdms.get("clear",  {}).get("fcnn_hidden")
    fcnn_chance_rdm = fcnn_rdms.get("chance", {}).get("fcnn_hidden")
    if fcnn_clear_rdm is not None and fcnn_chance_rdm is not None:
        rdm_plotter.plot_dual_state_fcnn(
            rdm_clear=fcnn_clear_rdm,
            rdm_noisy=fcnn_chance_rdm,
            save_name="rdm_dual_fcnn.png",
            subdir="phase2_rdms",
        )
        logger.info("Saved FCNN dual-state RDM figure.")

    # ── Human Dual-State RDM Figures – every subject × every ROI ─────────
    for subj in subjects:
        sid    = subj.subject_id
        c_rdms = human_rdms.get(sid, {}).get("conscious",   {})
        u_rdms = human_rdms.get(sid, {}).get("unconscious", {})

        all_rois = sorted(set(c_rdms.keys()) | set(u_rdms.keys()))
        for roi in all_rois:
            c_rdm = c_rdms.get(roi)
            u_rdm = u_rdms.get(roi)
            if c_rdm is None or u_rdm is None:
                continue
            rdm_plotter.plot_dual_state(
                c_rdm, u_rdm,
                suptitle=(
                    f"Subject {sid}  ·  Representational Dissimilarity Matrix\n"
                    f"{roi.replace('_', ' ').title()}  (Conscious | Unconscious)"
                ),
                save_name=f"rdm_dual_{sid}_{roi}.png",
                subdir=f"phase2_rdms/{roi}",
            )
        if all_rois:
            logger.info(
                "Saved dual-state RDM plots for %s across %d ROIs.", sid, len(all_rois)
            )

    # ── Aggregate RDMs + sorted figures ──────────────────────────────────────
    #
    # Directory layout
    # ----------------
    # phase2_rdms/mean/{roi}/           – mean aggregate (category-sorted)
    # phase2_rdms/median/{roi}/         – median aggregate (category-sorted)
    # phase2_rdms/sorted_independent/{roi}/
    #     Every subject + both aggregates, each sorted by its OWN optimal
    #     Ward order.  Figures are NOT visually comparable across subjects
    #     but faithfully represent individual geometry.
    # phase2_rdms/sorted_consensus/{roi}/
    #     Every subject + both aggregates, ALL sorted by the SAME GW-barycenter
    #     consensus order.  Figures are directly visually comparable.
    #
    excluded_sids = set(settings.aggregate_exclude_subjects)
    included_sids = sorted(
        sid for sid in human_rdms
        if not isinstance(human_rdms[sid], dict) or sid not in excluded_sids
    )
    # For aggregate computation only include non-excluded, non-internal keys
    included_human_rdms = {
        sid: human_rdms[sid]
        for sid in included_sids
        if isinstance(human_rdms[sid], dict)
        and sid not in excluded_sids
    }

    agg_rdms: dict[str, dict[str, dict[str, RDM]]] = {
        m: {s: {} for s in ("conscious", "unconscious")}
        for m in _AGG_METHODS
    }

    for state in ("conscious", "unconscious"):
        all_rois = sorted({
            roi
            for sid in included_human_rdms
            if state in included_human_rdms[sid]
            for roi in included_human_rdms[sid][state]
        })

        included_labels = sorted(included_human_rdms.keys())

        for roi in all_rois:
            roi_rdm_list: list[RDM] = [
                included_human_rdms[sid][state][roi]
                for sid in included_human_rdms
                if state in included_human_rdms[sid]
                and roi in included_human_rdms[sid][state]
            ]
            if not roi_rdm_list:
                continue

            # ─ Aggregate figures (category-sorted, one per method) ─────────
            # Always recompute from roi_rdm_list (which is sized to the
            # current session's common_stims_by_state intersection).
            # _update_aggregate_figures has already written the full-cohort
            # caches; here we need the session-correct versions for the
            # sorted figures that follow, so we never read possibly-stale
            # caches whose stimulus count may differ from this session's RDMs.
            for method in _AGG_METHODS:
                agg = aggregate_rdm(
                    roi_rdm_list, roi_or_layer=roi, state=state, method=method
                )
                agg_cache = cache_dir / f"agg_{method}_{state}_{roi}.npy"
                RDMBuilder.save(agg, str(agg_cache))
                agg_rdms[method][state][roi] = agg
                rdm_plotter.plot_mean_rdm(
                    agg,
                    save_name=f"rdm_{method}_{state}_{roi}.png",
                    subdir=f"phase2_rdms/{method}/{roi}",
                    included_subjects=included_labels,
                )

            # ─ Independent sorted figures ───────────────────────────────────
            all_rdms_for_sort = roi_rdm_list + [
                agg_rdms[m][state][roi] for m in _AGG_METHODS
            ]
            for rdm_item in all_rdms_for_sort:
                # Only pass included_subjects annotation on aggregate items
                is_agg = rdm_item.subject_id in _AGG_METHODS
                rdm_plotter.plot_sorted_rdm(
                    rdm_item,
                    title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                    save_name=(
                        f"rdm_sorted_indep_{rdm_item.subject_id}_{state}_{roi}.png"
                    ),
                    subdir=f"phase2_rdms/sorted_independent/{roi}",
                    included_subjects=included_labels if is_agg else None,
                )

            # ─ Consensus sorted figures (GW-barycenter order) ─────────────
            # Always recompute: same reason as aggregates above — cached
            # consensus order may be from a different stimulus count.
            # EXCEPTION: when the current session has only 1 subject,
            # gw_consensus_matrix is meaningless (and will fail sanitisation).
            # In that case we prefer the cache written moments ago by
            # _update_aggregate_figures, which used all accumulated subjects.
            logger.info(
                "  Computing GW-barycenter consensus for ROI=%s state=%s …",
                roi, state,
            )
            consensus_cache = cache_dir / f"consensus_order_{state}_{roi}.npz"
            if len(roi_rdm_list) >= 2:
                bary_matrix = gw_consensus_matrix(roi_rdm_list)
                c_order, c_k, c_score = sorted_order(bary_matrix)
                np.savez(
                    str(consensus_cache),
                    c_order=c_order,
                    c_k=np.array(c_k),
                    c_score=np.array(c_score),
                )
            elif consensus_cache.exists():
                logger.info(
                    "  Only 1 session subject – loading consensus cache from "
                    "_update_aggregate_figures for ROI=%s state=%s.", roi, state,
                )
                loaded  = np.load(str(consensus_cache), allow_pickle=True)
                c_order = loaded["c_order"]
                c_k     = int(loaded["c_k"])
                c_score = float(loaded["c_score"])
            else:
                logger.warning(
                    "  Only 1 session subject and no consensus cache exists yet "
                    "for ROI=%s state=%s – falling back to independent sort.", roi, state,
                )
                c_order, c_k, c_score = None, None, None

            for rdm_item in all_rdms_for_sort:
                is_agg = rdm_item.subject_id in _AGG_METHODS
                rdm_plotter.plot_sorted_rdm(
                    rdm_item,
                    title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                    save_name=(
                        f"rdm_sorted_consensus_{rdm_item.subject_id}_{state}_{roi}.png"
                    ),
                    subdir=f"phase2_rdms/sorted_consensus/{roi}",
                    common_order=c_order,
                    best_k=c_k,
                    best_score=c_score,
                    included_subjects=included_labels if is_agg else None,
                )

        logger.info(
            "Aggregate & sorted RDM figures saved for state=%s (%d ROIs).",
            state, len(all_rois),
        )

    # Attach for downstream phases; legacy alias kept for Phase 4.
    human_rdms["_agg_rdms"]  = agg_rdms
    human_rdms["_mean_rdms"] = agg_rdms["mean"]

    # ── Dump aggregate + FCNN RDMs to disk (idempotent) ───────────────────
    if agg_rdms:
        dump_aggregate_rdms(agg_rdms, rdm_dump_dir)
    if fcnn_rdms:
        dump_fcnn_rdms(fcnn_rdms, rdm_dump_dir)

    logger.info("Phase 2 complete.\n")
    return human_rdms, fcnn_rdms


# ── Cross-subject aggregate update (c) ───────────────────────────────────────

def _update_aggregate_figures(
    settings: Settings,
    rdm_dump_dir: "Path",
    rdm_builder: "RDMBuilder",
    rdm_plotter: "RDMPlotter",
) -> None:
    """
    (c) Load ALL per-subject RDM archives currently on disk, recompute mean /
    median / GW-consensus aggregates, and regenerate every aggregate figure.

    Called at the end of each per-subject ``run()`` invocation so that
    aggregate figures always reflect the complete set of subjects processed
    so far, not just the subjects in the current pipeline call.

    Existing per-subject figures for subjects not in the current in-memory
    `subjects` list are also back-filled if their figure files are missing.
    """
    available = list_dumped_subjects(rdm_dump_dir)
    if not available:
        return

    # Load all per-subject RDMs from disk
    all_human_rdms: dict = {}
    for sid in available:
        try:
            all_human_rdms[sid] = load_subject_rdms(sid, rdm_dump_dir)
        except Exception as exc:
            logger.warning("_update_aggregate_figures: skipping %s – %s", sid, exc)

    if not all_human_rdms:
        return

    # Apply exclusion list — excluded subjects are still back-filled for
    # their individual figures, but are omitted from all aggregate computation.
    excluded_sids = set(settings.aggregate_exclude_subjects)
    included_human_rdms = {
        sid: v for sid, v in all_human_rdms.items() if sid not in excluded_sids
    }
    included_labels = sorted(included_human_rdms.keys())

    logger.info(
        "(c) Recomputing aggregate RDMs from %d cached subjects: %s",
        len(included_human_rdms), included_labels,
    )
    if excluded_sids & set(all_human_rdms.keys()):
        logger.info(
            "(c) Excluded from aggregates: %s",
            sorted(excluded_sids & set(all_human_rdms.keys())),
        )

    cache_dir = _rdm_cache_dir(settings)

    # Re-derive common stimuli per state across all cached subjects
    # (needed for alignment; each subject was already aligned during run() so
    # matrices share the same stimulus ordering within each state)
    agg_rdms: dict[str, dict[str, dict[str, RDM]]] = {
        m: {s: {} for s in ("conscious", "unconscious")}
        for m in _AGG_METHODS
    }

    # Back-fill any missing dual-state per-subject figures
    for sid, state_dict in all_human_rdms.items():
        c_rdms = state_dict.get("conscious", {})
        u_rdms = state_dict.get("unconscious", {})
        all_rois_subj = sorted(set(c_rdms.keys()) | set(u_rdms.keys()))
        for roi in all_rois_subj:
            c_rdm = c_rdms.get(roi)
            u_rdm = u_rdms.get(roi)
            if c_rdm is None or u_rdm is None:
                continue
            fig_path = (
                rdm_plotter._out_dir(f"phase2_rdms/{roi}")
                / f"rdm_dual_{sid}_{roi}.png"
            )
            if not fig_path.exists():
                rdm_plotter.plot_dual_state(
                    c_rdm, u_rdm,
                    suptitle=(
                        f"Subject {sid}  ·  Representational Dissimilarity Matrix\n"
                        f"{roi.replace('_', ' ').title()}  (Conscious | Unconscious)"
                    ),
                    save_name=f"rdm_dual_{sid}_{roi}.png",
                    subdir=f"phase2_rdms/{roi}",
                )

    # ── Common stimuli per state across included subjects ─────────────────
    # Mirror exactly what the original run() did: compute the intersection
    # of stimulus names across all subjects for each state, then use that
    # single shared ordering when building every aggregate.  Each cached
    # per-subject RDM already has stimulus_names stored; we just index into
    # its matrix rows/cols to extract the common subset.
    common_stims_by_state: dict[str, np.ndarray] = {}
    for state in ("conscious", "unconscious"):
        name_sets = [
            set(included_human_rdms[sid][state][
                next(iter(included_human_rdms[sid][state]))
            ].stimulus_names.tolist())
            for sid in included_human_rdms
            if state in included_human_rdms[sid] and included_human_rdms[sid][state]
        ]
        if not name_sets:
            common_stims_by_state[state] = np.array([])
            continue
        common_set = set.intersection(*name_sets)
        ref_sid  = next(s for s in included_human_rdms if state in included_human_rdms[s] and included_human_rdms[s][state])
        ref_rdm  = next(iter(included_human_rdms[ref_sid][state].values()))
        stim_to_label = dict(zip(ref_rdm.stimulus_names.tolist(), ref_rdm.labels.tolist()))
        common_stims_by_state[state] = np.array(
            sorted(common_set, key=lambda x: (-stim_to_label.get(x, 0), x))
        )

    for state in ("conscious", "unconscious"):
        common_stims = common_stims_by_state.get(state, np.array([]))
        if len(common_stims) == 0:
            continue

        all_rois = sorted({
            roi
            for sid in included_human_rdms
            if state in included_human_rdms[sid]
            for roi in included_human_rdms[sid][state]
        })

        for roi in all_rois:
            # Subset each subject's RDM to the common stimulus ordering.
            roi_rdm_list: list[RDM] = []
            for sid in included_human_rdms:
                if state not in included_human_rdms[sid]:
                    continue
                rdm = included_human_rdms[sid][state].get(roi)
                if rdm is None:
                    continue
                stim_to_idx = {s: i for i, s in enumerate(rdm.stimulus_names.tolist())}
                idx = np.array([stim_to_idx[s] for s in common_stims if s in stim_to_idx])
                if len(idx) != len(common_stims):
                    logger.warning(
                        "  (c) Subject %s missing some common stimuli for state=%s roi=%s – skipping.",
                        sid, state, roi,
                    )
                    continue
                sub_matrix = rdm.matrix[np.ix_(idx, idx)]
                sub_labels = rdm.labels[idx]
                aligned_rdm = RDMBuilder().build_from_matrix(
                    matrix         = sub_matrix,
                    stimulus_names = common_stims,
                    labels         = sub_labels,
                    roi_or_layer   = rdm.roi_or_layer,
                    subject_id     = rdm.subject_id,
                    state          = rdm.state,
                )
                roi_rdm_list.append(aligned_rdm)

            if not roi_rdm_list:
                continue

            # ── Recompute (overwrite) aggregate RDMs ─────────────────────
            for method in _AGG_METHODS:
                agg = aggregate_rdm(
                    roi_rdm_list, roi_or_layer=roi, state=state, method=method
                )
                agg_rdms[method][state][roi] = agg
                agg_cache = cache_dir / f"agg_{method}_{state}_{roi}.npy"
                RDMBuilder.save(agg, str(agg_cache))
                rdm_plotter.plot_mean_rdm(
                    agg,
                    save_name=f"rdm_{method}_{state}_{roi}.png",
                    subdir=f"phase2_rdms/{method}/{roi}",
                    included_subjects=included_labels,
                )

            # ── Recompute GW-consensus order (overwrite) ──────────────────
            logger.info(
                "  (c) Recomputing GW-barycenter consensus for ROI=%s state=%s …",
                roi, state,
            )
            try:
                from analysis.rsa.rdm_utils import gw_consensus_matrix, sorted_order
                bary_matrix = gw_consensus_matrix(roi_rdm_list)
                c_order, c_k, c_score = sorted_order(bary_matrix)
            except Exception as exc:
                logger.warning(
                    "  (c) GW consensus failed for %s/%s: %s – skipping consensus figs.",
                    roi, state, exc,
                )
                continue

            consensus_cache = cache_dir / f"consensus_order_{state}_{roi}.npz"
            np.savez(
                str(consensus_cache),
                c_order=c_order,
                c_k=np.array(c_k),
                c_score=np.array(c_score),
            )

            # ── Regenerate independent + consensus sorted figures ─────────
            all_rdms_for_sort = roi_rdm_list + [
                agg_rdms[m][state][roi] for m in _AGG_METHODS
            ]
            for rdm_item in all_rdms_for_sort:
                is_agg = rdm_item.subject_id in _AGG_METHODS
                rdm_plotter.plot_sorted_rdm(
                    rdm_item,
                    title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                    save_name=(
                        f"rdm_sorted_indep_{rdm_item.subject_id}_{state}_{roi}.png"
                    ),
                    subdir=f"phase2_rdms/sorted_independent/{roi}",
                    included_subjects=included_labels if is_agg else None,
                )
                rdm_plotter.plot_sorted_rdm(
                    rdm_item,
                    title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                    save_name=(
                        f"rdm_sorted_consensus_{rdm_item.subject_id}_{state}_{roi}.png"
                    ),
                    subdir=f"phase2_rdms/sorted_consensus/{roi}",
                    common_order=c_order,
                    best_k=c_k,
                    best_score=c_score,
                    included_subjects=included_labels if is_agg else None,
                )

    # Persist the updated aggregate RDMs
    if agg_rdms:
        dump_aggregate_rdms(agg_rdms, rdm_dump_dir)

    # Release large in-memory structures promptly
    del all_human_rdms, included_human_rdms, agg_rdms

    logger.info(
        "(c) Aggregate RDM figures updated for %d included subject(s) (%d total cached).",
        len(included_labels), len(available),
    )


# ── Cross-subject intersected-stimulus update ─────────────────────────────────

def _intersected_signature(rdm_dump_dir: Path, available: list[str]) -> str:
    """
    Return a string that uniquely identifies the current set of per-subject
    RDM archives (subject IDs + their file modification times).  Used to
    detect whether _update_intersected_figures needs to rerun.
    """
    parts = []
    for sid in sorted(available):
        p = rdm_dump_dir / f"{sid}.npz"
        mtime = int(p.stat().st_mtime) if p.exists() else 0
        parts.append(f"{sid}:{mtime}")
    return "|".join(parts)


def _update_intersected_figures(
    settings: Settings,
    rdm_plotter: "RDMPlotter",
) -> None:
    """
    Load ALL per-subject RDM archives currently on disk, compute the global
    cross-subject stimulus intersection for each state, subset every subject's
    RDM matrices to that shared stimulus space, dump them to the intersected
    directory, recompute aggregate RDMs + consensus figures, and save all
    figures under ``figures/intersected_stimuli_rdm/`` (mirroring the
    ``phase2_rdms/`` layout).

    Called at the end of each per-subject pipeline iteration so that the
    intersected archive always reflects all subjects accumulated so far.
    Excluded subjects (``settings.aggregate_exclude_subjects``) are omitted
    from aggregate computation but their individual intersected archives are
    still written so their figures remain available.

    Skips the full recompute if the set of on-disk subject archives has not
    changed since the last run (detected via a lightweight signature file).
    """
    from analysis.rsa.rdm_utils import aggregate_rdm, gw_consensus_matrix, sorted_order

    rdm_dump_dir     = settings.subject_rdm_dir
    intersected_dir  = settings.intersected_rdm_dir
    cache_dir        = _rdm_cache_dir(settings)
    intersected_dir.mkdir(parents=True, exist_ok=True)

    available = list_dumped_subjects(rdm_dump_dir)
    if len(available) < 1:
        return

    # ── Signature check: skip if nothing changed since last intersected run ──
    sig_file  = intersected_dir / ".subjects_signature"
    current_sig = _intersected_signature(rdm_dump_dir, available)
    if sig_file.exists() and sig_file.read_text().strip() == current_sig:
        logger.info(
            "(intersected) Signature unchanged (%d subjects) – skipping recompute.",
            len(available),
        )
        return

    # Load all per-subject RDMs
    all_human_rdms: dict = {}
    for sid in available:
        try:
            all_human_rdms[sid] = load_subject_rdms(sid, rdm_dump_dir)
        except Exception as exc:
            logger.warning("_update_intersected_figures: skipping %s – %s", sid, exc)
    if not all_human_rdms:
        return

    excluded_sids    = set(settings.aggregate_exclude_subjects)
    included_rdms    = {sid: v for sid, v in all_human_rdms.items() if sid not in excluded_sids}
    included_labels  = sorted(included_rdms.keys())

    logger.info(
        "(intersected) Computing global intersection for %d subject(s): %s",
        len(all_human_rdms), sorted(all_human_rdms.keys()),
    )

    # ── Compute global common stimuli per state ───────────────────────────────
    common_stims_by_state: dict[str, np.ndarray] = {}
    for state in ("conscious", "unconscious"):
        name_sets = [
            set(all_human_rdms[sid][state][
                next(iter(all_human_rdms[sid][state]))
            ].stimulus_names.tolist())
            for sid in all_human_rdms
            if state in all_human_rdms[sid] and all_human_rdms[sid][state]
        ]
        if not name_sets:
            common_stims_by_state[state] = np.array([])
            continue
        common_set   = set.intersection(*name_sets)
        ref_sid      = next(s for s in all_human_rdms if state in all_human_rdms[s] and all_human_rdms[s][state])
        ref_rdm      = next(iter(all_human_rdms[ref_sid][state].values()))
        stim_to_label = dict(zip(ref_rdm.stimulus_names.tolist(), ref_rdm.labels.tolist()))
        common_stims_by_state[state] = np.array(
            sorted(common_set, key=lambda x: (-stim_to_label.get(x, 0), x))
        )
        logger.info(
            "(intersected) state=%s: %d common stimuli across %d subjects",
            state, len(common_stims_by_state[state]), len(all_human_rdms),
        )

    # ── Subset every subject's RDMs and dump intersected archives ────────────
    all_intersected: dict = {}   # sid → {state → {roi → RDM}}
    for sid, state_dict in all_human_rdms.items():
        subj_intersected: dict = {}
        for state, common_stims in common_stims_by_state.items():
            if len(common_stims) == 0 or state not in state_dict:
                continue
            roi_dict = state_dict[state]
            subj_intersected[state] = {}
            for roi, rdm in roi_dict.items():
                stim_to_idx = {s: i for i, s in enumerate(rdm.stimulus_names.tolist())}
                idx = np.array([stim_to_idx[s] for s in common_stims if s in stim_to_idx])
                if len(idx) != len(common_stims):
                    logger.warning(
                        "(intersected) %s missing some common stimuli for state=%s roi=%s – skipping.",
                        sid, state, roi,
                    )
                    continue
                sub_matrix = rdm.matrix[np.ix_(idx, idx)]
                sub_labels = rdm.labels[idx]
                aligned = RDMBuilder().build_from_matrix(
                    matrix=sub_matrix,
                    stimulus_names=common_stims,
                    labels=sub_labels,
                    roi_or_layer=rdm.roi_or_layer,
                    subject_id=rdm.subject_id,
                    state=rdm.state,
                )
                subj_intersected[state][roi] = aligned
        if subj_intersected:
            dump_intersected_subject_rdms(sid, subj_intersected, intersected_dir)
            all_intersected[sid] = subj_intersected

    if not all_intersected:
        return

    # ── Build an RDMPlotter that targets the intersected figures subdirectory ─
    # We do this by temporarily monkey-patching the plotter's output root so
    # all save_name paths resolve under intersected_stimuli_rdm/ instead of
    # the default phase2_rdms/ prefix.  We use a thin wrapper to avoid any
    # permanent state change to the shared plotter instance.
    class _IntersectedPlotter:
        """Delegate all plot_* calls to rdm_plotter but prefix subdir."""
        _PREFIX = "intersected_stimuli_rdm"

        def __init__(self, base: "RDMPlotter") -> None:
            self._base = base

        def __getattr__(self, name):
            base_fn = getattr(self._base, name)
            if not name.startswith("plot_"):
                return base_fn
            import functools
            @functools.wraps(base_fn)
            def _wrapped(*args, **kwargs):
                if "subdir" in kwargs:
                    orig = kwargs["subdir"]
                    # replace leading "phase2_rdms" with our prefix if present
                    if orig.startswith("phase2_rdms"):
                        kwargs["subdir"] = orig.replace(
                            "phase2_rdms", self._PREFIX, 1
                        )
                    else:
                        kwargs["subdir"] = f"{self._PREFIX}/{orig}"
                return base_fn(*args, **kwargs)
            return _wrapped

    ip = _IntersectedPlotter(rdm_plotter)

    # ── Per-subject dual-state figures (back-fill missing) ───────────────────
    for sid, state_dict in all_intersected.items():
        c_rdms = state_dict.get("conscious",   {})
        u_rdms = state_dict.get("unconscious", {})
        for roi in sorted(set(c_rdms.keys()) | set(u_rdms.keys())):
            c_rdm = c_rdms.get(roi)
            u_rdm = u_rdms.get(roi)
            if c_rdm is None or u_rdm is None:
                continue
            fig_path = (
                rdm_plotter._out_dir(f"intersected_stimuli_rdm/{roi}")
                / f"rdm_dual_{sid}_{roi}.png"
            )
            if not fig_path.exists():
                ip.plot_dual_state(
                    c_rdm, u_rdm,
                    suptitle=(
                        f"Subject {sid}  ·  Intersected RDM\n"
                        f"{roi.replace('_', ' ').title()}  (Conscious | Unconscious)"
                    ),
                    save_name=f"rdm_dual_{sid}_{roi}.png",
                    subdir=f"intersected_stimuli_rdm/{roi}",
                )

    # ── Aggregates only over included subjects ────────────────────────────────
    included_intersected = {sid: v for sid, v in all_intersected.items() if sid not in excluded_sids}
    if not included_intersected:
        return

    agg_rdms: dict = {m: {s: {} for s in ("conscious", "unconscious")} for m in _AGG_METHODS}

    for state in ("conscious", "unconscious"):
        common_stims = common_stims_by_state.get(state, np.array([]))
        if len(common_stims) == 0:
            continue
        all_rois = sorted({
            roi
            for sid in included_intersected
            if state in included_intersected[sid]
            for roi in included_intersected[sid][state]
        })

        for roi in all_rois:
            roi_rdm_list: list[RDM] = [
                included_intersected[sid][state][roi]
                for sid in included_intersected
                if state in included_intersected[sid] and roi in included_intersected[sid][state]
            ]
            if not roi_rdm_list:
                continue

            # Aggregate figures
            for method in _AGG_METHODS:
                agg = aggregate_rdm(roi_rdm_list, roi_or_layer=roi, state=state, method=method)
                agg_rdms[method][state][roi] = agg
                agg_cache = cache_dir / f"intersected_agg_{method}_{state}_{roi}.npy"
                RDMBuilder.save(agg, str(agg_cache))
                ip.plot_mean_rdm(
                    agg,
                    save_name=f"rdm_{method}_{state}_{roi}.png",
                    subdir=f"intersected_stimuli_rdm/{method}/{roi}",
                    included_subjects=included_labels,
                )

            # Independent sorted figures
            all_rdms_for_sort = roi_rdm_list + [agg_rdms[m][state][roi] for m in _AGG_METHODS]
            for rdm_item in all_rdms_for_sort:
                is_agg = rdm_item.subject_id in _AGG_METHODS
                ip.plot_sorted_rdm(
                    rdm_item,
                    title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                    save_name=f"rdm_sorted_indep_{rdm_item.subject_id}_{state}_{roi}.png",
                    subdir=f"intersected_stimuli_rdm/sorted_independent/{roi}",
                    included_subjects=included_labels if is_agg else None,
                )

            # Consensus sorted figures
            logger.info(
                "  (intersected) GW consensus for ROI=%s state=%s …", roi, state,
            )
            consensus_cache = cache_dir / f"intersected_consensus_order_{state}_{roi}.npz"
            if len(roi_rdm_list) >= 2:
                bary_matrix = gw_consensus_matrix(roi_rdm_list)
                c_order, c_k, c_score = sorted_order(bary_matrix)
                np.savez(
                    str(consensus_cache),
                    c_order=c_order, c_k=np.array(c_k), c_score=np.array(c_score),
                )
            elif consensus_cache.exists():
                loaded  = np.load(str(consensus_cache), allow_pickle=True)
                c_order = loaded["c_order"]
                c_k     = int(loaded["c_k"])
                c_score = float(loaded["c_score"])
            else:
                c_order = c_k = c_score = None

            for rdm_item in all_rdms_for_sort:
                is_agg = rdm_item.subject_id in _AGG_METHODS
                ip.plot_sorted_rdm(
                    rdm_item,
                    title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                    save_name=f"rdm_sorted_consensus_{rdm_item.subject_id}_{state}_{roi}.png",
                    subdir=f"intersected_stimuli_rdm/sorted_consensus/{roi}",
                    common_order=c_order,
                    best_k=c_k,
                    best_score=c_score,
                    included_subjects=included_labels if is_agg else None,
                )

        logger.info(
            "(intersected) Figures updated for state=%s (%d ROIs, %d subjects).",
            state, len(all_rois), len(included_intersected),
        )

    # Dump intersected aggregates
    dump_intersected_aggregate_rdms(agg_rdms, intersected_dir)

    # Write signature so subsequent calls with the same subject set are skipped
    sig_file.write_text(current_sig)

    # Explicitly release large in-memory structures so the GC can reclaim RAM
    # before the next subject iteration begins.
    del all_human_rdms, all_intersected, included_intersected, agg_rdms

    logger.info(
        "(intersected) Archive updated: %d subjects, intersected dir: %s",
        len(available), intersected_dir,
    )


# ── Per-subject RDM load path (bypasses BOLD entirely) ───────────────────────

def load_rdms_from_disk(
    settings: Settings,
    subject_ids: list[str] | None = None,
) -> tuple[dict, dict]:
    """
    Reconstruct ``human_rdms`` and ``fcnn_rdms`` entirely from the on-disk
    per-subject archives written by a previous ``--dump-subject-rdms`` run.

    No BOLD data, no NIfTI loading, no Subject objects needed.

    Parameters
    ----------
    settings    : active Settings (used for paths and roi registration)
    subject_ids : restrict to these subjects; defaults to all dumped subjects

    Returns
    -------
    human_rdms : same structure as returned by ``run()``
    fcnn_rdms  : same structure as returned by ``run()``

    Raises
    ------
    RuntimeError  if no per-subject RDM archives are found on disk
    """
    logger.info("=" * 60)
    logger.info("PHASE 2 (disk-load mode) – Reconstructing RDMs from archives")
    logger.info("=" * 60)

    rdm_dump_dir = settings.subject_rdm_dir
    available    = list_dumped_subjects(rdm_dump_dir)

    if not available:
        raise RuntimeError(
            f"No per-subject RDM archives found under {rdm_dump_dir}.\n"
            "Run with --dump-subject-rdms for at least one subject first."
        )

    ids_to_load = subject_ids if subject_ids else available
    missing     = [sid for sid in ids_to_load if sid not in available]
    if missing:
        logger.warning(
            "The following subjects have no RDM archive and will be skipped: %s",
            missing,
        )

    human_rdms: dict = {}
    discovered_rois: set[str] = set()

    for sid in ids_to_load:
        if sid not in available:
            continue
        try:
            state_rdms = load_subject_rdms(sid, rdm_dump_dir)
            human_rdms[sid] = state_rdms
            for state_dict in state_rdms.values():
                discovered_rois.update(state_dict.keys())
        except Exception as exc:
            logger.error("Failed to load RDMs for %s: %s", sid, exc)

    if not human_rdms:
        raise RuntimeError("No subject RDMs could be loaded from disk.")

    logger.info(
        "Loaded RDMs for %d subjects: %s", len(human_rdms), sorted(human_rdms.keys())
    )

    # ── Register ROIs with settings (mirrors SubjectBuilder.register_rois) ─
    settings.register_active_rois(sorted(discovered_rois))
    logger.info("Registered %d ROIs: %s", len(discovered_rois), sorted(discovered_rois))

    # ── Aggregate RDMs ─────────────────────────────────────────────────────
    agg_rdms = load_aggregate_rdms(rdm_dump_dir)
    if not agg_rdms:
        logger.info(
            "No aggregate RDM archive found – downstream phases that need "
            "aggregate RDMs (e.g. phase 3 consensus figures) may be limited."
        )
        agg_rdms = {m: {s: {} for s in ("conscious", "unconscious")}
                    for m in ("mean", "median")}

    human_rdms["_agg_rdms"]  = agg_rdms
    human_rdms["_mean_rdms"] = agg_rdms.get("mean", {})

    # ── FCNN RDMs ──────────────────────────────────────────────────────────
    fcnn_rdms = load_fcnn_rdms(rdm_dump_dir)
    if not fcnn_rdms:
        logger.info("No FCNN RDM archives found – cross-modality phases will be skipped.")

    logger.info("Phase 2 (disk-load mode) complete.\n")
    return human_rdms, fcnn_rdms

