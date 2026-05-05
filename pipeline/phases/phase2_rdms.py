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
    agg_rdms: dict[str, dict[str, dict[str, RDM]]] = {
        m: {s: {} for s in ("conscious", "unconscious")}
        for m in _AGG_METHODS
    }

    for state in ("conscious", "unconscious"):
        all_rois = sorted({
            roi
            for sid in human_rdms
            if isinstance(human_rdms[sid], dict) and state in human_rdms[sid]
            for roi in human_rdms[sid][state]
        })

        for roi in all_rois:
            roi_rdm_list: list[RDM] = [
                human_rdms[sid][state][roi]
                for sid in human_rdms
                if isinstance(human_rdms[sid], dict)
                and state in human_rdms[sid]
                and roi in human_rdms[sid][state]
            ]
            if not roi_rdm_list:
                continue

            # ─ Aggregate figures (category-sorted, one per method) ─────────
            for method in _AGG_METHODS:
                agg_cache = cache_dir / f"agg_{method}_{state}_{roi}.npy"
                if agg_cache.exists():
                    logger.info(
                        "  Loading cached agg RDM: method=%s state=%s roi=%s",
                        method, state, roi,
                    )
                    agg = RDMBuilder.load(str(agg_cache))
                else:
                    agg = aggregate_rdm(
                        roi_rdm_list, roi_or_layer=roi, state=state, method=method
                    )
                    RDMBuilder.save(agg, str(agg_cache))
                    logger.info(
                        "  Cached agg RDM: method=%s state=%s roi=%s → %s",
                        method, state, roi, agg_cache,
                    )
                agg_rdms[method][state][roi] = agg
                rdm_plotter.plot_mean_rdm(
                    agg,
                    save_name=f"rdm_{method}_{state}_{roi}.png",
                    subdir=f"phase2_rdms/{method}/{roi}",
                )

            # ─ Independent sorted figures ───────────────────────────────────
            all_rdms_for_sort = roi_rdm_list + [
                agg_rdms[m][state][roi] for m in _AGG_METHODS
            ]
            for rdm_item in all_rdms_for_sort:
                rdm_plotter.plot_sorted_rdm(
                    rdm_item,
                    title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                    save_name=(
                        f"rdm_sorted_indep_{rdm_item.subject_id}_{state}_{roi}.png"
                    ),
                    subdir=f"phase2_rdms/sorted_independent/{roi}",
                )

            # ─ Consensus sorted figures (GW-barycenter order) ─────────────
            consensus_cache = cache_dir / f"consensus_order_{state}_{roi}.npz"
            if consensus_cache.exists():
                logger.info(
                    "  Loading cached GW consensus order: state=%s roi=%s", state, roi
                )
                loaded    = np.load(str(consensus_cache), allow_pickle=True)
                c_order   = loaded["c_order"]
                c_k       = int(loaded["c_k"])
                c_score   = float(loaded["c_score"])
            else:
                logger.info(
                    "  Computing GW-barycenter consensus for ROI=%s state=%s …",
                    roi, state,
                )
                bary_matrix = gw_consensus_matrix(roi_rdm_list)
                c_order, c_k, c_score = sorted_order(bary_matrix)
                np.savez(
                    str(consensus_cache),
                    c_order=c_order,
                    c_k=np.array(c_k),
                    c_score=np.array(c_score),
                )
                logger.info(
                    "  Cached GW consensus order → %s", consensus_cache
                )

            for rdm_item in all_rdms_for_sort:
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

