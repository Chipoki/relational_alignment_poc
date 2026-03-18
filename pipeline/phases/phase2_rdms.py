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
from visualization.rdm_plotter import RDMPlotter

logger = logging.getLogger(__name__)

_AGG_METHODS: tuple[Literal["mean", "median"], ...] = ("mean", "median")


def _normalize_name(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', Path(str(s)).stem.lower())


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

    rdm_dir = Path(settings.rdm_dir)
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
                agg = aggregate_rdm(
                    roi_rdm_list, roi_or_layer=roi, state=state, method=method
                )
                agg_rdms[method][state][roi] = agg
                rdm_plotter.plot_mean_rdm(
                    agg,
                    save_name=f"rdm_{method}_{state}_{roi}.png",
                    subdir=f"phase2_rdms/{method}/{roi}",
                )

            # ─ Independent sorted figures ───────────────────────────────────
            # Each subject and each aggregate sorted by its own optimal order.
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
                    # No common_order supplied → each RDM uses its own Ward sort
                )

            # ─ Consensus sorted figures (GW-barycenter order) ─────────────
            logger.info(
                "  Computing GW-barycenter consensus for ROI=%s state=%s …",
                roi, state,
            )
            bary_matrix = gw_consensus_matrix(roi_rdm_list)
            c_order, c_k, c_score = sorted_order(bary_matrix)

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

    # Attach for downstream phases; legacy alias kept for Phase 4 / 7.
    human_rdms["_agg_rdms"]  = agg_rdms
    human_rdms["_mean_rdms"] = agg_rdms["mean"]

    logger.info("Phase 2 complete.\n")
    return human_rdms, fcnn_rdms
