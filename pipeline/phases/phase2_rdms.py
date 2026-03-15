"""Phase 2 – Dual-State Intra-Modality RDM Construction."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from config.settings import Settings
from analysis.rsa.rdm import RDMBuilder, RDM
from embeddings.embedding_store import EmbeddingStore
from visualization.rdm_plotter import RDMPlotter
from utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)


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
        common_stims_by_state[state] = list(set.intersection(*name_sets)) if name_sets else []

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
            sorted_stims   = sorted(common_stims, key=lambda x: (-stim_to_label.get(x, 0), x))
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

    # ── FCNN RDMs ─────────────────────────────────────────────────────────
    human_state_map = {"clear": "conscious", "chance": "unconscious"}

    for noise_state, store_key in [("clear", "fcnn_clear"), ("chance", "fcnn_chance")]:
        names_key      = f"{store_key}_names"
        rdm_cache_path = rdm_dir / f"fcnn_rdm_{noise_state}.npy"

        if rdm_cache_path.exists():
            logger.info("Cached FCNN RDM found for '%s' – loading from disk.", noise_state)
            fcnn_rdm = RDMBuilder.load(str(rdm_cache_path))
            fcnn_rdms[noise_state] = {"fcnn_hidden": fcnn_rdm}
            continue

        if not embedding_store.exists(store_key) or not embedding_store.exists(names_key):
            logger.info("FCNN embeddings not found for '%s' – skipping.", noise_state)
            continue

        fcnn_emb   = embedding_store.load(store_key)
        fcnn_names = embedding_store.load(names_key)

        human_state = human_state_map[noise_state]
        ref_stims   = common_stims_by_state.get(human_state, [])
        if not ref_stims:
            logger.warning("No common stimuli for human state '%s'.", human_state)
            continue

        vis_data = next(
            (getattr(s, human_state) for s in subjects if getattr(s, human_state, None)),
            None,
        )
        if vis_data is None:
            continue

        stim_to_label    = dict(zip(vis_data.stimulus_names, vis_data.labels))
        sorted_ref_stims = sorted(ref_stims, key=lambda x: (-stim_to_label.get(x, 0), x))
        stim_to_fcnn_idx = {_normalize_name(n): i for i, n in enumerate(fcnn_names)}

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
        sid   = subj.subject_id
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

    logger.info("Phase 2 complete.\n")
    return human_rdms, fcnn_rdms
