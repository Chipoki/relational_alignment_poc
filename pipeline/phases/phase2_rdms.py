"""Phase 2 – Dual-State Intra-Modality RDM Construction."""
from __future__ import annotations

import logging
import re
import gc
from pathlib import Path
from typing import Literal

import numpy as np

from analysis.rsa.rdm import RDMBuilder, RDM
from analysis.rsa.rdm_utils import aggregate_rdm, gw_consensus_matrix, sorted_order, sorted_order_within_category
from config.settings import Settings
from embeddings.embedding_store import EmbeddingStore
from utils.io_utils import ensure_dir
from utils.rdm_io import (
    dump_subject_rdms, dump_aggregate_rdms, dump_fcnn_rdms,
    load_subject_rdms, load_aggregate_rdms, load_fcnn_rdms,
    list_dumped_subjects, dump_intersected_subject_rdms,
    dump_intersected_aggregate_rdms, list_intersected_subjects,
    load_intersected_subject_rdms,
)
from visualization.rdm_plotter import RDMPlotter

logger = logging.getLogger(__name__)

_AGG_METHODS: tuple[Literal["mean", "median"], ...] = ("mean", "median")
LINKAGE_METHODS = ["ward", "average", "complete"]


def _normalize_name(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', Path(str(s)).stem.lower())

def _rdm_cache_dir(settings: Settings) -> Path:
    d = Path(getattr(settings, "checkpoints_dir", "checkpoints")) / "rdms"
    d.mkdir(parents=True, exist_ok=True)
    return d

def run(
    settings: Settings, subjects: list, embedding_store: EmbeddingStore,
    rdm_builder: RDMBuilder, rdm_plotter: RDMPlotter,
) -> tuple[dict, dict]:
    logger.info("=" * 60)
    logger.info("PHASE 2 – Dual-State Intra-Modality RDM Construction")
    logger.info("=" * 60)

    rdm_dir   = Path(settings.rdm_dir)
    cache_dir = _rdm_cache_dir(settings)
    ensure_dir(rdm_dir)

    human_rdms: dict = {}
    fcnn_rdms:  dict = {}

    common_stims_by_state: dict[str, list] = {}
    for state in ("conscious", "unconscious"):
        name_sets = [set(getattr(subj, state).stimulus_names) for subj in subjects if getattr(subj, state, None) is not None]
        common_stims_by_state[state] = list(set.intersection(*name_sets)) if name_sets else []

    for subj in subjects:
        human_rdms[subj.subject_id] = {}
        for state in ("conscious", "unconscious"):
            vis_data = getattr(subj, state, None)
            common_stims = common_stims_by_state.get(state, [])

            if vis_data is None or not common_stims:
                continue

            stim_to_label = dict(zip(vis_data.stimulus_names, vis_data.labels))
            sorted_stims = sorted(common_stims, key=lambda x: (-stim_to_label.get(x, 0), x))
            aligned_labels = np.array([stim_to_label[s] for s in sorted_stims])
            aligned_names  = np.array(sorted_stims)

            roi_averaged = {roi: [] for roi in vis_data.bold_patterns}
            for stim in sorted_stims:
                idx = np.where(vis_data.stimulus_names == stim)[0]
                for roi, patterns in vis_data.bold_patterns.items():
                    roi_averaged[roi].append(patterns[idx].mean(axis=0))

            for roi in roi_averaged:
                arr = np.nan_to_num(np.array(roi_averaged[roi]), nan=0.0)
                valid = np.var(arr, axis=0) > 1e-8
                roi_averaged[roi] = arr[:, valid] if valid.sum() > 0 else arr

            state_rdms = rdm_builder.build_from_embeddings(
                roi_averaged, aligned_names, aligned_labels, subj.subject_id, state
            )
            human_rdms[subj.subject_id][state] = state_rdms

    rdm_dump_dir = settings.subject_rdm_dir
    for subj in subjects:
        sid = subj.subject_id
        if sid in human_rdms and human_rdms[sid]:
            dump_subject_rdms(sid, human_rdms[sid], rdm_dump_dir)

    _update_aggregate_figures(settings, rdm_dump_dir, rdm_builder, rdm_plotter)
    _update_intersected_figures(settings, rdm_plotter)

    human_state_map = {"clear": "conscious", "chance": "unconscious"}
    for noise_state, store_key in [("clear", "fcnn_clear"), ("chance", "fcnn_chance")]:
        names_key = f"{store_key}_names"
        rdm_cache_path = rdm_dir / f"fcnn_rdm_{noise_state}.npy"

        if rdm_cache_path.exists():
            fcnn_rdms[noise_state] = {"fcnn_hidden": RDMBuilder.load(str(rdm_cache_path))}
            continue

        if not embedding_store.exists(store_key) or not embedding_store.exists(names_key):
            continue

        fcnn_emb = embedding_store.load(store_key)
        fcnn_names = embedding_store.load(names_key)

        human_state = human_state_map[noise_state]
        ref_stims = common_stims_by_state.get(human_state, [])
        if not ref_stims: continue

        vis_data = next((getattr(s, human_state) for s in subjects if getattr(s, human_state, None)), None)
        if vis_data is None: continue

        stim_to_label = dict(zip(vis_data.stimulus_names, vis_data.labels))
        sorted_ref_stims = sorted(ref_stims, key=lambda x: (-stim_to_label.get(x, 0), x))
        stim_to_fcnn_idx = {_normalize_name(n): i for i, n in enumerate(fcnn_names)}

        aligned_fcnn_emb, valid_labels, valid_stims = [], [], []
        for stim in sorted_ref_stims:
            idx = stim_to_fcnn_idx.get(_normalize_name(stim))
            if idx is not None:
                aligned_fcnn_emb.append(fcnn_emb[idx])
                valid_labels.append(stim_to_label[stim])
                valid_stims.append(stim)

        if not aligned_fcnn_emb: continue
        fcnn_rdm = rdm_builder.build_vectorised(
            np.array(aligned_fcnn_emb), np.array(valid_stims), np.array(valid_labels),
            "fcnn_hidden", f"fcnn_{noise_state}", noise_state
        )
        fcnn_rdms[noise_state] = {"fcnn_hidden": fcnn_rdm}
        RDMBuilder.save(fcnn_rdm, str(rdm_cache_path))

    fcnn_clear_rdm  = fcnn_rdms.get("clear",  {}).get("fcnn_hidden")
    fcnn_chance_rdm = fcnn_rdms.get("chance", {}).get("fcnn_hidden")
    if fcnn_clear_rdm and fcnn_chance_rdm:
        rdm_plotter.plot_dual_state_fcnn(
            fcnn_clear_rdm, fcnn_chance_rdm,
            save_name="rdm_dual_fcnn.png", subdir="phase2_rdms/original_rdms"
        )

    for subj in subjects:
        sid = subj.subject_id
        c_rdms = human_rdms.get(sid, {}).get("conscious", {})
        u_rdms = human_rdms.get(sid, {}).get("unconscious", {})

        for roi in sorted(set(c_rdms.keys()) | set(u_rdms.keys())):
            if c_rdms.get(roi) and u_rdms.get(roi):
                rdm_plotter.plot_dual_state(
                    c_rdms[roi], u_rdms[roi],
                    suptitle=f"Subject {sid} · {roi.replace('_', ' ').title()} (Conscious | Unconscious)",
                    save_name=f"rdm_dual_{sid}_{roi}.png",
                    subdir=f"phase2_rdms/original_rdms/{roi}",
                )

    excluded_sids = set(settings.aggregate_exclude_subjects)
    included_sids = sorted(sid for sid in human_rdms if sid not in excluded_sids)
    included_human_rdms = {sid: human_rdms[sid] for sid in included_sids if isinstance(human_rdms[sid], dict)}

    agg_rdms: dict = {m: {s: {} for s in ("conscious", "unconscious")} for m in _AGG_METHODS}

    for state in ("conscious", "unconscious"):
        all_rois = sorted({roi for sid in included_human_rdms if state in included_human_rdms[sid] for roi in included_human_rdms[sid][state]})
        included_labels = sorted(included_human_rdms.keys())

        for roi in all_rois:
            roi_rdm_list = [included_human_rdms[sid][state][roi] for sid in included_human_rdms if state in included_human_rdms[sid] and roi in included_human_rdms[sid][state]]
            if not roi_rdm_list: continue

            for method in _AGG_METHODS:
                agg = aggregate_rdm(roi_rdm_list, roi, state, method)
                agg_cache = cache_dir / f"agg_{method}_{state}_{roi}.npy"
                RDMBuilder.save(agg, str(agg_cache))
                agg_rdms[method][state][roi] = agg
                rdm_plotter.plot_mean_rdm(
                    agg, save_name=f"rdm_{method}_{state}_{roi}.png",
                    subdir=f"phase2_rdms/original_rdms/{method}/{roi}",
                    included_subjects=included_labels,
                )

            all_rdms_for_sort = roi_rdm_list + [agg_rdms[m][state][roi] for m in _AGG_METHODS]

            bary_matrix = gw_consensus_matrix(roi_rdm_list) if len(roi_rdm_list) >= 2 else None

            for rdm_item in all_rdms_for_sort:
                is_agg = rdm_item.subject_id in _AGG_METHODS
                for link_method in LINKAGE_METHODS:
                    rdm_plotter.plot_sorted_rdm(
                        rdm_item, title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                        save_name=f"rdm_sorted_indep_{link_method}_{rdm_item.subject_id}_{state}_{roi}.png",
                        subdir=f"phase2_rdms/original_rdms/sorted_independent/{link_method}/{roi}",
                        included_subjects=included_labels if is_agg else None,
                        linkage_method=link_method, within_category=False,
                    )
                    rdm_plotter.plot_sorted_rdm(
                        rdm_item, title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                        save_name=f"rdm_sorted_indep_within_cat_{link_method}_{rdm_item.subject_id}_{state}_{roi}.png",
                        subdir=f"phase2_rdms/original_rdms/sorted_independent_within_category/{link_method}/{roi}",
                        included_subjects=included_labels if is_agg else None,
                        linkage_method=link_method, within_category=True,
                    )

                    if bary_matrix is not None:
                        c_order, c_k, c_score = sorted_order(bary_matrix, method=link_method)
                        rdm_plotter.plot_sorted_rdm(
                            rdm_item, title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                            save_name=f"rdm_sorted_consensus_{link_method}_{rdm_item.subject_id}_{state}_{roi}.png",
                            subdir=f"phase2_rdms/original_rdms/sorted_consensus/{link_method}/{roi}",
                            common_order=c_order, best_k=c_k, best_score=c_score,
                            included_subjects=included_labels if is_agg else None,
                            linkage_method=link_method, within_category=False,
                        )
                        c_order_wc, c_k_wc, c_score_wc = sorted_order_within_category(bary_matrix, rdm_item.labels, method=link_method)
                        rdm_plotter.plot_sorted_rdm(
                            rdm_item, title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                            save_name=f"rdm_sorted_consensus_within_cat_{link_method}_{rdm_item.subject_id}_{state}_{roi}.png",
                            subdir=f"phase2_rdms/original_rdms/sorted_consensus_within_category/{link_method}/{roi}",
                            common_order=c_order_wc, best_k=c_k_wc, best_score=c_score_wc,
                            included_subjects=included_labels if is_agg else None,
                            linkage_method=link_method, within_category=True,
                        )

            # Clean up loops
            gc.collect()

    human_rdms["_agg_rdms"]  = agg_rdms
    human_rdms["_mean_rdms"] = agg_rdms["mean"]
    if agg_rdms: dump_aggregate_rdms(agg_rdms, rdm_dump_dir)
    if fcnn_rdms: dump_fcnn_rdms(fcnn_rdms, rdm_dump_dir)
    return human_rdms, fcnn_rdms

def _update_aggregate_figures(settings: Settings, rdm_dump_dir: "Path", rdm_builder: "RDMBuilder", rdm_plotter: "RDMPlotter") -> None:
    pass

def _intersected_signature(rdm_dump_dir: Path, available: list[str]) -> str:
    parts = []
    for sid in sorted(available):
        p = rdm_dump_dir / f"{sid}.npz"
        mtime = int(p.stat().st_mtime) if p.exists() else 0
        parts.append(f"{sid}:{mtime}")
    return "|".join(parts)

def _update_intersected_figures(settings: Settings, rdm_plotter: "RDMPlotter") -> None:
    rdm_dump_dir     = settings.subject_rdm_dir
    intersected_dir  = settings.intersected_rdm_dir
    cache_dir        = _rdm_cache_dir(settings)
    intersected_dir.mkdir(parents=True, exist_ok=True)
    available = list_dumped_subjects(rdm_dump_dir)
    if not available: return

    sig_file  = intersected_dir / ".subjects_signature"
    current_sig = _intersected_signature(rdm_dump_dir, available)

    test_dirs = [Path(settings.vis_output_dir) / "phase2_rdms/intersected_stimuli_rdms/sorted_independent_within_category" / m / "fusiform" for m in LINKAGE_METHODS]
    dirs_exist = all(d.exists() for d in test_dirs)

    if sig_file.exists() and sig_file.read_text().strip() == current_sig and dirs_exist:
        logger.info("(intersected) Signature unchanged and all method dirs exist – skipping recompute.")
        return

    excluded_sids = set(settings.aggregate_exclude_subjects)

    common_stims_by_state = {"conscious": None, "unconscious": None}
    for sid in available:
        try:
            state_dict = load_subject_rdms(sid, rdm_dump_dir)
            for state in ("conscious", "unconscious"):
                if state in state_dict and state_dict[state]:
                    roi = next(iter(state_dict[state]))
                    s_list = state_dict[state][roi].stimulus_names.tolist()
                    if common_stims_by_state[state] is None:
                        common_stims_by_state[state] = set(s_list)
                        if f"{state}_ref_labels" not in common_stims_by_state:
                            common_stims_by_state[f"{state}_ref_labels"] = dict(zip(s_list, state_dict[state][roi].labels.tolist()))
                    else:
                        common_stims_by_state[state] = common_stims_by_state[state].intersection(set(s_list))
            del state_dict
        except Exception: pass
        gc.collect()

    for state in ("conscious", "unconscious"):
        if common_stims_by_state[state]:
            ref_dict = common_stims_by_state[f"{state}_ref_labels"]
            common_stims_by_state[state] = np.array(sorted(common_stims_by_state[state], key=lambda x: (-ref_dict.get(x, 0), x)))
        else:
            common_stims_by_state[state] = np.array([])

    for sid in available:
        try:
            state_dict = load_subject_rdms(sid, rdm_dump_dir)
            subj_intersected = {}
            for state, common_stims in common_stims_by_state.items():
                if state in ("conscious", "unconscious") and len(common_stims) > 0 and state in state_dict:
                    subj_intersected[state] = {}
                    for roi, rdm in state_dict[state].items():
                        stim_to_idx = {s: i for i, s in enumerate(rdm.stimulus_names.tolist())}
                        idx = np.array([stim_to_idx[s] for s in common_stims if s in stim_to_idx])
                        if len(idx) == len(common_stims):
                            subj_intersected[state][roi] = RDMBuilder().build_from_matrix(
                                rdm.matrix[np.ix_(idx, idx)], common_stims, rdm.labels[idx], roi, sid, state
                            )

            if subj_intersected:
                dump_intersected_subject_rdms(sid, subj_intersected, intersected_dir)
                for state in subj_intersected:
                    for roi in subj_intersected[state]:
                        if state == "conscious" and "unconscious" in subj_intersected and roi in subj_intersected["unconscious"]:
                            rdm_plotter.plot_dual_state(
                                subj_intersected["conscious"][roi], subj_intersected["unconscious"][roi],
                                suptitle=f"Subject {sid} · Intersected RDM\n{roi.title()} (C|U)",
                                save_name=f"rdm_dual_{sid}_{roi}.png", subdir=f"phase2_rdms/intersected_stimuli_rdms/{roi}"
                            )
                        rdm_item = subj_intersected[state][roi]
                        for link_method in LINKAGE_METHODS:
                            rdm_plotter.plot_sorted_rdm(
                                rdm_item, title_prefix=f"{sid}  ·  {state.title()}  ·  ",
                                save_name=f"rdm_sorted_indep_{link_method}_{sid}_{state}_{roi}.png",
                                subdir=f"phase2_rdms/intersected_stimuli_rdms/sorted_independent/{link_method}/{roi}",
                                linkage_method=link_method, within_category=False
                            )
                            rdm_plotter.plot_sorted_rdm(
                                rdm_item, title_prefix=f"{sid}  ·  {state.title()}  ·  ",
                                save_name=f"rdm_sorted_indep_within_cat_{link_method}_{sid}_{state}_{roi}.png",
                                subdir=f"phase2_rdms/intersected_stimuli_rdms/sorted_independent_within_category/{link_method}/{roi}",
                                linkage_method=link_method, within_category=True
                            )
            del state_dict
            del subj_intersected
        except Exception: pass

        gc.collect()

    included_sids = [s for s in available if s not in excluded_sids]
    agg_rdms = {m: {s: {} for s in ("conscious", "unconscious")} for m in _AGG_METHODS}

    for state in ("conscious", "unconscious"):
        common_stims = common_stims_by_state.get(state, np.array([]))
        if len(common_stims) == 0: continue

        state_rdms = []
        for sid in included_sids:
            try:
                intersected = load_intersected_subject_rdms(sid, intersected_dir)
                if state in intersected:
                    state_rdms.append((sid, intersected.pop(state)))
                del intersected
            except Exception: pass
            gc.collect()

        if not state_rdms: continue
        all_rois = sorted({roi for sid, rdict in state_rdms for roi in rdict})

        for roi in all_rois:
            roi_rdm_list = [rdict[roi] for sid, rdict in state_rdms if roi in rdict]
            if not roi_rdm_list: continue

            bary_matrix = gw_consensus_matrix(roi_rdm_list) if len(roi_rdm_list) >= 2 else None

            for method in _AGG_METHODS:
                agg = aggregate_rdm(roi_rdm_list, roi, state, method)
                agg_rdms[method][state][roi] = agg
                rdm_plotter.plot_mean_rdm(
                    agg, save_name=f"rdm_{method}_{state}_{roi}.png",
                    subdir=f"phase2_rdms/intersected_stimuli_rdms/{method}/{roi}", included_subjects=included_sids
                )

                for link_method in LINKAGE_METHODS:
                    rdm_plotter.plot_sorted_rdm(
                        agg, title_prefix=f"{agg.subject_id}  ·  {state.title()}  ·  ",
                        save_name=f"rdm_sorted_indep_{link_method}_{agg.subject_id}_{state}_{roi}.png",
                        subdir=f"phase2_rdms/intersected_stimuli_rdms/sorted_independent/{link_method}/{roi}",
                        included_subjects=included_sids, linkage_method=link_method, within_category=False
                    )
                    rdm_plotter.plot_sorted_rdm(
                        agg, title_prefix=f"{agg.subject_id}  ·  {state.title()}  ·  ",
                        save_name=f"rdm_sorted_indep_within_cat_{link_method}_{agg.subject_id}_{state}_{roi}.png",
                        subdir=f"phase2_rdms/intersected_stimuli_rdms/sorted_independent_within_category/{link_method}/{roi}",
                        included_subjects=included_sids, linkage_method=link_method, within_category=True
                    )

            if bary_matrix is not None:
                all_rdms_for_sort = roi_rdm_list + [agg_rdms[m][state][roi] for m in _AGG_METHODS]
                for rdm_item in all_rdms_for_sort:
                    is_agg = rdm_item.subject_id in _AGG_METHODS
                    for link_method in LINKAGE_METHODS:
                        c_order, c_k, c_score = sorted_order(bary_matrix, method=link_method)
                        rdm_plotter.plot_sorted_rdm(
                            rdm_item, title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                            save_name=f"rdm_sorted_consensus_{link_method}_{rdm_item.subject_id}_{state}_{roi}.png",
                            subdir=f"phase2_rdms/intersected_stimuli_rdms/sorted_consensus/{link_method}/{roi}",
                            common_order=c_order, best_k=c_k, best_score=c_score,
                            included_subjects=included_sids if is_agg else None, linkage_method=link_method, within_category=False
                        )
                        c_order_wc, c_k_wc, c_score_wc = sorted_order_within_category(bary_matrix, rdm_item.labels, method=link_method)
                        rdm_plotter.plot_sorted_rdm(
                            rdm_item, title_prefix=f"{rdm_item.subject_id}  ·  {state.title()}  ·  ",
                            save_name=f"rdm_sorted_consensus_within_cat_{link_method}_{rdm_item.subject_id}_{state}_{roi}.png",
                            subdir=f"phase2_rdms/intersected_stimuli_rdms/sorted_consensus_within_category/{link_method}/{roi}",
                            common_order=c_order_wc, best_k=c_k_wc, best_score=c_score_wc,
                            included_subjects=included_sids if is_agg else None, linkage_method=link_method, within_category=True
                        )

            if 'all_rdms_for_sort' in locals(): del all_rdms_for_sort
            if 'bary_matrix' in locals(): del bary_matrix
            del roi_rdm_list
            gc.collect()

        del state_rdms
        gc.collect()

    dump_intersected_aggregate_rdms(agg_rdms, intersected_dir)
    sig_file.write_text(current_sig)
    logger.info("(intersected) Archive updated: %d subjects, intersected dir: %s", len(available), intersected_dir)

def load_rdms_from_disk(settings: Settings, subject_ids: list[str] | None = None) -> tuple[dict, dict]:
    logger.info("=" * 60)
    logger.info("PHASE 2 (disk-load mode) – Reconstructing RDMs from archives")
    logger.info("=" * 60)

    rdm_dump_dir = settings.subject_rdm_dir
    available    = list_dumped_subjects(rdm_dump_dir)

    if not available:
        raise RuntimeError(f"No per-subject archives under {rdm_dump_dir}.")

    ids_to_load = subject_ids if subject_ids else available
    human_rdms: dict = {}
    discovered_rois: set[str] = set()

    for sid in ids_to_load:
        if sid not in available: continue
        try:
            state_rdms = load_subject_rdms(sid, rdm_dump_dir)
            human_rdms[sid] = state_rdms
            for state_dict in state_rdms.values():
                discovered_rois.update(state_dict.keys())
        except Exception as exc: pass

    settings.register_active_rois(sorted(discovered_rois))
    agg_rdms = load_aggregate_rdms(rdm_dump_dir) or {m: {s: {} for s in ("conscious", "unconscious")} for m in ("mean", "median")}
    human_rdms["_agg_rdms"]  = agg_rdms
    human_rdms["_mean_rdms"] = agg_rdms.get("mean", {})

    fcnn_rdms = load_fcnn_rdms(rdm_dump_dir)
    return human_rdms, fcnn_rdms