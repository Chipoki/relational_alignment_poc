"""scripts/run_pipeline.py – End-to-end POC pipeline orchestrator.

Runs all six phases of the action plan:
  Phase 1  – Dual-State Embedding Extraction
  Phase 2  – Dual-State Intra-Modality RDM Construction
  Phase 3  – Balanced Inter-Subject Representational Analysis
  Phase 4  – 2 × 2 Cross-Modality Alignment
  Phase 5  – The Structural Invariance Metric
  Phase 6  – Relational Visualizations

Usage
-----
    python scripts/run_pipeline.py --config config/config.yaml [--subjects sub-01 sub-02]
"""
from __future__ import annotations

import argparse
import logging
import sys
import re
from pathlib import Path

import numpy as np

# ── Ensure project root is on sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings
from data.loaders.subject import Subject
from data.preprocessors.subject_builder import SubjectBuilder
from embeddings.fcnn_embedder import FCNNEmbedder
from embeddings.fmri_embedder import FMRIEmbedder
from embeddings.embedding_store import EmbeddingStore
from analysis.rsa.rdm import RDMBuilder, RDM
from analysis.rsa.rsa_analyzer import RSAAnalyzer, RSAResult
from analysis.rsa.noise_ceiling import NoiseCeiling
from analysis.gromov_wasserstein.gw_aligner import GromovWassersteinAligner
from analysis.gromov_wasserstein.gw_result import GWDistanceMatrix
from visualization.rdm_plotter import RDMPlotter
from visualization.meta_mds_plotter import MetaMDSPlotter
from visualization.transport_plotter import TransportPlotter
from visualization.summary_plotter import SummaryPlotter
from utils.logging_utils import setup_logging
from utils.io_utils import save_json, ensure_dir

logger = logging.getLogger(__name__)


def _normalize_name(s: str) -> str:
    """
    Strip extensions, punctuation, and whitespace for robust matching
    between behavioural CSV labels and raw image filenames.
    """
    return re.sub(r'[^a-z0-9]', '', Path(str(s)).stem.lower())


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline class
# ─────────────────────────────────────────────────────────────────────────────

class POCPipeline:
    """
    Orchestrates the full six-phase POC analysis.

    Each phase is a self-contained method so individual phases can be
    re-run independently during development without re-processing earlier ones.
    """

    def __init__(self, settings: Settings) -> None:
        self._cfg = settings
        settings.ensure_output_dirs()

        # Component instances
        self._subject_builder = SubjectBuilder(settings)
        self._fcnn_embedder = FCNNEmbedder(settings)
        self._fmri_embedder = FMRIEmbedder(settings)
        self._embedding_store = EmbeddingStore(settings.embedding_dir)
        self._rdm_builder = RDMBuilder()
        self._rsa_analyzer = RSAAnalyzer(settings)
        self._noise_ceiling = NoiseCeiling()
        self._gw_aligner = GromovWassersteinAligner(settings)

        # Visualisers
        self._rdm_plotter = RDMPlotter(settings)
        self._meta_mds_plotter = MetaMDSPlotter(settings)
        self._transport_plotter = TransportPlotter(settings)
        self._summary_plotter = SummaryPlotter(settings)

        # State containers populated progressively
        self._subjects: list[Subject] = []
        self._human_rdms: dict[str, dict[str, dict[str, RDM]]] = {}
        # Layout: state → subject_id → roi → RDM
        self._fcnn_rdms: dict[str, dict[str, RDM]] = {}
        # Layout: noise_state → roi → RDM

    # ── Phase 0 – Data loading ───────────────────────────────────────────────

    def load_subjects(self, subject_ids: list[str] | None = None) -> None:
        """Discover and load all subjects from the data root directory."""
        root = Path(self._cfg.data["root"])
        if not root.exists():
            raise FileNotFoundError(
                f"Data root not found: {root}\n"
                "Please set data.root in config/config.yaml and populate it with subject folders."
            )

        ids = subject_ids or self._cfg.data.get("subject_ids") or []
        if not ids:
            # Auto-discover: any direct subfolder
            ids = sorted(p.name for p in root.iterdir() if p.is_dir())
        if not ids:
            raise ValueError(f"No subject directories found under {root}")

        logger.info("Loading %d subjects: %s", len(ids), ids)
        for sid in ids:
            subj_dir = root / sid
            try:
                subj = self._subject_builder.build(subj_dir, sid)
                self._subjects.append(subj)
                logger.info("✓ Loaded %s", subj)
            except Exception as exc:
                logger.error("✗ Failed to load subject %s: %s", sid, exc)

        logger.info("Loaded %d / %d subjects successfully", len(self._subjects), len(ids))

    # ── Phase 1 – Embedding extraction ──────────────────────────────────────

    def phase1_extract_embeddings(
        self,
        stimulus_image_dir: str | Path | None = None,
    ) -> None:
        """
        Phase 1: Extract FCNN hidden-layer embeddings for clear and noisy images.
        """
        logger.info("=" * 60)
        logger.info("PHASE 1 – Dual-State Embedding Extraction")
        logger.info("=" * 60)

        if stimulus_image_dir is None:
            logger.warning(
                "No stimulus image directory provided. "
                "FCNN embeddings will be skipped. "
                "Pass --stimulus-dir to enable."
            )
            return

        img_dir = Path(stimulus_image_dir)

        # Use rglob for recursive directory searching (finds images in nested class folders)
        image_paths = sorted(img_dir.rglob("*.jpg")) + sorted(img_dir.rglob("*.png"))
        if not image_paths:
            logger.warning("No images found in %s (or its subdirectories) – FCNN phase skipped.", img_dir)
            return

        logger.info("Found %d stimulus images in %s", len(image_paths), img_dir)

        for noise_state in ("clear", "chance"):
            store_key = f"fcnn_{noise_state}"
            names_key = f"fcnn_{noise_state}_names"

            # If we already have the embeddings and their associated names, skip extraction
            if self._embedding_store.exists(store_key) and self._embedding_store.exists(names_key):
                logger.info("Cached FCNN embeddings found for '%s' – skipping extraction.", noise_state)
            else:
                logger.info("Extracting FCNN embeddings (noise_state=%s)...", noise_state)
                embeddings = self._fcnn_embedder.extract_embeddings(image_paths, noise_state=noise_state)

                # Save the image stems (filenames without extension) for perfect alignment in Phase 2
                image_names = np.array([p.stem for p in image_paths])

                self._embedding_store.save(store_key, embeddings)
                self._embedding_store.save(names_key, image_names)
                logger.info("Saved FCNN embeddings (%s) → shape %s", noise_state, embeddings.shape)

        logger.info("Phase 1 complete.\n")

    # ── Phase 2 – RDM Construction ───────────────────────────────────────────

    def phase2_build_rdms(self) -> None:
        """
        Phase 2: Construct aligned RDMs for each subject × state × ROI,
        and for FCNN clear/chance states by averaging trials per unique stimulus.
        """
        logger.info("=" * 60)
        logger.info("PHASE 2 – Dual-State Intra-Modality RDM Construction")
        logger.info("=" * 60)

        rdm_dir = Path(self._cfg.rdm_dir)
        ensure_dir(rdm_dir)

        # 1. Find common unique stimuli across ALL subjects for each state
        common_stims_by_state = {}
        for state in ("conscious", "unconscious"):
            name_sets = []
            for subj in self._subjects:
                vis = getattr(subj, state, None)
                if vis is not None:
                    name_sets.append(set(vis.stimulus_names))
            if name_sets:
                # Intersect to guarantee identical items for inter-subject RSA
                common_stims_by_state[state] = list(set.intersection(*name_sets))
            else:
                common_stims_by_state[state] = []

        ref_subj = self._subjects[0] if self._subjects else None

        # ---------------------------------------------------------
        # Human RDMs
        # ---------------------------------------------------------
        for subj in self._subjects:
            self._human_rdms[subj.subject_id] = {}
            for state in ("conscious", "unconscious"):
                vis_data = getattr(subj, state, None)
                common_stims = common_stims_by_state.get(state, [])

                if vis_data is None or not common_stims:
                    logger.warning(
                        "Subject %s: no valid '%s' data or common stimuli – skipping RDMs",
                        subj.subject_id,
                        state,
                    )
                    continue

                # Build a mapping of stimulus name to category label for this subject
                stim_to_label = {
                    s: l for s, l in zip(vis_data.stimulus_names, vis_data.labels)
                }

                # Sort stimuli to keep bipartite heatmap logic: Living (1) first, then alphabetical
                sorted_stims = sorted(
                    common_stims, key=lambda x: (-stim_to_label.get(x, 0), x)
                )

                aligned_labels = np.array([stim_to_label[s] for s in sorted_stims])
                aligned_stim_names = np.array(sorted_stims)

                # Average trial repetitions per unique stimulus
                roi_averaged = {roi: [] for roi in vis_data.bold_patterns.keys()}
                for stim in sorted_stims:
                    idx = np.where(vis_data.stimulus_names == stim)[0]
                    for roi, patterns in vis_data.bold_patterns.items():
                        roi_averaged[roi].append(patterns[idx].mean(axis=0))

                # --- FIX: Sanitize fMRI patterns (Remove NaNs & Zero-Variance Voxels) ---
                for roi in roi_averaged:
                    arr = np.array(roi_averaged[roi])

                    # Clean Z-score division by zero
                    arr = np.nan_to_num(arr, nan=0.0)

                    # Drop voxels with absolutely zero variance to prevent NaN correlations
                    variances = np.var(arr, axis=0)
                    valid_mask = variances > 1e-8

                    if valid_mask.sum() > 0:
                        arr = arr[:, valid_mask]

                    roi_averaged[roi] = arr
                # -------------------------------------------------------------------------

                state_rdms = self._rdm_builder.build_from_embeddings(
                    embeddings=roi_averaged,
                    stimulus_names=aligned_stim_names,
                    labels=aligned_labels,
                    subject_id=subj.subject_id,
                    state=state,
                )
                self._human_rdms[subj.subject_id][state] = state_rdms
                logger.info(
                    "Built %d RDMs for subject %s, state=%s (n_stimuli=%d)",
                    len(state_rdms),
                    subj.subject_id,
                    state,
                    len(sorted_stims),
                )

        # ---------------------------------------------------------
        # FCNN RDMs
        # ---------------------------------------------------------
        human_state_map = {"clear": "conscious", "chance": "unconscious"}

        for noise_state, store_key in [("clear", "fcnn_clear"), ("chance", "fcnn_chance")]:
            names_key = f"{store_key}_names"
            if not self._embedding_store.exists(store_key) or not self._embedding_store.exists(names_key):
                logger.info("FCNN embeddings not found for '%s' – skipping FCNN RDMs.", noise_state)
                continue

            fcnn_emb = self._embedding_store.load(store_key)   # (n_images, n_units)
            fcnn_names = self._embedding_store.load(names_key)  # (n_images,) string array

            # FIX: Map FCNN noise_state to appropriate corresponding human visibility state stimuli
            human_state = human_state_map[noise_state]
            ref_stims = common_stims_by_state.get(human_state, [])

            if not ref_stims:
                logger.warning("No common stimuli found for human state '%s' to align with FCNN '%s'.", human_state, noise_state)
                continue

            # Need to get labels from a subject that has this state loaded
            vis_data = None
            for s in self._subjects:
                vis_data = getattr(s, human_state, None)
                if vis_data is not None:
                    break

            if vis_data is None:
                continue

            stim_to_label = {
                s: l for s, l in zip(
                    vis_data.stimulus_names, vis_data.labels
                )
            }

            sorted_ref_stims = sorted(
                ref_stims, key=lambda x: (-stim_to_label.get(x, 0), x)
            )

            # --- FIX: Robust String Matching ---
            # Map FCNN image stem to its row index in the embedding matrix using normalized strings
            stim_to_fcnn_idx = {_normalize_name(name): i for i, name in enumerate(fcnn_names)}

            aligned_fcnn_emb = []
            valid_labels = []
            valid_stims = []

            for stim in sorted_ref_stims:
                norm_stim = _normalize_name(stim)
                if norm_stim in stim_to_fcnn_idx:
                    idx = stim_to_fcnn_idx[norm_stim]
                    aligned_fcnn_emb.append(fcnn_emb[idx])
                    valid_labels.append(stim_to_label[stim])
                    valid_stims.append(stim)
            # -----------------------------------

            if not aligned_fcnn_emb:
                logger.warning(
                    "Failed to align FCNN stimuli with human stimuli for %s. "
                    "Check naming conventions.",
                    noise_state,
                )
                continue

            fcnn_rdm = self._rdm_builder.build_vectorised(
                patterns=np.array(aligned_fcnn_emb),
                stimulus_names=np.array(valid_stims),
                labels=np.array(valid_labels),
                roi_or_layer="fcnn_hidden",
                subject_id=f"fcnn_{noise_state}",
                state=noise_state,
            )
            self._fcnn_rdms[noise_state] = {"fcnn_hidden": fcnn_rdm}
            logger.info("Built FCNN RDM for noise_state=%s (n_stimuli=%d)", noise_state, len(valid_stims))

        # ---------------------------------------------------------
        # Phase 2 Visualization: Dual-state RDMs — ALL subjects
        # ---------------------------------------------------------
        # FIX: was `for subj in self._subjects[:2]` — a debug slice that was
        # never removed, causing plots to be saved for sub-01 and sub-02 only.
        # Changed to iterate over all subjects.
        for subj in self._subjects:
            sid = subj.subject_id
            c_rois = list(self._human_rdms.get(sid, {}).get("conscious", {}).keys())

            if not c_rois:
                continue

            # Prefer fusiform if it exists, otherwise use the first one available
            example_roi = "fusiform" if "fusiform" in c_rois else c_rois[0]

            c_rdm = self._human_rdms.get(sid, {}).get("conscious", {}).get(example_roi)
            u_rdm = self._human_rdms.get(sid, {}).get("unconscious", {}).get(example_roi)

            if c_rdm and u_rdm:
                # FIX: pass n_stimuli so the plotter can annotate each panel,
                # and use improved titles / axis labels (handled inside
                # RDMPlotter.plot_dual_state — see visualization/rdm_plotter.py).
                self._rdm_plotter.plot_dual_state(
                    c_rdm, u_rdm,
                    suptitle=(
                        f"Subject {sid}  ·  Representational Dissimilarity Matrix\n"
                        f"{example_roi.replace('_', ' ').title()}  "
                        f"(Conscious | Unconscious)"
                    ),
                    save_name=f"rdm_dual_{sid}_{example_roi}.png",
                )
                logger.info("Saved dual-state RDM plot for %s / %s", sid, example_roi)

        logger.info("Phase 2 complete.\n")

    # ── Phase 3 – Inter-subject RSA ──────────────────────────────────────────

    def phase3_inter_subject_rsa(self) -> dict:
        """
        Phase 3: Compute all inter-subject RSA correlations and noise ceiling.
        Returns summary statistics dict.
        """
        logger.info("=" * 60)
        logger.info("PHASE 3 – Balanced Inter-Subject Representational Analysis")
        logger.info("=" * 60)

        summary: dict = {}

        # Supervised RSA and Noise Ceiling
        for state in ("conscious", "unconscious"):
            logger.info("Computing inter-subject RSA for state=%s", state)
            state_summary: dict = {}

            for roi in self._cfg.roi_names:
                roi_rdms = [
                    self._human_rdms[sid][state][roi]
                    for sid in self._human_rdms
                    if state in self._human_rdms[sid]
                    and roi in self._human_rdms[sid][state]
                ]
                if len(roi_rdms) < 2:
                    continue

                rsa_results = self._rsa_analyzer.inter_subject_rsa(roi_rdms)
                mean_rho = self._rsa_analyzer.mean_rho(rsa_results)
                nc = self._noise_ceiling.compute(roi_rdms)

                state_summary[roi] = {
                    "mean_rho": mean_rho,
                    "n_pairs": len(rsa_results),
                    "n_significant": sum(r.significant for r in rsa_results),
                    "noise_ceiling_upper": nc["upper"],
                    "noise_ceiling_lower": nc["lower"],
                }
                logger.info(
                    "  ROI %s | state=%s | mean ρ=%.3f | NC upper=%.3f lower=%.3f",
                    roi, state, mean_rho, nc["upper"], nc["lower"],
                )

            summary[state] = state_summary

        # Inter-subject GW alignment (unsupervised baseline)
        for state in ("conscious", "unconscious"):
            logger.info("Computing inter-subject GW alignment (state=%s)…", state)
            for roi in self._cfg.roi_names:
                roi_rdms = [
                    self._human_rdms[sid][state][roi]
                    for sid in self._human_rdms
                    if state in self._human_rdms[sid]
                    and roi in self._human_rdms[sid][state]
                ]
                if len(roi_rdms) < 2:
                    continue

                ids = [
                    f"{sid}_{state}"
                    for sid in self._human_rdms
                    if state in self._human_rdms[sid]
                ]
                gw_matrix = self._gw_aligner.build_pairwise_distance_matrix(roi_rdms, ids)
                logger.info(
                    "  ROI %s | GW mean distance=%.4f",
                    roi,
                    gw_matrix.matrix[gw_matrix.matrix > 0].mean(),
                )

        save_json(summary, Path(self._cfg.stats_dir) / "phase3_inter_subject_rsa.json")

        # ---------------------------------------------------------
        # Phase 3 Visualization: Summary RSA bar chart
        # ---------------------------------------------------------
        c_results: list = []
        u_results: list = []

        # Reconstruct flat RSAResult-like objects for plotting
        for roi, metrics in summary.get("conscious", {}).items():
            c_results.append(RSAResult(
                subject_a="avg", subject_b="avg", roi_or_layer=roi,
                state_a="conscious", state_b="conscious",
                rho=metrics["mean_rho"], p_value=0.0, significant=True,
            ))
        for roi, metrics in summary.get("unconscious", {}).items():
            u_results.append(RSAResult(
                subject_a="avg", subject_b="avg", roi_or_layer=roi,
                state_a="unconscious", state_b="unconscious",
                rho=metrics["mean_rho"], p_value=0.0, significant=True,
            ))

        if c_results or u_results:
            self._summary_plotter.plot_rsa_by_roi(
                c_results, u_results,
                roi_names=self._cfg.roi_names,
                save_name="phase3_rsa_by_roi.png",
            )
            logger.info("Saved inter-subject RSA bar chart.")

        logger.info("Phase 3 complete.\n")
        return summary

    # ── Phase 4 – Cross-modality alignment ──────────────────────────────────

    def phase4_cross_modality_alignment(self) -> dict:
        """
        Phase 4: 2×2 Cross-Modality Alignment (C-C and U-U).
        """
        logger.info("=" * 60)
        logger.info("PHASE 4 – 2×2 Cross-Modality Alignment")
        logger.info("=" * 60)

        summary: dict = {}

        alignment_pairs = [
            ("clear", "conscious", "C-C"),
            ("chance", "unconscious", "U-U"),
        ]

        for noise_state, human_state, label in alignment_pairs:
            if noise_state not in self._fcnn_rdms:
                logger.info(
                    "FCNN '%s' RDMs not available – skipping %s alignment.",
                    noise_state,
                    label,
                )
                continue

            logger.info(
                "Running %s alignment (FCNN %s ↔ human %s)...",
                label,
                noise_state,
                human_state,
            )
            summary[label] = {}

            for roi in self._cfg.roi_names:
                fcnn_rdm = self._fcnn_rdms[noise_state].get("fcnn_hidden")
                if fcnn_rdm is None:
                    continue

                human_rdms = [
                    self._human_rdms[sid][human_state][roi]
                    for sid in self._human_rdms
                    if human_state in self._human_rdms[sid]
                    and roi in self._human_rdms[sid][human_state]
                ]

                if not human_rdms:
                    continue

                # Supervised RSA
                rsa_results = self._rsa_analyzer.cross_modality_rsa(human_rdms, fcnn_rdm)
                mean_rho = self._rsa_analyzer.mean_rho(rsa_results)

                # Unsupervised GW
                all_rdms = human_rdms + [fcnn_rdm]
                ids = [f"{r.subject_id}_{r.state}" for r in all_rdms]
                gw_matrix = self._gw_aligner.build_pairwise_distance_matrix(all_rdms, ids)

                # Top-k matching rate (from GW results involving FCNN)
                fcnn_gw_results = [
                    res for res in gw_matrix.results
                    if "fcnn" in res.source_id or "fcnn" in res.target_id
                ]
                mean_top_k = (
                    float(
                        sum(r.top_k_matching_rate for r in fcnn_gw_results)
                        / len(fcnn_gw_results)
                    )
                    if fcnn_gw_results else 0.0
                )

                summary[label][roi] = {
                    "mean_rsa_rho": mean_rho,
                    "mean_top_k_matching_rate": mean_top_k,
                    "n_rsa_significant": sum(r.significant for r in rsa_results),
                }
                logger.info(
                    "  %s | ROI %s | RSA ρ=%.3f | Top-k rate=%.2f",
                    label, roi, mean_rho, mean_top_k,
                )

        save_json(summary, Path(self._cfg.stats_dir) / "phase4_cross_modality.json")
        logger.info("Phase 4 complete.\n")
        return summary

    # ── Phase 5 – Structural Invariance ────────────────────────────────────

    def phase5_structural_invariance(self) -> dict:
        """
        Phase 5: The Structural Invariance Metric.
        ΔGW_human (conscious→unconscious) vs. ΔGW_FCNN (clear→noisy).
        """
        logger.info("=" * 60)
        logger.info("PHASE 5 – Structural Invariance Metric")
        logger.info("=" * 60)

        if "clear" not in self._fcnn_rdms or "chance" not in self._fcnn_rdms:
            logger.warning("FCNN RDMs not available – Phase 5 skipped.")
            return {}

        summary: dict = {}

        for roi in self._cfg.roi_names:
            c_rdms = [
                self._human_rdms[sid]["conscious"][roi]
                for sid in self._human_rdms
                if "conscious" in self._human_rdms[sid]
                and roi in self._human_rdms[sid]["conscious"]
            ]
            u_rdms = [
                self._human_rdms[sid]["unconscious"][roi]
                for sid in self._human_rdms
                if "unconscious" in self._human_rdms[sid]
                and roi in self._human_rdms[sid]["unconscious"]
            ]
            fcnn_clear = self._fcnn_rdms["clear"].get("fcnn_hidden")
            fcnn_noisy = self._fcnn_rdms["chance"].get("fcnn_hidden")

            if not c_rdms or not u_rdms or fcnn_clear is None or fcnn_noisy is None:
                continue

            metrics = self._gw_aligner.structural_invariance_metric(
                c_rdms, u_rdms, fcnn_clear, fcnn_noisy
            )
            summary[roi] = metrics
            logger.info(
                "  ROI %s | ΔGW_human=%.4f ± %.4f | ΔGW_FCNN=%.4f | Bioplausible=%s",
                roi,
                metrics["delta_gw_human_mean"],
                metrics["delta_gw_human_std"],
                metrics["delta_gw_fcnn"],
                metrics["bioplausibility_check"],
            )

        save_json(summary, Path(self._cfg.stats_dir) / "phase5_structural_invariance.json")

        # ---------------------------------------------------------
        # Phase 5 Visualization: Structural invariance scatter
        # ---------------------------------------------------------
        if summary:
            self._summary_plotter.plot_structural_invariance(
                summary, self._cfg.roi_names,
                save_name="phase5_structural_invariance.png",
            )
            logger.info("Saved structural invariance scatter.")

        logger.info("Phase 5 complete.\n")
        return summary

    # ── Phase 6 – Visualisations ────────────────────────────────────────────

    def phase6_visualize(self) -> None:
        """
        Phase 6: Relational Visualizations (Meta-MDS).
        """
        logger.info("=" * 60)
        logger.info("PHASE 6 – Relational Visualizations (Meta-MDS)")
        logger.info("=" * 60)

        # Dynamic fallback for Meta-MDS ROI mapping
        sid0 = self._subjects[0].subject_id if self._subjects else None
        c_rois = (
            list(self._human_rdms.get(sid0, {}).get("conscious", {}).keys())
            if sid0 else []
        )
        example_roi = (
            "fusiform" if "fusiform" in c_rois else (c_rois[0] if c_rois else "wholebrain")
        )

        # Meta-MDS for key ROIs
        for roi in [example_roi, "lateral_occipital"]:
            if roi not in c_rois:
                continue

            all_rdms: list[RDM] = []
            ids: list[str] = []

            # Gather human RDMs
            for sid in self._human_rdms:
                for state in ("conscious", "unconscious"):
                    rdm = self._human_rdms[sid].get(state, {}).get(roi)
                    if rdm:
                        all_rdms.append(rdm)
                        ids.append(f"{sid}_{state[:3]}")

            # Gather AI RDMs
            for noise_state in ("clear", "chance"):
                rdm = self._fcnn_rdms.get(noise_state, {}).get("fcnn_hidden")
                if rdm:
                    all_rdms.append(rdm)
                    ids.append(f"fcnn_{noise_state}")

            if len(all_rdms) >= 3:
                gw_mat = self._gw_aligner.build_pairwise_distance_matrix(all_rdms, ids)
                human_ids = [i for i in ids if not i.startswith("fcnn")]
                model_ids = [i for i in ids if i.startswith("fcnn")]

                self._meta_mds_plotter.plot(
                    gw_mat, human_ids, model_ids,
                    title=f"Meta-MDS | {roi}",
                    save_name=f"phase6_meta_mds_{roi}.png",
                )
                logger.info("Saved meta-MDS for ROI %s.", roi)

        logger.info("Phase 6 complete.\n")

    # ── Full pipeline convenience method ────────────────────────────────────

    def run(
        self,
        subject_ids: list[str] | None = None,
        stimulus_image_dir: str | Path | None = None,
    ) -> None:
        """Run all phases end-to-end."""
        logger.info("Starting POC Pipeline")
        self.load_subjects(subject_ids)
        self.phase1_extract_embeddings(stimulus_image_dir)
        self.phase2_build_rdms()
        self.phase3_inter_subject_rsa()
        self.phase4_cross_modality_alignment()
        self.phase5_structural_invariance()
        self.phase6_visualize()
        logger.info("Pipeline complete. Results saved to: %s", self._cfg.results_dir)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the POC Relational Alignment Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--subjects", nargs="*", default=None,
        help="Subject IDs to process (default: all in data.root)",
    )
    parser.add_argument(
        "--stimulus-dir",
        default="./../soto_data/unconfeats/data/experiment_images_greyscaled",
        help="Directory containing stimulus PNG/JPG images for FCNN embedding",
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        choices=[1, 2, 3, 4, 5, 6],
        help="Run a single phase only (requires prior phases' outputs on disk)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging output level",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings(args.config)
    log_file = Path(settings.log_dir) / "pipeline.log"
    setup_logging(level=args.log_level, log_file=str(log_file))
    pipeline = POCPipeline(settings)

    if args.phase is None:
        pipeline.run(
            subject_ids=args.subjects,
            stimulus_image_dir=args.stimulus_dir,
        )
    else:
        # Phase-specific execution (assumes earlier phase outputs exist on disk)
        pipeline.load_subjects(args.subjects)
        dispatch = {
            1: lambda: pipeline.phase1_extract_embeddings(args.stimulus_dir),
            2: pipeline.phase2_build_rdms,
            3: pipeline.phase3_inter_subject_rsa,
            4: pipeline.phase4_cross_modality_alignment,
            5: pipeline.phase5_structural_invariance,
            6: pipeline.phase6_visualize,
        }
        dispatch[args.phase]()


if __name__ == "__main__":
    main()