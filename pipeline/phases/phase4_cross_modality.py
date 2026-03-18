"""Phase 4 – 2×2 Cross-Modality Alignment.

Contains two independent analyses:

1. Per-ROI cross-modality RSA & GW  (individual subjects ↔ FCNN)
   – Unchanged from original pipeline.
   – Mean/median aggregate RDMs are intentionally NOT used here because
     averaging compresses noise and inflates ρ relative to individual subjects,
     making the numbers non-comparable with the per-subject baseline.

2. ROI × ROI second-order RDM  (Option B)
   – For each aggregation method (mean, median) and each visibility state,
     build an n_roi × n_roi matrix whose cell (i, j) is Spearman ρ between
     the aggregate RDM of ROI i and ROI j.
   – Reveals which brain regions share similar representational geometries.
   – Noise-ceiling-normalised ρ values are stored in the stats JSON.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from config.settings import Settings
from analysis.rsa.rsa_analyzer import RSAAnalyzer
from analysis.rsa.noise_ceiling import NoiseCeiling
from analysis.gromov_wasserstein.gw_aligner import GromovWassersteinAligner
from utils.io_utils import save_json

logger = logging.getLogger(__name__)


def run(
    settings: Settings,
    human_rdms: dict,
    fcnn_rdms: dict,
    rsa_analyzer: RSAAnalyzer,
    gw_aligner: GromovWassersteinAligner,
    noise_ceiling: NoiseCeiling | None = None,
) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 4 – 2×2 Cross-Modality Alignment")
    logger.info("=" * 60)

    from visualization.phase3_plotter import Phase3Plotter
    from visualization.rdm_plotter import RDMPlotter
    p3         = Phase3Plotter(settings)
    rdm_plt    = RDMPlotter(settings)

    summary: dict        = {}
    alignment_pairs      = [("clear", "conscious", "C-C"), ("chance", "unconscious", "U-U")]
    agg_rdms: dict       = human_rdms.get("_agg_rdms", {})

    # ────────────────────────────────────────────────────────────────
    # 1.  Per-ROI cross-modality RSA & GW  (individual subjects)
    # ────────────────────────────────────────────────────────────────
    for noise_state, human_state, label in alignment_pairs:
        if noise_state not in fcnn_rdms:
            logger.info(
                "FCNN '%s' RDMs not available – skipping %s.", noise_state, label
            )
            continue

        summary[label] = {}
        for roi in settings.active_roi_names:
            fcnn_rdm = fcnn_rdms[noise_state].get("fcnn_hidden")
            if fcnn_rdm is None:
                continue

            human = [
                human_rdms[sid][human_state][roi]
                for sid in human_rdms
                if isinstance(human_rdms[sid], dict)
                and human_state in human_rdms[sid]
                and roi in human_rdms[sid][human_state]
            ]
            if not human:
                continue

            rsa_results = rsa_analyzer.cross_modality_rsa(human, fcnn_rdm)
            mean_rho    = rsa_analyzer.mean_rho(rsa_results)

            all_rdms  = human + [fcnn_rdm]
            ids       = [f"{r.subject_id}_{r.state}" for r in all_rdms]
            gw_matrix = gw_aligner.build_pairwise_distance_matrix(all_rdms, ids)

            fcnn_gw = [
                r for r in gw_matrix.results
                if "fcnn" in r.source_id or "fcnn" in r.target_id
            ]
            mean_top_k = (
                float(sum(r.top_k_matching_rate for r in fcnn_gw) / len(fcnn_gw))
                if fcnn_gw else 0.0
            )

            # Noise-ceiling-normalised ρ (stored in JSON only; not plotted)
            nc_rho: float | None = None
            if noise_ceiling is not None:
                nc = noise_ceiling.compute(human)
                nc_upper = nc.get("upper", None)
                if nc_upper and nc_upper > 1e-8:
                    nc_rho = round(mean_rho / nc_upper, 4)

            summary[label][roi] = {
                "mean_rsa_rho":              mean_rho,
                "mean_top_k_matching_rate":  mean_top_k,
                "n_rsa_significant":         sum(r.significant for r in rsa_results),
                "nc_normalised_rho":         nc_rho,
            }
            logger.info(
                "  %s | ROI %s | RSA ρ=%.3f | Top-k rate=%.2f%s",
                label, roi, mean_rho, mean_top_k,
                f" | NC-norm ρ={nc_rho:.3f}" if nc_rho is not None else "",
            )

            gw_matrix.state        = f"{label}_{roi}"
            gw_matrix.roi_or_layer = roi
            p3.plot_gw_matrix(
                gw_matrix,
                title=f"GW Distance Matrix  ({label})\nROI: {roi}",
                save_name=f"phase4_gw_matrix_{roi}_{label}.png",
                subdir=f"phase4_cross_modality/{roi}",
            )

    # ────────────────────────────────────────────────────────────────
    # 2.  ROI × ROI second-order RDM  (mean & median)
    # ────────────────────────────────────────────────────────────────
    # Cell (i, j) = Spearman ρ between upper-triangle vectors of ROI_i and ROI_j
    # aggregate RDMs.  High ρ means the two ROIs encode stimuli similarly.
    # Separate matrices for each (method × state) combination.
    #
    roi_x_roi_stats: dict = {}

    for method, state_dict in agg_rdms.items():
        for state, roi_dict in state_dict.items():
            rois_available = [
                r for r in settings.active_roi_names if r in roi_dict
            ]
            n = len(rois_available)
            if n < 2:
                continue

            rho_matrix = np.full((n, n), np.nan)
            p_matrix   = np.full((n, n), np.nan)

            # Vectorise each aggregate RDM upper triangle once
            vecs = [
                _upper_tri(roi_dict[roi].matrix) for roi in rois_available
            ]

            for i in range(n):
                rho_matrix[i, i] = 1.0
                p_matrix[i, i]   = 0.0
                for j in range(i + 1, n):
                    rho, pval = spearmanr(vecs[i], vecs[j])
                    rho_matrix[i, j] = rho_matrix[j, i] = float(rho)
                    p_matrix[i, j]   = p_matrix[j, i]   = float(pval)

            rdm_plt.plot_roi_x_roi_rdm(
                rho_matrix=rho_matrix,
                p_matrix=p_matrix,
                roi_names=rois_available,
                method=method,
                state=state,
                save_name=f"phase4_roi_x_roi_{method}_{state}.png",
                subdir="phase4_cross_modality/roi_x_roi",
            )

            roi_x_roi_stats[f"{method}_{state}"] = {
                rois_available[i]: {
                    rois_available[j]: {
                        "rho": round(float(rho_matrix[i, j]), 4),
                        "p":   round(float(p_matrix[i, j]),   6),
                    }
                    for j in range(n) if i != j
                }
                for i in range(n)
            }
            logger.info(
                "  ROI×ROI second-order RDM saved: method=%s state=%s (%d ROIs).",
                method, state, n,
            )

    summary["roi_x_roi"] = roi_x_roi_stats
    save_json(summary, Path(settings.stats_dir) / "phase4_cross_modality.json")
    logger.info("Phase 4 complete.\n")
    return summary


def _upper_tri(matrix: np.ndarray) -> np.ndarray:
    """Return the upper-triangle (k=1) values of a square matrix as a 1-D vector."""
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]
