"""Phase 4 – 2×2 Cross-Modality Alignment."""
from __future__ import annotations

import logging
from pathlib import Path

from config.settings import Settings
from analysis.rsa.rsa_analyzer import RSAAnalyzer
from analysis.gromov_wasserstein.gw_aligner import GromovWassersteinAligner
from utils.io_utils import save_json

logger = logging.getLogger(__name__)


def run(
    settings: Settings,
    human_rdms: dict,
    fcnn_rdms: dict,
    rsa_analyzer: RSAAnalyzer,
    gw_aligner: GromovWassersteinAligner,
) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 4 – 2×2 Cross-Modality Alignment")
    logger.info("=" * 60)

    from visualization.phase3_plotter import Phase3Plotter
    p3 = Phase3Plotter(settings)

    summary: dict = {}
    alignment_pairs = [("clear", "conscious", "C-C"), ("chance", "unconscious", "U-U")]

    for noise_state, human_state, label in alignment_pairs:
        if noise_state not in fcnn_rdms:
            logger.info("FCNN '%s' RDMs not available – skipping %s.", noise_state, label)
            continue

        summary[label] = {}
        for roi in settings.active_roi_names:              # ← active_roi_names
            fcnn_rdm = fcnn_rdms[noise_state].get("fcnn_hidden")
            if fcnn_rdm is None:
                continue

            human = [
                human_rdms[sid][human_state][roi]
                for sid in human_rdms
                if human_state in human_rdms[sid] and roi in human_rdms[sid][human_state]
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

            summary[label][roi] = {
                "mean_rsa_rho":             mean_rho,
                "mean_top_k_matching_rate": mean_top_k,
                "n_rsa_significant":        sum(r.significant for r in rsa_results),
            }
            logger.info(
                "  %s | ROI %s | RSA ρ=%.3f | Top-k rate=%.2f",
                label, roi, mean_rho, mean_top_k,
            )

            gw_matrix.state        = f"{label}_{roi}"
            gw_matrix.roi_or_layer = roi
            p3.plot_gw_matrix(
                gw_matrix,
                title=f"GW Distance Matrix  ({label})\nROI: {roi}",
                save_name=f"phase4_gw_matrix_{roi}_{label}.png",
                subdir=f"phase4_cross_modality/{roi}",
            )

    save_json(summary, Path(settings.stats_dir) / "phase4_cross_modality.json")
    logger.info("Phase 4 complete.\n")
    return summary
