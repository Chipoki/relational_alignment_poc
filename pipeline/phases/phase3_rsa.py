"""Phase 3 – Balanced Inter-Subject Representational Analysis."""
from __future__ import annotations

import logging
from pathlib import Path

from config.settings import Settings
from analysis.rsa.rdm import RDM
from analysis.rsa.rsa_analyzer import RSAAnalyzer, RSAResult
from analysis.rsa.noise_ceiling import NoiseCeiling
from analysis.gromov_wasserstein.gw_aligner import GromovWassersteinAligner
from visualization.summary_plotter import SummaryPlotter
from utils.io_utils import save_json

logger = logging.getLogger(__name__)


def run(
    settings: Settings,
    human_rdms: dict,
    rsa_analyzer: RSAAnalyzer,
    noise_ceiling: NoiseCeiling,
    gw_aligner: GromovWassersteinAligner,
    summary_plotter: SummaryPlotter,
) -> dict:
    """Compute inter-subject RSA + noise ceiling + GW alignment."""
    logger.info("=" * 60)
    logger.info("PHASE 3 – Balanced Inter-Subject Representational Analysis")
    logger.info("=" * 60)

    summary: dict = {}

    for state in ("conscious", "unconscious"):
        state_summary: dict = {}
        for roi in settings.roi_names:
            roi_rdms = [
                human_rdms[sid][state][roi]
                for sid in human_rdms
                if state in human_rdms[sid] and roi in human_rdms[sid][state]
            ]
            if len(roi_rdms) < 2:
                continue

            #  TODO - I would like to visualize second order RDMs of instances with p < 0.02
            #   also - it seems (down the line) that the RSAResult of c_results and u_results is filled in
            #   a hardcoded manner? requires inspection
            rsa_results = rsa_analyzer.inter_subject_rsa(roi_rdms)
            mean_rho = rsa_analyzer.mean_rho(rsa_results)
            nc = noise_ceiling.compute(roi_rdms)

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

    # GW alignment baseline
    for state in ("conscious", "unconscious"):
        for roi in settings.roi_names:
            roi_rdms = [
                human_rdms[sid][state][roi]
                for sid in human_rdms
                if state in human_rdms[sid] and roi in human_rdms[sid][state]
            ]
            if len(roi_rdms) < 2:
                continue
            ids = [f"{sid}_{state}" for sid in human_rdms if state in human_rdms[sid]]
            gw_matrix = gw_aligner.build_pairwise_distance_matrix(roi_rdms, ids)
            logger.info(
                "  ROI %s | GW mean distance=%.4f",
                roi, gw_matrix.matrix[gw_matrix.matrix > 0].mean(),
            )

    save_json(summary, Path(settings.stats_dir) / "phase3_inter_subject_rsa.json")

    # Visualization
    c_results = [
        RSAResult(subject_a="avg", subject_b="avg", roi_or_layer=roi,
                  state_a="conscious", state_b="conscious",
                  rho=m["mean_rho"], p_value=0.0, significant=True)
        for roi, m in summary.get("conscious", {}).items()
    ]
    u_results = [
        RSAResult(subject_a="avg", subject_b="avg", roi_or_layer=roi,
                  state_a="unconscious", state_b="unconscious",
                  rho=m["mean_rho"], p_value=0.0, significant=True)
        for roi, m in summary.get("unconscious", {}).items()
    ]
    if c_results or u_results:
        summary_plotter.plot_rsa_by_roi(
            c_results, u_results,
            roi_names=settings.roi_names,
            save_name="phase3_rsa_by_roi.png",
        )
        logger.info("Saved inter-subject RSA bar chart.")

    logger.info("Phase 3 complete.\n")
    return summary
