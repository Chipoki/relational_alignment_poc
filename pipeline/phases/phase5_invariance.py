"""Phase 5 – Structural Invariance Metric."""
from __future__ import annotations

import logging
from pathlib import Path

from config.settings import Settings
from analysis.gromov_wasserstein.gw_aligner import GromovWassersteinAligner
from visualization.summary_plotter import SummaryPlotter
from utils.io_utils import save_json

logger = logging.getLogger(__name__)


def run(
    settings: Settings,
    human_rdms: dict,
    fcnn_rdms: dict,
    gw_aligner: GromovWassersteinAligner,
    summary_plotter: SummaryPlotter,
) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE 5 – Structural Invariance Metric")
    logger.info("=" * 60)

    if "clear" not in fcnn_rdms or "chance" not in fcnn_rdms:
        logger.warning("FCNN RDMs not available – Phase 5 skipped.")
        return {}

    summary: dict = {}

    for roi in settings.active_roi_names:                  # ← active_roi_names
        c_rdms = [
            human_rdms[sid]["conscious"][roi]
            for sid in human_rdms
            if "conscious" in human_rdms[sid] and roi in human_rdms[sid]["conscious"]
        ]
        u_rdms = [
            human_rdms[sid]["unconscious"][roi]
            for sid in human_rdms
            if "unconscious" in human_rdms[sid] and roi in human_rdms[sid]["unconscious"]
        ]
        fcnn_clear = fcnn_rdms["clear"].get("fcnn_hidden")
        fcnn_noisy = fcnn_rdms["chance"].get("fcnn_hidden")

        if not c_rdms or not u_rdms or fcnn_clear is None or fcnn_noisy is None:
            continue

        metrics = gw_aligner.structural_invariance_metric(
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

    save_json(summary, Path(settings.stats_dir) / "phase5_structural_invariance.json")

    if summary:
        summary_plotter.plot_structural_invariance(
            summary, settings.active_roi_names,            # ← active_roi_names
            save_name="phase5_structural_invariance.png",
            subdir="phase5_invariance",
        )
        logger.info("Saved structural invariance scatter (all ROIs).")

    logger.info("Phase 5 complete.\n")
    return summary
