"""Phase 6 – Relational Visualizations (Meta-MDS)."""
from __future__ import annotations

import logging

from config.settings import Settings
from analysis.gromov_wasserstein.gw_aligner import GromovWassersteinAligner
from visualization.meta_mds_plotter import MetaMDSPlotter

logger = logging.getLogger(__name__)


def run(
    subjects: list,
    human_rdms: dict,
    fcnn_rdms: dict,
    gw_aligner: GromovWassersteinAligner,
    meta_mds_plotter: MetaMDSPlotter,
    settings: Settings | None = None,
) -> None:
    """Build a 2-D Meta-MDS map of GWD distances for every ROI."""
    logger.info("=" * 60)
    logger.info("PHASE 6 – Relational Visualizations (Meta-MDS)")
    logger.info("=" * 60)

    # Collect all available ROIs from the first subject that has data
    sid0   = subjects[0].subject_id if subjects else None
    c_rois = list(human_rdms.get(sid0, {}).get("conscious", {}).keys()) if sid0 else []

    # Honour settings.roi_names order when available, else use what exists
    if settings is not None:
        all_rois = [r for r in settings.roi_names if r in c_rois]
    else:
        all_rois = c_rois

    if not all_rois:
        logger.warning("No ROIs available for Phase 6 – skipping.")
        return

    for roi in all_rois:
        all_rdms, ids = [], []

        for sid in human_rdms:
            for state in ("conscious", "unconscious"):
                rdm = human_rdms[sid].get(state, {}).get(roi)
                if rdm:
                    all_rdms.append(rdm)
                    ids.append(f"{sid}_{state[:3]}")

        for noise_state in ("clear", "chance"):
            rdm = fcnn_rdms.get(noise_state, {}).get("fcnn_hidden")
            if rdm:
                all_rdms.append(rdm)
                ids.append(f"fcnn_{noise_state}")

        if len(all_rdms) < 3:
            logger.info("  ROI %s – too few RDMs (%d), skipping.", roi, len(all_rdms))
            continue

        gw_mat    = gw_aligner.build_pairwise_distance_matrix(all_rdms, ids)
        human_ids = [i for i in ids if not i.startswith("fcnn")]
        model_ids = [i for i in ids if i.startswith("fcnn")]

        meta_mds_plotter.plot(
            gw_mat, human_ids, model_ids,
            title=f"Meta-MDS | {roi}",
            save_name=f"phase6_meta_mds_{roi}.png",
            subdir=f"phase6_meta_mds/{roi}",
        )
        logger.info("Saved meta-MDS for ROI %s.", roi)

    logger.info("Phase 6 complete.\n")
