"""Phase 6 – Relational Visualizations (Meta-MDS)."""
from __future__ import annotations

import logging

from analysis.gromov_wasserstein.gw_aligner import GromovWassersteinAligner
from visualization.meta_mds_plotter import MetaMDSPlotter

logger = logging.getLogger(__name__)


def run(
    subjects: list,
    human_rdms: dict,
    fcnn_rdms: dict,
    gw_aligner: GromovWassersteinAligner,
    meta_mds_plotter: MetaMDSPlotter,
) -> None:
    """Build a 2-D Meta-MDS map of GWD distances between all RDMs."""
    logger.info("=" * 60)
    logger.info("PHASE 6 – Relational Visualizations (Meta-MDS)")
    logger.info("=" * 60)

    sid0 = subjects[0].subject_id if subjects else None
    c_rois = list(human_rdms.get(sid0, {}).get("conscious", {}).keys()) if sid0 else []
    example_roi = "fusiform" if "fusiform" in c_rois else (c_rois[0] if c_rois else "wholebrain")

    for roi in [example_roi, "lateral_occipital"]:
        if roi not in c_rois:
            continue

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

        if len(all_rdms) >= 3:
            gw_mat = gw_aligner.build_pairwise_distance_matrix(all_rdms, ids)
            human_ids = [i for i in ids if not i.startswith("fcnn")]
            model_ids = [i for i in ids if i.startswith("fcnn")]
            meta_mds_plotter.plot(
                gw_mat, human_ids, model_ids,
                title=f"Meta-MDS | {roi}",
                save_name=f"phase6_meta_mds_{roi}.png",
            )
            logger.info("Saved meta-MDS for ROI %s.", roi)

    logger.info("Phase 6 complete.\n")
