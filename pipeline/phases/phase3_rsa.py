"""Phase 3 – Balanced Inter-Subject Representational Analysis.

Extended analytics (added):
  1. Second-order RDMs (RSA matrices) for pairs with raw p < 0.02,
     regardless of Bonferroni correction status.
  2. GW distance matrices for three inter-state comparisons:
       C×C (intra-conscious), U×U (intra-unconscious),
       C×U (inter-state) – three heatmaps per ROI.
  3. Bonus visualizations:
       (a) ρ distribution violin plots by state.
       (b) ρ vs. p-value significance landscape scatter.
       (c) Noise ceiling overlay bar chart.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config.settings import Settings
from analysis.rsa.rdm import RDM
from analysis.rsa.rsa_analyzer import RSAAnalyzer, RSAResult
from analysis.rsa.noise_ceiling import NoiseCeiling
from analysis.gromov_wasserstein.gw_aligner import GromovWassersteinAligner
from analysis.gromov_wasserstein.gw_result import GWDistanceMatrix
from visualization.summary_plotter import SummaryPlotter
from visualization.phase3_plotter import Phase3Plotter
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
    """Compute inter-subject RSA + noise ceiling + GW alignment (Phase 3)."""
    logger.info("=" * 60)
    logger.info("PHASE 3 – Balanced Inter-Subject Representational Analysis")
    logger.info("=" * 60)

    phase3_plotter = Phase3Plotter(settings)

    summary: dict = {}

    # ── Per-ROI per-state RSA + noise ceiling ─────────────────────────────────
    # Also accumulate results for bonus plots
    all_rsa_results: dict[str, dict[str, list[RSAResult]]] = {}   # {roi: {state: [...]}}
    all_roi_rdms:    dict[str, dict[str, list[RDM]]]       = {}   # {roi: {state: [...]}}

    for state in ("conscious", "unconscious"):
        state_summary: dict = {}
        for roi in settings.roi_names:
            roi_rdms: list[RDM] = [
                human_rdms[sid][state][roi]
                for sid in human_rdms
                if state in human_rdms[sid] and roi in human_rdms[sid][state]
            ]
            if len(roi_rdms) < 2:
                continue

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

            # Accumulate for extended plots
            all_roi_rdms.setdefault(roi, {})[state] = roi_rdms
            all_rsa_results.setdefault(roi, {})[state] = rsa_results

            # ── Feature 1: second-order RDMs for pairs with p < 0.02 ────────
            n_below = sum(r.p_value < Phase3Plotter.P_THRESHOLD for r in rsa_results)
            if n_below > 0:
                logger.info(
                    "  → %d pairs below p=%.2f for ROI %s | %s – plotting second-order RDM.",
                    n_below, Phase3Plotter.P_THRESHOLD, roi, state,
                )
                phase3_plotter.plot_second_order_rdm(
                    rdms=roi_rdms,
                    rsa_results=rsa_results,
                    roi=roi,
                    state=state,
                    save_prefix=f"phase3_2nd_order_rdm_{roi}_{state}",
                )
            else:
                logger.info(
                    "  → No pairs below p=%.2f for ROI %s | %s (skipping 2nd-order RDM).",
                    Phase3Plotter.P_THRESHOLD, roi, state,
                )

        summary[state] = state_summary

    # ── Feature 2: GW distance matrices – C×C, U×U, and C×U per ROI ─────────
    for roi in settings.roi_names:
        rdms_c = all_roi_rdms.get(roi, {}).get("conscious", [])
        rdms_u = all_roi_rdms.get(roi, {}).get("unconscious", [])

        # 2a. Intra-conscious GW (C×C)
        if len(rdms_c) >= 2:
            ids_c = [f"{rdm.subject_id}_con" for rdm in rdms_c]
            gw_cc = gw_aligner.build_pairwise_distance_matrix(rdms_c, ids_c)
            gw_cc.state = "conscious"
            gw_cc.roi_or_layer = roi
            phase3_plotter.plot_gw_matrix(
                gw_cc,
                title=f"GW Distance Matrix  (Conscious × Conscious)\nROI: {roi}",
                save_name=f"phase3_gw_matrix_{roi}_conscious_x_conscious.png",
            )
            mean_gw_cc = gw_cc.matrix[gw_cc.matrix > 0].mean() if gw_cc.matrix.any() else 0.0
            logger.info("  ROI %s | GW C×C mean distance=%.4f", roi, mean_gw_cc)

        # 2b. Intra-unconscious GW (U×U)
        if len(rdms_u) >= 2:
            ids_u = [f"{rdm.subject_id}_unc" for rdm in rdms_u]
            gw_uu = gw_aligner.build_pairwise_distance_matrix(rdms_u, ids_u)
            gw_uu.state = "unconscious"
            gw_uu.roi_or_layer = roi
            phase3_plotter.plot_gw_matrix(
                gw_uu,
                title=f"GW Distance Matrix  (Unconscious × Unconscious)\nROI: {roi}",
                save_name=f"phase3_gw_matrix_{roi}_unconscious_x_unconscious.png",
            )
            mean_gw_uu = gw_uu.matrix[gw_uu.matrix > 0].mean() if gw_uu.matrix.any() else 0.0
            logger.info("  ROI %s | GW U×U mean distance=%.4f", roi, mean_gw_uu)

        # 2c. Inter-state GW (C×U) — concatenate all RDMs, then slice cross-block
        if len(rdms_c) >= 1 and len(rdms_u) >= 1:
            all_rdms_cu = rdms_c + rdms_u
            ids_cu = (
                [f"{rdm.subject_id}_con" for rdm in rdms_c]
                + [f"{rdm.subject_id}_unc" for rdm in rdms_u]
            )
            gw_cu = gw_aligner.build_pairwise_distance_matrix(all_rdms_cu, ids_cu)
            gw_cu.state = "conscious_vs_unconscious"
            gw_cu.roi_or_layer = roi
            # Use the dedicated inter-state plotter to show only the cross-block
            phase3_plotter.plot_inter_state_gw_matrix(
                gw_cu,
                roi=roi,
                save_name=f"phase3_gw_matrix_{roi}_conscious_vs_unconscious.png",
            )
            # Also log mean GW in the cross-block
            n_c = len(rdms_c)
            n_u = len(rdms_u)
            cross_block = gw_cu.matrix[:n_c, n_c:]
            mean_gw_cu = float(cross_block.mean()) if cross_block.size > 0 else 0.0
            logger.info("  ROI %s | GW C×U mean distance=%.4f", roi, mean_gw_cu)

    # ── Feature 3a: ρ violin plots by state ──────────────────────────────────
    results_by_roi_for_violin: dict[str, dict[str, list[RSAResult]]] = {
        roi: {
            state: all_rsa_results.get(roi, {}).get(state, [])
            for state in ("conscious", "unconscious")
        }
        for roi in settings.roi_names
        if roi in all_rsa_results
    }
    if results_by_roi_for_violin:
        phase3_plotter.plot_rho_violins(
            results_by_roi_for_violin,
            save_name="phase3_rho_violins.png",
        )
        logger.info("Saved ρ violin plots.")

    # ── Feature 3b: ρ vs. p-value scatter ───────────────────────────────────
    all_c_results: list[RSAResult] = [
        r for roi in all_rsa_results
        for r in all_rsa_results[roi].get("conscious", [])
    ]
    all_u_results: list[RSAResult] = [
        r for roi in all_rsa_results
        for r in all_rsa_results[roi].get("unconscious", [])
    ]
    n_comparisons_total = len(all_c_results) + len(all_u_results)
    if all_c_results or all_u_results:
        phase3_plotter.plot_rho_vs_pvalue(
            results_conscious=all_c_results,
            results_unconscious=all_u_results,
            n_comparisons=n_comparisons_total or None,
            save_name="phase3_rho_vs_pvalue.png",
        )
        logger.info("Saved ρ vs. p-value scatter.")

    # ── Feature 3c: noise ceiling overlay bar chart ──────────────────────────
    if summary:
        phase3_plotter.plot_noise_ceiling_bars(
            summary=summary,
            roi_names=settings.roi_names,
            save_name="phase3_noise_ceiling_bars.png",
        )
        logger.info("Saved noise ceiling bar chart.")

    # ── Original summary bar chart (kept for backward compat) ───────────────
    c_results_summary = [
        RSAResult(
            subject_a="avg", subject_b="avg", roi_or_layer=roi,
            state_a="conscious", state_b="conscious",
            rho=m["mean_rho"], p_value=0.0, significant=True,
        )
        for roi, m in summary.get("conscious", {}).items()
    ]
    u_results_summary = [
        RSAResult(
            subject_a="avg", subject_b="avg", roi_or_layer=roi,
            state_a="unconscious", state_b="unconscious",
            rho=m["mean_rho"], p_value=0.0, significant=True,
        )
        for roi, m in summary.get("unconscious", {}).items()
    ]
    if c_results_summary or u_results_summary:
        summary_plotter.plot_rsa_by_roi(
            c_results_summary, u_results_summary,
            roi_names=settings.roi_names,
            save_name="phase3_rsa_by_roi.png",
        )
        logger.info("Saved inter-subject RSA bar chart.")

    save_json(summary, Path(settings.stats_dir) / "phase3_inter_subject_rsa.json")
    logger.info("Phase 3 complete.\n")
    return summary
