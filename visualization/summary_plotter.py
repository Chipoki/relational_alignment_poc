"""visualization/summary_plotter.py – Summary bar/scatter plots for RSA and GW results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.rsa.rsa_analyzer import RSAResult
from analysis.gromov_wasserstein.gw_result import GWDistanceMatrix
from config.settings import Settings


class SummaryPlotter:
    """Generates overview summary figures for RSA and GW results."""

    def __init__(self, settings: Settings) -> None:
        self._out_dir = Path(settings.visualization_dir)
        self._dpi = settings.visualization.get("dpi", 150)

    # ── RSA bar plot ─────────────────────────────────────────────────────────

    def plot_rsa_by_roi(
        self,
        results_conscious: list[RSAResult],
        results_unconscious: list[RSAResult],
        roi_names: list[str],
        title: str = "Inter-subject RSA by ROI",
        save_name: str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Bar chart of mean Spearman ρ per ROI, comparing conscious vs. unconscious.
        """
        def _mean_rho_per_roi(results: list[RSAResult], rois: list[str]) -> np.ndarray:
            rho_arr = np.zeros(len(rois))
            for i, roi in enumerate(rois):
                subset = [r.rho for r in results if r.roi_or_layer == roi]
                rho_arr[i] = np.mean(subset) if subset else 0.0
            return rho_arr

        rho_c = _mean_rho_per_roi(results_conscious, roi_names)
        rho_u = _mean_rho_per_roi(results_unconscious, roi_names)

        x = np.arange(len(roi_names))
        width = 0.38

        fig, ax = plt.subplots(figsize=(12, 4.5))
        bars_c = ax.bar(x - width / 2, rho_c, width, label="Conscious", color="#4878D0", alpha=0.85)
        bars_u = ax.bar(x + width / 2, rho_u, width, label="Unconscious", color="#EE854A", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(roi_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Mean Spearman ρ")
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color="black", lw=0.8)
        ax.legend()
        plt.tight_layout()

        if save_name:
            fig.savefig(self._out_dir / save_name, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── Structural invariance metric ─────────────────────────────────────────

    def plot_structural_invariance(
        self,
        invariance_results: dict[str, dict],   # roi_name → metrics dict from Phase 5
        roi_names: list[str],
        save_name: str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """Scatter: ΔGW_human vs ΔGW_FCNN per ROI."""
        human_deltas, fcnn_deltas, labels = [], [], []

        for roi in roi_names:
            if roi in invariance_results:
                m = invariance_results[roi]
                human_deltas.append(m["delta_gw_human_mean"])
                fcnn_deltas.append(m["delta_gw_fcnn"])
                labels.append(roi)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(human_deltas, fcnn_deltas, s=70, c="#2CA02C", zorder=3)
        for x, y, lbl in zip(human_deltas, fcnn_deltas, labels):
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)

        lim = max(max(human_deltas, default=1), max(fcnn_deltas, default=1)) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Identity line")
        ax.set_xlabel("ΔGW Human (Conscious → Unconscious)")
        ax.set_ylabel("ΔGW FCNN (Clear → Noisy)")
        ax.set_title("Phase 5 – Structural Invariance Metric", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_name:
            fig.savefig(self._out_dir / save_name, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig
