"""visualization/phase3_plotter.py – Phase 3 extended analytics visualizations.

Provides:
  1. Second-order RDM (RSA matrix) heatmaps for pairs with p < 0.02
     (regardless of Bonferroni correction status).
  2. GW distance matrix heatmaps for all three inter-state comparisons:
       - Conscious × Conscious (intra-state)
       - Unconscious × Unconscious (intra-state)
       - Conscious × Unconscious (inter-state)
  3. Bonus visualizations:
       a. Spearman ρ violin plots by state.
       b. ρ vs. p-value scatter (p-threshold annotated).
       c. Noise ceiling overlay bar chart per ROI.
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from analysis.rsa.rsa_analyzer import RSAResult
from analysis.rsa.rdm import RDM
from analysis.gromov_wasserstein.gw_result import GWDistanceMatrix
from config.settings import Settings


class Phase3Plotter:
    """All extended Phase 3 visualizations."""

    # p-value threshold used for second-order RDM selection
    P_THRESHOLD: float = 0.02

    def __init__(self, settings: Settings) -> None:
        self._out_dir = Path(settings.visualization_dir)
        self._dpi = settings.visualization.get("dpi", 150)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.  Second-order RDMs (RSA matrices) for pairs with p < 0.02
    # ─────────────────────────────────────────────────────────────────────────

    def plot_second_order_rdm(
        self,
        rdms: list[RDM],
        rsa_results: list[RSAResult],
        roi: str,
        state: str,
        save_prefix: str | None = None,
        show: bool = False,
    ) -> list[plt.Figure]:
        """
        Build and plot a second-order RDM (subject × subject Spearman ρ matrix)
        for every pair whose p-value < P_THRESHOLD, even if Bonferroni-insignificant.

        Each figure is one ROI × state combination.  The full N×N subject matrix
        is drawn as a heatmap; cells meeting the threshold are annotated with ρ
        and a star when also Bonferroni-significant.

        Parameters
        ----------
        rdms         : list of RDMs for this roi/state.
        rsa_results  : pairwise RSAResult objects (same roi/state).
        roi          : ROI name (used in title / filename).
        state        : 'conscious' or 'unconscious'.
        save_prefix  : filename stem; defaults to 'phase3_2nd_order_rdm_{roi}_{state}'.
        show         : call plt.show().

        Returns
        -------
        List of Figure objects (one per qualifying plot).
        """
        # Filter pairs below the raw p-value threshold
        qualifying = [r for r in rsa_results if r.p_value < self.P_THRESHOLD]
        if not qualifying:
            return []

        subject_ids = [rdm.subject_id for rdm in rdms]
        n = len(subject_ids)
        sid_to_idx = {sid: i for i, sid in enumerate(subject_ids)}

        # Build full ρ matrix and p matrix
        rho_mat = np.full((n, n), np.nan)
        p_mat = np.full((n, n), np.nan)
        sig_mat = np.zeros((n, n), dtype=bool)  # Bonferroni significance
        np.fill_diagonal(rho_mat, 1.0)  # self-correlation = 1

        for r in rsa_results:
            i = sid_to_idx.get(r.subject_a)
            j = sid_to_idx.get(r.subject_b)
            if i is not None and j is not None:
                rho_mat[i, j] = r.rho
                rho_mat[j, i] = r.rho
                p_mat[i, j] = r.p_value
                p_mat[j, i] = r.p_value
                sig_mat[i, j] = r.significant
                sig_mat[j, i] = r.significant

        figs: list[plt.Figure] = []

        # ── Full second-order RDM heatmap ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(max(5, n * 0.7 + 1.5), max(4.5, n * 0.65 + 1.5)))

        # Mask self-diagonal for colour scaling (they are all 1.0)
        plot_mat = np.copy(rho_mat)
        np.fill_diagonal(plot_mat, np.nan)

        vmax = np.nanmax(np.abs(plot_mat))
        vmax = max(vmax, 0.01)  # avoid degenerate range
        im = ax.imshow(
            plot_mat,
            cmap="RdYlBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Spearman ρ")

        # Annotate cells: ρ value + star if Bonferroni-significant
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="grey")
                elif not np.isnan(rho_mat[i, j]):
                    below_thresh = (not np.isnan(p_mat[i, j])) and p_mat[i, j] < self.P_THRESHOLD
                    bonf = sig_mat[i, j]
                    label = f"{rho_mat[i, j]:.2f}"
                    if bonf:
                        label += "*"
                    color = "white" if abs(rho_mat[i, j]) > 0.6 * vmax else "black"
                    fontweight = "bold" if below_thresh else "normal"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=7, color=color, fontweight=fontweight)

        # Highlight cells below p < P_THRESHOLD with a rectangle
        for i in range(n):
            for j in range(n):
                if i != j and not np.isnan(p_mat[i, j]) and p_mat[i, j] < self.P_THRESHOLD:
                    rect = plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        linewidth=1.8, edgecolor="#222222", facecolor="none"
                    )
                    ax.add_patch(rect)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(subject_ids, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(subject_ids, fontsize=8)
        ax.set_title(
            f"Second-order RDM (RSA matrix)\n"
            f"ROI: {roi}  |  State: {state}  |  "
            f"Bold/boxed: p < {self.P_THRESHOLD}  |  *: Bonferroni-sig",
            fontsize=9,
        )

        legend_elems = [
            mpatches.Patch(edgecolor="#222222", facecolor="none",
                           linewidth=1.8, label=f"p < {self.P_THRESHOLD} (raw)"),
            mpatches.Patch(facecolor="none", edgecolor="none",
                           label="* = Bonferroni-significant"),
        ]
        ax.legend(handles=legend_elems, loc="lower right", fontsize=7, framealpha=0.75)

        plt.tight_layout()
        prefix = save_prefix or f"phase3_2nd_order_rdm_{roi}_{state}"
        save_path = self._out_dir / f"{prefix}.png"
        fig.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        figs.append(fig)
        return figs

    # ─────────────────────────────────────────────────────────────────────────
    # 2.  GW distance matrix heatmaps (C×C, U×U, C×U)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_gw_matrix(
        self,
        gw_matrix: GWDistanceMatrix,
        title: str | None = None,
        save_name: str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot a single GW distance matrix as an annotated heatmap.

        Parameters
        ----------
        gw_matrix : GWDistanceMatrix (symmetric, zeros on diagonal).
        title     : figure title; auto-generated if None.
        save_name : filename; auto-generated if None.
        show      : call plt.show().
        """
        mat = gw_matrix.matrix
        labels = gw_matrix.labels
        n = len(labels)

        plot_mat = mat.copy().astype(float)
        np.fill_diagonal(plot_mat, np.nan)   # hide trivial diagonal

        vmax = np.nanmax(plot_mat)
        vmax = max(vmax, 1e-6)

        fig, ax = plt.subplots(figsize=(max(5, n * 0.7 + 1.5), max(4.5, n * 0.65 + 1.5)))
        im = ax.imshow(plot_mat, cmap="viridis_r", vmin=0, vmax=vmax, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("GW Distance")

        # Annotate each cell with value
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "0", ha="center", va="center", fontsize=7, color="grey")
                else:
                    val = mat[i, j]
                    brightness = val / vmax
                    color = "white" if brightness > 0.55 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=max(5, 8 - n // 3), color=color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        auto_title = title or (
            f"GW Distance Matrix\n"
            f"ROI: {gw_matrix.roi_or_layer}  |  State: {gw_matrix.state}"
        )
        ax.set_title(auto_title, fontsize=9)

        plt.tight_layout()
        fname = save_name or (
            f"phase3_gw_matrix_{gw_matrix.roi_or_layer}_{gw_matrix.state}.png"
        )
        fig.savefig(self._out_dir / fname, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def plot_inter_state_gw_matrix(
        self,
        gw_matrix: GWDistanceMatrix,
        roi: str,
        save_name: str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot a rectangular (conscious × unconscious) GW distance matrix.

        The matrix is rectangular: rows = conscious subjects,
        columns = unconscious subjects.  gw_matrix.state should be
        'conscious_vs_unconscious'.
        """
        mat = gw_matrix.matrix
        labels = gw_matrix.labels
        n = len(labels)
        half = n // 2
        c_labels = labels[:half]
        u_labels = labels[half:]

        # Slice the cross-block (top-right quadrant of the symmetric matrix)
        cross_mat = mat[:half, half:].copy().astype(float)

        vmax = cross_mat.max()
        vmax = max(vmax, 1e-6)

        fig, ax = plt.subplots(
            figsize=(max(5, half * 0.8 + 1.5), max(4.5, half * 0.75 + 1.5))
        )
        im = ax.imshow(cross_mat, cmap="viridis_r", vmin=0, vmax=vmax, aspect="auto")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("GW Distance")

        for i in range(cross_mat.shape[0]):
            for j in range(cross_mat.shape[1]):
                val = cross_mat[i, j]
                brightness = val / vmax
                color = "white" if brightness > 0.55 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=max(5, 8 - half // 3), color=color)

        ax.set_xticks(range(len(u_labels)))
        ax.set_yticks(range(len(c_labels)))
        ax.set_xticklabels(u_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(c_labels, fontsize=8)
        ax.set_xlabel("Unconscious subjects", fontsize=9)
        ax.set_ylabel("Conscious subjects", fontsize=9)
        ax.set_title(
            f"Inter-State GW Distance  (Conscious × Unconscious)\nROI: {roi}",
            fontsize=9,
        )

        plt.tight_layout()
        fname = save_name or f"phase3_gw_matrix_{roi}_conscious_vs_unconscious.png"
        fig.savefig(self._out_dir / fname, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # 3a.  Bonus – Spearman ρ violin plots by state
    # ─────────────────────────────────────────────────────────────────────────

    def plot_rho_violins(
        self,
        results_by_roi: dict[str, dict[str, list[RSAResult]]],
        save_name: str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Violin plot of pairwise Spearman ρ distributions per ROI, split by
        state (conscious = blue, unconscious = orange).

        Parameters
        ----------
        results_by_roi : {roi: {state: [RSAResult, ...]}}
        """
        roi_names = list(results_by_roi.keys())
        n_rois = len(roi_names)
        if n_rois == 0:
            return plt.figure()

        fig, axes = plt.subplots(
            1, n_rois,
            figsize=(max(3 * n_rois, 6), 4.5),
            sharey=True,
        )
        if n_rois == 1:
            axes = [axes]

        state_colors = {"conscious": "#4878D0", "unconscious": "#EE854A"}

        for ax, roi in zip(axes, roi_names):
            data, colors, labels_vp = [], [], []
            for state in ("conscious", "unconscious"):
                rho_vals = [r.rho for r in results_by_roi[roi].get(state, [])]
                if rho_vals:
                    data.append(rho_vals)
                    colors.append(state_colors[state])
                    labels_vp.append(state.capitalize())

            if not data:
                ax.set_title(roi, fontsize=8)
                continue

            parts = ax.violinplot(data, showmedians=True, showextrema=True)
            for i, (body, col) in enumerate(zip(parts["bodies"], colors)):
                body.set_facecolor(col)
                body.set_alpha(0.75)
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(0.9)

            ax.set_xticks(range(1, len(labels_vp) + 1))
            ax.set_xticklabels(labels_vp, fontsize=8)
            ax.axhline(0, color="black", lw=0.7, ls="--")
            ax.set_title(roi, fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        axes[0].set_ylabel("Spearman ρ", fontsize=9)
        fig.suptitle(
            "Pairwise ρ Distribution per ROI  |  Phase 3 Inter-Subject RSA",
            fontsize=10,
            y=1.01,
        )

        # Shared legend
        legend_handles = [
            mpatches.Patch(color=c, label=s.capitalize())
            for s, c in state_colors.items()
        ]
        fig.legend(
            handles=legend_handles, loc="upper right",
            fontsize=8, bbox_to_anchor=(1.0, 1.0),
        )

        plt.tight_layout()
        fname = save_name or "phase3_rho_violins.png"
        fig.savefig(self._out_dir / fname, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # 3b.  Bonus – ρ vs. p-value scatter (significance landscape)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_rho_vs_pvalue(
        self,
        results_conscious: list[RSAResult],
        results_unconscious: list[RSAResult],
        n_comparisons: int | None = None,
        save_name: str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Scatter plot of Spearman ρ (x) vs. raw p-value (y, log-scale).

        Horizontal reference lines are drawn at:
          - p = 0.05  (uncorrected alpha)
          - p = 0.02  (visualisation threshold used in second-order RDMs)
          - p = 0.05 / n_comparisons  (Bonferroni threshold, if n_comparisons given)

        Point colour encodes state; shape encodes Bonferroni significance.
        """
        fig, ax = plt.subplots(figsize=(7, 5))

        state_cfg = [
            (results_conscious, "#4878D0", "Conscious"),
            (results_unconscious, "#EE854A", "Unconscious"),
        ]

        for results, color, label in state_cfg:
            if not results:
                continue
            rhos = np.array([r.rho for r in results])
            pvals = np.array([r.p_value for r in results])
            sigs = np.array([r.significant for r in results])

            # Clip p=0 to a small value for log scale
            pvals = np.clip(pvals, 1e-4, 1.0)

            ax.scatter(
                rhos[~sigs], pvals[~sigs],
                c=color, marker="o", s=45, alpha=0.65,
                label=f"{label} (not Bonf-sig)",
            )
            ax.scatter(
                rhos[sigs], pvals[sigs],
                c=color, marker="*", s=120, alpha=0.9,
                label=f"{label} (Bonf-sig ✱)",
            )

        ax.set_yscale("log")
        ax.invert_yaxis()

        # Reference lines
        ax.axhline(0.05, color="grey", lw=1.0, ls="--", label="p = 0.05 (uncorrected)")
        ax.axhline(self.P_THRESHOLD, color="#AA0000", lw=1.2, ls="-.",
                   label=f"p = {self.P_THRESHOLD} (2nd-order RDM threshold)")
        if n_comparisons:
            bonf_thresh = 0.05 / n_comparisons
            ax.axhline(bonf_thresh, color="purple", lw=1.2, ls=":",
                       label=f"p = {bonf_thresh:.4f} (Bonferroni, n={n_comparisons})")

        ax.set_xlabel("Spearman ρ", fontsize=10)
        ax.set_ylabel("p-value (log scale)", fontsize=10)
        ax.set_title("RSA Significance Landscape  |  Phase 3", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
        plt.tight_layout()

        fname = save_name or "phase3_rho_vs_pvalue.png"
        fig.savefig(self._out_dir / fname, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # 3c.  Bonus – Noise ceiling overlay bar chart
    # ─────────────────────────────────────────────────────────────────────────

    def plot_noise_ceiling_bars(
        self,
        summary: dict,
        roi_names: list[str],
        save_name: str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Bar chart of mean Spearman ρ per ROI for each state, with noise
        ceiling shaded as a semi-transparent band (lower–upper NC).

        Parameters
        ----------
        summary   : phase3 summary dict  {state: {roi: {mean_rho, nc_upper, nc_lower, ...}}}
        roi_names : ordered list of ROI names.
        """
        states = ["conscious", "unconscious"]
        state_colors = {"conscious": "#4878D0", "unconscious": "#EE854A"}
        nc_colors    = {"conscious": "#A8C4E8", "unconscious": "#F4C49A"}

        n = len(roi_names)
        x = np.arange(n)
        width = 0.38
        offsets = {"conscious": -width / 2, "unconscious": width / 2}

        fig, ax = plt.subplots(figsize=(max(10, n * 1.1 + 2), 5))

        for state in states:
            state_data = summary.get(state, {})
            rho_vals = np.array([state_data.get(roi, {}).get("mean_rho", 0.0) for roi in roi_names])
            nc_upper = np.array([state_data.get(roi, {}).get("noise_ceiling_upper", np.nan) for roi in roi_names])
            nc_lower = np.array([state_data.get(roi, {}).get("noise_ceiling_lower", np.nan) for roi in roi_names])

            bar_x = x + offsets[state]
            ax.bar(
                bar_x, rho_vals, width,
                color=state_colors[state], alpha=0.85,
                label=f"{state.capitalize()} mean ρ",
            )

            # Noise ceiling band
            for xi, low, high in zip(bar_x, nc_lower, nc_upper):
                if not (np.isnan(low) or np.isnan(high)):
                    ax.fill_between(
                        [xi - width / 2, xi + width / 2],
                        [low, low], [high, high],
                        color=nc_colors[state], alpha=0.50, zorder=0,
                    )

        # Legend entries for noise ceiling bands
        for state in states:
            ax.fill_between(
                [], [], [],
                color=nc_colors[state], alpha=0.50,
                label=f"{state.capitalize()} noise ceiling",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(roi_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Mean Spearman ρ", fontsize=10)
        ax.set_title(
            "Inter-Subject RSA with Noise Ceiling  |  Phase 3",
            fontsize=11,
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        fname = save_name or "phase3_noise_ceiling_bars.png"
        fig.savefig(self._out_dir / fname, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return fig
