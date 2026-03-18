"""visualization/rdm_plotter.py – Plot single, dual-state, aggregate and second-order RDMs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from analysis.rsa.rdm import RDM
from config.settings import Settings


class RDMPlotter:
    """Generates RDM heatmaps with category-sorted stimulus ordering."""

    def __init__(self, settings: Settings) -> None:
        self._base_dir = Path(settings.visualization_dir)
        self._dpi = settings.visualization.get("dpi", 150)

    def _out_dir(self, subdir: str) -> Path:
        p = self._base_dir / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Single RDM ────────────────────────────────────────────────────────────

    def plot_rdm(
        self,
        rdm: RDM,
        title: str | None = None,
        save_name: str | None = None,
        subdir: str = "phase2_rdms",
        show: bool = False,
    ) -> plt.Figure:
        """Plot a single RDM heatmap sorted by category."""
        sort_idx = np.argsort(1 - rdm.labels)
        mat = np.copy(rdm.matrix[np.ix_(sort_idx, sort_idx)])
        np.fill_diagonal(mat, np.nan)
        sorted_labels = rdm.labels[sort_idx]
        n_living = int(sorted_labels.sum())
        n_total  = len(sorted_labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, label="Dissimilarity (1 − Spearman ρ)")
        ax.axhline(n_living - 0.5, color="black", lw=1.5)
        ax.axvline(n_living - 0.5, color="black", lw=1.5)
        ax.set_xlabel("Stimuli (Living → Non-living)")
        ax.set_ylabel("Stimuli (Living → Non-living)")
        ax.set_title(
            title or f"{rdm.subject_id} | {rdm.state} | {rdm.roi_or_layer}",
            fontsize=10, pad=8,
        )
        ax.legend(
            handles=[
                mpatches.Patch(color="none", label=f"Living (n={n_living})"),
                mpatches.Patch(color="none", label=f"Non-living (n={n_total - n_living})"),
            ],
            loc="lower right", fontsize=7, framealpha=0.7,
        )
        plt.tight_layout()
        if save_name:
            fig.savefig(
                self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight"
            )
        if show:
            plt.show()
        plt.close(fig)
        return fig

    # ── Dual-state ───────────────────────────────────────────────────────────

    def plot_dual_state(
        self,
        rdm_conscious: RDM,
        rdm_unconscious: RDM,
        suptitle: str | None = None,
        save_name: str | None = None,
        subdir: str = "phase2_rdms",
        show: bool = False,
    ) -> plt.Figure:
        """Plot conscious and unconscious human RDMs side by side."""
        return self._plot_dual(
            rdm_left=rdm_conscious, rdm_right=rdm_unconscious,
            label_left="Conscious", label_right="Unconscious",
            suptitle=suptitle, save_name=save_name, subdir=subdir, show=show,
        )

    def plot_dual_state_fcnn(
        self,
        rdm_clear: RDM,
        rdm_noisy: RDM,
        suptitle: str | None = None,
        save_name: str | None = None,
        subdir: str = "phase2_rdms",
        show: bool = False,
    ) -> plt.Figure:
        """Plot FCNN clear (0-noise) and noisy (chance-level) RDMs side by side."""
        return self._plot_dual(
            rdm_left=rdm_clear, rdm_right=rdm_noisy,
            label_left="Clear (0-noise)", label_right="Noisy (chance-level)",
            suptitle=suptitle or "FCNN  ·  Representational Dissimilarity Matrix\n(Clear | Noisy)",
            save_name=save_name, subdir=subdir, show=show,
        )

    # ── Aggregate RDM (mean / median) ───────────────────────────────────────

    def plot_mean_rdm(
        self,
        rdm: RDM,
        rsa_rho: float | None = None,
        rsa_p: float | None = None,
        save_name: str | None = None,
        subdir: str | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot a cross-subject aggregate RDM (mean or median).

        Title and default subdir are derived from ``rdm.subject_id``
        (``'mean'`` or ``'median'``), so no extra parameter is required.
        """
        method = rdm.subject_id          # 'mean' or 'median'
        subdir = subdir or f"phase2_rdms/{method}"

        sort_idx = np.argsort(1 - rdm.labels)
        mat      = np.copy(rdm.matrix[np.ix_(sort_idx, sort_idx)])
        np.fill_diagonal(mat, np.nan)
        n_living = int(rdm.labels.sum())
        n_total  = len(rdm.labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, label="Dissimilarity (1 − Spearman ρ)")
        ax.axhline(n_living - 0.5, color="black", lw=1.5)
        ax.axvline(n_living - 0.5, color="black", lw=1.5)
        ax.set_xlabel("Stimuli (Living → Non-living)")
        ax.set_ylabel("Stimuli (Living → Non-living)")

        title = (
            f"{method.title()} RDM  ·  {rdm.roi_or_layer}  ·  "
            f"{rdm.state.title()}\n(n_stimuli={n_total})"
        )
        if rsa_rho is not None:
            sig_str = f"p={rsa_p:.4f}" if rsa_p is not None else ""
            title  += f"\nFCNN cross-modality RSA: ρ={rsa_rho:.3f}  {sig_str}"
        ax.set_title(title, fontsize=9, pad=8)

        ax.legend(
            handles=[
                mpatches.Patch(color="none", label=f"Living (n={n_living})"),
                mpatches.Patch(
                    color="none", label=f"Non-living (n={n_total - n_living})"
                ),
            ],
            loc="lower right", fontsize=7, framealpha=0.7,
        )
        sorted_labels = rdm.labels[sort_idx]
        self._draw_animacy_sidebar(ax, sorted_labels, side="right")
        plt.tight_layout()

        if save_name:
            fig.savefig(
                self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight"
            )
        if show:
            plt.show()
        plt.close(fig)
        return fig

    # ── Sorted RDM (Ward + silhouette) ───────────────────────────────────────

    def plot_sorted_rdm(
        self,
        rdm: RDM,
        title_prefix: str = "",
        save_name: str | None = None,
        subdir: str = "phase2_rdms/sorted_independent",
        show: bool = False,
        common_order: np.ndarray | None = None,
        best_k: int | None = None,
        best_score: float | None = None,
        k_min: int = 2,
        k_max: int = 8,
    ) -> plt.Figure:
        """
        Plot a cluster-sorted RDM with Ward/silhouette boundary lines.

        ``common_order`` absent  →  independent sort (sorted_independent/).
        ``common_order`` present →  GW-consensus sort (sorted_consensus/).
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        if common_order is not None:
            order       = common_order
            _best_k     = best_k
            _best_score = best_score
            order_src   = "GW-consensus"
        else:
            from analysis.rsa.rdm_utils import sorted_order
            order, _best_k, _best_score = sorted_order(
                rdm.matrix, k_min=k_min, k_max=k_max
            )
            order_src = "independent"

        condensed  = np.clip(squareform(rdm.matrix, checks=False), 0, None)
        Z          = linkage(condensed, method="ward")
        labels_arr = fcluster(Z, _best_k, criterion="maxclust")[order] - 1

        mat = rdm.matrix[np.ix_(order, order)]
        np.fill_diagonal(mat, np.nan)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, label="Dissimilarity (1 − Spearman ρ)")

        boundaries = np.where(np.diff(labels_arr))[0] + 0.5
        for b in boundaries:
            ax.axhline(b, color="black", lw=1.2, ls="--", alpha=0.7)
            ax.axvline(b, color="black", lw=1.2, ls="--", alpha=0.7)

        ax.set_xlabel(f"Stimuli ({order_src} order)")
        ax.set_ylabel(f"Stimuli ({order_src} order)")
        ax.set_title(
            f"{title_prefix}Cluster-Sorted RDM  ·  {rdm.roi_or_layer}  ·  "
            f"{rdm.state.title()}\n"
            f"Ward  |  k*={_best_k}  |  silhouette={_best_score:.3f}  "
            f"({order_src})",
            fontsize=9, pad=8,
        )
        sorted_labels = rdm.labels[order]
        self._draw_animacy_sidebar(ax, sorted_labels, side="right")
        plt.tight_layout()

        if save_name:
            fig.savefig(
                self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight"
            )
        if show:
            plt.show()
        plt.close(fig)
        return fig

    # ── ROI × ROI second-order RDM ──────────────────────────────────────────

    def plot_roi_x_roi_rdm(
        self,
        rho_matrix: np.ndarray,
        p_matrix: np.ndarray,
        roi_names: list[str],
        method: str,
        state: str,
        save_name: str | None = None,
        subdir: str = "phase4_cross_modality/roi_x_roi",
        alpha: float = 0.05,
        show: bool = False,
    ) -> plt.Figure:
        """
        Heatmap of Spearman ρ between every pair of ROI aggregate RDMs.

        Each cell shows the ρ value.  Cells that do NOT reach Bonferroni-
        corrected significance are hatched so the pattern is immediately
        apparent.  The diagonal (self-correlation = 1) is shown in a
        neutral grey to avoid visual bias.

        Parameters
        ----------
        rho_matrix : (n, n) Spearman ρ values
        p_matrix   : (n, n) corresponding p-values
        roi_names  : ordered list of ROI labels
        method     : 'mean' or 'median'  (used in title)
        state      : 'conscious' or 'unconscious'
        alpha      : significance threshold before Bonferroni
        """
        n          = len(roi_names)
        n_pairs    = n * (n - 1) // 2
        bonf_alpha = alpha / max(n_pairs, 1)
        sig_mask   = p_matrix < bonf_alpha
        np.fill_diagonal(sig_mask, True)   # diagonal always "significant"

        # Mask diagonal for colour mapping (show as grey)
        plot_mat = rho_matrix.copy().astype(float)
        diag_val = np.nan
        np.fill_diagonal(plot_mat, diag_val)

        fig, ax = plt.subplots(
            figsize=(max(5, n * 0.65 + 1.5), max(4.5, n * 0.65 + 1.2))
        )

        # Diverging colourmap centred at 0
        vabs = np.nanmax(np.abs(rho_matrix[~np.eye(n, dtype=bool)]))
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
        im   = ax.imshow(plot_mat, cmap="RdYlBu_r", norm=norm, aspect="auto")
        plt.colorbar(im, ax=ax, label="Spearman ρ")

        # Grey diagonal
        for i in range(n):
            ax.add_patch(
                plt.Rectangle(
                    (i - 0.5, i - 0.5), 1, 1,
                    color="lightgrey", zorder=2,
                )
            )
            ax.text(i, i, "1.00", ha="center", va="center",
                    fontsize=7, color="dimgrey", zorder=3)

        # Cell annotations + significance hatching
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rho = rho_matrix[i, j]
                ax.text(
                    j, i, f"{rho:.2f}",
                    ha="center", va="center",
                    fontsize=max(5, 8 - n // 4),
                    color="black" if abs(rho) < 0.7 else "white",
                    zorder=3,
                )
                if not sig_mask[i, j]:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, hatch="///",
                            edgecolor="grey", lw=0.5, zorder=4,
                        )
                    )

        labels = [r.replace("_", "\n") for r in roi_names]
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(
            f"ROI × ROI Second-Order RDM  ·  {method.title()}  ·  {state.title()}\n"
            f"Spearman ρ between aggregate RDM upper triangles\n"
            f"(hatching = non-significant after Bonferroni, α={alpha})",
            fontsize=9, pad=10,
        )
        plt.tight_layout()
        if save_name:
            fig.savefig(
                self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight"
            )
        if show:
            plt.show()
        plt.close(fig)
        return fig

    # ── Private helpers ───────────────────────────────────────────────────────

    def _plot_dual(
        self,
        rdm_left: RDM,
        rdm_right: RDM,
        label_left: str,
        label_right: str,
        suptitle: str | None,
        save_name: str | None,
        subdir: str,
        show: bool,
    ) -> plt.Figure:
        """Shared implementation for any side-by-side dual RDM figure."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        for ax, rdm, label in zip(
            axes, [rdm_left, rdm_right], [label_left, label_right]
        ):
            sort_idx = np.argsort(1 - rdm.labels)
            mat = np.copy(rdm.matrix[np.ix_(sort_idx, sort_idx)])
            np.fill_diagonal(mat, np.nan)
            n_living = int(rdm.labels.sum())

            im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto")
            ax.axhline(n_living - 0.5, color="black", lw=1.5)
            ax.axvline(n_living - 0.5, color="black", lw=1.5)
            ax.set_title(f"{label}\n{rdm.roi_or_layer}", fontsize=9)
            ax.set_xlabel("Stimuli (Living → Non-living)")
            ax.set_ylabel("Stimuli (Living → Non-living)")
            cb = plt.colorbar(im, ax=ax, shrink=0.8)
            cb.set_label("Dissimilarity (1 − Spearman ρ)")

        if suptitle:
            fig.suptitle(suptitle, y=1.01, fontsize=11)

        plt.tight_layout()
        if save_name:
            fig.savefig(
                self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight"
            )
        if show:
            plt.show()
        plt.close(fig)
        return fig

    @staticmethod
    def _draw_animacy_sidebar(
        ax: plt.Axes,
        sorted_labels: np.ndarray,
        side: str = "right",
        bar_width: str = "2%",
        bar_pad: str = "1%",
    ) -> None:
        """Append a thin binary sidebar to `ax` (black=animate, white=inanimate)."""
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        sidebar_img = np.where(sorted_labels[:, None] == 1, 0.0, 1.0)
        divider = make_axes_locatable(ax)
        sidebar_ax = divider.append_axes(side, size=bar_width, pad=bar_pad)
        sidebar_ax.imshow(
            sidebar_img, cmap="gray", vmin=0.0, vmax=1.0,
            aspect="auto", interpolation="nearest",
        )
        sidebar_ax.set_xticks([])
        sidebar_ax.set_yticks([])
        sidebar_ax.set_xlabel("A/I", fontsize=6, labelpad=2)


