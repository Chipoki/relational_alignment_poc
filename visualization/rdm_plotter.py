"""visualization/rdm_plotter.py – Plot single and dual-state RDMs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

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

    # ── Public API ─────────────────────────────────────────────────────────────

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
            fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return fig

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
            rdm_left=rdm_conscious,
            rdm_right=rdm_unconscious,
            label_left="Conscious",
            label_right="Unconscious",
            suptitle=suptitle,
            save_name=save_name,
            subdir=subdir,
            show=show,
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
            rdm_left=rdm_clear,
            rdm_right=rdm_noisy,
            label_left="Clear (0-noise)",
            label_right="Noisy (chance-level)",
            suptitle=suptitle or "FCNN  ·  Representational Dissimilarity Matrix\n(Clear | Noisy)",
            save_name=save_name,
            subdir=subdir,
            show=show,
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

        The title is derived from ``rdm.subject_id`` (``'mean'`` or
        ``'median'``), so no extra parameter is needed.  An optional
        cross-modality RSA annotation is appended when supplied.

        ``subdir`` defaults to ``phase2_rdms/<method>`` if not given.
        """
        method   = rdm.subject_id          # 'mean' or 'median'
        subdir   = subdir or f"phase2_rdms/{method}"
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
                mpatches.Patch(color="none",
                               label=f"Non-living (n={n_total - n_living})"),
            ],
            loc="lower right", fontsize=7, framealpha=0.7,
        )
        plt.tight_layout()
        if save_name:
            fig.savefig(self._out_dir(subdir) / save_name,
                        dpi=self._dpi, bbox_inches="tight")
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
        subdir: str = "phase2_rdms/sorted",
        show: bool = False,
        # Optional pre-computed ordering (pass from Phase 2 to enforce a
        # common stimulus arrangement across all subjects and aggregates).
        common_order: np.ndarray | None = None,
        best_k: int | None = None,
        best_score: float | None = None,
        # Only used when common_order is NOT supplied.
        k_min: int = 3,
        k_max: int = 40,
    ) -> plt.Figure:
        """
        Plot a cluster-sorted RDM with Ward/silhouette boundary lines.

        When ``common_order`` is provided (pre-computed from the mean RDM
        in Phase 2), the same stimulus arrangement is applied verbatim —
        making all sorted figures within a ROI visually comparable.
        When omitted, the optimal order is derived independently from this
        RDM's own dissimilarity structure.
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        if common_order is not None:
            # Use the supplied consensus ordering.
            order      = common_order
            _best_k    = best_k
            _best_score = best_score
            # Derive cluster labels by re-cutting the *own* dendrogram at k*
            # so boundaries reflect this RDM's local geometry while stimulus
            # positions are fixed by the consensus order.
            condensed  = np.clip(squareform(rdm.matrix, checks=False), 0, None)
            Z          = linkage(condensed, method="ward")
            labels_arr = fcluster(Z, _best_k, criterion="maxclust")[order] - 1
        else:
            from analysis.rsa.rdm_utils import sorted_order
            order, _best_k, _best_score = sorted_order(
                rdm.matrix, k_min=k_min, k_max=k_max
            )
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

        order_src = "consensus" if common_order is not None else "independent"
        ax.set_xlabel(f"Stimuli (cluster-sorted, {order_src})")
        ax.set_ylabel(f"Stimuli (cluster-sorted, {order_src})")
        ax.set_title(
            f"{title_prefix}Cluster-Sorted RDM  ·  {rdm.roi_or_layer}  ·  "
            f"{rdm.state.title()}\n"
            f"Ward linkage  |  k*={_best_k}  |  silhouette={_best_score:.3f}  "
            f"({order_src} order)",
            fontsize=9, pad=8,
        )
        plt.tight_layout()
        if save_name:
            fig.savefig(self._out_dir(subdir) / save_name,
                        dpi=self._dpi, bbox_inches="tight")
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
                self._out_dir(subdir) / save_name,
                dpi=self._dpi, bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.close(fig)
        return fig
