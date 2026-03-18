"""visualization/rdm_plotter.py – Plot single and dual-state RDMs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

    # ── Public API ───────────────────────────────────────────────────────────

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
        n_living = sorted_labels.sum()
        n_total = len(sorted_labels)

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
        patches = [
            mpatches.Patch(color="none", label=f"Living (n={n_living})"),
            mpatches.Patch(color="none", label=f"Non-living (n={n_total - n_living})"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.7)
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

    # ── Mean RDM ──────────────────────────────────────────────────────────────

    def plot_mean_rdm(
        self,
        rdm: RDM,
        rsa_rho: float | None = None,
        rsa_p: float | None = None,
        save_name: str | None = None,
        subdir: str = "phase2_rdms/mean",
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot the cross-subject mean RDM for a given ROI & state.
        Optionally annotates with cross-modality RSA ρ and p-value.
        """
        sort_idx = np.argsort(1 - rdm.labels)
        mat = np.copy(rdm.matrix[np.ix_(sort_idx, sort_idx)])
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
            f"Mean RDM  ·  {rdm.roi_or_layer}  ·  {rdm.state.title()}\n"
            f"(averaged across subjects, n_stimuli={n_total})"
        )
        if rsa_rho is not None:
            sig_str = f"p={rsa_p:.4f}" if rsa_p is not None else ""
            title += f"\nFCNN cross-modality RSA: ρ={rsa_rho:.3f}  {sig_str}"
        ax.set_title(title, fontsize=9, pad=8)

        patches = [
            mpatches.Patch(color="none", label=f"Living (n={n_living})"),
            mpatches.Patch(color="none", label=f"Non-living (n={n_total - n_living})"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.7)
        plt.tight_layout()
        if save_name:
            fig.savefig(self._out_dir(subdir) / save_name,
                        dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return fig

    # ── Sorted RDM (Ward + silhouette) ────────────────────────────────────────

    def plot_sorted_rdm(
        self,
        rdm: RDM,
        title_prefix: str = "",
        save_name: str | None = None,
        subdir: str = "phase2_rdms/sorted",
        show: bool = False,
        k_min: int = 3,
        k_max: int = 40,
    ) -> plt.Figure:
        """
        Plot a Ward/silhouette-sorted RDM with cluster boundaries.

        The rows/columns are reordered so that maximally similar stimuli
        are grouped together (Mei et al. 2022 method, generalised).
        """
        from analysis.rsa.rdm_utils import sorted_order

        order, best_k, best_score = sorted_order(
            rdm.matrix, k_min=k_min, k_max=k_max
        )
        mat = rdm.matrix[np.ix_(order, order)]
        np.fill_diagonal(mat, np.nan)
        sorted_labels = rdm.labels[order]

        # Cluster boundary positions
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        condensed = squareform(rdm.matrix, checks=False)
        condensed = np.clip(condensed, 0, None)
        Z = linkage(condensed, method="ward")
        labels_arr = fcluster(Z, best_k, criterion="maxclust")[order] - 1

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, label="Dissimilarity (1 − Spearman ρ)")

        # Draw cluster boundaries
        boundaries = np.where(np.diff(labels_arr))[0] + 0.5
        for b in boundaries:
            ax.axhline(b, color="black", lw=1.2, ls="--", alpha=0.7)
            ax.axvline(b, color="black", lw=1.2, ls="--", alpha=0.7)

        ax.set_xlabel("Stimuli (cluster-sorted)")
        ax.set_ylabel("Stimuli (cluster-sorted)")
        ax.set_title(
            f"{title_prefix}Cluster-Sorted RDM  ·  {rdm.roi_or_layer}  ·  "
            f"{rdm.state.title()}\n"
            f"Ward linkage  |  k*={best_k}  |  silhouette={best_score:.3f}",
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

    # ── Private helpers ──────────────────────────────────────────────────────

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

        for ax, rdm, label in zip(axes, [rdm_left, rdm_right], [label_left, label_right]):
            sort_idx = np.argsort(1 - rdm.labels)
            mat = np.copy(rdm.matrix[np.ix_(sort_idx, sort_idx)])
            np.fill_diagonal(mat, np.nan)
            n_living = rdm.labels.sum()

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
            fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return fig
