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
        self._out_dir = Path(settings.visualization_dir)
        self._dpi = settings.visualization.get("dpi", 150)

    # ── Public API ───────────────────────────────────────────────────────────

    def plot_rdm(
        self,
        rdm: RDM,
        title: str | None = None,
        save_name: str | None = None,
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
        ax.set_title(title or f"{rdm.subject_id} | {rdm.state} | {rdm.roi_or_layer}", fontsize=10, pad=8)
        patches = [
            mpatches.Patch(color="none", label=f"Living (n={n_living})"),
            mpatches.Patch(color="none", label=f"Non-living (n={n_total - n_living})"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.7)
        plt.tight_layout()
        if save_name:
            fig.savefig(self._out_dir / save_name, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    def plot_dual_state(
        self,
        rdm_conscious: RDM,
        rdm_unconscious: RDM,
        suptitle: str | None = None,
        save_name: str | None = None,
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
            show=show,
        )

    def plot_dual_state_fcnn(
        self,
        rdm_clear: RDM,
        rdm_noisy: RDM,
        suptitle: str | None = None,
        save_name: str | None = None,
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
            show=show,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _plot_dual(
        self,
        rdm_left: RDM,
        rdm_right: RDM,
        label_left: str,
        label_right: str,
        suptitle: str | None,
        save_name: str | None,
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
            fig.savefig(self._out_dir / save_name, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig
