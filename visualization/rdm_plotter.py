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
        sort_idx = np.argsort(1 - rdm.labels)  # Living first
        mat = rdm.matrix[np.ix_(sort_idx, sort_idx)]
        sorted_labels = rdm.labels[sort_idx]

        n_living = sorted_labels.sum()
        n_total = len(sorted_labels)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="RdYlBu_r", vmin=0, vmax=1.0, aspect="auto")
        plt.colorbar(im, ax=ax, label="Dissimilarity (1 − Spearman ρ)")

        # Draw category boundary
        ax.axhline(n_living - 0.5, color="black", lw=1.5)
        ax.axvline(n_living - 0.5, color="black", lw=1.5)

        # Labels
        ax.set_xlabel("Stimuli (sorted by category)")
        ax.set_ylabel("Stimuli (sorted by category)")
        _title = title or f"{rdm.subject_id} | {rdm.state} | {rdm.roi_or_layer}"
        ax.set_title(_title, fontsize=10, pad=8)

        # Category legend
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
        """Plot conscious and unconscious RDMs side by side."""
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        for ax, rdm, label in zip(
            axes, [rdm_conscious, rdm_unconscious], ["Conscious", "Unconscious"]
        ):
            sort_idx = np.argsort(1 - rdm.labels)
            mat = rdm.matrix[np.ix_(sort_idx, sort_idx)]
            n_living = rdm.labels.sum()
            im = ax.imshow(mat, cmap="RdYlBu_r", vmin=0, vmax=1.0, aspect="auto")
            ax.axhline(n_living - 0.5, color="black", lw=1.5)
            ax.axvline(n_living - 0.5, color="black", lw=1.5)
            ax.set_title(f"{label}\n{rdm.roi_or_layer}", fontsize=9)
            ax.set_xlabel("Stimuli"); ax.set_ylabel("Stimuli")
            plt.colorbar(im, ax=ax, shrink=0.8)

        if suptitle:
            fig.suptitle(suptitle, y=1.01, fontsize=11)

        plt.tight_layout()
        if save_name:
            fig.savefig(self._out_dir / save_name, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig
