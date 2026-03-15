"""visualization/meta_mds_plotter.py – Dual-state meta-MDS scatter (Phase 6)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from analysis.gromov_wasserstein.gw_result import GWDistanceMatrix
from config.settings import Settings


class MetaMDSPlotter:
    """
    Produces a 2D MDS embedding of GW distances between all subjects
    and FCNN states.
    """

    def __init__(self, settings: Settings) -> None:
        self._base_dir = Path(settings.visualization_dir)
        self._dpi = settings.visualization.get("dpi", 150)
        self._n_components = settings.visualization.get("meta_mds_n_components", 2)

    def _out_dir(self, subdir: str) -> Path:
        p = self._base_dir / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Public API ───────────────────────────────────────────────────────────

    def plot(
        self,
        gw_matrix: GWDistanceMatrix,
        human_ids: list[str],
        model_ids: list[str],
        title: str | None = None,
        save_name: str | None = None,
        subdir: str = "phase6_meta_mds",
        show: bool = False,
    ) -> plt.Figure:
        from sklearn.manifold import MDS

        mds = MDS(
            n_components=self._n_components,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress=False,
        )
        coords = mds.fit_transform(gw_matrix.matrix)

        fig, ax = plt.subplots(figsize=(7, 6))

        human_color = "#4878D0"
        model_color = "#EE854A"

        for idx, label in enumerate(gw_matrix.labels):
            color  = human_color if label in human_ids else model_color
            marker = "o"         if label in human_ids else "^"
            ax.scatter(coords[idx, 0], coords[idx, 1],
                       c=color, marker=marker, s=90, zorder=3)
            ax.annotate(
                label, (coords[idx, 0], coords[idx, 1]),
                textcoords="offset points", xytext=(5, 3), fontsize=7, alpha=0.8,
            )

        legend_handles = [
            mpatches.Patch(color=human_color, label="Human subjects"),
            mpatches.Patch(color=model_color, label="FCNN states"),
        ]
        ax.legend(handles=legend_handles, fontsize=9)
        ax.set_xlabel("MDS Dimension 1")
        ax.set_ylabel("MDS Dimension 2")
        _title = title or (
            f"Meta-MDS of GW Distances | {gw_matrix.roi_or_layer} | {gw_matrix.state}"
        )
        ax.set_title(_title, fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_name:
            fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return fig
