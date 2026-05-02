"""visualization/transport_plotter.py – Bipartite transport plan heatmaps."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.gromov_wasserstein.gw_result import GWResult
from config.settings import Settings


class TransportPlotter:
    """Plots the optimal transport plan as a bipartite matching heatmap."""

    def __init__(self, settings: Settings) -> None:
        self._base_dir = Path(settings.vis_output_dir)
        self._dpi = settings.vis_dpi

    def _out_dir(self, subdir: str) -> Path:
        p = self._base_dir / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def plot(
        self,
        result: GWResult,
        source_labels: np.ndarray,
        target_labels: np.ndarray,
        title: str | None = None,
        save_name: str | None = None,
        subdir: str = "phase4_cross_modality",
        show: bool = False,
    ) -> plt.Figure:
        src_sort = np.argsort(1 - source_labels)
        tgt_sort = np.argsort(1 - target_labels)

        T = result.transport_plan[np.ix_(src_sort, tgt_sort)]

        n_src_living = source_labels.sum()
        n_tgt_living = target_labels.sum()

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(T, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Transport mass")

        ax.axhline(n_src_living - 0.5, color="white", lw=1.5, ls="--")
        ax.axvline(n_tgt_living - 0.5, color="white", lw=1.5, ls="--")

        ax.set_xlabel(f"Target: {result.target_id} (sorted by category)")
        ax.set_ylabel(f"Source: {result.source_id} (sorted by category)")
        _title = title or (
            f"Transport Plan | {result.roi_or_layer}\n"
            f"GWD={result.gw_distance:.4f}, Top-k rate={result.top_k_matching_rate:.2f}"
        )
        ax.set_title(_title, fontsize=9)
        plt.tight_layout()

        if save_name:
            fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return fig
