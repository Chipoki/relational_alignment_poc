"""visualization/rdm_plotter.py – Plot single, dual-state, aggregate and second-order RDMs."""
from __future__ import annotations

from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from analysis.rsa.rdm import RDM
from config.settings import Settings


class RDMPlotter:
    """Generates RDM heatmaps with category-sorted stimulus ordering (Memory Safe OO API)."""

    def __init__(self, settings: Settings) -> None:
        self._base_dir = Path(settings.vis_output_dir)
        self._dpi = settings.vis_dpi

    def _out_dir(self, subdir: str) -> Path:
        p = self._base_dir / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def plot_rdm(
        self, rdm: RDM, title: str | None = None, save_name: str | None = None,
        subdir: str = "phase2_rdms/original_rdms", show: bool = False,
    ) -> None:
        sort_idx = np.argsort(1 - rdm.labels)
        mat = np.copy(rdm.matrix[np.ix_(sort_idx, sort_idx)])
        np.fill_diagonal(mat, np.nan)
        sorted_labels = rdm.labels[sort_idx]
        n_living = int(sorted_labels.sum())

        # Object-Oriented Figure Generation
        fig = Figure(figsize=(6, 5))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto")
        fig.colorbar(im, ax=ax, label="Dissimilarity (1 − Spearman ρ)")
        ax.axhline(n_living - 0.5, color="black", lw=1.5)
        ax.axvline(n_living - 0.5, color="black", lw=1.5)
        ax.set_xlabel("Stimuli (Living → Non-living)")
        ax.set_ylabel("Stimuli (Living → Non-living)")
        ax.set_title(title or f"{rdm.subject_id} | {rdm.state} | {rdm.roi_or_layer}", fontsize=10, pad=8)

        fig.tight_layout()
        if save_name:
            fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight")

        fig.clf()
        return None

    def plot_dual_state(
        self, rdm_conscious: RDM, rdm_unconscious: RDM, suptitle: str | None = None,
        save_name: str | None = None, subdir: str = "phase2_rdms/original_rdms", show: bool = False,
    ) -> None:
        return self._plot_dual(rdm_conscious, rdm_unconscious, "Conscious", "Unconscious", suptitle, save_name, subdir, show)

    def plot_dual_state_fcnn(
        self, rdm_clear: RDM, rdm_noisy: RDM, suptitle: str | None = None,
        save_name: str | None = None, subdir: str = "phase2_rdms/original_rdms", show: bool = False,
    ) -> None:
        return self._plot_dual(rdm_clear, rdm_noisy, "Clear (0-noise)", "Noisy (chance-level)", suptitle, save_name, subdir, show)

    def plot_mean_rdm(
        self, rdm: RDM, rsa_rho: float | None = None, rsa_p: float | None = None,
        save_name: str | None = None, subdir: str | None = None, show: bool = False, included_subjects: list[str] | None = None,
    ) -> None:
        method = rdm.subject_id
        subdir = subdir or f"phase2_rdms/original_rdms/{method}"
        sort_idx = np.argsort(1 - rdm.labels)
        mat      = np.copy(rdm.matrix[np.ix_(sort_idx, sort_idx)])
        np.fill_diagonal(mat, np.nan)
        n_living = int(rdm.labels.sum())

        fig = Figure(figsize=(6, 5))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto")
        fig.colorbar(im, ax=ax, label="Dissimilarity (1 − Spearman ρ)")
        ax.axhline(n_living - 0.5, color="black", lw=1.5)
        ax.axvline(n_living - 0.5, color="black", lw=1.5)
        ax.set_title(f"{method.title()} RDM  ·  {rdm.roi_or_layer}  ·  {rdm.state.title()}", fontsize=9, pad=8)

        self._draw_animacy_sidebar(ax, rdm.labels[sort_idx], side="right")
        if included_subjects: _annotate_included_subjects(fig, included_subjects)

        fig.tight_layout()
        if save_name:
            fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight")

        fig.clf()
        return None

    def plot_sorted_rdm(
        self,
        rdm: RDM,
        title_prefix: str = "",
        save_name: str | None = None,
        subdir: str = "phase2_rdms/original_rdms/sorted_independent",
        show: bool = False,
        common_order: np.ndarray | None = None,
        best_k: int | None = None,
        best_score: float | None = None,
        k_min: int = 2,
        k_max: int = 8,
        included_subjects: list[str] | None = None,
        linkage_method: str = "ward",
        within_category: bool = False,
    ) -> None:
        from analysis.rsa.rdm_utils import sorted_order, sorted_order_within_category

        if common_order is not None and len(common_order) != rdm.n_stimuli:
            print(f"Shape mismatch for {rdm.subject_id}. Falling back to independent.")
            common_order = None

        if common_order is not None:
            order, _best_k, _best_score = common_order, best_k, best_score
            order_src = "GW-consensus"
        else:
            if within_category:
                order, _best_k, _best_score = sorted_order_within_category(
                    rdm.matrix, rdm.labels, k_min=k_min, k_max=k_max, method=linkage_method
                )
            else:
                order, _best_k, _best_score = sorted_order(
                    rdm.matrix, k_min=k_min, k_max=k_max, method=linkage_method
                )
            order_src = "independent"

        mat = rdm.matrix[np.ix_(order, order)]
        np.fill_diagonal(mat, np.nan)

        fig = Figure(figsize=(6.5, 5.5))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto")
        fig.colorbar(im, ax=ax, label="Dissimilarity (1 − Spearman ρ)")

        if within_category:
            n_living = int(rdm.labels.sum())
            ax.axhline(n_living - 0.5, color="black", lw=1.5, ls="-")
            ax.axvline(n_living - 0.5, color="black", lw=1.5, ls="-")

        wc_text = "\n(Within-Category)" if within_category else ""
        ax.set_title(
            f"{title_prefix}Cluster-Sorted RDM  ·  {rdm.roi_or_layer}  ·  {rdm.state.title()}{wc_text}\n"
            f"{linkage_method.title()}  |  k*={_best_k}  |  silh={_best_score:.3f}  ({order_src})",
            fontsize=9, pad=8,
        )

        self._draw_animacy_sidebar(ax, rdm.labels[order], side="right")
        if included_subjects: _annotate_included_subjects(fig, included_subjects)

        fig.tight_layout()
        if save_name:
            fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight")

        fig.clf()
        return None

    def plot_roi_x_roi_rdm(
        self, rho_matrix: np.ndarray, p_matrix: np.ndarray, roi_names: list[str],
        method: str, state: str, save_name: str | None = None,
        subdir: str = "phase4_cross_modality/roi_x_roi", alpha: float = 0.05, show: bool = False,
    ) -> None:
        n = len(roi_names)
        n_pairs = n * (n - 1) // 2
        bonf_alpha = alpha / max(n_pairs, 1)
        sig_mask = p_matrix < bonf_alpha
        np.fill_diagonal(sig_mask, True)

        plot_mat = rho_matrix.copy().astype(float)
        np.fill_diagonal(plot_mat, np.nan)

        fig = Figure(figsize=(max(5, n * 0.65 + 1.5), max(4.5, n * 0.65 + 1.2)))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        vabs = np.nanmax(np.abs(rho_matrix[~np.eye(n, dtype=bool)]))
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
        im = ax.imshow(plot_mat, cmap="RdYlBu_r", norm=norm, aspect="auto")
        fig.colorbar(im, ax=ax, label="Spearman ρ")

        for i in range(n):
            ax.add_patch(mpatches.Rectangle((i - 0.5, i - 0.5), 1, 1, color="lightgrey", zorder=2))
            ax.text(i, i, "1.00", ha="center", va="center", fontsize=7, color="dimgrey", zorder=3)
            for j in range(n):
                if i == j: continue
                rho = rho_matrix[i, j]
                ax.text(j, i, f"{rho:.2f}", ha="center", va="center", fontsize=max(5, 8 - n // 4), color="black" if abs(rho) < 0.7 else "white", zorder=3)
                if not sig_mask[i, j]:
                    ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor="grey", lw=0.5, zorder=4))

        labels = [r.replace("_", "\n") for r in roi_names]
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)

        fig.tight_layout()
        if save_name:
            fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight")

        fig.clf()
        return None

    def _plot_dual(self, rdm_left: RDM, rdm_right: RDM, label_left: str, label_right: str, suptitle: str | None, save_name: str | None, subdir: str, show: bool) -> None:
        fig = Figure(figsize=(11, 4.5))
        canvas = FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2)

        for ax, rdm, label in zip(axes, [rdm_left, rdm_right], [label_left, label_right]):
            sort_idx = np.argsort(1 - rdm.labels)
            mat = np.copy(rdm.matrix[np.ix_(sort_idx, sort_idx)])
            np.fill_diagonal(mat, np.nan)
            n_living = int(rdm.labels.sum())

            im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto")
            ax.axhline(n_living - 0.5, color="black", lw=1.5)
            ax.axvline(n_living - 0.5, color="black", lw=1.5)
            ax.set_title(f"{label}\n{rdm.roi_or_layer}", fontsize=9)
            cb = fig.colorbar(im, ax=ax, shrink=0.8)

        if suptitle:
            fig.suptitle(suptitle, y=1.01, fontsize=11)

        fig.tight_layout()
        if save_name:
            fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi, bbox_inches="tight")

        fig.clf()
        return None

    @staticmethod
    def _draw_animacy_sidebar(ax, sorted_labels: np.ndarray, side: str = "right", bar_width: str = "2%", bar_pad: str = "1%") -> None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        sidebar_img = np.where(sorted_labels[:, None] == 1, 0.0, 1.0)
        divider = make_axes_locatable(ax)
        sidebar_ax = divider.append_axes(side, size=bar_width, pad=bar_pad)
        sidebar_ax.imshow(sidebar_img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
        sidebar_ax.set_xticks([]); sidebar_ax.set_yticks([])

def _annotate_included_subjects(fig: "Figure", subjects: list) -> None:
    label = "Included subjects: " + ", ".join(sorted(str(s) for s in subjects))
    fig.text(0.5, -0.02, label, ha="center", va="top", fontsize=6.5, color="#444444", wrap=True)