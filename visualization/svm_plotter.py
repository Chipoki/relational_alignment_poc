"""visualization/svm_plotter.py – Figures for SVM decoding results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from analysis.svm.svm_decoder import SVMResult
from config.settings import Settings


class SVMPlotter:
    """Generates bar charts and generalization heatmaps for SVM decoding."""

    def __init__(self, settings: Settings) -> None:
        self._base_dir = Path(settings.vis_output_dir)
        self._dpi = settings.vis_dpi

    def _out_dir(self, subdir: str) -> Path:
        p = self._base_dir / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── ROI bar chart: AUC per ROI ──────────────────────────────────────────

    def plot_decoding_by_roi(
        self,
        results: list[SVMResult],
        state: str,
        roi_names: list[str],
        subject_id: str,
        save_name: str,
        subdir: str = "phase0b_svm",
    ) -> plt.Figure:
        """
        Bar chart of mean AUC per ROI for a single subject × state.
        Overlays per-fold AUC as scatter, marks chance level and significance.
        Mirrors Figure 3 layout from Mei et al. (2022).
        """
        res_by_roi = {r.roi: r for r in results if r.state == state}
        rois = [r for r in roi_names if r in res_by_roi]
        if not rois:
            return None

        x = np.arange(len(rois))
        mean_aucs   = [res_by_roi[r].mean_auc    for r in rois]
        mean_chance = [res_by_roi[r].mean_chance  for r in rois]
        sigs        = [res_by_roi[r].significant  for r in rois]
        fold_aucs   = [res_by_roi[r].auc_scores   for r in rois]

        fig, ax = plt.subplots(figsize=(max(8, len(rois) * 0.9), 4.5))
        colors = ["#2196F3" if s else "#90CAF9" for s in sigs]
        bars = ax.bar(x, mean_aucs, color=colors, width=0.6, zorder=2,
                      label="Mean AUC (SVM)")
        ax.bar(x, mean_chance, color="lightgrey", width=0.6, zorder=1,
               alpha=0.5, label="Chance (Dummy)")

        # Scatter per-fold AUCs
        for i, folds in enumerate(fold_aucs):
            jitter = self._rng_jitter(len(folds))
            ax.scatter(x[i] + jitter, folds, s=6, color="black",
                       alpha=0.3, zorder=3)

        # Significance markers
        for i, (sig, auc) in enumerate(zip(sigs, mean_aucs)):
            if sig:
                ax.text(x[i], auc + 0.01, "*", ha="center",
                        va="bottom", fontsize=10, color="black")

        ax.axhline(0.5, color="red", lw=1.2, ls="--", label="0.5 chance")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [r.replace("_", "\n") for r in rois],
            fontsize=7, rotation=45, ha="right",
        )
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.3, 1.05)
        ax.set_title(
            f"SVM Decoding  ·  {subject_id}  ·  {state.title()}\n"
            f"Leave-one-pair-out CV  (blue = p<0.05 Bonferroni)",
            fontsize=9,
        )
        sig_patch  = mpatches.Patch(color="#2196F3",  label="Significant")
        ns_patch   = mpatches.Patch(color="#90CAF9",  label="Non-significant")
        ax.legend(handles=[sig_patch, ns_patch], fontsize=7, loc="lower right")

        plt.tight_layout()
        fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi,
                    bbox_inches="tight")
        plt.close(fig)
        return fig

    # ── Generalisation heatmap across subjects & ROIs ──────────────────────

    def plot_generalisation_heatmap(
        self,
        results_by_subject: dict[str, list[SVMResult]],
        roi_names: list[str],
        save_name: str = "phase0b_svm_generalisation_heatmap.png",
        subdir: str = "phase0b_svm",
    ) -> plt.Figure:
        """
        Subjects × ROI heatmap of delta-AUC (SVM − chance) for C→U transfer.
        Cells are hatched if non-significant.
        Closely mirrors Fig. 3 layout from the paper.
        """
        subjects = sorted(results_by_subject.keys())
        rois     = roi_names
        data     = np.full((len(subjects), len(rois)), np.nan)
        sig_mask = np.zeros_like(data, dtype=bool)

        for i, sid in enumerate(subjects):
            res_map = {r.roi: r for r in results_by_subject[sid]
                       if r.state == "c_to_u"}
            for j, roi in enumerate(rois):
                if roi in res_map:
                    data[i, j]     = res_map[roi].delta_auc
                    sig_mask[i, j] = res_map[roi].significant

        fig, ax = plt.subplots(figsize=(max(8, len(rois) * 0.8),
                                        max(3, len(subjects) * 0.7)))
        im = ax.imshow(data, cmap="RdYlBu_r", vmin=-0.2, vmax=0.4,
                       aspect="auto")
        plt.colorbar(im, ax=ax, label="ΔAUC (SVM − chance)")

        # Hatch non-significant cells
        for i in range(len(subjects)):
            for j in range(len(rois)):
                if not sig_mask[i, j] and not np.isnan(data[i, j]):
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch="///", edgecolor="grey", lw=0.5,
                    ))

        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(
            [r.replace("_", "\n") for r in rois],
            fontsize=7, rotation=45, ha="right",
        )
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels(subjects, fontsize=8)
        ax.set_title(
            "Cross-State Generalisation: Train=Conscious → Test=Unconscious\n"
            "ΔAUC  (hatching = non-significant after Bonferroni)",
            fontsize=9,
        )
        plt.tight_layout()
        fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi,
                    bbox_inches="tight")
        plt.close(fig)
        return fig

    # ── Summary across subjects ─────────────────────────────────────────────

    def plot_group_summary(
        self,
        all_results: dict[str, dict[str, list[SVMResult]]],
        roi_names: list[str],
        states: tuple[str, ...] = ("conscious", "unconscious", "c_to_u"),
        save_name: str = "phase0b_svm_group_summary.png",
        subdir: str = "phase0b_svm",
    ) -> plt.Figure:
        """
        Group-level mean AUC (±SEM across subjects) per ROI and state.
        """
        state_colors = {
            "conscious":   "#1565C0",
            "unconscious": "#E65100",
            "c_to_u":      "#2E7D32",
        }
        state_labels = {
            "conscious":   "Conscious",
            "unconscious": "Unconscious",
            "c_to_u":      "Conscious → Unconscious",
        }
        n_rois = len(roi_names)
        n_states = len(states)
        fig, ax  = plt.subplots(figsize=(max(9, n_rois * 1.0), 5))
        x = np.arange(n_rois)
        bar_w = 0.25

        for s_idx, state in enumerate(states):
            means, sems = [], []
            for roi in roi_names:
                vals = [
                    r.mean_auc
                    for sid in all_results
                    for r in all_results[sid].get(state, [])
                    if r.roi == roi and r.n_folds > 0
                ]
                means.append(np.mean(vals) if vals else np.nan)
                sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            offset = (s_idx - n_states / 2 + 0.5) * bar_w
            ax.bar(x + offset, means, bar_w,
                   color=state_colors[state], label=state_labels[state],
                   yerr=sems, capsize=3, alpha=0.85)

        ax.axhline(0.5, color="red", lw=1.2, ls="--", label="Chance (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [r.replace("_", "\n") for r in roi_names],
            fontsize=7, rotation=45, ha="right",
        )
        ax.set_ylabel("Mean ROC-AUC  (± SEM across subjects)")
        ax.set_ylim(0.35, 1.0)
        ax.set_title("Group SVM Decoding Summary  ·  All ROIs", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        fig.savefig(self._out_dir(subdir) / save_name, dpi=self._dpi,
                    bbox_inches="tight")
        plt.close(fig)
        return fig

    @staticmethod
    def _rng_jitter(n: int, scale: float = 0.18) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.uniform(-scale / 2, scale / 2, n)
