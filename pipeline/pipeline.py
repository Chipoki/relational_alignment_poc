"""pipeline/pipeline.py – POCPipeline: state container + orchestrator."""
from __future__ import annotations

import logging
from pathlib import Path

from config.settings import Settings
from data.preprocessors.subject_builder import SubjectBuilder
from embeddings.fcnn_embedder import FCNNEmbedder
from embeddings.fmri_embedder import FMRIEmbedder
from embeddings.embedding_store import EmbeddingStore
from analysis.rsa.rdm import RDMBuilder
from analysis.rsa.rsa_analyzer import RSAAnalyzer
from analysis.rsa.noise_ceiling import NoiseCeiling
from analysis.gromov_wasserstein.gw_aligner import GromovWassersteinAligner
from visualization.rdm_plotter import RDMPlotter
from visualization.meta_mds_plotter import MetaMDSPlotter
from visualization.transport_plotter import TransportPlotter
from visualization.summary_plotter import SummaryPlotter
from pipeline.phases import phase7_svm


from pipeline.phases import (
    phase0_finetune,
    phase1_embeddings,
    phase2_rdms,
    phase3_rsa,
    phase4_cross_modality,
    phase5_invariance,
    phase6_visualize,
)

logger = logging.getLogger(__name__)


class POCPipeline:
    """
    Thin orchestrator: owns shared state and component instances,
    delegates all logic to the per-phase modules in pipeline/phases/.
    """

    def __init__(self, settings: Settings) -> None:
        self._cfg = settings
        settings.ensure_output_dirs()

        self._subject_builder   = SubjectBuilder(settings)
        self._fcnn_embedder     = FCNNEmbedder(settings)
        self._fmri_embedder     = FMRIEmbedder(settings)
        self._embedding_store   = EmbeddingStore(settings.embedding_dir)
        self._rdm_builder       = RDMBuilder()
        self._rsa_analyzer      = RSAAnalyzer(settings)
        self._noise_ceiling     = NoiseCeiling()
        self._gw_aligner        = GromovWassersteinAligner(settings)

        self._rdm_plotter       = RDMPlotter(settings)
        self._meta_mds_plotter  = MetaMDSPlotter(settings)
        self._transport_plotter = TransportPlotter(settings)
        self._summary_plotter   = SummaryPlotter(settings)

        self._subjects:   list = []
        self._human_rdms: dict = {}
        self._fcnn_rdms:  dict = {}

    # ── Subject loading ──────────────────────────────────────────────────────

    def load_subjects(self, subject_ids: list[str] | None = None) -> None:
        """Discover and load all subjects, then register actual ROIs with settings."""
        root = Path(self._cfg.data["root"])
        if not root.exists():
            raise FileNotFoundError(
                f"Data root not found: {root}\n"
                "Please set data.root in config/config.yaml."
            )

        ids = subject_ids or self._cfg.data.get("subject_ids") or []
        if not ids:
            ids = sorted(p.name for p in root.iterdir() if p.is_dir())
        if not ids:
            raise ValueError(f"No subject directories found under {root}")

        logger.info("Loading %d subjects: %s", len(ids), ids)
        for sid in ids:
            try:
                subj = self._subject_builder.build(root / sid, sid)
                self._subjects.append(subj)
                logger.info("✓ Loaded %s", subj)
            except Exception as exc:
                logger.error("✗ Failed to load subject %s: %s", sid, exc)

        logger.info(
            "Loaded %d / %d subjects successfully", len(self._subjects), len(ids)
        )

        # ── Register the ROIs that actually exist in the data ─────────────
        # This makes settings.active_roi_names authoritative for all downstream
        # phases.  wholebrain is always included; any FreeSurfer region masks
        # found in func_masks/ are added on top.
        self._subject_builder.register_rois_with_settings()

    # ── Phase dispatch ───────────────────────────────────────────────────────

    def phase0_finetune_fcnn(self, stimulus_image_dir=None) -> None:
        phase0_finetune.run(
            self._cfg, self._fcnn_embedder, self._subjects, stimulus_image_dir
        )

    def phase1_extract_embeddings(self, stimulus_image_dir=None) -> None:
        phase1_embeddings.run(
            self._fcnn_embedder, self._embedding_store, stimulus_image_dir
        )

    def phase2_build_rdms(self) -> None:
        self._human_rdms, self._fcnn_rdms = phase2_rdms.run(
            self._cfg, self._subjects, self._embedding_store,
            self._rdm_builder, self._rdm_plotter,
        )

    def phase3_inter_subject_rsa(self) -> dict:
        return phase3_rsa.run(
            self._cfg, self._human_rdms,
            self._rsa_analyzer, self._noise_ceiling,
            self._gw_aligner, self._summary_plotter,
        )

    def phase4_cross_modality_alignment(self) -> dict:
        return phase4_cross_modality.run(
            self._cfg, self._human_rdms, self._fcnn_rdms,
            self._rsa_analyzer, self._gw_aligner,
        )

    def phase5_structural_invariance(self) -> dict:
        return phase5_invariance.run(
            self._cfg, self._human_rdms, self._fcnn_rdms,
            self._gw_aligner, self._summary_plotter,
        )

    def phase6_visualize(self) -> None:
        phase6_visualize.run(
            self._subjects, self._human_rdms, self._fcnn_rdms,
            self._gw_aligner, self._meta_mds_plotter,
            settings=self._cfg,
        )

    def phase7_svm_decoding(self) -> dict:
        from pipeline.phases import phase7_svm
        return phase7_svm.run(
            self._cfg, self._subjects, self._human_rdms,
        )

    # ── Full pipeline ────────────────────────────────────────────────────────

    def run(self, subject_ids=None, stimulus_image_dir=None) -> None:
        logger.info("Starting POC Pipeline")
        self.load_subjects(subject_ids)
        self.phase0_finetune_fcnn(stimulus_image_dir)
        self.phase1_extract_embeddings(stimulus_image_dir)
        self.phase2_build_rdms()
        self.phase3_inter_subject_rsa()
        self.phase4_cross_modality_alignment()
        self.phase5_structural_invariance()
        self.phase6_visualize()
        self.phase7_svm_decoding()
        logger.info(
            "Pipeline complete. Results saved to: %s", self._cfg.results_dir
        )
