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

from pipeline.phases import (
    phase0_finetune,
    phase0b_svm,
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
        """
        Discover and load all subjects using the active data_source mode.

        In ``derivatives`` mode subjects are found under
        ``settings.derivatives_root/<sub-XX>/``.
        In ``replication`` mode subjects are found under
        ``settings.replication_root/MRI/<sub-XX>/``.

        After loading, the actual ROI set present in the data is registered
        with settings so all downstream phases use it.
        """
        cfg = self._cfg
        ids = subject_ids or cfg.subject_ids or []

        if cfg.data_source == "derivatives":
            root = cfg.derivatives_root
            logger.info(
                "Loading %d subjects from derivatives root: %s", len(ids), root
            )
            if not root.exists():
                raise FileNotFoundError(
                    f"derivatives_root does not exist: {root}\n"
                    "Check data.derivatives_root in config/config.yaml."
                )
            for sid in ids:
                subject_dir = root / sid
                if not subject_dir.exists():
                    logger.warning("Subject directory not found: %s – skipping", subject_dir)
                    continue
                try:
                    subj = self._subject_builder.build(subject_dir, sid)
                    self._subjects.append(subj)
                    logger.info("✓ Loaded %s", subj)
                except Exception as exc:
                    logger.error("✗ Failed to load subject %s: %s", sid, exc)

        else:  # replication mode
            mri_root = cfg.replication_root / "MRI"
            logger.info(
                "Loading %d subjects from replication MRI root: %s", len(ids), mri_root
            )
            if not mri_root.exists():
                raise FileNotFoundError(
                    f"replication MRI root does not exist: {mri_root}\n"
                    "Check data.replication_root in config/config.yaml."
                )
            for sid in ids:
                try:
                    # In replication mode SubjectBuilder._build_replication()
                    # resolves all paths internally; subject_root arg is unused.
                    subj = self._subject_builder.build(mri_root / sid, sid)
                    self._subjects.append(subj)
                    logger.info("✓ Loaded %s", subj)
                except Exception as exc:
                    logger.error("✗ Failed to load subject %s: %s", sid, exc)

        logger.info(
            "Loaded %d / %d subjects successfully",
            len(self._subjects), len(ids)
        )

        # Register the ROIs that actually exist in the loaded data
        self._subject_builder.register_rois_with_settings()

    # ── Phase dispatch ───────────────────────────────────────────────────────

    def phase0_finetune_fcnn(self, stimulus_image_dir=None) -> None:
        phase0_finetune.run(
            self._cfg, self._fcnn_embedder, self._subjects, stimulus_image_dir
        )

    def phase0b_svm_decoding(self) -> dict:
        """Phase 0.2 – SVM decoding; runs right after FCNN fine-tuning."""
        return phase0b_svm.run(
            self._cfg, self._subjects, self._human_rdms,
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
            noise_ceiling=self._noise_ceiling,
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

    # ── Full pipeline ────────────────────────────────────────────────────────

    def run(self, subject_ids=None, stimulus_image_dir=None) -> None:
        logger.info("Starting POC Pipeline  (data_source=%r)", self._cfg.data_source)
        self.load_subjects(subject_ids)
        self.phase0_finetune_fcnn(stimulus_image_dir)   # Phase 0.1
        self.phase0b_svm_decoding()                     # Phase 0.2
        self.phase1_extract_embeddings(stimulus_image_dir)
        self.phase2_build_rdms()
        self.phase3_inter_subject_rsa()
        self.phase4_cross_modality_alignment()
        self.phase5_structural_invariance()
        self.phase6_visualize()
        logger.info(
            "Pipeline complete. Results saved to: %s", self._cfg.results_dir
        )
