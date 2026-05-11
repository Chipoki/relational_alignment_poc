"""run_pipeline.py – CLI entry point for the POC pipeline.

Usage
-----
    # Full all-subjects run (original behaviour, unchanged):
    python run_pipeline.py --config config/config.yaml

    # ── Per-subject (low-RAM) workflow ────────────────────────────────────
    # Stage A – run once per subject to dump RDMs + SVM patterns to disk:
    python run_pipeline.py --subjects sub-01 --dump-subject-rdms

    # Stage B – load dumped RDMs for all subjects and run phases 3-6:
    python run_pipeline.py --from-subject-rdms

    # Stage B – single phase (e.g. inter-subject RSA only):
    python run_pipeline.py --from-subject-rdms --phase 3

    # Single-phase run using subjects loaded in memory (original behaviour):
    python run_pipeline.py --subjects sub-01 --phase 2

Phase numbering
---------------
    0   – FCNN fine-tuning        (phase 0.1)
    0b  – SVM decoding            (phase 0.2, runs right after FCNN fine-tune)
    1   – Embedding extraction
    2   – RDM construction
    3   – Inter-subject RSA
    4   – Cross-modality alignment
    5   – Structural invariance
    6   – Visualisation
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings
from pipeline.pipeline import POCPipeline
from utils.logging_utils import setup_logging

_VALID_PHASES = ["0", "0b", "1", "2", "3", "4", "5", "6"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the POC Relational Alignment Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--stimulus-dir",
        default="./../soto_data/unconfeats/data/experiment_images_greyscaled",
    )
    parser.add_argument(
        "--phase",
        default=None,
        choices=_VALID_PHASES,
        help=(
            "Run a single phase (requires prior phases' outputs on disk). "
            "Valid values: " + ", ".join(_VALID_PHASES)
        ),
    )

    # ── Per-subject / low-RAM flags ──────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dump-subject-rdms",
        action="store_true",
        help=(
            "After loading BOLD and building RDMs for the given subject(s), "
            "dump per-subject RDM archives to disk and exit.  "
            "Run once per subject to build up the archive.  "
            "Incompatible with --from-subject-rdms."
        ),
    )
    mode_group.add_argument(
        "--from-subject-rdms",
        action="store_true",
        help=(
            "Skip BOLD loading entirely.  Load per-subject RDM archives "
            "written by previous --dump-subject-rdms runs and continue with "
            "phases 3-6 (or the single --phase specified).  "
            "All subjects whose archives exist on disk are used unless "
            "--subjects restricts the list."
        ),
    )

    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args     = _parse_args()
    settings = Settings(args.config)

    setup_logging(
        level=args.log_level,
        log_file=str(settings.log_dir / "pipeline.log"),
    )

    pipeline = POCPipeline(settings)

    # ── Replication mode: fully automatic per-subject iteration ─────────────
    # When data_source == "replication" and neither legacy flag is set,
    # the pipeline iterates over each subject individually, then runs the
    # downstream phases on the intersected archives.
    if (
        settings.data_source == "replication"
        and not args.dump_subject_rdms
        and not args.from_subject_rdms
        and args.phase is None
    ):
        _run_replication_mode(pipeline, args, settings)
        return

    # ── Mode A: dump per-subject RDMs and exit (legacy, still works) ────────
    if args.dump_subject_rdms:
        _run_dump_mode(pipeline, args, settings)
        return

    # ── Mode B: load from disk and run downstream phases (legacy) ───────────
    if args.from_subject_rdms:
        _run_from_disk_mode(pipeline, args, settings)
        return

    # ── Original mode: full in-memory run ────────────────────────────────────
    if args.phase is None:
        pipeline.run(subject_ids=args.subjects, stimulus_image_dir=args.stimulus_dir)
        return

    # Single-phase run (original behaviour)
    pipeline.load_subjects(args.subjects)
    dispatch = {
        "0":  lambda: pipeline.phase0_finetune_fcnn(args.stimulus_dir),
        "0b": pipeline.phase0b_svm_decoding,
        "1":  lambda: pipeline.phase1_extract_embeddings(args.stimulus_dir),
        "2":  pipeline.phase2_build_rdms,
        "3":  pipeline.phase3_inter_subject_rsa,
        "4":  pipeline.phase4_cross_modality_alignment,
        "5":  pipeline.phase5_structural_invariance,
        "6":  pipeline.phase6_visualize,
    }
    dispatch[args.phase]()


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_replication_mode(
    pipeline: "POCPipeline",
    args: argparse.Namespace,
    settings: Settings,
) -> None:
    """
    Fully automatic replication-mode pipeline.

    Part 1 – per-subject loop
    -------------------------
    For each subject (in order):
      * Skip fMRI loading entirely if all Stage-A outputs already exist on
        disk (per-subject RDM archive + SVM checkpoint).
      * Otherwise: load BOLD, run phases 0-2 (fine-tune, embeddings, RDMs,
        SVM), dump all outputs, then free RAM before the next subject.
      * After each subject's dump: update aggregate figures AND intersected
        figures/caches so they always reflect every subject processed so far.

    Part 2 – cohort-level downstream
    ---------------------------------
    After every subject has been processed, load the intersected RDMs and
    run phases 3-6 on the globally-consistent stimulus space.
    """
    log = logging.getLogger(__name__)
    subject_ids = args.subjects or settings.subject_ids

    log.info(
        "Replication mode: processing %d subject(s) sequentially: %s",
        len(subject_ids), subject_ids,
    )

    # ── Part 1: per-subject loop ─────────────────────────────────────────────
    for sid in subject_ids:
        log.info("─" * 60)
        log.info("Subject %s", sid)

        if settings.subject_phase1_complete(sid):
            log.info(
                "  Stage-A outputs already on disk for %s – skipping fMRI load. "
                "Running intersected update only.", sid,
            )
        else:
            # Load one subject
            pipeline.load_subjects([sid])

            if not pipeline._subjects:
                log.error("  Could not load subject %s – skipping.", sid)
                pipeline.clear_subject_data()
                continue

            # Phase 0.1: FCNN fine-tune (idempotent)
            pipeline.phase0_finetune_fcnn(args.stimulus_dir)

            # Phase 1: embeddings (idempotent)
            pipeline.phase1_extract_embeddings(args.stimulus_dir)

            # Phase 2: build + dump RDMs (also calls _update_aggregate_figures
            # and _update_intersected_figures internally)
            pipeline.phase2_build_rdms()

            # Phase 0.2: SVM decoding + dump
            pipeline.phase0b_svm_decoding()

            # Free RAM before next subject
            pipeline.clear_subject_data()
            import gc; gc.collect()

        # Even when skipping fMRI load, re-run the intersected update so that
        # re-running any single subject always produces updated group figures.
        pipeline.phase2_update_intersected()

    # ── Part 2: cohort-level downstream phases ───────────────────────────────
    log.info("=" * 60)
    log.info("All subjects processed – starting downstream phases (3-6)")
    log.info("=" * 60)

    pipeline.load_intersected_rdms_for_downstream(subject_ids=None)

    pipeline.phase0b_svm_decoding_from_disk(subject_ids=None)
    pipeline.phase3_inter_subject_rsa()
    pipeline.phase4_cross_modality_alignment()
    pipeline.phase5_structural_invariance()
    pipeline.phase6_visualize()

    log.info(
        "Pipeline (replication mode) complete. Results saved to: %s",
        settings.results_dir,
    )

def _run_dump_mode(
    pipeline: POCPipeline,
    args: argparse.Namespace,
    settings: Settings,
) -> None:
    """
    Stage A – per-subject run.

    Loads BOLD for the given subject(s), runs phases 0b + 2 (SVM patterns +
    RDMs), dumps both to disk, then exits.  Phases 3-6 are NOT run.

    Typical usage (loop over subjects in a shell script):
        for sub in sub-01 sub-02 sub-03; do
            python run_pipeline.py --subjects $sub --dump-subject-rdms
        done
    """
    if not args.subjects:
        logging.warning(
            "--dump-subject-rdms requires at least one subject via --subjects. "
            "Auto-discovering from config."
        )
    pipeline.load_subjects(args.subjects)

    if not pipeline._subjects:
        logging.error("No subjects loaded – nothing to dump.")
        return

    # Phase 0.1: FCNN fine-tuning (idempotent checkpoint)
    pipeline.phase0_finetune_fcnn(args.stimulus_dir)

    # Phase 1: embeddings (idempotent checkpoint)
    pipeline.phase1_extract_embeddings(args.stimulus_dir)

    # Phase 2: build RDMs + dump per-subject archives automatically
    # (dump_subject_rdms is called inside phase2_rdms.run() for every subject)
    pipeline.phase2_build_rdms()

    # Phase 0.2: SVM decoding + dump SVM patterns
    # (dump_svm_patterns is called inside phase0b_svm.run() for every subject)
    pipeline.phase0b_svm_decoding()

    n_dumped = len(pipeline._subjects)
    rdm_dir  = settings.subject_rdm_dir
    logging.getLogger(__name__).info(
        "Dump complete.  %d subject(s) archived under %s.\n"
        "Re-run with --from-subject-rdms (optionally --phase 3|4|5|6) "
        "once all subjects are dumped.",
        n_dumped, rdm_dir,
    )


def _run_from_disk_mode(
    pipeline: POCPipeline,
    args: argparse.Namespace,
    settings: Settings,
) -> None:
    """
    Stage B – cohort-level run from archived RDMs.

    Loads per-subject RDM archives (no BOLD), then runs the requested
    downstream phases.  Defaults to phases 0b + 3 + 4 + 5 + 6.
    """
    subject_ids = args.subjects  # None → use all available on disk

    # Reconstruct human_rdms + fcnn_rdms from disk
    pipeline.phase2_load_rdms_from_disk(subject_ids)

    if args.phase is not None:
        # Single-phase dispatch
        dispatch_disk = {
            "0b": lambda: pipeline.phase0b_svm_decoding_from_disk(subject_ids),
            "3":  pipeline.phase3_inter_subject_rsa,
            "4":  pipeline.phase4_cross_modality_alignment,
            "5":  pipeline.phase5_structural_invariance,
            "6":  pipeline.phase6_visualize,
        }
        if args.phase not in dispatch_disk:
            logging.getLogger(__name__).error(
                "Phase %s cannot be run in --from-subject-rdms mode "
                "(it requires BOLD data).  Valid phases: %s",
                args.phase, sorted(dispatch_disk.keys()),
            )
            return
        dispatch_disk[args.phase]()
        return

    # Default: run everything that doesn't need BOLD
    pipeline.phase0b_svm_decoding_from_disk(subject_ids)
    pipeline.phase3_inter_subject_rsa()
    pipeline.phase4_cross_modality_alignment()
    pipeline.phase5_structural_invariance()
    pipeline.phase6_visualize()
    logging.getLogger(__name__).info(
        "Pipeline (from-disk mode) complete.  Results saved to: %s",
        settings.results_dir,
    )


if __name__ == "__main__":
    main()

