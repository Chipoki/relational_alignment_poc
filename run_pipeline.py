"""run_pipeline.py – CLI entry point for the POC pipeline.

Usage
-----
    python run_pipeline.py --config config/config.yaml [--subjects sub-01] [--phase 2]

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
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings(args.config)
    setup_logging(level=args.log_level, log_file=str(Path(settings.log_dir) / "pipeline.log"))

    pipeline = POCPipeline(settings)

    if args.phase is None:
        pipeline.run(subject_ids=args.subjects, stimulus_image_dir=args.stimulus_dir)
        return

    pipeline.load_subjects(args.subjects)
    dispatch = {
        "0":  lambda: pipeline.phase0_finetune_fcnn(args.stimulus_dir),
        "0b": lambda: pipeline.phase0b_svm_decoding(),
        "1":  lambda: pipeline.phase1_extract_embeddings(args.stimulus_dir),
        "2":  pipeline.phase2_build_rdms,
        "3":  pipeline.phase3_inter_subject_rsa,
        "4":  pipeline.phase4_cross_modality_alignment,
        "5":  pipeline.phase5_structural_invariance,
        "6":  pipeline.phase6_visualize,
    }
    dispatch[args.phase]()


if __name__ == "__main__":
    main()
