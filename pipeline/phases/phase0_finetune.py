"""Phase 0 – FCNN Fine-Tuning."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from config.settings import Settings
from embeddings.fcnn_embedder import FCNNEmbedder

logger = logging.getLogger(__name__)


def _normalize_name(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', Path(str(s)).stem.lower())


def run(
    settings: Settings,
    fcnn_embedder: FCNNEmbedder,
    subjects: list,
    stimulus_image_dir: str | Path | None,
) -> None:
    """
    Fine-tune the FCNN hidden + classifier layers on clear stimuli.
    Fully idempotent: skips immediately if a checkpoint already exists.
    """
    logger.info("=" * 60)
    logger.info("PHASE 0 – FCNN Fine-Tuning")
    logger.info("=" * 60)

    if fcnn_embedder.is_finetuned():
        logger.info("Fine-tuned FCNN checkpoint already on disk – Phase 0 skipped.")
        return

    if stimulus_image_dir is None:
        logger.warning("No stimulus_image_dir provided – FCNN fine-tuning skipped.")
        return

    img_dir = Path(stimulus_image_dir)
    image_paths = sorted(img_dir.rglob("*.jpg")) + sorted(img_dir.rglob("*.png"))

    if not image_paths:
        logger.warning("No images found in %s – FCNN fine-tuning skipped.", img_dir)
        return

    # Build label map from subject behavioural data
    label_map: dict[str, int] = {}
    for subj in subjects:
        for state in ("conscious", "unconscious"):
            vis = getattr(subj, state, None)
            if vis is None:
                continue
            for name, lbl in zip(vis.stimulus_names, vis.labels):
                label_map[_normalize_name(name)] = int(lbl)

    train_paths, train_labels = [], []
    missing = 0
    for p in image_paths:
        norm = _normalize_name(p.stem)
        label = label_map.get(norm, 0)
        if norm not in label_map:
            missing += 1
        train_paths.append(p)
        train_labels.append(label)

    if missing:
        logger.warning("%d images had no matching subject label – assigned label=0.", missing)

    cfg = settings.fcnn
    fcnn_embedder.finetune(
        train_paths=train_paths,
        train_labels=train_labels,
        n_epochs=cfg.get("finetune_epochs", 30),
        lr=cfg.get("finetune_lr", 1e-4),
    )
    logger.info("Phase 0 complete.\n")
