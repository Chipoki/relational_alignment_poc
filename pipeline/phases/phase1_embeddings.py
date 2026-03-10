"""Phase 1 – Dual-State Embedding Extraction."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from embeddings.fcnn_embedder import FCNNEmbedder
from embeddings.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


def run(
    fcnn_embedder: FCNNEmbedder,
    embedding_store: EmbeddingStore,
    stimulus_image_dir: str | Path | None,
) -> None:
    """Extract FCNN hidden-layer embeddings for clear and noisy images."""
    logger.info("=" * 60)
    logger.info("PHASE 1 – Dual-State Embedding Extraction")
    logger.info("=" * 60)

    if stimulus_image_dir is None:
        logger.warning("No stimulus image directory provided – FCNN embeddings skipped.")
        return

    img_dir = Path(stimulus_image_dir)
    image_paths = sorted(img_dir.rglob("*.jpg")) + sorted(img_dir.rglob("*.png"))

    if not image_paths:
        logger.warning("No images found in %s – FCNN phase skipped.", img_dir)
        return

    logger.info("Found %d stimulus images in %s", len(image_paths), img_dir)

    for noise_state in ("clear", "chance"):
        store_key = f"fcnn_{noise_state}"
        names_key = f"fcnn_{noise_state}_names"

        if embedding_store.exists(store_key) and embedding_store.exists(names_key):
            logger.info("Cached FCNN embeddings found for '%s' – skipping.", noise_state)
            continue

        logger.info("Extracting FCNN embeddings (noise_state=%s)...", noise_state)
        embeddings = fcnn_embedder.extract_embeddings(image_paths, noise_state=noise_state)
        image_names = np.array([p.stem for p in image_paths])

        embedding_store.save(store_key, embeddings)
        embedding_store.save(names_key, image_names)
        logger.info("Saved FCNN embeddings (%s) → shape %s", noise_state, embeddings.shape)

    logger.info("Phase 1 complete.\n")
