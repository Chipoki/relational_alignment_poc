"""embeddings/embedding_store.py – Save / load embeddings from disk."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingStore:
    """Thin persistence layer for numpy embedding arrays."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, array: np.ndarray) -> None:
        path = self._root / f"{key}.npy"
        np.save(path, array)

    def load(self, key: str) -> np.ndarray:
        path = self._root / f"{key}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Embedding not found: {path}")
        return np.load(path, allow_pickle=False)

    def exists(self, key: str) -> bool:
        return (self._root / f"{key}.npy").exists()

    def save_dict(self, prefix: str, data: dict[str, np.ndarray]) -> None:
        for key, arr in data.items():
            self.save(f"{prefix}__{key}", arr)

    def load_dict(self, prefix: str, keys: list[str]) -> dict[str, np.ndarray]:
        return {k: self.load(f"{prefix}__{k}") for k in keys}
