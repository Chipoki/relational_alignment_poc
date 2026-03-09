"""config/settings.py – Centralised configuration loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Settings:
    """
    Immutable-ish settings object built from a YAML config file.
    Attribute access mirrors the YAML hierarchy.
    """

    _instance: "Settings | None" = None

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        self._raw = raw
        # Expose top-level keys as attributes for ergonomic access
        for key, value in raw.items():
            setattr(self, key, value)

    # ── Convenience helpers ──────────────────────────────────────────────────

    @property
    def roi_names(self) -> list[str]:
        return self._raw["rois"]

    @property
    def visibility_states(self) -> list[str]:
        """Visibility states included in analysis (e.g. conscious, unconscious)."""
        return self._raw["data"].get("visibility_states", ["conscious", "unconscious"])

    @property
    def nifti_prefix(self) -> str:
        """Stem prefix for per-state NIfTI files, e.g. 'wholebrain'."""
        return self._raw["data"].get("nifti_prefix", "wholebrain")

    @property
    def mask_filename(self) -> str:
        """Filename of the single whole-brain mask, e.g. 'mask.nii.gz'."""
        return self._raw["data"].get("mask_filename", "mask.nii.gz")

    @property
    def results_root(self) -> Path:
        return Path(self._raw["output"]["results_dir"])

    def ensure_output_dirs(self) -> None:
        for key in ("results_dir", "rdm_dir", "embedding_dir", "stats_dir"):
            Path(self._raw["output"][key]).mkdir(parents=True, exist_ok=True)
        Path(self._raw["visualization"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Singleton factory (optional convenience) ────────────────────────────

    @classmethod
    def get(cls, config_path: str | Path | None = None) -> "Settings":
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    def __repr__(self) -> str:
        return f"Settings(keys={list(self._raw.keys())})"
