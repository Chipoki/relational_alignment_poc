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
        """Root output directory (results_dir)."""
        output_cfg = self._raw["output"]
        if "results_dir" in output_cfg:
            return Path(output_cfg["results_dir"])
        parent = Path(output_cfg.get("parent_dir", "results"))
        return parent

    @property
    def output_parent_dir(self) -> Path:
        """Parent directory for all pipeline outputs."""
        return Path(self._raw["output"].get("parent_dir", "results"))

    def _get_output_subdir(self, key: str, default_subdir: str) -> Path:
        """
        Get output subdirectory path.
        If explicitly set in config, use that.
        Otherwise, derive from parent_dir + default_subdir.
        """
        output_cfg = self._raw["output"]
        if key in output_cfg:
            return Path(output_cfg[key])
        parent = self.output_parent_dir
        return parent / default_subdir

    @property
    def results_dir(self) -> Path:
        """Results root directory."""
        return self.results_root

    @property
    def rdm_dir(self) -> Path:
        """RDM storage directory."""
        return self._get_output_subdir("rdm_dir", "rdms")

    @property
    def embedding_dir(self) -> Path:
        """Embedding storage directory."""
        return self._get_output_subdir("embedding_dir", "embeddings")

    @property
    def stats_dir(self) -> Path:
        """Statistics output directory."""
        return self._get_output_subdir("stats_dir", "stats")

    @property
    def visualization_dir(self) -> Path:
        """Visualization output directory."""
        viz_cfg = self._raw["visualization"]
        viz_dir = viz_cfg.get("output_dir", "figures")
        # If it's an absolute or already contains parent path, use as-is
        viz_path = Path(viz_dir)
        if viz_path.is_absolute() or str(viz_path).startswith("results"):
            return viz_path
        # Otherwise, place it under the output parent directory
        return self.output_parent_dir / viz_dir

    @property
    def log_dir(self) -> Path:
        """Logging directory."""
        return self.output_parent_dir / "logs"

    def ensure_output_dirs(self) -> None:
        """Create all necessary output directories."""
        # Create parent and all subdirectories
        for path in [
            self.results_root,
            self.rdm_dir,
            self.embedding_dir,
            self.stats_dir,
            self.visualization_dir,
            self.log_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    # ── Singleton factory (optional convenience) ────────────────────────────

    @classmethod
    def get(cls, config_path: str | Path | None = None) -> "Settings":
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    def __repr__(self) -> str:
        return f"Settings(keys={list(self._raw.keys())})"
