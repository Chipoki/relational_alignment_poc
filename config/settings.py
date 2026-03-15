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

        # Runtime-discovered ROIs (populated by SubjectBuilder after data loading).
        # None means "not yet discovered" — phases must call active_roi_names.
        self._active_roi_names: list[str] | None = None

    # ── ROI helpers ──────────────────────────────────────────────────────────

    @property
    def roi_names(self) -> list[str]:
        """Static list from config.yaml. Includes 'wholebrain' always."""
        static = list(self._raw["rois"])
        if "wholebrain" not in static:
            static = ["wholebrain"] + static
        return static

    @property
    def active_roi_names(self) -> list[str]:
        """
        ROIs actually present in the loaded data.

        Populated by SubjectBuilder.register_active_rois() after subjects are
        loaded.  Falls back to roi_names if registration has not happened yet
        (e.g. unit-tests or single-phase runs).
        """
        if self._active_roi_names is not None:
            return self._active_roi_names
        return self.roi_names

    def register_active_rois(self, roi_names: list[str]) -> None:
        """
        Called by SubjectBuilder (or pipeline) once the actual ROI keys
        present in human_rdms are known.

        Rules
        -----
        * 'wholebrain' is always included (it is the whole-brain fallback).
        * ROIs are de-duplicated while preserving the config.yaml order for
          any ROIs that appear in both lists (config ROIs first, then any
          extras discovered in the data).
        """
        ordered: list[str] = []
        # Config order first, keeping only those that exist in data
        for r in self.roi_names:
            if r in roi_names:
                ordered.append(r)
        # Any extra ROIs discovered in data but not in config
        for r in roi_names:
            if r not in ordered:
                ordered.append(r)
        # wholebrain always present
        if "wholebrain" not in ordered:
            ordered.insert(0, "wholebrain")
        self._active_roi_names = ordered

    # ── Convenience helpers ──────────────────────────────────────────────────

    @property
    def visibility_states(self) -> list[str]:
        return self._raw["data"].get("visibility_states", ["conscious", "unconscious"])

    @property
    def nifti_prefix(self) -> str:
        return self._raw["data"].get("nifti_prefix", "wholebrain")

    @property
    def mask_filename(self) -> str:
        return self._raw["data"].get("mask_filename", "mask.nii.gz")

    @property
    def results_root(self) -> Path:
        output_cfg = self._raw["output"]
        if "results_dir" in output_cfg:
            return Path(output_cfg["results_dir"])
        parent = Path(output_cfg.get("parent_dir", "results"))
        return parent

    @property
    def output_parent_dir(self) -> Path:
        return Path(self._raw["output"].get("parent_dir", "results"))

    def _get_output_subdir(self, key: str, default_subdir: str) -> Path:
        output_cfg = self._raw["output"]
        if key in output_cfg:
            return Path(output_cfg[key])
        parent = self.output_parent_dir
        return parent / default_subdir

    @property
    def results_dir(self) -> Path:
        return self.results_root

    @property
    def rdm_dir(self) -> Path:
        return self._get_output_subdir("rdm_dir", "rdms")

    @property
    def embedding_dir(self) -> Path:
        return self._get_output_subdir("embedding_dir", "embeddings")

    @property
    def stats_dir(self) -> Path:
        return self._get_output_subdir("stats_dir", "stats")

    @property
    def visualization_dir(self) -> Path:
        viz_cfg  = self._raw["visualization"]
        viz_dir  = viz_cfg.get("output_dir", "figures")
        viz_path = Path(viz_dir)
        if viz_path.is_absolute() or str(viz_path).startswith("results"):
            return viz_path
        return self.output_parent_dir / viz_dir

    @property
    def log_dir(self) -> Path:
        return self.output_parent_dir / "logs"

    def ensure_output_dirs(self) -> None:
        for path in [
            self.results_root,
            self.rdm_dir,
            self.embedding_dir,
            self.stats_dir,
            self.visualization_dir,
            self.log_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    # ── Singleton factory ────────────────────────────────────────────────────

    @classmethod
    def get(cls, config_path: str | Path | None = None) -> "Settings":
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance

    def __repr__(self) -> str:
        return f"Settings(keys={list(self._raw.keys())})"
