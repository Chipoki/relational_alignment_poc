"""
config/settings.py – centralised path & config resolution.

Supports two data_source modes:
  "derivatives"  – pre-stacked wholebrain_*.nii.gz + wholebrain_*.csv under
                   ds003927/derivatives/<sub-XX>/
  "replication"  – per-run ICAed_filtered/filtered.nii.gz under
                   author_replication/MRI/<sub-XX>/func/session-<SS>/
                   with bilateral ROI masks in anat/ROI_BOLD/ and
                   raw event .tsv files read from ds003927/<sub>/ses-XX/func/
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

import yaml


# ── locate config.yaml ────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_raw_config(config_path: Path = _CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── top-level settings object ────────────────────────────────────────────────
class Settings:
    """
    Parsed, validated configuration.

    Key attributes
    --------------
    data_source : str
        "derivatives" or "replication"
    derivatives_root : Path
    replication_root : Path
    ds003927_root : Path
        Raw BIDS root used for event .tsv files in replication mode and as the
        parent of derivatives/ in derivatives mode.
    subject_ids : list[str]
        Populated at construction; auto-discovered if config list is empty.
    active_roi_names : list[str]
        Populated after subjects are built via register_active_rois().

    Notes
    -----
    ``self.data`` exposes the raw ``data:`` section of config.yaml as a dict
    so that legacy callers (BehavioralLoader, FMRILoader, SubjectBuilder, and
    the pipeline) can still do ``settings.data["key"]`` without change.
    """

    def __init__(self, config_path: Path = _CONFIG_PATH):
        raw = load_raw_config(config_path)

        # ── data source ──────────────────────────────────────────────────────
        self.data_source: str = raw.get("data_source", "derivatives").strip().lower()
        assert self.data_source in ("derivatives", "replication"), (
            f"data_source must be 'derivatives' or 'replication', got '{self.data_source}'"
        )

        # ── raw data section (kept as dict for legacy callers) ────────────────
        data = raw["data"]
        self.data: dict = data          # ← IMPORTANT: keeps settings.data["key"] working

        # ── roots ────────────────────────────────────────────────────────────
        self.derivatives_root = Path(data["derivatives_root"])
        self.replication_root = Path(data["replication_root"])
        self.ds003927_root    = Path(data["ds003927_root"])

        # Convenience: active root for subject discovery
        self._active_root = (
            self.derivatives_root if self.data_source == "derivatives"
            else self.replication_root / "MRI"
        )

        # ── subject ids ──────────────────────────────────────────────────────
        configured_ids: list = data.get("subject_ids") or []
        if configured_ids:
            self.subject_ids: List[str] = [str(s) for s in configured_ids]
        else:
            self.subject_ids = self._discover_subjects()

        # ── scalar data params ───────────────────────────────────────────────
        self.n_subjects:        int   = data.get("n_subjects", len(self.subject_ids))
        self.n_sessions:        int   = data.get("n_sessions", 7)
        self.session_start:     int   = data.get("session_start", 1)  # first session number (sessions are 2-7, not 1-6)
        self.n_runs_per_session: int  = data.get("n_runs_per_session", 9)
        self.tr:                float = float(data["tr"])
        self.hrf_delay_seconds: float = float(data.get("hrf_delay_seconds", 5.0))
        self.n_hrf_volumes:     int   = int(data.get("n_hrf_volumes", 1))

        # Replication-mode pre-processing flags
        self.replication_detrend:      bool = bool(data.get("replication_detrend", True))
        self.replication_zscore_per_run: bool = bool(data.get("replication_zscore_per_run", True))

        col = data.get("stimuli_csv_col", {})
        self.col_label:      str = col.get("label",      "labels")
        self.col_category:   str = col.get("category",   "targets")
        self.col_visibility: str = col.get("visibility", "visibility")
        self.col_volume:     str = col.get("volume",     "volume_interest")
        self.col_onset:      str = col.get("onset",      "onset")

        self.visibility_states: List[str] = data.get(
            "visibility_states", ["conscious", "unconscious"]
        )
        self.nifti_prefix:  str = data.get("nifti_prefix",  "wholebrain")
        self.mask_filename: str = data.get("mask_filename", "mask.nii.gz")

        # ── ROIs ─────────────────────────────────────────────────────────────
        self.rois: List[str] = raw.get("rois", [])
        # _active_roi_names is populated later via register_active_rois()
        self._active_roi_names: List[str] = list(self.rois)

        # ── FCNN / RDM / RSA / GW ────────────────────────────────────────────
        self.fcnn               = raw.get("fcnn", {})
        self.rdm                = raw.get("rdm",  {})
        self.rsa                = raw.get("rsa",  {})
        self.gromov_wasserstein = raw.get("gromov_wasserstein", {})

        # ── output ───────────────────────────────────────────────────────────
        out = raw.get("output", {})
        self.output_parent_dir = Path(out.get(
            "parent_dir",
            "/home/tomerd/Documents/projects/MSc/lab/thesis_practice/"
            "relational_alignment/poc_output"
        ))

        # ── visualisation ────────────────────────────────────────────────────
        viz = raw.get("visualization", {})
        self.vis_output_dir        = os.path.join(self.output_parent_dir, viz.get("output_dir", "figures"))
        self.vis_dpi               = int(viz.get("dpi", 150))
        self.vis_mds_n_components  = int(viz.get("meta_mds_n_components", 2))

    # ── ROI registration (called by SubjectBuilder after all subjects built) ─

    @property
    def roi_names(self) -> List[str]:
        """Alias for self.rois – used by ROIExtractor."""
        return self.rois

    @property
    def active_roi_names(self) -> List[str]:
        """ROI names actually present in the loaded data (set by register_active_rois)."""
        return self._active_roi_names

    def register_active_rois(self, roi_names: List[str]) -> None:
        """
        Called by SubjectBuilder.register_rois_with_settings() after all
        subjects are loaded.  Overwrites the config-level list with the ROIs
        that actually exist in the data.
        """
        self._active_roi_names = list(roi_names)

    # ── subject discovery ────────────────────────────────────────────────────

    def _discover_subjects(self) -> List[str]:
        """Return sorted list of sub-XX folders found under the active root."""
        root = self._active_root
        if not root.exists():
            raise FileNotFoundError(
                f"Active data root does not exist: {root}\n"
                f"(data_source='{self.data_source}')"
            )
        pattern = re.compile(r"^sub-\d+$")
        subjects = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and pattern.match(d.name)
        )
        if not subjects:
            raise RuntimeError(f"No sub-XX directories found under {root}")
        return subjects

    # ── path helpers: derivatives mode ───────────────────────────────────────

    def deriv_subject_dir(self, subject: str) -> Path:
        return self.derivatives_root / subject

    def deriv_nifti(self, subject: str, state: str) -> Path:
        return self.deriv_subject_dir(subject) / f"{self.nifti_prefix}_{state}.nii.gz"

    def deriv_csv(self, subject: str, state: str) -> Path:
        return self.deriv_subject_dir(subject) / f"{self.nifti_prefix}_{state}.csv"

    def deriv_mask(self, subject: str) -> Path:
        return self.deriv_subject_dir(subject) / self.mask_filename

    # ── path helpers: replication mode ───────────────────────────────────────

    def replic_subject_dir(self, subject: str) -> Path:
        return self.replication_root / "MRI" / subject

    def replic_func_dir(self, subject: str) -> Path:
        return self.replic_subject_dir(subject) / "func"

    def replic_anat_dir(self, subject: str) -> Path:
        return self.replic_subject_dir(subject) / "anat"

    def replic_run_dir(self, subject: str, session: int, run: int) -> Path:
        """
        Returns the run-level directory inside author_replication:
            .../func/session-<SS>/<subject>_unfeat_run-<R>/
        """
        return (
            self.replic_func_dir(subject)
            / f"session-{session:02d}"
            / f"{subject}_unfeat_run-{run}"
        )

    def replic_run_dirs(self, subject: str) -> List[Path]:
        """
        Return sorted list of all per-run directories, e.g.
        .../func/session-01/sub-01_unfeat_run-1/
        """
        func_dir = self.replic_func_dir(subject)
        run_dirs: List[Path] = []
        for ses_dir in sorted(func_dir.glob("session-*")):
            for run_dir in sorted(ses_dir.glob(f"{subject}_unfeat_run-*")):
                run_dirs.append(run_dir)
        return run_dirs

    def replic_filtered_bold(self, run_dir: Path) -> Path:
        return run_dir / "outputs" / "func" / "ICAed_filtered" / "filtered.nii.gz"

    def replic_run_mask(self, run_dir: Path) -> Path:
        """Per-run func mask (may not always exist; caller should check)."""
        return run_dir / "outputs" / "func" / "mask.nii.gz"

    def replic_mc_par(self, run_dir: Path) -> Path:
        return run_dir / "outputs" / "func" / "MC" / "MCflirt.par"

    def replic_roi_bold_mask(self, subject: str, roi_name: str,
                              hemisphere: str = "lh") -> Path:
        """
        e.g. .../anat/ROI_BOLD/ctx-lh-fusiform_BOLD.nii.gz
        hemisphere: "lh" | "rh"
        """
        fname = f"ctx-{hemisphere}-{roi_name}_BOLD.nii.gz"
        return self.replic_anat_dir(subject) / "ROI_BOLD" / fname

    def replic_roi_bold_dir(self, subject: str) -> Path:
        return self.replic_anat_dir(subject) / "ROI_BOLD"

    # ── path helpers: ds003927 raw BIDS (event TSVs) ─────────────────────────

    def ds003927_subject_dir(self, subject: str) -> Path:
        return self.ds003927_root / subject

    def ds003927_session_func_dir(self, subject: str, session: int) -> Path:
        return self.ds003927_subject_dir(subject) / f"ses-{session:02d}" / "func"

    def ds003927_events_tsv(self, subject: str, session: int, run: int) -> Path:
        """
        BIDS event file:
            <ds003927_root>/<subject>/ses-<SS>/func/<subject>_ses-<SS>_task-recog_run-<R>_events.tsv
        """
        fname = f"{subject}_ses-{session:02d}_task-recog_run-{run}_events.tsv"
        return self.ds003927_session_func_dir(subject, session) / fname

    # Event CSVs (derivatives mode, pre-processed, already separated by state)
    def event_csv(self, subject: str, state: str) -> Path:
        return self.deriv_csv(subject, state)

    # ── output subdirs ────────────────────────────────────────────────────────

    @property
    def results_dir(self) -> Path:
        return self.output_parent_dir / "results"

    @property
    def rdm_dir(self) -> Path:
        return self.output_parent_dir / "results" / "rdms"

    @property
    def embedding_dir(self) -> Path:
        return self.output_parent_dir / "results" / "embeddings"

    @property
    def stats_dir(self) -> Path:
        return self.output_parent_dir / "results" / "stats"

    @property
    def figures_dir(self) -> Path:
        return self.output_parent_dir / self.vis_output_dir

    @property
    def log_dir(self) -> Path:
        return self.output_parent_dir / "logs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_parent_dir / "checkpoints"

    @property
    def subject_rdm_dir(self) -> Path:
        """
        Directory that holds the per-subject RDM archives written by
        ``--dump-subject-rdms`` and read by ``--from-subject-rdms``.

        Default: ``<checkpoints_dir>/subject_rdms/``
        Override via ``output.subject_rdm_dir`` in config.yaml.
        """
        raw = load_raw_config()
        custom = raw.get("output", {}).get("subject_rdm_dir", None)
        if custom:
            return Path(custom)
        return self.checkpoints_dir / "subject_rdms"

    def ensure_output_dirs(self) -> None:
        for d in [
            self.results_dir,
            self.rdm_dir,
            self.embedding_dir,
            self.stats_dir,
            self.figures_dir,
            self.log_dir,
            self.checkpoints_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"Settings(data_source={self.data_source!r}, "
            f"n_subjects={len(self.subject_ids)}, "
            f"subjects={self.subject_ids})"
        )


# ── Module-level singleton (lazy so imports don't fail before config exists) ─
_settings: Optional[Settings] = None


def get_settings(config_path: Path = _CONFIG_PATH) -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings(config_path)
    return _settings


def reload_settings(config_path: Path = _CONFIG_PATH) -> Settings:
    global _settings
    _settings = Settings(config_path)
    return _settings
