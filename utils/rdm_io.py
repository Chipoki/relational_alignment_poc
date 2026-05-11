"""utils/rdm_io.py – Per-subject RDM serialisation / deserialisation.

This module provides the I/O layer that lets the pipeline run in two stages:

  Stage A  (BOLD-intensive, per-subject)
      python run_pipeline.py --subjects sub-01 --dump-subject-rdms
      → loads BOLD, builds RDMs, dumps them to disk, then exits.
      Repeat for every subject individually.  Peak RAM = one subject's BOLD.

  Stage B  (RDM-only, all subjects)
      python run_pipeline.py --from-subject-rdms --phase 3
      → loads only the dumped RDMs, never touches BOLD, runs phases 3-6.

Format
------
Each subject is stored in a single ``.npz`` archive under::

    <checkpoints_dir>/subject_rdms/<subject_id>.npz

The archive contains flat arrays whose names encode the hierarchy::

    rdm__<state>__<roi>__matrix          float32 (n, n)
    rdm__<state>__<roi>__stimulus_names  U-string (n,)
    rdm__<state>__<roi>__labels          int32    (n,)
    rdm__<state>__<roi>__roi_or_layer    scalar str
    rdm__<state>__<roi>__subject_id      scalar str
    rdm__<state>__<roi>__state           scalar str

Aggregate and consensus arrays (``_agg_rdms``, ``_mean_rdms``) are stored
in a separate aggregate archive::

    <checkpoints_dir>/subject_rdms/_aggregates.npz

FCNN RDMs are stored per noise-state::

    <checkpoints_dir>/subject_rdms/_fcnn_<noise_state>.npz
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from analysis.rsa.rdm import RDM

logger = logging.getLogger(__name__)

# Separator used in npz key names; must not appear in state/roi/subject names.
_SEP = "__"
_PREFIX = "rdm"


# ── low-level helpers ─────────────────────────────────────────────────────────

def _encode_rdm(rdm: RDM, state: str, roi: str) -> dict:
    """Return a flat dict of arrays suitable for np.savez."""
    base = f"{_PREFIX}{_SEP}{state}{_SEP}{roi}"
    return {
        f"{base}{_SEP}matrix":         rdm.matrix.astype(np.float32),
        f"{base}{_SEP}stimulus_names": rdm.stimulus_names.astype(str),
        f"{base}{_SEP}labels":         rdm.labels.astype(np.int32),
        f"{base}{_SEP}roi_or_layer":   np.array(rdm.roi_or_layer),
        f"{base}{_SEP}subject_id":     np.array(rdm.subject_id),
        f"{base}{_SEP}state":          np.array(rdm.state),
    }


def _decode_rdms(npz) -> dict:
    """
    Reconstruct ``state → roi → RDM`` from a loaded npz archive.

    Returns a nested dict: {state: {roi: RDM}}.
    """
    # Collect all (state, roi) pairs present in the archive
    entries: dict[tuple, dict] = {}
    suffix_fields = {"matrix", "stimulus_names", "labels",
                     "roi_or_layer", "subject_id", "state"}

    for key in npz.files:
        parts = key.split(_SEP)
        if len(parts) != 4 or parts[0] != _PREFIX:
            continue
        _, state, roi, field = parts
        if field not in suffix_fields:
            continue
        entries.setdefault((state, roi), {})[field] = npz[key]

    result: dict[str, dict[str, RDM]] = {}
    for (state, roi), fields in entries.items():
        if not all(f in fields for f in ("matrix", "stimulus_names", "labels")):
            logger.warning("Incomplete RDM entry for state=%s roi=%s – skipping.", state, roi)
            continue
        rdm = RDM(
            matrix         = fields["matrix"].astype(np.float32),
            stimulus_names = fields["stimulus_names"].astype(str),
            labels         = fields["labels"].astype(np.int32),
            roi_or_layer   = str(fields.get("roi_or_layer", roi)),
            subject_id     = str(fields.get("subject_id", "unknown")),
            state          = str(fields.get("state", state)),
        )
        result.setdefault(state, {})[roi] = rdm

    return result


# ── public dump API ───────────────────────────────────────────────────────────

def subject_rdm_path(rdm_dump_dir: Path, subject_id: str) -> Path:
    """Return the canonical path for one subject's RDM archive."""
    return rdm_dump_dir / f"{subject_id}.npz"


def dump_subject_rdms(
    subject_id: str,
    state_rdms: Dict[str, Dict[str, RDM]],   # state → roi → RDM
    rdm_dump_dir: Path,
) -> Path:
    """
    Serialise one subject's RDMs (all states × all ROIs) into a single npz file.

    Parameters
    ----------
    subject_id  : e.g. "sub-01"
    state_rdms  : {state: {roi: RDM}}  — the per-subject slice of human_rdms
    rdm_dump_dir: directory to write into (created if absent)

    Returns the written path.
    """
    rdm_dump_dir.mkdir(parents=True, exist_ok=True)
    out_path = subject_rdm_path(rdm_dump_dir, subject_id)

    payload: dict = {}
    for state, roi_dict in state_rdms.items():
        for roi, rdm in roi_dict.items():
            payload.update(_encode_rdm(rdm, state, roi))

    np.savez_compressed(str(out_path), **payload)
    logger.info(
        "Dumped RDMs for %s → %s  (%d state×roi pairs)",
        subject_id, out_path,
        sum(len(rois) for rois in state_rdms.values()),
    )
    return out_path


def load_subject_rdms(
    subject_id: str,
    rdm_dump_dir: Path,
) -> Dict[str, Dict[str, RDM]]:
    """
    Load one subject's RDMs from the per-subject npz archive.

    Returns {state: {roi: RDM}}.
    Raises FileNotFoundError if the archive does not exist.
    """
    path = subject_rdm_path(rdm_dump_dir, subject_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Per-subject RDM archive not found for {subject_id}: {path}\n"
            "Run with --dump-subject-rdms first."
        )
    with np.load(str(path), allow_pickle=False) as npz:
        result = _decode_rdms(npz)
    logger.info(
        "Loaded RDMs for %s from %s  (%d states)",
        subject_id, path, len(result),
    )
    return result


def list_dumped_subjects(rdm_dump_dir: Path) -> list[str]:
    """Return sorted list of subject IDs whose RDM archives exist on disk."""
    if not rdm_dump_dir.exists():
        return []
    return sorted(
        p.stem for p in rdm_dump_dir.glob("*.npz")
        if not p.stem.startswith("_")
    )


# ── aggregate / FCNN helpers ──────────────────────────────────────────────────

def _agg_archive_path(rdm_dump_dir: Path) -> Path:
    return rdm_dump_dir / "_aggregates.npz"


def _fcnn_archive_path(rdm_dump_dir: Path, noise_state: str) -> Path:
    return rdm_dump_dir / f"_fcnn_{noise_state}.npz"


def dump_aggregate_rdms(
    agg_rdms: Dict[str, Dict[str, Dict[str, RDM]]],   # method → state → roi → RDM
    rdm_dump_dir: Path,
) -> None:
    """
    Serialise the aggregate RDMs (mean / median, all states × ROIs).

    The archive key layout adds the aggregation method as an extra segment::

        rdm__<method>__<state>__<roi>__<field>
    """
    rdm_dump_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {}
    for method, state_dict in agg_rdms.items():
        for state, roi_dict in state_dict.items():
            for roi, rdm in roi_dict.items():
                base = f"{_PREFIX}{_SEP}{method}{_SEP}{state}{_SEP}{roi}"
                payload[f"{base}{_SEP}matrix"]         = rdm.matrix.astype(np.float32)
                payload[f"{base}{_SEP}stimulus_names"] = rdm.stimulus_names.astype(str)
                payload[f"{base}{_SEP}labels"]         = rdm.labels.astype(np.int32)
                payload[f"{base}{_SEP}roi_or_layer"]   = np.array(rdm.roi_or_layer)
                payload[f"{base}{_SEP}subject_id"]     = np.array(rdm.subject_id)
                payload[f"{base}{_SEP}state"]          = np.array(rdm.state)

    out = _agg_archive_path(rdm_dump_dir)
    np.savez_compressed(str(out), **payload)
    logger.info("Dumped aggregate RDMs → %s", out)


def load_aggregate_rdms(
    rdm_dump_dir: Path,
) -> Dict[str, Dict[str, Dict[str, RDM]]]:
    """
    Load aggregate RDMs from disk.

    Returns {method: {state: {roi: RDM}}}.
    Returns an empty dict if the archive does not exist.
    """
    path = _agg_archive_path(rdm_dump_dir)
    if not path.exists():
        logger.debug("No aggregate RDM archive found at %s.", path)
        return {}

    result: dict = {}
    with np.load(str(path), allow_pickle=False) as npz:
        for key in npz.files:
            parts = key.split(_SEP)
            # expect: rdm__method__state__roi__field  (5 parts)
            if len(parts) != 5 or parts[0] != _PREFIX:
                continue
            _, method, state, roi, field = parts
            entry = result.setdefault(method, {}).setdefault(state, {}).setdefault(roi, {})
            entry[field] = npz[key]

    # Convert raw field dicts → RDM objects
    out: dict = {}
    for method, state_dict in result.items():
        for state, roi_dict in state_dict.items():
            for roi, fields in roi_dict.items():
                if "matrix" not in fields:
                    continue
                rdm = RDM(
                    matrix         = fields["matrix"].astype(np.float32),
                    stimulus_names = fields["stimulus_names"].astype(str),
                    labels         = fields["labels"].astype(np.int32),
                    roi_or_layer   = str(fields.get("roi_or_layer", roi)),
                    subject_id     = str(fields.get("subject_id", f"agg_{method}")),
                    state          = str(fields.get("state", state)),
                )
                out.setdefault(method, {}).setdefault(state, {})[roi] = rdm

    logger.info("Loaded aggregate RDMs from %s", path)
    return out


def dump_fcnn_rdms(fcnn_rdms: Dict[str, Dict[str, RDM]], rdm_dump_dir: Path) -> None:
    """Serialise FCNN RDMs (noise_state → roi → RDM) per noise state."""
    rdm_dump_dir.mkdir(parents=True, exist_ok=True)
    for noise_state, roi_dict in fcnn_rdms.items():
        payload: dict = {}
        for roi, rdm in roi_dict.items():
            payload.update(_encode_rdm(rdm, noise_state, roi))
        out = _fcnn_archive_path(rdm_dump_dir, noise_state)
        np.savez_compressed(str(out), **payload)
        logger.info("Dumped FCNN RDMs for noise_state=%s → %s", noise_state, out)


def load_fcnn_rdms(rdm_dump_dir: Path) -> Dict[str, Dict[str, RDM]]:
    """
    Load FCNN RDMs from per-noise-state archives.

    Returns {noise_state: {roi: RDM}}.  Missing archives are silently skipped.
    """
    result: dict = {}
    for noise_state in ("clear", "chance"):
        path = _fcnn_archive_path(rdm_dump_dir, noise_state)
        if not path.exists():
            continue
        with np.load(str(path), allow_pickle=False) as npz:
            decoded = _decode_rdms(npz)
        # _decode_rdms returns {state: {roi: RDM}}; here state == noise_state
        for state, roi_dict in decoded.items():
            result[state] = roi_dict
    if result:
        logger.info("Loaded FCNN RDMs for noise states: %s", sorted(result.keys()))
    return result


# ── intersected-stimulus RDM helpers ─────────────────────────────────────────
#
# These mirror the per-subject + aggregate helpers above but target a separate
# directory (<checkpoints_dir>/subject_rdms_intersected/) that stores RDMs
# already subsetted to the global cross-subject stimulus intersection.
# Phases 3-6 always read from this directory when running in per-subject mode.

def intersected_rdm_path(intersected_dir: Path, subject_id: str) -> Path:
    """Return the canonical path for one subject's intersected RDM archive."""
    return intersected_dir / f"{subject_id}.npz"


def dump_intersected_subject_rdms(
    subject_id: str,
    state_rdms: Dict[str, Dict[str, RDM]],
    intersected_dir: Path,
) -> Path:
    """
    Serialise one subject's stimulus-intersection-aligned RDMs.

    Identical format to dump_subject_rdms(); stored in a separate directory
    so the original (per-subject-only) archives are preserved.
    """
    intersected_dir.mkdir(parents=True, exist_ok=True)
    out_path = intersected_rdm_path(intersected_dir, subject_id)
    payload: dict = {}
    for state, roi_dict in state_rdms.items():
        for roi, rdm in roi_dict.items():
            payload.update(_encode_rdm(rdm, state, roi))
    np.savez_compressed(str(out_path), **payload)
    logger.info(
        "Dumped intersected RDMs for %s → %s  (%d state×roi pairs)",
        subject_id, out_path,
        sum(len(rois) for rois in state_rdms.values()),
    )
    return out_path


def load_intersected_subject_rdms(
    subject_id: str,
    intersected_dir: Path,
) -> Dict[str, Dict[str, RDM]]:
    """
    Load one subject's intersected RDMs.  Returns {state: {roi: RDM}}.
    Raises FileNotFoundError if the archive does not exist.
    """
    path = intersected_rdm_path(intersected_dir, subject_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Intersected RDM archive not found for {subject_id}: {path}"
        )
    with np.load(str(path), allow_pickle=False) as npz:
        result = _decode_rdms(npz)
    logger.info(
        "Loaded intersected RDMs for %s from %s  (%d states)",
        subject_id, path, len(result),
    )
    return result


def list_intersected_subjects(intersected_dir: Path) -> list[str]:
    """Return sorted list of subject IDs with intersected RDM archives on disk."""
    if not intersected_dir.exists():
        return []
    return sorted(
        p.stem for p in intersected_dir.glob("*.npz")
        if not p.stem.startswith("_")
    )


def dump_intersected_aggregate_rdms(
    agg_rdms: Dict[str, Dict[str, Dict[str, RDM]]],
    intersected_dir: Path,
) -> None:
    """Serialise intersected aggregate RDMs (mean/median) to the intersected dir."""
    dump_aggregate_rdms(agg_rdms, intersected_dir)


def load_intersected_aggregate_rdms(
    intersected_dir: Path,
) -> Dict[str, Dict[str, Dict[str, RDM]]]:
    """Load intersected aggregate RDMs from the intersected dir."""
    return load_aggregate_rdms(intersected_dir)


# ── SVM pattern dump (for per-subject SVM runs) ───────────────────────────────

def _svm_patterns_path(rdm_dump_dir: Path, subject_id: str) -> Path:
    return rdm_dump_dir / f"_svm_patterns_{subject_id}.npz"


def dump_svm_patterns(
    subject_id: str,
    roi_patterns_by_state: Dict[str, Dict[str, np.ndarray]],  # state → roi → (n,v)
    labels_by_state: Dict[str, np.ndarray],                    # state → (n,)
    stim_names_by_state: Dict[str, np.ndarray],                # state → (n,)
    rdm_dump_dir: Path,
) -> Path:
    """
    Dump the minimal per-subject data needed for SVM decoding.

    Stored as::
        pat__<state>__<roi>  float32 (n_trials, n_voxels)
        lbl__<state>         int32   (n_trials,)
        stm__<state>         str     (n_trials,)
    """
    rdm_dump_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {}
    for state, roi_dict in roi_patterns_by_state.items():
        for roi, arr in roi_dict.items():
            payload[f"pat{_SEP}{state}{_SEP}{roi}"] = arr.astype(np.float32)
        if state in labels_by_state:
            payload[f"lbl{_SEP}{state}"] = labels_by_state[state].astype(np.int32)
        if state in stim_names_by_state:
            payload[f"stm{_SEP}{state}"] = stim_names_by_state[state].astype(str)

    out = _svm_patterns_path(rdm_dump_dir, subject_id)
    np.savez_compressed(str(out), **payload)
    logger.info("Dumped SVM patterns for %s → %s", subject_id, out)
    return out


def load_svm_patterns(
    subject_id: str,
    rdm_dump_dir: Path,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load SVM patterns for one subject.

    Returns (roi_patterns_by_state, labels_by_state, stim_names_by_state).
    Raises FileNotFoundError if the archive is missing.
    """
    path = _svm_patterns_path(rdm_dump_dir, subject_id)
    if not path.exists():
        raise FileNotFoundError(
            f"SVM patterns archive not found for {subject_id}: {path}"
        )

    roi_patterns: dict[str, dict[str, np.ndarray]] = {}
    labels:       dict[str, np.ndarray] = {}
    stim_names:   dict[str, np.ndarray] = {}

    with np.load(str(path), allow_pickle=False) as npz:
        for key in npz.files:
            parts = key.split(_SEP)
            if len(parts) == 3 and parts[0] == "pat":
                _, state, roi = parts
                roi_patterns.setdefault(state, {})[roi] = npz[key].astype(np.float32)
            elif len(parts) == 2 and parts[0] == "lbl":
                labels[parts[1]] = npz[key].astype(np.int32)
            elif len(parts) == 2 and parts[0] == "stm":
                stim_names[parts[1]] = npz[key].astype(str)

    logger.info("Loaded SVM patterns for %s from %s", subject_id, path)
    return roi_patterns, labels, stim_names
