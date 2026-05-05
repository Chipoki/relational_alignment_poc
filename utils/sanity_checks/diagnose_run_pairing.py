#!/usr/bin/env python3
"""
diagnose_run_pairing.py  –  Verify TSV↔BOLD run pairing for every subject.

Run from the project root:
    python utils/sanity_checks/diagnose_run_pairing.py

Uses the same two-stage matching logic as the fixed SubjectBuilder:
  1. Direct match:     TSV run-N  →  BOLD folder run-N  (if it exists)
  2. Positional match: unmatched TSVs paired 1:1 with unmatched BOLD folders
                       in sort order (handles sub-01 ses-04 runs 61/62 → 6/7)

✓  = run will be processed
✗  = run has no matching BOLD and will be skipped
~  = positional fallback (folder number ≠ TSV number, but BOLD exists)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings

SETTINGS_PATH = PROJECT_ROOT / "config" / "config.yaml"
settings = Settings(SETTINGS_PATH)


def _parse_tsv_run(name: str) -> int | None:
    m = re.search(r"_run-(\d+)_events\.tsv$", name)
    return int(m.group(1)) if m else None


def check_subject(subject_id: str) -> None:
    cfg = settings
    print(f"\n{'='*72}")
    print(f"  Subject: {subject_id}")
    print(f"{'='*72}")

    total_tsv = bold_found = bold_missing = positional = 0

    for ses in range(cfg.session_start, cfg.session_start + cfg.n_sessions):
        ses_func_dir = cfg.ds003927_session_func_dir(subject_id, ses)
        if not ses_func_dir.exists():
            continue

        tsv_entries: list[tuple[int, Path]] = []
        for f in ses_func_dir.iterdir():
            if not (f.name.endswith("_events.tsv") and "task-recog" in f.name):
                continue
            run_num = _parse_tsv_run(f.name)
            if run_num is not None:
                tsv_entries.append((run_num, f))
        tsv_entries.sort(key=lambda x: x[0])
        if not tsv_entries:
            continue

        # Discover valid BOLD folders
        ses_bold_dir = cfg.replic_func_dir(subject_id) / f"session-{ses:02d}"
        bold_entries: list[tuple[int, Path]] = []
        if ses_bold_dir.exists():
            folder_pat = re.compile(rf"^{re.escape(subject_id)}_unfeat_run-(\d+)$")
            for run_dir in ses_bold_dir.iterdir():
                fm = folder_pat.match(run_dir.name)
                if not fm:
                    continue
                bold_path = cfg.replic_filtered_bold(run_dir)
                if bold_path.exists():
                    bold_entries.append((int(fm.group(1)), bold_path))
        bold_entries.sort(key=lambda x: x[0])
        bold_by_num = {n: bp for n, bp in bold_entries}

        # Stage 1: direct match
        matched_tsv: set[int] = set()
        matched_bold: set[int] = set()
        pairs: list[tuple[int, int | None, Path | None, str]] = []  # (tsv_run, bold_run, bold_path, status)

        for tsv_run, _ in tsv_entries:
            if tsv_run in bold_by_num:
                pairs.append((tsv_run, tsv_run, bold_by_num[tsv_run], "✓"))
                matched_tsv.add(tsv_run)
                matched_bold.add(tsv_run)

        # Stage 2: positional fallback
        unmatched_tsvs  = [(n, p) for n, p in tsv_entries  if n not in matched_tsv]
        unmatched_bolds = [(n, p) for n, p in bold_entries if n not in matched_bold]
        for (tsv_run, _), (bold_num, bp) in zip(unmatched_tsvs, unmatched_bolds):
            pairs.append((tsv_run, bold_num, bp, "~"))
        for (tsv_run, _) in unmatched_tsvs[len(unmatched_bolds):]:
            pairs.append((tsv_run, None, None, "✗"))

        pairs.sort(key=lambda x: x[0])

        print(f"\n  Session {ses:02d}:  ({len(tsv_entries)} TSV files, "
              f"{len(bold_entries)} valid BOLD folders)")
        print(f"  {'TSV run':>8}  {'BOLD folder':>11}  {'Status':>6}  BOLD path")
        print(f"  {'-'*8}  {'-'*11}  {'-'*6}  ---------")

        for tsv_run, bold_num, bp, status in pairs:
            total_tsv += 1
            folder_str = str(bold_num) if bold_num is not None else "—"
            path_str   = str(bp) if bp is not None else "NO MATCH"
            print(f"  {tsv_run:>8}  {folder_str:>11}  {status:>6}  {path_str}")
            if status == "✓":
                bold_found += 1
            elif status == "~":
                bold_found  += 1
                positional  += 1
            else:
                bold_missing += 1

    print(f"\n  Summary: {bold_found}/{total_tsv} TSV runs matched to BOLD "
          f"({positional} via positional fallback, {bold_missing} missing/skipped)")


def main() -> None:
    if settings.data_source != "replication":
        print("This script only applies to data_source='replication'.  Exiting.")
        sys.exit(0)

    subject_ids = settings.subject_ids
    if not subject_ids:
        print("No subject IDs configured.  Check config.yaml.")
        sys.exit(1)

    print(f"Config:  {SETTINGS_PATH}")
    print(f"ds003927_root:    {settings.ds003927_root}")
    print(f"replication_root: {settings.replication_root}")
    print(f"session_start={settings.session_start}  n_sessions={settings.n_sessions}")
    print(f"Checking {len(subject_ids)} subject(s): {subject_ids}")

    for sid in subject_ids:
        check_subject(sid)

    print("\nDone.")


if __name__ == "__main__":
    main()