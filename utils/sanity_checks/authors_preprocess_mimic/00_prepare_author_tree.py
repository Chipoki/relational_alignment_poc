#!/usr/bin/env python3
"""
00_prepare_author_tree.py
=========================
STEP 0 — Build the author-style working tree from the BIDS dataset.

What this script does
---------------------
1. Creates the directory skeleton under author_replication/MRI/<subject>/:
       anat/   func/session-<SS>/<subject>_unfeat_run-<R>/
2. Symlinks each BIDS bold .nii.gz into the author run directory.
3. Symlinks the T1w NIfTI into anat/ under two names so later scripts
   can find it regardless of which naming convention they use.
4. Writes a JSON manifest listing every (session, run, bold) triple.

Run ONCE before any other script.

Usage
-----
    python 00_prepare_author_tree.py [--root ROOT] [--subject sub-01]

After this script:
    author_replication/MRI/sub-01/func/session-02/sub-01_unfeat_run-1/
        sub-01_unfeat_run-1_bold.nii.gz  →  (symlink to BIDS NIfTI)
    author_replication/MRI/sub-01/anat/
        sub-01_t1.nii.gz                 →  (symlink to BIDS T1w)
        sub-01T1w.nii.gz                 →  (same symlink, alt name)
"""
from __future__ import annotations

from pathlib import Path
from pipeline_common import (
    make_parser,
    ds_root,
    subject_bids_dir,
    subject_deriv_dir,
    subject_anat_dir,
    subject_func_dir,
    replica_root,
    ensure_dir,
    symlink_force,
    write_manifest,
    first_session,
    _norm_ses,
    _norm_run,
    _BIDS_SES_RE,
    _BIDS_RUN_RE,
)


def _collect_bids_runs(root: Path, subject: str):
    """
    Walk BIDS ses-*/func/ and return sorted list of
    (ses_padded, run_bare, bold_path).
    Excludes any run whose filename contains 'wrong'.
    """
    bids_dir = subject_bids_dir(root, subject)
    runs = []
    for ses_dir in sorted(bids_dir.glob('ses-*')):
        sm = _BIDS_SES_RE.fullmatch(ses_dir.name)
        if not sm:
            continue
        ses = _norm_ses(sm.group(1))
        func_dir = ses_dir / 'func'
        if not func_dir.exists():
            continue
        for bold in sorted(func_dir.glob(f'{subject}_ses-*_*run-*_bold.nii.gz')):
            if 'wrong' in bold.name:
                continue
            rm = _BIDS_RUN_RE.search(bold.name)
            if not rm:
                continue
            run = _norm_run(rm.group(1))
            runs.append((ses, run, bold.resolve()))
    return sorted(runs, key=lambda x: (int(x[0]), int(x[1])))


def main():
    ap = make_parser(
        'Build the author-style working tree from ds003927 for one subject.'
    )
    args = ap.parse_args()

    root    = Path(args.root).resolve()
    subject = args.subject

    # ── discover BIDS runs ────────────────────────────────────────────────────
    runs = _collect_bids_runs(root, subject)
    if not runs:
        raise RuntimeError(
            f'No BIDS bold files found for {subject} under {subject_bids_dir(root, subject)}'
        )

    # ── create anat directory & symlinks ──────────────────────────────────────
    anat_out = ensure_dir(subject_anat_dir(root, subject))

    # T1w skull-on (BIDS)
    bids_anat = subject_bids_dir(root, subject) / 'anat'
    t1_candidates = sorted(bids_anat.glob(f'{subject}_*T1w.nii.gz'))
    if not t1_candidates:
        raise RuntimeError(f'No T1w .nii.gz found in {bids_anat}')
    t1_bids = t1_candidates[0].resolve()

    # Two symlink names so utils.py can find it by either convention
    symlink_force(t1_bids, anat_out / f'{subject}_t1.nii.gz')
    symlink_force(t1_bids, anat_out / f'{subject}T1w.nii.gz')

    # BET brain (derivatives) — symlink so find_anat_brain() resolves it
    deriv_anat = subject_deriv_dir(root, subject) / 'anat'
    brain_candidates = sorted(deriv_anat.glob('*brain*.nii*'))
    if brain_candidates:
        symlink_force(brain_candidates[0].resolve(),
                      anat_out / brain_candidates[0].name)
    else:
        print(
            f'WARNING: No BET brain found in {deriv_anat}. '
            'Step 01 (preprocessing) requires it for registration. '
            'Run FSL BET manually if needed.'
        )

    # ── create func tree & bold symlinks ─────────────────────────────────────
    func_out = ensure_dir(subject_func_dir(root, subject))

    for ses, run, bold in runs:
        run_dir = func_out / f'session-{ses}' / f'{subject}_unfeat_run-{run}'
        ensure_dir(run_dir)
        out_name = f'{subject}_unfeat_run-{run}_bold.nii.gz'
        dst = run_dir / out_name
        if dst.exists() or dst.is_symlink():
            if not args.overwrite:
                continue
        symlink_force(bold, dst)
        print(f'  linked session-{ses} run-{run} → {bold}')

    # ── write manifest ────────────────────────────────────────────────────────
    # Re-import list_runs now that the tree exists
    from pipeline_common import list_runs
    manifest = write_manifest(root, subject)

    print(f'\nPrepared author tree for {subject}')
    print(f'First functional session : {first_session(root, subject)}')
    print(f'Total runs               : {len(runs)}')
    print(f'Manifest written to      : {manifest}')
    print(f'\nNext step → python 01_preprocess_fmri.py --subject {subject}')


if __name__ == '__main__':
    main()
