#!/usr/bin/env python3
"""
03_freesurfer_reconall.py
=========================
STEP 3 — FreeSurfer recon-all surface reconstruction.

Mirrors the author's '2.freesurfer reconall.py'.

Input : author_replication/MRI/<subject>/anat/<subject>_t1.nii.gz
Output: author_replication/MRI/<subject>/anat/<subject>/
           mri/aparc+aseg.mgz    ← needed by step 04/05
           mri/orig/001.mgz      ← needed by step 04/05
           (full FS surface outputs)

NOTE ON DERIVATIVES
-------------------
The ds003927 dataset ships a partial FreeSurfer workspace at:
    derivatives/<subject>/fs_workspace/<subject>/
but it only contains labels, surf, scripts etc — NOT mri/aparc+aseg.mgz.
Therefore recon-all must be run from scratch using the raw T1w.

Runtime: recon-all takes ~8–12 hours per subject on a modern machine.
Use --openmp N to control the number of threads recon-all uses internally
(default 8, safe for 16 GB RAM with 20 cores).

Usage
-----
    python 03_freesurfer_reconall.py [--subject sub-01] [--openmp 8] [--overwrite]

FreeSurfer must be installed and FREESURFER_HOME + SUBJECTS_DIR must be
handled by this script (SUBJECTS_DIR is set to the author anat directory).
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path

from nipype.interfaces.freesurfer import ReconAll

from pipeline_common import (
    DEFAULT_ROOT,
    subject_anat_dir,
    find_anat_head,
    ensure_dir,
)


def main():
    ap = argparse.ArgumentParser(
        description='Run FreeSurfer recon-all (author-style).'
    )
    ap.add_argument('--root',    default=str(DEFAULT_ROOT))
    ap.add_argument('--subject', default='sub-01')
    ap.add_argument('--openmp',  type=int, default=8,
                    help='Threads for recon-all (default 8; safe for 16 GB)')
    ap.add_argument('--overwrite', action='store_true',
                    help='Re-run even if subject directory already exists')
    args = ap.parse_args()

    root    = Path(args.root).resolve()
    subjects = [args.subject]

    if subjects == ["-1"]:
        subjects = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07"]

    for subject in subjects:
        anat_dir = ensure_dir(subject_anat_dir(root, subject))
        subj_dir = anat_dir / subject   # FS SUBJECTS_DIR/<subject>

        if subj_dir.exists() and (subj_dir / 'mri' / 'aparc+aseg.mgz').exists():
            if not args.overwrite:
                print(f'skip: {subj_dir}/mri/aparc+aseg.mgz already exists')
                return

        # Find raw T1w (skull-on)
        t1 = find_anat_head(root, subject)
        print(f'[03] Running recon-all on {t1}')
        print(f'     SUBJECTS_DIR = {anat_dir}')
        print(f'     Expected runtime: 8–12 hours with --openmp {args.openmp}')

        # Set SUBJECTS_DIR so FreeSurfer writes into our anat directory
        os.environ['SUBJECTS_DIR'] = str(anat_dir.resolve())

        reconall = ReconAll()
        reconall.inputs.subject_id   = subject
        reconall.inputs.directive    = 'all'
        reconall.inputs.subjects_dir = str(anat_dir.resolve())
        reconall.inputs.T1_files     = str(t1.resolve())
        reconall.inputs.openmp       = args.openmp
        reconall.run()

        aparc = subj_dir / 'mri' / 'aparc+aseg.mgz'
        if aparc.exists():
            print(f'[03] Done. aparc+aseg.mgz: {aparc}')
        else:
            print(f'WARNING: recon-all finished but aparc+aseg.mgz not found at {aparc}')

        print(f'\nNext step → python 04_extract_roi_bold.py --subject {subject}')


if __name__ == '__main__':
    main()
