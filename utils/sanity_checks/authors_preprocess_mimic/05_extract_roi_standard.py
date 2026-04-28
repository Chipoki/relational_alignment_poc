#!/usr/bin/env python3
"""
05_extract_roi_standard.py
==========================
STEP 5 — Warp ROI masks from structural space to MNI standard space.

Mirrors the author's '3.2.extract ROI and convert to standard space.py'.

This step re-uses the _fsl.nii.gz masks already created in step 04 and
warps them to MNI space using the FNIRT warp field produced during the
registration step (01).

Pipeline per ROI mask:
  utils.create_simple_highres2standard nipype workflow
    FLIRT highres → standard, FNIRT warp, applywarp
    → <label>_fsl_standard.nii.gz

Inputs (must exist):
  • author_replication/MRI/<sub>/anat/ROIs_anat/*_fsl.nii.gz
    (created by step 04)
  • author_replication/MRI/<sub>/func/session-<fs>/
      <sub>_unfeat_run-1/outputs/reg/
        highres2standard.mat
        highres2standard_warp.nii.gz
    (created by step 01)

Output directory:
  author_replication/MRI/<sub>/anat/ROI_standard/

Usage
-----
    python 05_extract_roi_standard.py [--subject sub-01] [--overwrite]
"""
from __future__ import annotations

import os
from pathlib import Path
from glob import glob
from shutil import copyfile

from nipype.interfaces import freesurfer, fsl

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

from pipeline_common import (
    make_parser,
    ROI_LABELS,
    first_session,
    subject_anat_dir,
    author_reg_dir,
    ensure_dir,
    load_utils,
    _norm_ses,
    _norm_run,
)


def main():
    ap   = make_parser('Warp ROI masks from structural space to MNI standard space.')
    args = ap.parse_args()

    root    = Path(args.root).resolve()
    subjects = [args.subject]
    utils   = load_utils(root)

    if subjects == ["-1"]:
        subjects = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07"]

    for subject in subjects:
        anat_dir = subject_anat_dir(root, subject)
        fs_ses   = _norm_ses(first_session(root, subject))

        # ── FreeSurfer paths ──────────────────────────────────────────────────────
        os.environ['SUBJECTS_DIR'] = str(anat_dir.resolve())
        in_file  = anat_dir / subject / 'mri' / 'aparc+aseg.mgz'
        original = anat_dir / subject / 'mri' / 'orig' / '001.mgz'

        # ── registration dir (from step 01, reference run) ────────────────────────
        # The author's utils.create_simple_highres2standard uses the reg/ folder
        # from the first session's first run.
        reg_dir = author_reg_dir(root, subject, fs_ses, '1')
        if not reg_dir.exists():
            raise FileNotFoundError(
                f'Registration directory not found: {reg_dir}\n'
                'Run step 01 first.'
            )

        # ── output directories ────────────────────────────────────────────────────
        roi_anat_dir     = ensure_dir(anat_dir / 'ROIs_anat')
        roi_standard_dir = ensure_dir(anat_dir / 'ROI_standard')

        # ── Step A: (re-)binarize/swapdim/reslice if any _fsl masks are missing ──
        # In most cases step 04 already created all *_fsl.nii.gz files.
        for label_name, idx in ROI_LABELS.items():
            binary_file = roi_anat_dir / f'{label_name}.nii.gz'
            fsl_file    = roi_anat_dir / f'{label_name}_fsl.nii.gz'

            if fsl_file.exists() and not args.overwrite:
                continue

            print(f'[05] Re-creating structural mask: {label_name}')

            if not binary_file.exists() or args.overwrite:
                binarizer = freesurfer.Binarize(
                    in_file     = str(in_file.resolve()),
                    match       = [idx],
                    binary_file = str(binary_file.resolve()),
                )
                binarizer.run()

            fsl_swapdim = fsl.SwapDimensions(new_dims=('x', 'z', '-y'))
            fsl_swapdim.inputs.in_file  = str(binary_file.resolve())
            fsl_swapdim.inputs.out_file = str(fsl_file.resolve())
            fsl_swapdim.run()

            mc = freesurfer.MRIConvert()
            mc.inputs.in_file      = str(fsl_file.resolve())
            mc.inputs.reslice_like = str(original.resolve())
            mc.inputs.out_file     = str(fsl_file.resolve())
            mc.run()

        # ── Step B: warp each structural mask to standard space ───────────────────
        roi_in_structural = sorted(glob(str(roi_anat_dir / '*_fsl.nii.gz')))
        print(f'[05] Warping {len(roi_in_structural)} masks to MNI standard space …')

        # The author passes the reg/ directory as the "preprocessed_functional_dir"
        # argument to create_simple_highres2standard. Inside that workflow the code
        # looks for highres2standard.mat and highres2standard_warp.nii.gz.
        preprocessed_functional_dir = str(reg_dir.parent.resolve())

        for roi in roi_in_structural:
            roi      = str(Path(roi).resolve())
            roi_name = Path(roi).name
            expected = roi_standard_dir / roi_name.replace('.nii.gz', '_standard.nii.gz')

            if expected.exists() and not args.overwrite:
                continue

            wf = utils.create_simple_highres2standard(
                roi                         = roi,
                roi_name                    = roi_name,
                preprocessed_functional_dir = preprocessed_functional_dir,
                output_dir                  = str(roi_standard_dir.resolve()),
            )
            wf.base_dir = str(roi_standard_dir.resolve())
            wf.write_graph(dotfilename=f'{Path(roi_name).stem}.dot')
            wf.run()

        # ── Step C: combine pars* into ventrolateralPFC ───────────────────────────
        pars_masks = sorted(glob(str(roi_standard_dir / '*pars*.nii.gz')))
        print(f'[05] Combining pars* masks into ventrolateralPFC (standard) …')

        for direction in ['lh', 'rh']:
            parts = sorted([
                str(Path(m).resolve())
                for m in pars_masks
                if f'ctx-{direction}-' in m
            ])
            if len(parts) < 3:
                print(f'     WARNING: expected 3 pars* for {direction}, '
                      f'found {len(parts)}. Skipping.')
                continue

            out_file = roi_standard_dir / f'ctx-{direction}-ventrolateralPFC_standard.nii.gz'

            merger = fsl.ImageMaths(
                in_file  = parts[0],
                in_file2 = parts[1],
                op_string = '-add',
            )
            merger.inputs.out_file = str(out_file.resolve())
            merger.run()

            merger2 = fsl.ImageMaths(
                in_file  = str(out_file.resolve()),
                in_file2 = parts[2],
                op_string = '-add',
            )
            merger2.inputs.out_file = str(out_file.resolve())
            merger2.run()

            binarize = fsl.ImageMaths(op_string='-bin')
            binarize.inputs.in_file  = str(out_file.resolve())
            binarize.inputs.out_file = str(out_file.resolve())
            binarize.run()

        # move individual pars* to achieved/
        achieved = ensure_dir(roi_standard_dir / 'achieved')
        for part in pars_masks:
            dst = achieved / Path(part).name
            if not dst.exists():
                copyfile(part, str(dst))
            os.remove(part)

        print(f'[05] Done. ROI standard masks in: {roi_standard_dir}')
        print(f'\nNext step → python 06_highpass_filter.py --subject {subject}')


if __name__ == '__main__':
    main()
