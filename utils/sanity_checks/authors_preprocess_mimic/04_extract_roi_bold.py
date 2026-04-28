#!/usr/bin/env python3
"""
04_extract_roi_bold.py
======================
STEP 4 — Extract ROI masks from FreeSurfer aparc+aseg and warp to BOLD space.

Mirrors the author's '3.extract ROI and convert to BOLD space.py'.

Pipeline per ROI label:
  1. mri_binarize  — threshold aparc+aseg.mgz at the integer label index
                     → <label>.nii.gz  (FreeSurfer RAS space)
  2. fslswapdim    — reorder axes ('x', 'z', '-y') to match FSL convention
                     → <label>_fsl.nii.gz
  3. mri_convert   — reslice to match orig/001.mgz voxel grid
                     → <label>_fsl.nii.gz  (overwrite)
  4. FLIRT         — linear warp from structural to BOLD space
     (via utils.create_simple_struc2BOLD nipype workflow)
     → <label>_fsl_BOLD.nii.gz

Post-processing:
  The three pars* labels (parsopercularis, parsorbitalis, parstriangularis)
  are added and binarised into a single "ventrolateralPFC" mask per hemisphere.
  The individual pars* masks are moved to an 'achieved/' subfolder.

Inputs (all must exist before running):
  • author_replication/MRI/<sub>/anat/<sub>/mri/aparc+aseg.mgz
  • author_replication/MRI/<sub>/anat/<sub>/mri/orig/001.mgz
  • author_replication/MRI/<sub>/func/session-<fs>/
      <sub>_unfeat_run-1/outputs/  (reg/ mat + func/)

Output directory:
  author_replication/MRI/<sub>/anat/ROI_BOLD/

Usage
-----
    python 04_extract_roi_bold.py [--subject sub-01] [--overwrite]

Note: ROI extraction is fast (< 1 min per label) so no parallelism needed here.
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
    author_output_dir,
    ensure_dir,
    load_utils,
    _norm_ses,
    _norm_run,
)


def main():
    ap   = make_parser('Extract ROIs from aparc+aseg and convert to BOLD space.')
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

        for f in [in_file, original]:
            if not f.exists():
                raise FileNotFoundError(
                    f'Required FreeSurfer file missing: {f}\n'
                    'Run step 03 (recon-all) first.'
                )

        # ── output directories ────────────────────────────────────────────────────
        roi_anat_dir = ensure_dir(anat_dir / 'ROIs_anat')
        roi_bold_dir = ensure_dir(anat_dir / 'ROI_BOLD')

        # preprocessed_functional_dir = the outputs/ folder of the reference run
        preprocessed_functional_dir = author_output_dir(root, subject, fs_ses, '1')
        if not preprocessed_functional_dir.exists():
            raise FileNotFoundError(
                f'Reference run outputs/ not found: {preprocessed_functional_dir}\n'
                'Run step 01 first.'
            )

        # ── Step A: binarize → swapdim → reslice for each ROI ────────────────────
        print(f'[04] Binarizing {len(ROI_LABELS)} ROI labels …')
        for label_name, idx in ROI_LABELS.items():
            binary_file = roi_anat_dir / f'{label_name}.nii.gz'
            fsl_file    = roi_anat_dir / f'{label_name}_fsl.nii.gz'

            if fsl_file.exists() and not args.overwrite:
                continue

            print(f'     {label_name}  (index {idx})')

            # mri_binarize
            if not binary_file.exists() or args.overwrite:
                binarizer = freesurfer.Binarize(
                    in_file     = str(in_file.resolve()),
                    match       = [idx],
                    binary_file = str(binary_file.resolve()),
                )
                binarizer.run()

            # fslswapdim
            fsl_swapdim = fsl.SwapDimensions(new_dims=('x', 'z', '-y'))
            fsl_swapdim.inputs.in_file  = str(binary_file.resolve())
            fsl_swapdim.inputs.out_file = str(fsl_file.resolve())
            fsl_swapdim.run()

            # mri_convert (reslice to orig space)
            mc = freesurfer.MRIConvert()
            mc.inputs.in_file      = str(fsl_file.resolve())
            mc.inputs.reslice_like = str(original.resolve())
            mc.inputs.out_file     = str(fsl_file.resolve())
            mc.run()

        # ── Step B: register each structural ROI to BOLD space ───────────────────
        roi_in_structural = sorted(glob(str(roi_anat_dir / '*_fsl.nii.gz')))
        print(f'[04] Warping {len(roi_in_structural)} masks to BOLD space …')

        for roi in roi_in_structural:
            roi      = str(Path(roi).resolve())
            roi_name = Path(roi).name
            expected = roi_bold_dir / roi_name.replace('.nii.gz', '_BOLD.nii.gz')

            if expected.exists() and not args.overwrite:
                continue

            wf = utils.create_simple_struc2BOLD(
                roi                         = roi,
                roi_name                    = roi_name,
                preprocessed_functional_dir = str(preprocessed_functional_dir.resolve()),
                output_dir                  = str(roi_bold_dir.resolve()),
            )
            wf.base_dir = str(roi_bold_dir.resolve())
            wf.write_graph(dotfilename=f'{Path(roi_name).stem}.dot')
            wf.run()

        # ── Step C: combine pars* into ventrolateralPFC (inferior frontal gyrus) ─
        pars_masks = sorted(glob(str(roi_bold_dir / '*pars*.nii.gz')))
        print(f'[04] Combining pars* masks into ventrolateralPFC …')

        for direction in ['lh', 'rh']:
            parts = sorted([
                str(Path(m).resolve())
                for m in pars_masks
                if f'ctx-{direction}-' in m
            ])
            if len(parts) < 3:
                print(f'     WARNING: expected 3 pars* masks for {direction}, '
                      f'found {len(parts)}. Skipping combine.')
                continue

            out_file = roi_bold_dir / f'ctx-{direction}-ventrolateralPFC_BOLD.nii.gz'

            # add part0 + part1
            merger = fsl.ImageMaths(
                in_file  = parts[0],
                in_file2 = parts[1],
                op_string = '-add',
            )
            merger.inputs.out_file = str(out_file.resolve())
            merger.run()

            # add part2
            merger2 = fsl.ImageMaths(
                in_file  = str(out_file.resolve()),
                in_file2 = parts[2],
                op_string = '-add',
            )
            merger2.inputs.out_file = str(out_file.resolve())
            merger2.run()

            # binarise
            binarize = fsl.ImageMaths(op_string='-bin')
            binarize.inputs.in_file  = str(out_file.resolve())
            binarize.inputs.out_file = str(out_file.resolve())
            binarize.run()

        # move individual pars* to achieved/
        achieved = ensure_dir(roi_bold_dir / 'achieved')
        for part in pars_masks:
            dst = achieved / Path(part).name
            if not dst.exists():
                copyfile(part, str(dst))
            os.remove(part)

        print(f'[04] Done. ROI BOLD masks in: {roi_bold_dir}')
        print(f'\nNext step → python 05_extract_roi_standard.py --subject {subject}')


if __name__ == '__main__':
    main()
