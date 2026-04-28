#!/usr/bin/env python3
"""
02_ica_aroma.py
===============
STEP 2 — ICA-AROMA non-aggressive denoising.

Mirrors the author's '1.ICA_AROMA.py'.

For every run:
  Input : outputs/func/prefiltered_func.nii.gz
  Output: outputs/func/ICA_AROMA/denoised_func_data_nonaggr.nii.gz

All runs use the registration files produced by step 01 for the
reference (first session, run-1):
    reg/example_func2highres.mat
    reg/highres2standard_warp.nii.gz

Parallelism
-----------
All runs are independent at this stage → fully parallel.
Default --workers 4 (conservative for 16 GB RAM; ICA-AROMA is
memory-light but CPU-heavy).

Usage
-----
    python 02_ica_aroma.py [--subject sub-01] [--workers 4] [--overwrite]
"""
from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from nipype.interfaces.fsl import ICA_AROMA

from pipeline_common import (
    make_parser,
    list_runs,
    first_session,
    author_func_output_dir,
    author_reg_dir,
    ensure_dir,
    _norm_ses,
    _norm_run,
)


def _run_one(
    root_s:    str,
    subject:   str,
    ses:       str,
    run:       str,
    ref_ses:   str,
    ref_run:   str,
    overwrite: bool = False,
    dry_run:   bool = False,
) -> str:
    root = Path(root_s).resolve()
    ses  = _norm_ses(ses)
    run  = _norm_run(run)

    func_dir = author_func_output_dir(root, subject, ses, run)
    in_file  = func_dir / 'prefiltered_func.nii.gz'
    out_dir  = ensure_dir(func_dir / 'ICA_AROMA')
    denoised = out_dir / 'denoised_func_data_nonaggr.nii.gz'

    if denoised.exists() and not overwrite:
        return f'skip session-{ses} run-{run}: ICA_AROMA output exists'
    if dry_run:
        return f'dry-run session-{ses} run-{run}'

    # registration files come from reference run
    reg_dir = author_reg_dir(root, subject, ref_ses, ref_run)
    mat_file    = reg_dir / 'example_func2highres.mat'
    warp_file   = reg_dir / 'highres2standard_warp.nii.gz'
    movpar_file = func_dir / 'MC' / 'MCflirt.par'
    mask_file   = func_dir / 'mask.nii.gz'

    for f in [in_file, mat_file, warp_file, movpar_file, mask_file]:
        if not f.exists():
            raise FileNotFoundError(f'Required input missing: {f}')

    aroma = ICA_AROMA()
    aroma.inputs.in_file           = str(in_file.resolve())
    aroma.inputs.mat_file          = str(mat_file.resolve())
    aroma.inputs.fnirt_warp_file   = str(warp_file.resolve())
    aroma.inputs.motion_parameters = str(movpar_file.resolve())
    aroma.inputs.mask              = str(mask_file.resolve())
    aroma.inputs.denoise_type      = 'nonaggr'   # ← author setting
    aroma.inputs.out_dir           = str(out_dir.resolve())
    aroma.run()

    return f'done session-{ses} run-{run}'


def main():
    ap   = make_parser('ICA-AROMA non-aggressive denoising on all preprocessed runs.')
    args = ap.parse_args()

    root    = Path(args.root).resolve()
    subjects = [args.subject]

    if subjects == ["-1"]:
        subjects = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07"]

    for subject in subjects:
        runs    = list_runs(root, subject)

        fs  = _norm_ses(first_session(root, subject))
        ref_run_id = '1'

        max_workers = max(1, args.workers)
        print(f'[02] ICA-AROMA: {len(runs)} runs, {max_workers} workers')

        if max_workers == 1:
            for r in runs:
                print(_run_one(str(root), subject, r.ses, r.run,
                               fs, ref_run_id, args.overwrite, args.dry_run))
            return

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_one, str(root), subject, r.ses, r.run,
                          fs, ref_run_id, args.overwrite, args.dry_run): (r.ses, r.run)
                for r in runs
            }
            for fut in as_completed(futures):
                try:
                    print(fut.result())
                except Exception as exc:
                    ses, run = futures[fut]
                    print(f'ERROR session-{ses} run-{run}: {exc}')


if __name__ == '__main__':
    main()
