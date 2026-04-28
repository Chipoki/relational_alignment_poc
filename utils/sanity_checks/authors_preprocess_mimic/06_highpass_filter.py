#!/usr/bin/env python3
"""
06_highpass_filter.py
=====================
STEP 6 — High-pass temporal filtering of ICA-AROMA denoised data.

Mirrors the author's '4.highpass filter.py'.

Parameters (exactly as in the paper & utils.py):
  HP_freq = 60 seconds  (cutoff period)
  TR      = 0.85 s
  → sigma = HP_freq / 2 / TR = 35.29 volumes
  → FSL -bptf argument = 35.2941176471  (passed inside utils.py)

Per-run pipeline (via utils.create_highpass_filter_workflow):
  1. img2float       — cast to float32
  2. getthreshold    — 2nd / 98th percentile
  3. thresholding    — 10 % of 98th percentile mask (char)
  4. dilatemask      — dilate with -dilF
  5. apply_dilatemask— mask the float data
  6. cal_intensity_scale_factor — median inside mask
  7. meanscale       — intensity normalise (median → 10 000)
  8. meanfunc        — temporal mean of normalised data
  9. highpass_filering — FSL -bptf (zero-phase Gaussian HPF)
  10. addmean        — add back the temporal mean (remove DC offset)
  → output: ICAed_filtered/filtered.nii.gz

Nipype working dirs are placed under author_replication/work/hpf/ and
cleaned up after each run to avoid disk bloat.

Parallelism
-----------
All runs are independent → fully parallel.
Default --workers 4.

Usage
-----
    python 06_highpass_filter.py [--subject sub-01] [--workers 4] [--overwrite]
"""
from __future__ import annotations

from pathlib import Path
from glob import glob
from shutil import copyfile, rmtree
from concurrent.futures import ProcessPoolExecutor, as_completed

from pipeline_common import (
    make_parser,
    list_runs,
    author_func_output_dir,
    replica_root,
    load_utils,
    ensure_dir,
    _norm_ses,
    _norm_run,
)

HP_FREQ = 60      # seconds  — high-pass cutoff period
TR      = 0.85    # seconds  — repetition time


def _run_one(
    root_s:    str,
    subject:   str,
    ses:       str,
    run:       str,
    overwrite: bool = False,
    dry_run:   bool = False,
) -> str:
    root = Path(root_s).resolve()
    ses  = _norm_ses(ses)
    run  = _norm_run(run)

    utils    = load_utils(root)
    func_dir = author_func_output_dir(root, subject, ses, run)

    in_file    = func_dir / 'ICA_AROMA' / 'denoised_func_data_nonaggr.nii.gz'
    output_dir = ensure_dir(func_dir / 'ICAed_filtered')
    target     = output_dir / 'filtered.nii.gz'

    if target.exists() and not overwrite:
        return f'skip session-{ses} run-{run}: filtered.nii.gz already exists'
    if dry_run:
        return f'dry-run session-{ses} run-{run}'

    if not in_file.exists():
        raise FileNotFoundError(
            f'ICA-AROMA output missing: {in_file}\nRun step 02 first.'
        )

    # nipype working directory — kept outside ds003927
    wf_work = ensure_dir(
        replica_root(root) / 'work' / 'hpf' / subject / f's{ses}_r{run}'
    )

    hpf = utils.create_highpass_filter_workflow(
        HP_freq       = HP_FREQ,
        TR            = TR,
        workflow_name = f'highpassfiler_{ses}_{run}',
    )
    hpf.base_dir = str(wf_work)
    hpf.write_graph(dotfilename=f'session_{ses}_run_{run}.dot')

    hpf.inputs.inputspec.ICAed_file  = str(in_file.resolve())
    hpf.inputs.addmean.out_file      = str(target.resolve())

    hpf.run()

    # copy log files
    for log in glob(str(wf_work / '*' / '*' / '*' / '*' / '*' / 'report.rst')):
        log_name = Path(log).parts[-5]
        copyfile(log, str(output_dir / f'log_{log_name}.rst'))

    # clean up nipype working directory
    if wf_work.exists():
        rmtree(wf_work)

    return f'done session-{ses} run-{run}'


def main():
    ap   = make_parser('High-pass filter ICA-AROMA denoised outputs (60 s cutoff, TR=0.85 s).')
    args = ap.parse_args()

    root    = Path(args.root).resolve()
    subjects = [args.subject]
    if subjects == ["-1"]:
        subjects = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07"]

    for subject in subjects:

        runs    = list_runs(root, subject)

        max_workers = max(1, args.workers)
        print(
            f'[06] High-pass filter: {len(runs)} runs, {max_workers} workers\n'
            f'     HP_freq={HP_FREQ} s, TR={TR} s, '
            f'sigma={HP_FREQ/2/TR:.4f} volumes'
        )

        if max_workers == 1:
            for r in runs:
                print(_run_one(str(root), subject, r.ses, r.run,
                               args.overwrite, args.dry_run))
            return

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_one, str(root), subject, r.ses, r.run,
                          args.overwrite, args.dry_run): (r.ses, r.run)
                for r in runs
            }
            for fut in as_completed(futures):
                try:
                    print(fut.result())
                except Exception as exc:
                    ses, run = futures[fut]
                    print(f'ERROR session-{ses} run-{run}: {exc}')

        print(f'\n[06] All runs filtered. Final outputs in:')
        print(f'     author_replication/MRI/{subject}/func/session-*/*/outputs/func/ICAed_filtered/filtered.nii.gz')


if __name__ == '__main__':
    main()
