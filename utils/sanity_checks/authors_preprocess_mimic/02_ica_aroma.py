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

Special batch mode:
    --subject -1 → run on sub-02 … sub-07
"""
from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

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

# Max allowed deviation from session median file size for
# denoised_func_data_nonaggr.nii.gz before treating an existing file as broken.
DENOISED_SIZE_TOL = 0.15  # 15%


def compute_denoised_medians(root: Path, subject: str, runs) -> dict[str, int]:
    """
    Compute per-session median file size (bytes) of existing
    denoised_func_data_nonaggr.nii.gz for this subject.
    Used to detect obviously truncated outputs.
    """
    sizes_by_ses: dict[str, list[int]] = defaultdict(list)
    for r in runs:
        ses = _norm_ses(r.ses)
        func_dir = author_func_output_dir(root, subject, ses, r.run)
        denoised = func_dir / 'ICA_AROMA' / 'denoised_func_data_nonaggr.nii.gz'
        if denoised.exists():
            try:
                sizes_by_ses[ses].append(denoised.stat().st_size)
            except OSError:
                continue

    medians: dict[str, int] = {}
    for ses, sizes in sizes_by_ses.items():
        if len(sizes) < 2:
            continue
        sizes.sort()
        medians[ses] = sizes[len(sizes) // 2]
    return medians


def _run_one(
    root_s: str,
    subject: str,
    ses: str,
    run: str,
    ref_ses: str,
    ref_run: str,
    overwrite: bool = False,
    dry_run: bool = False,
    size_medians: dict[str, int] | None = None,
) -> str:
    root = Path(root_s).resolve()
    ses = _norm_ses(ses)
    run = _norm_run(run)

    if size_medians is None:
        size_medians = {}

    func_dir = author_func_output_dir(root, subject, ses, run)
    in_file = func_dir / 'prefiltered_func.nii.gz'
    out_dir = ensure_dir(func_dir / 'ICA_AROMA')
    denoised = out_dir / 'denoised_func_data_nonaggr.nii.gz'

    # Robust size check: if an existing denoised file is >15% smaller than the
    # session median, treat it as broken and recompute.
    denoised_ok = denoised.exists()
    median_size = size_medians.get(ses)
    if denoised_ok and median_size is not None:
        try:
            s = denoised.stat().st_size
            if s < (1.0 - DENOISED_SIZE_TOL) * float(median_size):
                print(
                    f"[02] session-{ses} run-{run}: denoised_func_data_nonaggr.nii.gz "
                    f"({s} bytes) is >{int(DENOISED_SIZE_TOL*100)}% smaller than "
                    f"session median ({median_size} bytes) – "
                    f"treating as broken and recomputing."
                )
                denoised_ok = False
        except OSError:
            denoised_ok = False

    # If we already have a reasonable output and not overwriting, skip
    if denoised_ok and not overwrite:
        return f'skip session-{ses} run-{run}: ICA_AROMA output exists'
    if dry_run:
        return f'dry-run session-{ses} run-{run}'

    # We are (re)running ICA-AROMA for this run. Clean out any stale contents
    # in the ICA_AROMA directory (especially partial melodic.ica without
    # melodic_IC.nii.gz) so MELODIC starts from a clean state.
    if out_dir.exists():
        for child in list(out_dir.iterdir()):
            if child.is_dir():
                rmtree(child)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
    ensure_dir(out_dir)

    # Registration files come from reference run
    reg_dir = author_reg_dir(root, subject, ref_ses, ref_run)
    mat_file = reg_dir / 'example_func2highres.mat'
    warp_file = reg_dir / 'highres2standard_warp.nii.gz'
    movpar_file = func_dir / 'MC' / 'MCflirt.par'
    mask_file = func_dir / 'mask.nii.gz'

    for f in [in_file, mat_file, warp_file, movpar_file, mask_file]:
        if not f.exists():
            raise FileNotFoundError(f'Required input missing: {f}')

    aroma = ICA_AROMA()
    aroma.inputs.in_file = str(in_file.resolve())
    aroma.inputs.mat_file = str(mat_file.resolve())
    aroma.inputs.fnirt_warp_file = str(warp_file.resolve())
    aroma.inputs.motion_parameters = str(movpar_file.resolve())
    aroma.inputs.mask = str(mask_file.resolve())
    aroma.inputs.denoise_type = 'nonaggr'  # ← author setting
    aroma.inputs.out_dir = str(out_dir.resolve())

    # Run ICA-AROMA; if MELODIC/ICA-AROMA fails (e.g., no melodic_IC), catch
    # and report the error so the pipeline can continue.
    try:
        aroma.run()
    except Exception as exc:
        return (
            f'ERROR session-{ses} run-{run}: ICA-AROMA failed: {exc}'
        )

    # Sanity check: ensure the expected denoised file was produced
    if not denoised.exists():
        return (
            f'ERROR session-{ses} run-{run}: ICA-AROMA finished but '
            f'{denoised} is missing (check ICA_AROMA/melodic.ica/log.txt)'
        )

    return f'done session-{ses} run-{run}'


def main():
    ap = make_parser('ICA-AROMA non-aggressive denoising on all preprocessed runs.')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    subjects = [args.subject]

    # Special batch mode: -1 → run on sub-02…sub-07
    if subjects == ["-1"]:
        subjects = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07"]

    for subject in subjects:
        runs = list_runs(root, subject)
        fs = _norm_ses(first_session(root, subject))
        ref_run_id = '1'

        # Precompute per-session median denoised sizes for this subject
        size_medians = compute_denoised_medians(root, subject, runs)

        max_workers = max(1, args.workers)
        print(f'\n[02] ===== Subject {subject} =====')
        print(f'[02] ICA-AROMA: {len(runs)} runs, {max_workers} workers')

        if max_workers == 1:
            # Sequential processing, but robust to per-run failures
            for r in runs:
                try:
                    msg = _run_one(
                        str(root), subject, r.ses, r.run,
                        fs, ref_run_id, args.overwrite, args.dry_run, size_medians
                    )
                    print(msg)
                except Exception as exc:
                    print(f'ERROR session-{r.ses} run-{r.run}: {exc}')
            continue

        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _run_one,
                    str(root), subject, r.ses, r.run,
                    fs, ref_run_id, args.overwrite, args.dry_run, size_medians
                ): (r.ses, r.run)
                for r in runs
            }

            for fut in as_completed(futures):
                ses, run = futures[fut]
                try:
                    print(fut.result())
                except Exception as exc:
                    print(f'ERROR session-{ses} run-{run}: {exc}')


if __name__ == '__main__':
    main()