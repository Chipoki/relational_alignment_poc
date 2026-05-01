#!/usr/bin/env python3
"""
01_preprocess_fmri.py
=====================
STEP 1 — Nipype/FSL FEAT-like functional preprocessing.

Exactly mirrors the author's '0.preprocess fmri.py' logic:

Per-run steps (via utils.create_fsl_FEAT_workflow_func):
1. img2float — cast to float32
2. remove_volumes— discard first 8 TRs (n_vol_remove=8, total_vol=495)
3. extractref — extract middle volume as example_func (first run only)
4. MCFlirt — motion correction (spline interpolation)
   • first run: ref = extracted middle volume
   • all other runs: ref = example_func from ref run
5. meanfunc — temporal mean of motion-corrected data
6. BET — brain mask from mean func (frac=0.3)
7. SUSAN smooth — 3 mm FWHM spatial smoothing
8. meanscale — intensity normalisation (median → 10 000)
   → outputs: prefiltered_func.nii.gz, mask.nii.gz, mean_func.nii.gz

Registration step (reference run only):
FLIRT func→struct, FNIRT struct→MNI, concat warp
→ outputs: example_func2highres.mat, highres2standard_warp.nii.gz, …

Parallelism
-----------
The reference run (first session, run-1) is always processed first and
sequentially — every other run needs its example_func.nii.gz.
After the reference is done, all remaining runs can be parallelised.

Usage
-----
python 01_preprocess_fmri.py [--subject sub-01] [--workers 4] [--overwrite]

--workers 4 (default) ← safe for 16 GB / 20 cores
Each FSL/nipype job peaks at ~2–3 GB.
Raise to 8 if your htop shows headroom.
"""
from __future__ import annotations

import os
import nibabel as nib
from pathlib import Path
from glob import glob
from shutil import copyfile, rmtree
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

from pipeline_common import (
    make_parser,
    list_runs,
    first_session,
    author_run_dir,
    author_func_output_dir,
    author_reg_dir,
    author_bold_file,
    load_utils,
    standard_files,
    ensure_dir,
    find_anat_head,
    find_anat_brain,
    _norm_ses,
    _norm_run,
)

# Maximum allowed deviation (fraction) from session median file size for
# prefiltered_func.nii.gz before treating an existing file as "broken".
PREFILTERED_SIZE_TOL = 0.15  # 15%


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _copy_first(pattern: str, dest: str) -> bool:
    matches = glob(pattern)
    if not matches:
        return False
    copyfile(matches[0], dest)
    return True


def _resolve_example_func(preproc, root: Path, subject: str,
                          ses: str, run: str) -> Path:
    """
    After the reference-run workflow finishes, find example_func.nii.gz and
    copy it into the canonical location so later runs can reference it.
    """
    canonical = author_func_output_dir(root, subject, ses, run) / 'example_func.nii.gz'
    if canonical.exists():
        return canonical.resolve()

    # search the nipype working dir tree
    for candidate in sorted(Path(preproc.base_dir).glob('**/example_func.nii.gz')):
        ensure_dir(canonical.parent)
        if candidate.resolve() != canonical.resolve():
            copyfile(str(candidate), str(canonical))
        return canonical.resolve()

    raise FileNotFoundError(
        f'example_func.nii.gz not found after reference run completed. '
        f'Searched under {preproc.base_dir}'
    )


def compute_prefiltered_medians(root: Path, subject: str, runs) -> dict[str, int]:
    """
    Compute per-session median file size (in bytes) of existing
    prefiltered_func.nii.gz files for this subject.

    Used to detect obviously truncated outputs: if an existing file is more
    than PREFILTERED_SIZE_TOL smaller than the session median, we treat it as
    broken and recompute it.
    """
    sizes_by_ses: dict[str, list[int]] = defaultdict(list)
    for r in runs:
        ses = _norm_ses(r.ses)
        out_func_dir = author_func_output_dir(root, subject, ses, r.run)
        pf = out_func_dir / 'prefiltered_func.nii.gz'
        if pf.exists():
            try:
                sizes_by_ses[ses].append(pf.stat().st_size)
            except OSError:
                # If the file is unreadable, skip it but allow recomputation later.
                continue

    medians: dict[str, int] = {}
    for ses, sizes in sizes_by_ses.items():
        if len(sizes) < 2:
            # Not enough data points to define a robust median – skip.
            continue
        sizes.sort()
        medians[ses] = sizes[len(sizes) // 2]
    return medians


# ─────────────────────────────────────────────────────────────────────────────
# single-run worker
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(
    root_s: str,
    subject: str,
    ses: str,
    run: str,
    overwrite: bool = False,
    dry_run: bool = False,
    size_medians: dict[str, int] | None = None,
) -> str:
    root = Path(root_s).resolve()
    ses = _norm_ses(ses)
    run = _norm_run(run)

    if size_medians is None:
        size_medians = {}

    utils = load_utils(root)

    # ── reference session/run flags ───────────────────────────────────────────
    fs = _norm_ses(first_session(root, subject))
    is_ref = (int(ses) == int(fs) and int(run) == 1)

    # ── check outputs ─────────────────────────────────────────────────────────
    out_func_dir = ensure_dir(author_func_output_dir(root, subject, ses, run))
    prefiltered = out_func_dir / 'prefiltered_func.nii.gz'

    # Robust size check: if an existing prefiltered_func is >15% smaller than
    # the session median, treat it as broken and force recomputation.
    prefiltered_ok = prefiltered.exists()
    median_size = size_medians.get(ses)
    if prefiltered_ok and median_size is not None:
        try:
            s = prefiltered.stat().st_size
            if s < (1.0 - PREFILTERED_SIZE_TOL) * float(median_size):
                print(
                    f"[01] session-{ses} run-{run}: prefiltered_func.nii.gz "
                    f"({s} bytes) is >{int(PREFILTERED_SIZE_TOL*100)}% smaller "
                    f"than session median ({median_size} bytes) – "
                    f"treating as broken and recomputing."
                )
                prefiltered_ok = False
        except OSError:
            # Cannot stat file ⇒ treat as broken.
            prefiltered_ok = False

    skip_feat = False
    if is_ref:
        # For the reference run, also require the registration warp to exist.
        # If FEAT finished but registration crashed, skip FEAT and redo only reg.
        warp = author_reg_dir(root, subject, ses, run) / 'highres2standard_warp.nii.gz'
        ef_mat = author_reg_dir(root, subject, ses, run) / 'example_func2highres.mat'
        reg_complete = warp.exists() and ef_mat.exists()
        if prefiltered_ok and reg_complete and not overwrite:
            return f'skip session-{ses} run-{run}: FEAT and registration already complete'
        if prefiltered_ok and not overwrite:
            # FEAT done, registration crashed — skip FEAT, redo registration only
            skip_feat = True
    else:
        if prefiltered_ok and not overwrite:
            return f'skip session-{ses} run-{run}: prefiltered_func already exists'

    if dry_run:
        return f'dry-run session-{ses} run-{run}'

    # ── input ─────────────────────────────────────────────────────────────────
    func_data_file = author_bold_file(root, subject, ses, run)
    img = nib.load(str(func_data_file.resolve()))

    # ── reference example_func / geometry check ───────────────────────────────
    ref_example = author_func_output_dir(root, subject, fs, '1') / 'example_func.nii.gz'
    if is_ref:
        # Reference run always uses its own example_func as reference
        first_run_arg = True
    else:
        if not ref_example.exists():
            raise FileNotFoundError(
                f'Reference example_func.nii.gz not ready: {ref_example}. '
                f'Process session-{fs} run-1 first.'
            )

        # Prefer the global reference example_func when geometry matches;
        # fall back to per-run example_func if shapes differ (e.g. 88x94x66 vs 88x88x66)
        same_geom = False
        try:
            ref_img = nib.load(str(ref_example))
            same_geom = (ref_img.shape[:3] == img.shape[:3])
        except Exception:
            same_geom = False

        if same_geom:
            first_run_arg = str(ref_example.resolve())
        else:
            first_run_arg = True

    # ── build & run preprocessing workflow ───────────────────────────────────
    nvols = img.shape[3]

    n_remove = 10
    keep = nvols - n_remove
    if keep <= 0:
        raise ValueError(f"{func_data_file} has only {nvols} volumes; cannot remove {n_remove}")

    if not skip_feat:
        preproc, MC_dir, output_dir = utils.create_fsl_FEAT_workflow_func(
            workflow_name='nipype_workflow',
            first_run=first_run_arg,
            func_data_file=str(func_data_file.resolve()),
            fwhm=3,
            n_vol_remove=n_remove,
            total_vol=keep,
        )

        # FIX 1: per-run Nipype work dir under author_replication/
        feat_work_dir = ensure_dir(
            root / 'author_replication' / 'work' / 'feat'
            / subject / f's{ses}_r{run}'
        )
        preproc.base_dir = str(feat_work_dir)

        # FIX 2: ensure all outputs go to the per-run outputs/func/ dir
        per_run_func_dir = str(out_func_dir)
        preproc.inputs.dilatemask.out_file = os.path.join(per_run_func_dir, 'mask.nii.gz')
        preproc.inputs.meanscale.out_file = os.path.join(per_run_func_dir, 'prefiltered_func.nii.gz')
        preproc.inputs.gen_mean_func_img.out_file = os.path.join(per_run_func_dir, 'mean_func.nii.gz')
        if is_ref or first_run_arg is True:
            # When this run is acting as a reference (true ref run or
            # geometry-mismatched run using its own example_func), ensure
            # extractref writes example_func into the per-run outputs.
            preproc.inputs.extractref.roi_file = os.path.join(per_run_func_dir, 'example_func.nii.gz')

        preproc.write_graph()
        preproc.run()

        # ── copy MCflirt artefacts & clean up (only if FEAT was run) ────────────────
        wf_base = str(feat_work_dir)
        per_run_mc_dir = ensure_dir(out_func_dir / 'MC')
        for pattern, dest_name in [
            ('*.par', 'MCflirt.par'),
            ('*rot*', 'rot.png'),
            ('*trans*', 'trans.png'),
            ('*disp*', 'disp.png'),
        ]:
            _copy_first(
                os.path.join(
                    wf_base, 'nipype_workflow', 'MCFlirt',
                    'mapflow', '_MCFlirt0', pattern
                ),
                os.path.join(per_run_mc_dir, dest_name),
            )

        graph_hits = glob(os.path.join(wf_base, 'nipype_workflow', 'graph.png'))
        if graph_hits:
            copyfile(
                graph_hits[0],
                os.path.join(output_dir, f'session_{int(ses)}_run_{int(run)}.png')
            )

        for log in glob(os.path.join(wf_base, '*', '*', '*', '*', '*', 'report.rst')):
            log_name = log.split('/')[-5]
            copyfile(log, os.path.join(output_dir, f'log_{log_name}.rst'))

        # clean up the isolated work dir — outputs are already in outputs/func/
        if feat_work_dir.exists():
            rmtree(feat_work_dir)

    # ── resolve example_func for reference run ────────────────────────────────
    if is_ref:
        # example_func was written to output_dir by FEAT; look there first
        resolved_ef = out_func_dir / 'example_func.nii.gz'
        if not resolved_ef.exists() and not skip_feat:
            resolved_ef = _resolve_example_func(preproc, root, subject, ses, run)
        if not resolved_ef.exists():
            raise FileNotFoundError(
                f'example_func.nii.gz not found at {resolved_ef}. '
                f'Did step 01 FEAT complete successfully?'
            )

        # ── registration workflow ─────────────────────────────────────────────
        standard_brain, standard_head, standard_mask = standard_files(root)
        anat_brain = str(find_anat_brain(root, subject))
        anat_head = str(find_anat_head(root, subject))

        # reg_out_dir is where all registration output files are written
        reg_out_dir = str(ensure_dir(author_reg_dir(root, subject, ses, run)))

        # Override base_dir so nipype temp files go to work/, not inside reg/
        reg_work_dir = ensure_dir(
            root / 'author_replication' / 'work' / 'registration'
            / subject / f's{ses}_r{run}'
        )

        registration = utils.create_registration_workflow(
            anat_brain,
            anat_head,
            str(resolved_ef),
            standard_brain,
            standard_head,
            standard_mask,
            workflow_name='registration',
            output_dir=reg_out_dir,
        )
        registration.base_dir = str(reg_work_dir)

        registration.write_graph()
        registration.run()

        # clean up the nipype working dir (reg outputs are already in reg_out_dir)
        if reg_work_dir.exists():
            rmtree(reg_work_dir)

    return f'done session-{ses} run-{run}'


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = make_parser('Run the author FEAT-like preprocessing for all runs.')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    subjects = [args.subject]

    # Special batch mode: -1 → run on sub-02…sub-07
    if subjects == ["-1"]:
        subjects = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07"]

    for subject in subjects:
        all_runs = list_runs(root, subject)
        fs = _norm_ses(first_session(root, subject))

        # Precompute per-session median prefiltered sizes for this subject
        size_medians = compute_prefiltered_medians(root, subject, all_runs)

        # separate reference run from the rest
        ref_run = next(
            r for r in all_runs if int(r.ses) == int(fs) and int(r.run) == 1
        )
        others = [r for r in all_runs
                  if not (int(r.ses) == int(fs) and int(r.run) == 1)]

        print(f'\n[01] ===== Subject {subject} =====')
        # ── always process reference run first, sequentially ─────────────────────
        print(f'[01] Processing reference run: session-{ref_run.ses} run-{ref_run.run}')
        result = _run_one(
            str(root), subject, ref_run.ses, ref_run.run,
            args.overwrite, args.dry_run, size_medians
        )
        print(result)

        # verify example_func exists before launching parallel jobs
        ref_ef = author_func_output_dir(root, subject, fs, '1') / 'example_func.nii.gz'
        if not args.dry_run and not ref_ef.exists():
            # last-ditch search
            hits = sorted(
                author_run_dir(root, subject, fs, '1').glob('**/example_func.nii.gz')
            )
            if hits:
                ensure_dir(ref_ef.parent)
                copyfile(str(hits[0]), str(ref_ef))
        if not ref_ef.exists():
            raise FileNotFoundError(
                f'Reference run finished but example_func.nii.gz is missing: {ref_ef}'
            )

        if not others:
            print('[01] No additional runs to process for this subject.')
            continue

        # ── parallel processing of remaining runs ───────────────────────────────
        max_workers = max(1, args.workers)
        print(f'[01] Processing {len(others)} remaining runs with {max_workers} workers …')

        if max_workers == 1:
            for r in others:
                print(_run_one(
                    str(root), subject, r.ses, r.run,
                    args.overwrite, args.dry_run, size_medians
                ))
            # IMPORTANT: do not return here; continue to next subject if any
            continue

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _run_one,
                    str(root), subject, r.ses, r.run,
                    args.overwrite, args.dry_run, size_medians
                ): (r.ses, r.run)
                for r in others
            }

            for fut in as_completed(futures):
                try:
                    print(fut.result())
                except Exception as exc:
                    ses, run = futures[fut]
                    print(f'ERROR session-{ses} run-{run}: {exc}')


if __name__ == '__main__':
    main()