"""
scripts/extract_rois.py – Extract functional ROIs from FreeSurfer segmentations.
requires a parcellation file (aparc+aseg.mgz) and a functional reference image (example_func.nii.gz)
"""
from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

# FreeSurfer Desikan-Killiany atlas (aparc+aseg) integer codes
# Reference: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
ROI_MAPPING = {
    "pericalcarine": [1021, 2021],
    "lingual": [1013, 2013],
    "lateral_occipital": [1011, 2011],
    "fusiform": [1007, 2007],
    "inferior_temporal": [1009, 2009],
    "parahippocampal": [1016, 2016],
    "precuneus": [1025, 2025],
    "inferior_parietal": [1008, 2008],
    "superior_parietal": [1029, 2029],
    "superior_frontal": [1028, 2028],
    # FreeSurfer splits middle frontal into caudal (1003/2003) and rostral (1027/2027)
    "middle_frontal": [1003, 2003, 1027, 2027],
    # FreeSurfer splits inferior frontal into opercularis (1018), orbitalis (1019), and triangularis (1020)
    "inferior_frontal": [1018, 2018, 1019, 2019, 1020, 2020]
}


def main() -> None:
    # Set this to your exact derivatives folder path
    data_root = Path(
        "/home/tomerd/Documents/projects/MSc/lab/thesis_practice/relational_alignment/soto_data/ds003927/derivatives")

    # Find all subject directories
    subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("sub-")])

    if not subjects:
        logging.error(f"No subject directories found in {data_root}")
        return

    for subj_dir in subjects:
        subj_id = subj_dir.name
        logging.info(f"Processing {subj_id}...")

        # Expected file paths based on your screenshot
        aparc_path = subj_dir / "anat" / subj_id / "mri" / "aparc+aseg.mgz"
        func_ref_path = subj_dir / "example_func.nii.gz"
        func_mask_path = subj_dir / "mask.nii.gz"
        out_dir = subj_dir / "rois"

        if not aparc_path.exists():
            logging.warning(f"Missing parcellation {aparc_path} – Skipping.")
            continue
        if not func_ref_path.exists():
            logging.warning(f"Missing functional ref {func_ref_path} – Skipping.")
            continue

        # Create the rois/ directory where the pipeline expects them
        out_dir.mkdir(exist_ok=True)

        # 1. Load images
        aparc_img = nib.load(str(aparc_path))
        func_ref_img = nib.load(str(func_ref_path))

        func_mask_img = nib.load(str(func_mask_path))
        func_mask_data = func_mask_img.get_fdata() > 0

        # 2. Resample anatomical parcellation to functional space
        # CRITICAL: We must use 'nearest' interpolation so the integer category labels aren't blended into decimals
        logging.info("  Resampling parcellation to functional space...")
        aparc_resampled = resample_to_img(
            source_img=aparc_img,
            target_img=func_ref_img,
            interpolation="nearest"
        )
        aparc_data = aparc_resampled.get_fdata()

        # 3. Extract and save each ROI
        count = 0
        for roi_name, codes in ROI_MAPPING.items():
            # Create boolean mask where voxel value matches any of the target FreeSurfer codes
            roi_mask = np.isin(aparc_data, codes)

            # Clean it up by intersecting with the functional whole-brain mask
            final_mask = roi_mask & func_mask_data

            if final_mask.sum() == 0:
                logging.warning(f"  ROI {roi_name} resulted in 0 voxels after masking.")
                continue

            # Save as NIfTI
            out_img = nib.Nifti1Image(
                final_mask.astype(np.int16),
                affine=func_ref_img.affine,
                header=func_ref_img.header
            )
            out_path = out_dir / f"{roi_name}_mask.nii.gz"
            nib.save(out_img, str(out_path))
            count += 1

        logging.info(f"  ✓ Saved {count} ROI masks to {out_dir}\n")


if __name__ == "__main__":
    main()