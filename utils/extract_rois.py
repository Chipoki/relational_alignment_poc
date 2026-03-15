import os
import subprocess
import glob
import sys
import threading
import queue
import tkinter as tk
from tkinter import scrolledtext
import shutil

roi_mapping = {
    "pericalcarine": ["pericalcarine"],
    "lingual": ["lingual"],
    "lateral_occipital": ["lateraloccipital"],
    "fusiform": ["fusiform"],
    "parahippocampal": ["parahippocampal"],
    "inferior_temporal": ["inferiortemporal"],
    "inferior_parietal": ["inferiorparietal"],
    "precuneus": ["precuneus"],
    "superior_parietal": ["superiorparietal"],
    "superior_frontal": ["superiorfrontal"],
    "middle_frontal": ["rostralmiddlefrontal", "caudalmiddlefrontal"],
    "inferior_frontal": ["parsopercularis", "parstriangularis", "parsorbitalis"]
}


def run_subprocess(cmd, log_queue, env=None):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=env)
    for line in process.stdout:
        log_queue.put(line)
    process.wait()
    return process.returncode


def processing_thread(log_queue, root_window):
    log_queue.put("Starting preprocessing pipeline with Paranoia Locks active...\n")

    if subprocess.run("command -v flirt", shell=True, capture_output=True).returncode != 0:
        log_queue.put("ERROR: FSL 'flirt' not found in system paths.\n")
        return
    if subprocess.run("command -v recon-all", shell=True, capture_output=True).returncode != 0:
        log_queue.put("ERROR: FreeSurfer 'recon-all' not found in system paths.\n")
        return

    data_root = "./ds003927/derivatives"

    if not os.path.exists(data_root):
        log_queue.put(f"ERROR: Cannot find data root at {data_root}\n")
        return

    subjects = sorted(
        [d for d in os.listdir(data_root) if d.startswith("sub-") and os.path.isdir(os.path.join(data_root, d))])

    if not subjects:
        log_queue.put("CRITICAL: No subject folders found. Check your directory path.\n")
        return

    for sub in subjects:
        log_queue.put(f"\n{'=' * 20} Processing {sub} {'=' * 20}\n")
        sub_dir = os.path.join(data_root, sub)
        anat_dir = os.path.join(sub_dir, "anat")

        fs_workspace = os.path.join(sub_dir, "fs_workspace")
        out_label_dir = os.path.join(sub_dir, "fs_labels")
        out_mask_dir = os.path.join(sub_dir, "func_masks")

        os.makedirs(fs_workspace, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        # STRICT READ-ONLY targeting of raw data
        t1_raw_search = glob.glob(os.path.join(anat_dir, f"{sub}_T1w.nii.gz"))
        t1_brain_search = glob.glob(os.path.join(anat_dir, f"{sub}_T1w_*_brain.nii.gz"))
        example_func = os.path.join(sub_dir, "example_func.nii.gz")

        if not t1_raw_search or not t1_brain_search or not os.path.exists(example_func):
            log_queue.put(f"[{sub}] Missing required NIfTI files. Skipping to protect pipeline.\n")
            continue

        t1_raw = t1_raw_search[0]
        t1_brain = t1_brain_search[0]

        env = os.environ.copy()
        env["SUBJECTS_DIR"] = fs_workspace

        fs_sub_dir = os.path.join(fs_workspace, sub)
        lh_annot = os.path.join(fs_sub_dir, "label", "lh.aparc.annot")

        # STEP 0: Fresh recon-all sequence mapped to the safe workspace
        if not os.path.exists(lh_annot):
            log_queue.put(f"[{sub}] aparc.annot missing. Preparing workspace...\n")

            # PARANOIA LOCK: Auto-wipe dead folders ONLY if safely inside the new workspace
            if os.path.exists(fs_sub_dir):
                if "fs_workspace" in fs_sub_dir and "anat" not in fs_sub_dir:
                    log_queue.put(f"  -> Clearing dead folder from previous interrupted run in safe workspace...\n")
                    shutil.rmtree(fs_sub_dir, ignore_errors=True)
                else:
                    log_queue.put(
                        f"  -> CRITICAL SAFETY ABORT: Cleanup target path is outside safe bounds. Skipping {sub}.\n")
                    continue

            log_queue.put(f"[{sub}] Initiating fresh recon-all...\n")
            cmd_recon = ["recon-all", "-all", "-s", sub, "-i", t1_raw, "-sd", fs_workspace]

            ret = run_subprocess(cmd_recon, log_queue, env=env)
            if ret != 0:
                log_queue.put(f"[{sub}] recon-all failed. Check terminal for memory or license errors.\n")
                continue
        else:
            log_queue.put(f"[{sub}] FreeSurfer v6.0.0 workspace found. Skipping recon-all.\n")

        # STEP 1: Extract FreeSurfer .label files
        log_queue.put(f"[{sub}] Extracting FreeSurfer .label files...\n")
        for hemi in ["lh", "rh"]:
            cmd_annot = ["mri_annotation2label", "--subject", sub, "--hemi", hemi,
                         "--annotation", "aparc", "--outdir", out_label_dir]
            run_subprocess(cmd_annot, log_queue, env=env)

        # STEP 2: FSL FLIRT Registration
        log_queue.put(f"[{sub}] Calculating 7-DOF registration matrix...\n")
        anat2func_mat = os.path.join(out_mask_dir, "anat2func.mat")
        cmd_flirt = ["flirt", "-in", t1_brain, "-ref", example_func, "-omat", anat2func_mat, "-dof", "7"]
        run_subprocess(cmd_flirt, log_queue, env=env)

        # STEP 3: Create volumetric masks and binarize
        log_queue.put(f"[{sub}] Projecting and binarizing 12 functional ROIs...\n")
        for roi_name, fs_labels in roi_mapping.items():
            # Paths for intermediate anatomical files
            lh_anat = os.path.join(out_mask_dir, f"{roi_name}_lh_anat.nii.gz")
            rh_anat = os.path.join(out_mask_dir, f"{roi_name}_rh_anat.nii.gz")
            combined_anat = os.path.join(out_mask_dir, f"{roi_name}_anat.nii.gz")
            func_mask = os.path.join(out_mask_dir, f"{roi_name}_mask.nii.gz")

            # A. Process LH and RH separately
            for hemi, out_vol in [("lh", lh_anat), ("rh", rh_anat)]:
                hemi_labels = [os.path.join(out_label_dir, f"{hemi}.{lbl}.label")
                               for lbl in fs_labels]
                hemi_labels = [l for l in hemi_labels if os.path.exists(l)]

                if not hemi_labels:
                    continue

                label_args = []
                for lbl in hemi_labels:
                    label_args.extend(["--label", lbl])

                cmd_label2vol = ["mri_label2vol"] + label_args + [
                    "--temp", t1_brain, "--subject", sub, "--hemi", hemi,  # Added --hemi
                    "--regheader", t1_brain, "--o", out_vol,
                    "--proj", "frac", "0", "1", ".1", "--fillthresh", ".3"
                ]
                run_subprocess(cmd_label2vol, log_queue, env=env)

            # B. Merge Hemispheres
            if os.path.exists(lh_anat) and os.path.exists(rh_anat):
                run_subprocess(["fslmaths", lh_anat, "-add", rh_anat, combined_anat], log_queue, env=env)
            elif os.path.exists(lh_anat):
                os.rename(lh_anat, combined_anat)
            elif os.path.exists(rh_anat):
                os.rename(rh_anat, combined_anat)
            else:
                continue

            # C. Register to functional space and binarize
            cmd_applyxfm = ["flirt", "-in", combined_anat, "-ref", example_func, "-applyxfm",
                            "-init", anat2func_mat, "-interp", "nearestneighbour", "-out", func_mask]
            run_subprocess(cmd_applyxfm, log_queue, env=env)
            run_subprocess(["fslmaths", func_mask, "-bin", func_mask], log_queue, env=env)

            # Clean up
            for f in [lh_anat, rh_anat, combined_anat]:
                if os.path.exists(f): os.remove(f)

    log_queue.put("\nPipeline execution fully completed!\n")


def process_queue(root, text_widget, log_queue):
    try:
        while True:
            msg = log_queue.get_nowait()
            text_widget.insert(tk.END, msg)
            text_widget.see(tk.END)
    except queue.Empty:
        pass
    root.after(100, process_queue, root, text_widget, log_queue)


def main():
    root = tk.Tk()
    root.title("fMRI Pre-processing (Paranoia Locks Active)")
    root.geometry("800x600")

    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Courier", 10))
    text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    log_queue = queue.Queue()
    t = threading.Thread(target=processing_thread, args=(log_queue, root), daemon=True)
    t.start()

    root.after(100, process_queue, root, text_area, log_queue)
    root.mainloop()


if __name__ == "__main__":
    main()