#!/usr/bin/env python
"""
Download and prepare mini datasets for Digital-Limb-Gen.

Usage examples
--------------
# pull one dataset
python scripts/download_data.py --mrn

# pull everything
python scripts/download_data.py --all
"""
import argparse, subprocess, pathlib, json, os, shutil, sys, textwrap, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]   # repo root
RAW  = ROOT / "data/raw"
IMG  = ROOT / "data/ImagesTr"
META = ROOT / "data/Metadata"
RAW.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)

def _write_meta(case, h, w):
    bmi = round(w / ((h/100)**2), 1)
    meta = dict(height=h, weight=w, bmi=bmi,
                plane=dict(point=[0,0,0], normal=[1,0,0]))
    (META / f"{case}.json").write_text(json.dumps(meta, indent=2))

import shutil, subprocess, pathlib
def fetch_openneuro(dataset="ds002766", out=pathlib.Path("data/raw")):
    out = out / dataset
    if out.exists(): return out
    if shutil.which("openneuro-py"):
        SUBJECT = "sub-cast1"
        SESSION = "ses-01"
        INCLUDE = f"{SUBJECT}/{SESSION}"
        cmd = [
            "openneuro-py", "download",
            f"--dataset={dataset}",
            f"--include={INCLUDE}",
            f"--target-dir={out}"
        ]
        subprocess.run(cmd, check=True)
    elif shutil.which("openneuro"):
        SUBJECT = "sub-cast1"
        SESSION = "ses-01"
        INCLUDE = f"{SUBJECT}/{SESSION}"
        cmd = [
            "openneuro", "download",
            f"--dataset={dataset}",
            f"--include={INCLUDE}",
            f"--target-dir={out}"
        ]
        subprocess.run(cmd, check=True)
    else:
        raise RuntimeError("Install openneuro-py (pip) or @openneuro/cli (npm)")
    return out

def convert_dicom_to_nifti(dicom_dir: pathlib.Path, out_fp: pathlib.Path):
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["dcm2niix", "-z", "y", "-o", str(out_fp.parent),
                    "-f", out_fp.stem, str(dicom_dir)], check=True)

def download_mrn():
    print("Downloading MR-neurography scan (OpenNeuro ds002766)…")
    dicom_root = fetch_openneuro()

    # Prepare output path
    out_nii = IMG / "case001_mrn.nii.gz"

    # 1. Check for raw DICOM files
    dcm_files = list(dicom_root.rglob("*.dcm"))
    if dcm_files:
        # Convert first DICOM series
        dicom_dir = dcm_files[0].parent
        convert_dicom_to_nifti(dicom_dir, out_nii)
    else:
        # 2. Fallback: use existing NIfTI directly
        nii_files = list(dicom_root.rglob("*.nii.gz"))
        if not nii_files:
            raise RuntimeError(f"No DICOM or NIfTI files found in {dicom_root}")
        print(f"No DICOM files found; copying NIfTI {nii_files[0].name}")
        shutil.copy(nii_files[0], out_nii)

    # 3. Write metadata as before
    _write_meta("case001", 180, 80)

def download_visible_ct():
    """
    Download the Visible Human CT volume via FTP (anonymous),
    fetch all PNG slices from the correct directory,
    stack them into a 3D NumPy array, and save as NIfTI.
    """
    print("Downloading Visible Human CT volume from PNG slices…")

    from ftplib import FTP
    import io, re
    import imageio
    import numpy as np
    import SimpleITK as sitk

    # 1) Connect and login anonymously
    ftp = FTP('ftp.nlm.nih.gov')
    ftp.login()

    # 2) Change to the exact absolute directory on the FTP server
    ftp.cwd('/public/Visible-Human/Male-Images/PNG_format/radiological/frozenCT')

    # 3) List and sort all PNG filenames
    names = []
    ftp.retrlines('NLST', names.append)         # get directory listing
    png_files = sorted([n for n in names if n.lower().endswith('.png')])
    if not png_files:
        raise RuntimeError("No PNG slices found in FTP directory")

    # 4) Download each slice into memory and read via imageio
    volume_slices = []
    for fname in png_files:
        bio = io.BytesIO()
        ftp.retrbinary(f'RETR {fname}', bio.write)  # fetch file bytes
        bio.seek(0)
        arr = imageio.imread(bio)                   # read PNG from buffer
        volume_slices.append(arr)

    ftp.quit()

    # 5) Stack into a 3D volume and save as NIfTI
    volume = np.stack(volume_slices, axis=0)
    sitk_image = sitk.GetImageFromArray(volume)
    sitk.WriteImage(sitk_image, str(IMG / "case002_ct.nii.gz"))

    # 6) Write metadata side-car
    _write_meta("case002", 178, 77)



def download_ultrasound():
    print("Downloading Kaggle Ultrasound Nerve Segmentation subset…")
    url = ("https://raw.githubusercontent.com/openmedlab/"
           "Awesome-Medical-Dataset/main/resources/UNS.md")
    md = RAW / "us_nerve.md"
    if not md.exists():
        subprocess.run(["wget", "-q", "-O", md, url], check=True)
    # this MD lists links to individual PNGs; grab first 100
    import re, requests, io, numpy as np, SimpleITK as sitk
    links = re.findall(r"https://[^\s)]+png", md.read_text())[:100]
    vol = []
    for i, l in enumerate(links):
        print(f" {i:03}", end="\r"); sys.stdout.flush()
        img_arr = sitk.GetArrayFromImage(
            sitk.ReadImage(io.BytesIO(requests.get(l).content)))
        vol.append(img_arr)
    vol = np.stack(vol)
    sitk.WriteImage(sitk.GetImageFromArray(vol),
                    str(IMG / "case003_us.nii.gz"))
    _write_meta("case003", 172, 82)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mrn", action="store_true")
    p.add_argument("--ct", action="store_true")
    p.add_argument("--us", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    if args.all or args.mrn: download_mrn()
    if args.all or args.ct:  download_visible_ct()
    if args.all or args.us:  download_ultrasound()
    print("✅  Data download complete. Check data/ImagesTr")

if __name__ == "__main__":
    main()
