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
        subprocess.run(["openneuro-py", "download",
                        f"--dataset={dataset}", "--include=sub-01/ses-01/anat",
                        f"--directory={out}"], check=True)
    elif shutil.which("openneuro"):
        subprocess.run(["openneuro", "download", dataset,
                        "-d", out, "--include=sub-01/ses-01/anat"], check=True)
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
    # pick first DICOM series
    dicom_dir = next(dicom_root.rglob("*.dcm")).parent
    out_nii = IMG / "case001_mrn.nii.gz"
    convert_dicom_to_nifti(dicom_dir, out_nii)
    _write_meta("case001", 180, 80)

def download_visible_ct():
    print("Downloading Visible Human CT slice…")
    url = ("https://ftp.nlm.nih.gov/visible_human_datasets/"
           "male/frozen/Frozen_Frames/head/ct/ct_001.tif")
    out_tif = RAW / "vh_ct_001.tif"
    if not out_tif.exists():
        subprocess.run(["wget", "-q", "-O", out_tif, url], check=True)
    # SimpleITK can read TIFF stack as volume
    import SimpleITK as sitk
    img = sitk.ReadImage(str(out_tif))
    sitk.WriteImage(img, str(IMG / "case002_ct.nii.gz"))
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
