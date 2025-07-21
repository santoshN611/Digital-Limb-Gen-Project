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
    Visible Human Male – Frozen CT (PNG → NIfTI)

    • Downloads the current ZIP bundle published by NLM
      (CT Scans After Freezing.zip, ~440 kB sample stack).
    • Extracts with std-lib zipfile → no /usr/bin/unzip needed.
    • Stacks PNG slices, writes case002_ct.nii.gz + metadata.
    """
    import requests, zipfile, glob, imageio.v3 as iio
    import numpy as np, SimpleITK as sitk

    # ------------------------------------------------------------------ #
    # 1) Fetch the ZIP (spaces MUST be %20-encoded!)
    # ------------------------------------------------------------------ #
    url = ("https://data.lhncbc.nlm.nih.gov/public/Visible-Human/"
           "Sample-Data/CT%20Scans%20After%20Freezing.zip")
    zip_path = RAW / "vh_ct.zip"
    if not zip_path.exists():
        print("Downloading Visible Human CT ZIP …")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        zip_path.write_bytes(r.content)

    # ------------------------------------------------------------------ #
    # 2) Extract with Python’s zipfile (cross-platform)
    # ------------------------------------------------------------------ #
    extract_dir = RAW / "vh_ct"
    if not extract_dir.exists():
        print("Extracting PNG slices …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # ------------------------------------------------------------------ #
    # 3) Stack PNGs → volume
    # ------------------------------------------------------------------ #
    png_files = sorted(glob.glob(str(extract_dir / "*.png")))
    if not png_files:
        raise RuntimeError(
            "ZIP extracted but contained no PNG files - "
            "check the download URL or permissions."
        )
    print(f"Found {len(png_files)} PNG slices; building volume …")
    volume = np.stack([iio.imread(p) for p in png_files], axis=0)

    # ------------------------------------------------------------------ #
    # 4) Save NIfTI + metadata
    # ------------------------------------------------------------------ #
    out_nii = IMG / "case002_ct.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(volume), str(out_nii))
    _write_meta("case002", 178, 77)
    print(f"✓ Visible Human CT saved to {out_nii}")



def download_ultrasound(n_slices: int = 128):
    """
    Mini-subset from openmedlab/Awesome-Medical-Dataset/UNS.md
    ---------------------------------------------------------
    • Works with both old markdown link lines and new <img src="…"> tags.
    • Normalises GitHub blob URLs to raw.githubusercontent form.
    • Saves case003_us.nii.gz (Z,Y,X) + JSON meta.
    """
    import requests, re, io, numpy as np, imageio.v3 as iio, SimpleITK as sitk

    md_url = ("https://raw.githubusercontent.com/openmedlab/"
              "Awesome-Medical-Dataset/main/resources/UNS.md")
    txt = requests.get(md_url, timeout=30).text

    # regex matches: 1) markdown (…)png)   2)  <img src="…png">
    pattern = r"(https://[^\s\"')]+\.png)"
    links = re.findall(pattern, txt, flags=re.IGNORECASE)

    # normalise any GitHub blob URLs → raw.githubusercontent URLs
    norm = []
    for u in links:
        if "github.com" in u and "/blob/" in u:
            u = u.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        norm.append(u)
    links = norm[:n_slices]

    if not links:
        raise RuntimeError("UNS.md contains no downloadable PNG links")

    print(f"Stacking {len(links)} ultrasound frames …")
    vol = np.stack([iio.imread(io.BytesIO(requests.get(u, timeout=30).content))
                    for u in links], axis=0)

    out_nii = IMG / "case003_us.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(vol), str(out_nii))
    _write_meta("case003", 172, 82)
    print(f"✓ Ultrasound stack saved to {out_nii}")



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
