#!/usr/bin/env python
"""
CLI: python scripts/dicom2nii.py /path/to/dicom_dir --out data/ImagesTr/case0001_ct.nii.gz
"""
import argparse, subprocess, pathlib, json, shutil

parser = argparse.ArgumentParser()
parser.add_argument("dicom_dir")
parser.add_argument("--out", required=True)
parser.add_argument("--height", type=int, default=170)
parser.add_argument("--weight", type=int, default=70)
args = parser.parse_args()

dicom = pathlib.Path(args.dicom_dir)
nifti = pathlib.Path(args.out)
nifti.parent.mkdir(parents=True, exist_ok=True)

subprocess.run(["dcm2niix", "-z", "y", "-o", nifti.parent, dicom], check=True)  # :contentReference[oaicite:5]{index=5}

meta = {
    "height": args.height,
    "weight": args.weight,
    "bmi": args.weight / ((args.height/100)**2),
    "plane": {"point":[0,0,0],"normal":[1,0,0]}
}
json_fp = nifti.with_suffix(".json")
json_fp.write_text(json.dumps(meta, indent=2))
