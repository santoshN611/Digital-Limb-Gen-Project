from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from uuid import uuid4
import shutil, pathlib, subprocess, json

from pipelines.pre import preprocess_volume, SEG_MODEL   # local import

TMP = pathlib.Path("/tmp/limbgen")
TMP.mkdir(exist_ok=True, parents=True)

app = FastAPI(title="Digital-Limb-Gen Inference API")

jobs = {}  # rudimentary in-mem job store


def _run_inference(job_id: str, dicom_dir: pathlib.Path, meta: dict):
    """Background task: DICOM → NIfTI → tensor → segmentation mask."""
    nifti = dicom_dir.with_suffix(".nii.gz")
    # 1. convert
    subprocess.run(["dcm2niix", "-z", "y", "-o", dicom_dir.parent, dicom_dir],
                   check=True)                                         # :contentReference[oaicite:0]{index=0}
    # 2. preprocess & segment
    tensor = preprocess_volume(nifti, meta)
    preds  = SEG_MODEL(tensor.unsqueeze(0)).argmax(1).squeeze(0)
    out_nifti = nifti.with_name(f"{job_id}_seg.nii.gz")
    preprocess_volume.save_nifti(preds, out_nifti)  # helper in same module
    jobs[job_id]["status"] = "done"
    jobs[job_id]["result"] = out_nifti


@app.post("/upload")
async def upload_scan(background_tasks: BackgroundTasks,
                      file: UploadFile,
                      metadata: UploadFile):
    job_id = uuid4().hex
    job_dir = TMP / job_id
    job_dir.mkdir(parents=True)
    # save DICOM zip
    dicom_zip = job_dir / f"{job_id}.zip"
    with dicom_zip.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)
    # save meta JSON
    meta_path = job_dir / "meta.json"
    json_meta = json.loads(await metadata.read())
    meta_path.write_text(json.dumps(json_meta))
    # unzip DICOM
    subprocess.run(["unzip", "-qq", dicom_zip, "-d", job_dir])
    dicom_dir = job_dir / "dicom"
    # background inference
    jobs[job_id] = {"status": "running"}
    background_tasks.add_task(_run_inference, job_id, dicom_dir, json_meta)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    return jobs.get(job_id, {"error": "unknown id"})


@app.get("/result/{job_id}")
def result(job_id: str):
    record = jobs.get(job_id)
    if not record or record["status"] != "done":
        raise HTTPException(404, "Not ready")
    return FileResponse(record["result"])
