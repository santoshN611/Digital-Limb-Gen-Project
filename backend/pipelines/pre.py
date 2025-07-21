import torchio as tio                       # 3-D augmentation  :contentReference[oaicite:1]{index=1}
import simpleitk as sitk                    # DICOM/NIfTI IO   :contentReference[oaicite:2]{index=2}
import nibabel as nib
from pathlib import Path
import torch

spatial_aug = tio.transforms.RandAffine(scales=0.1, degrees=15)
elastic_aug = tio.transforms.Rand3DElastic()   # default α, σ  :contentReference[oaicite:3]{index=3}
gamma_aug   = tio.transforms.RandomGamma()

def read_nifti(fp: Path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(fp)))

def save_nifti(arr: torch.Tensor, out_fp: Path):
    img = sitk.GetImageFromArray(arr.numpy().astype("uint8"))
    sitk.WriteImage(img, str(out_fp))

def preprocess_volume(nifti_fp: Path, meta: dict):
    vol = read_nifti(nifti_fp)
    subject = tio.Subject(ct=tio.ScalarImage(tensor=vol[None]))
    subject = spatial_aug(elastic_aug(gamma_aug(subject)))
    tensor  = torch.as_tensor(subject.ct.data, dtype=torch.float32)
    # normalise simple Z-score
    tensor = (tensor - tensor.mean()) / tensor.std()
    return tensor  # shape [1, D, H, W]

# lazy-load CPU checkpoint
import torch, pathlib
MODEL_PATH = pathlib.Path(__file__).parent.parent / "models/seg_cpu.pt"
SEG_MODEL  = torch.jit.load(MODEL_PATH, map_location="cpu").eval()
