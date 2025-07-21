import torchio as tio, pathlib, json

class LimbDataset(tio.SubjectsDataset):
    def __init__(self, root="data/ImagesTr", training=True):
        imgs = sorted(pathlib.Path(root).glob("*.nii.gz"))
        subjects = []
        for img in imgs:
            label = img.with_name(img.stem + "_seg.nii.gz")
            meta  = img.with_suffix(".json")
            subj = tio.Subject(
                image = tio.ScalarImage(img),
                label = tio.LabelMap(label),
                meta  = json.loads(meta.read_text()) if meta.exists() else {},
            )
            subjects.append(subj)
        super().__init__(subjects)
