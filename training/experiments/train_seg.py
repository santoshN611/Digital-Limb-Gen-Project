import hydra, mlflow, torch, pathlib
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from datasets.limb_dataset import LimbDataset
from nets.dynunet import get_dynunet

@hydra.main(config_path="../configs", config_name="seg.yaml")
def main(cfg: DictConfig):
    mlflow.set_experiment("seg_baseline")
    with mlflow.start_run():
        ds = LimbDataset(cfg.data.root)
        loader = DataLoader(ds, batch_size=1, shuffle=True)
        net = get_dynunet().to("cpu")
        optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(cfg.train.epochs):
            for subj in loader:
                x = subj["image"][tio.DATA].float()
                y = subj["label"][tio.DATA].squeeze(1).long()
                logits = net(x)
                loss = loss_fn(logits, y)
                optim.zero_grad(); loss.backward(); optim.step()
        model_path = pathlib.Path("../../backend/models/seg_cpu.pt")
        torch.jit.script(net).save(model_path)
        mlflow.log_artifact(model_path.as_posix())

if __name__ == "__main__":
    main()
