from datetime import datetime
import logging.config
from pathlib import Path
import logging
import json

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader as GraphDataLoader
import torch
import numpy as np

from src.data.datasets import Tox21
from src.data.splitters import scaffold_split
from src.models.graph_mae import LitGraphMAE, LitFinetuneGraphMAE
from torch_geometric.seed import seed_everything

logger = logging.getLogger(__name__)


def train_and_predict_with_seed(logging_dir: str, checkpoint_encoder: Path, seed: int):
    seed_everything(seed)

    l_logger = TensorBoardLogger(f"{logging_dir}/tensorboard", name="graph_mae")

    # setup data
    dataset = Tox21("dataset/tox21/raw/tox21.csv")
    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, dataset.smiles)
    train_dataloader = GraphDataLoader(
        train_dataset, shuffle=True, batch_size=32, num_workers=8
    )
    valid_dataloader = GraphDataLoader(
        valid_dataset, shuffle=False, batch_size=32, num_workers=8
    )
    test_dataloader = GraphDataLoader(
        test_dataset, shuffle=False, batch_size=32, num_workers=8
    )

    # Load model
    model = LitFinetuneGraphMAE(output_features=dataset.num_target)
    # model.encoder.load_state_dict(torch.load("GraphMAE/chem/init_weights/pretrained.pth"))
    model.encoder.load_state_dict(torch.load(checkpoint_encoder))

    # Train
    model_checkpoint = ModelCheckpoint(
        dirpath=f"{logging_dir}/checkpoints",
        filename="{epoch}-{val_loss:.2f}-{val_roc_auc:.4f}",
        save_top_k=5,
        monitor="val_roc_auc",
        mode="max",
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=[1],
        # max_steps=2,
        max_epochs=100,
        logger=l_logger,
        log_every_n_steps=100,
        callbacks=[
            model_checkpoint,
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar(),
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    # Load best model and inference
    state_dict = model_checkpoint.state_dict()
    model.load_state_dict(torch.load(state_dict["best_model_path"])["state_dict"])

    trainer.test(model=model, dataloaders=test_dataloader)

    return model.test_roc_auc


def extract_and_save_encoder(pretrained_model_checkpoint: str) -> Path:
    path_checkpoint = Path(pretrained_model_checkpoint)
    pretrained_model = LitGraphMAE.load_from_checkpoint(path_checkpoint)
    path_pretrained_encoder = (
        path_checkpoint.parent / f"{path_checkpoint.stem}.encoder.ckpt"
    )
    torch.save(pretrained_model.graph_mae.encoder.state_dict(), path_pretrained_encoder)
    return path_pretrained_encoder


def setup_logging(experiment_dir: str):
    with open("data/config/logging.json", "r") as file:
        config = json.load(file)
    config["handlers"]["file"]["filename"] = f"{experiment_dir}/finetuning.log"
    logging.config.dictConfig(config=config)


if __name__ == "__main__":
    experiment_dir = "logs/finetuning/graph_mae/"
    experiment_dir = f"{experiment_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(experiment_dir)

    path_pretrained_encoder = extract_and_save_encoder(
        pretrained_model_checkpoint="logs/pretraining/graph_mae/2024-06-06_10-14/checkpoints/epoch=86-step=679731.ckpt"
    )
    roc_auc = []
    for seed in range(10):
        logging_dir = f"{experiment_dir}/{seed}"
        test_roc_auc = train_and_predict_with_seed(
            logging_dir=logging_dir,
            checkpoint_encoder=path_pretrained_encoder,
            seed=seed,
        )
        roc_auc.append(test_roc_auc)

    logger.info(f"RESULT TEST ROC AUC: {np.mean(roc_auc)} ± {np.std(roc_auc)}")
