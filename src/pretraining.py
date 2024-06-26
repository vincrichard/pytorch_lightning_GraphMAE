from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader as GraphDataLoader

from src.data.datasets import Zinc
from src.models.graph_mae import LitGraphMAE


logging_dir = f"logs/pretraining/graph_mae/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"


# setup data
dataset = Zinc("data/datasets/zinc.csv")
train_dataloader = GraphDataLoader(
    dataset, persistent_workers=True, shuffle=True, batch_size=256, num_workers=8
)

trainer = L.Trainer(
    accelerator="gpu",
    devices=[2],
    # limit_train_batches=50,
    max_epochs=100,
    logger=TensorBoardLogger(f"{logging_dir}/tensorboard", name="graphmae"),
    log_every_n_steps=1,
    # default_root_dir="some/path/",
    # enable_checkpointing=False
    callbacks=[
        ModelCheckpoint(
            dirpath=f"{logging_dir}/checkpoints", save_top_k=5, monitor="train_loss"
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ],
    #  Debug
    # fast_dev_run=5 # disable callbacks
    # limit_train_batches=0.1
    # limit_val_batches=5
    # num_sanity_val_steps=2 # Run at the start of training
    #  Performance
    # profiler="simple",  # / "advanced"
)

trainer.fit(
    model=LitGraphMAE(),
    train_dataloaders=train_dataloader,
    # val_dataloaders=,
)
