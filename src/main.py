import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    ModelCheckpoint,
)
from aim.pytorch_lightning import AimLogger
from torch_geometric.loader import DataLoader as GraphDataLoader

from src.data import Zinc250
from src.models.graph_mae import LitGraphMAE
from src.utils.aim_callback import (
    AimParamGradientCallback,
    AimLayerOutputDisctributionCallback,
    get_run_folder_aim_logger,
)

# track experimental data by using Aim
aim_logger = AimLogger(
    repo="./logs",
    experiment="GraphMAE",
    train_metric_prefix="train_",
    val_metric_prefix="val_",
)

logging_directory = get_run_folder_aim_logger(aim_logger)

# setup data
dataset = Zinc250()
train_dataloader = GraphDataLoader(dataset, batch_size=256, num_workers=8)

model = LitGraphMAE()
model.to_onnx(next(iter(train_dataloader)), logging_directory)

trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    # limit_train_batches=100,
    max_epochs=100,
    logger=aim_logger,
    log_every_n_steps=1,
    # default_root_dir="some/path/",
    # enable_checkpointing=False
    callbacks=[
        ModelCheckpoint(dirpath=logging_directory, save_top_k=5, monitor="train_loss"),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
        AimParamGradientCallback(),
        AimLayerOutputDisctributionCallback(),
    ],
    #  Debug
    # fast_dev_run=5 # disable callbacks
    # limit_train_batches=0.1
    # limit_val_batches=5
    # num_sanity_val_steps=2 # Run at the start of training
    #  Performance
    # profiler="simple" / "advanced"
)

trainer.fit(
    model=LitGraphMAE(),
    train_dataloaders=train_dataloader,
    # val_dataloaders=,
)
