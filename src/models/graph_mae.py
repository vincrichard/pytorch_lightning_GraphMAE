from typing import Union

import torch
from torch import optim
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MeanAggregation
import lightning as L
from sklearn.metrics import roc_auc_score


from src.models.gnn import GNN, GINConv
from src.criterion import sce_loss
from src.utils.onnx_export import save_model_onnx


class GraphMAE(torch.nn.Module):
    """For inference outside Pytorch Lightning"""

    def __init__(self):
        super().__init__()
        self.num_layer = 5
        self.emb_dim = 300
        self.encoder = GNN(num_layer=self.num_layer, emb_dim=self.emb_dim, drop_ratio=0)
        self.encoder_to_decoder = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Linear(
                in_features=self.emb_dim, out_features=self.emb_dim, bias=False
            ),
        )
        self.decoder = GINConv(emb_dim=self.emb_dim, out_dim=119, aggr="add")

    def forward(self, x, edge_index, edge_attr, masked_atom_mask):

        node_rep = self.encoder(x, edge_index, edge_attr)

        decoder_input = self.encoder_to_decoder(node_rep)
        decoder_input[masked_atom_mask] = 0

        return self.decoder(decoder_input, edge_index, edge_attr)


class LitGraphMAE(L.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.graph_mae = GraphMAE()
        self.criterion = sce_loss
        self.save_hyperparameters()

    def forward(self, batch: Data):
        raise self.graph_mae.forward(
            batch.x, batch.edge_index, batch.edge_attr, batch.masked_atom_mask
        )

    def training_step(self, batch: Data, batch_idx: int):
        masked_atom_mask = batch.masked_atom_mask
        pred_node = self.graph_mae(
            batch.x, batch.edge_index, batch.edge_attr, masked_atom_mask
        )

        loss = self.criterion(batch.node_attr_label, pred_node[masked_atom_mask])
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss.cpu().item(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0)
        return optimizer

    def to_onnx(self, batch, directory):
        list_names = ["x", "edge_index", "edge_attr", "masked_atom_mask"]
        batch = batch.to(self.device)
        self.graph_mae.to(self.device)
        input_values = tuple([batch.to_dict()[k] for k in list_names])
        save_model_onnx(
            self.graph_mae, input_values, list_names, ["node_predictions"], directory
        )


class LitFinetuneGraphMAE(L.LightningModule):

    def __init__(self, output_features):
        super().__init__()
        self.num_layer = 5
        self.emb_dim = 300
        self.encoder = GNN(
            num_layer=self.num_layer, emb_dim=self.emb_dim, drop_ratio=0.5
        )
        self.pooling = MeanAggregation()
        self.prediction_head = torch.nn.Linear(self.emb_dim, output_features)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, batch: Data) -> torch.Any:
        pred, _ = self.shared_step(batch, batch.y)
        return pred

    def shared_step(self, data: Data, target: torch.Tensor):
        node_rep = self.encoder(data.x, data.edge_index, data.edge_attr)
        pred = self.prediction_head(self.pooling(node_rep, data.batch))

        is_not_nan = target == target
        loss = self.criterion(pred[is_not_nan], target[is_not_nan])

        return pred, loss

    def training_step(self, batch: Union[Data, torch.Tensor], batch_idx: int):
        pred, loss = self.shared_step(batch, batch.y)
        self.log(
            "train_loss", loss.cpu().item(), prog_bar=True, batch_size=pred.size(0)
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self.preds, self.target = [], []

    def validation_step(self, batch: Data, batch_idx):
        data, target = batch, batch.y
        pred, loss = self.shared_step(data, target)
        self.log(
            "val_loss", loss.cpu().item(), prog_bar=True, batch_size=target.size(0)
        )
        self.preds.append(pred)
        self.target.append(target)

        return loss

    def on_validation_epoch_end(self):
        roc_auc = self.compute_mask_roc_auc()
        self.log("val_roc_auc", roc_auc, prog_bar=True, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.preds, self.target = [], []

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        roc_auc = self.compute_mask_roc_auc()
        self.log("test_roc_auc", roc_auc, prog_bar=True, on_epoch=True)
        self.test_roc_auc = roc_auc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0)
        return optimizer

    def compute_mask_roc_auc(self):
        preds, target = (
            torch.concat(self.preds).cpu().numpy(),
            torch.concat(self.target).cpu().numpy(),
        )
        is_nan = target == target
        roc_auc = 0
        for i in range(target.shape[1]):
            mask = is_nan[:, i]
            roc_auc += roc_auc_score(target[mask, i], preds[mask, i])
        return roc_auc / target.shape[1]
