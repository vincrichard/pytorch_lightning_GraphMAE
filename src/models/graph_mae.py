import torch
from torch import optim
from torch_geometric.data import Data
import lightning as L

from src.models.gnn import GNN, GNNDecoder
from src.criterion import sce_loss
from src.utils.onnx_export import save_model_onnx


class GraphMAE(torch.nn.Module):
    """For inference outside Pytorch Lightning"""

    def __init__(self):
        super().__init__()
        self.encoder = GNN(num_layer=5, emb_dim=500, drop_ratio=0)
        self.decoder = GNNDecoder(hidden_dim=500, out_dim=119)

    def forward(self, x, edge_index, edge_attr, masked_atom_mask):
        node_rep = self.encoder(x, edge_index, edge_attr)
        # masking of the representation is done in the decoder
        return self.decoder(node_rep, edge_index, edge_attr, masked_atom_mask)


# define the LightningModule
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

        node_rep = self.graph_mae.encoder(batch.x, batch.edge_index, batch.edge_attr)
        # masking of the representation is done in the decoder
        pred_node = self.graph_mae.decoder(
            node_rep, batch.edge_index, batch.edge_attr, masked_atom_mask
        )

        loss = self.criterion(batch.node_attr_label, pred_node[masked_atom_mask])

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss.cpu().item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        # trainer.fit(model, train_loader, valid_loader)
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        # this is the test loop
        # trainer.test(model, dataloaders=DataLoader(test_set))
        raise NotImplementedError

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
