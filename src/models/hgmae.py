import torch
from torch import optim
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, add_self_loops, to_dense_adj
import lightning as L

from src.models.gnn import GINConv, GNN
from src.criterion import sce_loss
from src.utils.onnx_export import save_model_onnx


class HGMAE(torch.nn.Module):
    """For inference outside Pytorch Lightning"""

    def __init__(self):
        super().__init__()
        self.encoder = GNN(num_layer=5, emb_dim=500, drop_ratio=0)
        self.node_encoder_to_decoder = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Linear(in_features=500, out_features=500, bias=False),
        )
        self.edge_encoder_to_decoder = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Linear(in_features=500, out_features=500, bias=False),
        )
        self.decoder = GINConv(emb_dim=500, out_dim=119, aggr="add")

    def forward(self, x, edge_index, edge_attr, masked_atom_mask):

        node_rep = self.encoder(x, edge_index, edge_attr)

        decoder_input = self.encoder_to_decoder(node_rep)
        decoder_input[masked_atom_mask] = 0

        return self.decoder(decoder_input, edge_index, edge_attr)


class LitHGMAE(L.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()
        self.hgmae = HGMAE()
        self.criterion = sce_loss
        self.save_hyperparameters()

    def training_step(self, batch: Data, batch_idx: int):

        node_loss = self.get_loss_node_reconstruction(batch)
        edge_loss = self.get_loss_edge_reconstruction(batch)

        # Logging to TensorBoard (if installed) by default
        self.log("train_edge_loss", edge_loss.cpu().item(), prog_bar=True)
        self.log("train_node_loss", node_loss.cpu().item(), prog_bar=True)

        loss = node_loss + 0.5 * edge_loss

        self.log("train_loss", loss.cpu().item(), prog_bar=True)

        return loss

    def get_loss_node_reconstruction(self, batch: Data):
        masked_atom_mask = batch.masked_atom_mask

        # Masking latent space
        node_rep = self.hgmae.encoder(batch.x, batch.edge_index, batch.edge_attr)
        decoder_input = self.hgmae.node_encoder_to_decoder(node_rep)
        decoder_input[masked_atom_mask] = 0

        pred_node = self.hgmae.decoder(decoder_input, batch.edge_index, batch.edge_attr)

        return self.criterion(batch.node_attr_label, pred_node[masked_atom_mask])

    def get_loss_edge_reconstruction(self, batch: Data):
        # Create mask data
        _, retain_mask = dropout_edge(batch.edge_index, p=0.5)
        mask_data = batch.clone()
        mask_data.edge_index = mask_data.edge_index[:, retain_mask]
        mask_data.edge_attr = mask_data.edge_attr[retain_mask]

        # Add self loop (present in the implementation)
        mask_data.edge_index, mask_data.edge_attr = add_self_loops(
            mask_data.edge_index,
            edge_attr=mask_data.edge_attr,
            fill_value=1,
            num_nodes=batch.x.shape[0],
        )

        # predicting node feat
        node_rep = self.hgmae.encoder(
            mask_data.x, mask_data.edge_index, mask_data.edge_attr
        )
        decoder_edge_input = self.hgmae.edge_encoder_to_decoder(node_rep)

        edge_prediction = self.hgmae.decoder(
            decoder_edge_input, mask_data.edge_index, mask_data.edge_attr
        )
        edge_prediction = torch.mm(edge_prediction, edge_prediction.T)

        return self.criterion(edge_prediction, to_dense_adj(batch.edge_index))

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
