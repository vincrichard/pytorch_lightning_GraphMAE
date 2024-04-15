from functools import partial

import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from src.featurizer import SimpleGraph2dFeaturizer, RandomAtomMask


class Zinc250(Dataset):
    BASE_PATH = "/nasa/shared_homes/vincent/sandbox/GraphMAE/"
    # ZINC_PATH = f"{BASE_PATH}src/datasets/zinc250k.csv"
    ZINC_PATH = f"{BASE_PATH}dataset/zinc_standard_agent/processed/smiles.csv"

    def __init__(self):
        super().__init__()
        # self.smiles = pd.read_csv(self.ZINC_PATH)["smiles"].tolist()
        self.smiles = pd.read_csv(self.ZINC_PATH, header=None)[0].tolist()
        # self.featurizer = mol_to_graph_data_obj_simple#SimpleGraph2dFeaturizer()
        self.featurizer = SimpleGraph2dFeaturizer()
        self.masking_strategy = RandomAtomMask(prob=0.25)
        # self.masking_strategy = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = 0.25, mask_edge=0)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.masking_strategy(self.featurizer(self.smiles[idx]))


Zinc250_DataLoader = partial(DataLoader, Zinc250())
