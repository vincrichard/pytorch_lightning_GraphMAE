import pandas as pd
from torch.utils.data import Dataset

from src.featurizer import SimpleGraph2dFeaturizer, RandomAtomMask


class Zinc(Dataset):

    def __init__(self, path):
        super().__init__()
        self.smiles = pd.read_csv(path, header=None)[0]
        self.featurizer = SimpleGraph2dFeaturizer()
        self.masking_strategy = RandomAtomMask(prob=0.25)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.masking_strategy(self.featurizer(self.smiles[idx]))
