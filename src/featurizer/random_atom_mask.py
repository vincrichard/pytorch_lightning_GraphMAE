import random
from copy import deepcopy
from typing import Optional
import math

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


class RandomAtomMask:
    TOKEN_VALUE = 119

    def __init__(self, prob: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0. If
                  a list is passed in, the value will be randomly and sampled between the two
                  end points.
        @param input_field: the name of the generated input containing the smile information
            should be a PyG_Data object
        """
        if isinstance(prob, list):
            assert 0 <= prob[0] <= 1.0
            assert 0 <= prob[1] <= 1.0
            assert prob[0] < prob[1]
        else:
            assert 0 <= prob <= 1.0, "p must be a value in the range [0, 1]"
        self.prob = prob

    def __call__(self, data: Data, seed=None) -> Data:
        """
        @param mol_graph: PyG Data to be augmented
        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned
        @returns: Augmented PyG Data
        """
        if isinstance(self.prob, list):
            self.p = random.uniform(self.prob[0], self.prob[1])
        else:
            self.p = self.prob

        assert isinstance(self.p, (float, int))
        assert isinstance(data, Data), "mol_graph passed in must be a PyG Data"
        output = self.apply_transform(data, seed)
        return output

    def apply_transform(self, data: Data, seed: Optional[None] = None) -> Data:
        """
        Transform that randomly mask atoms given a certain ratio
        @param mol_graph: PyG Data to be augmented
        @param seed:
        @returns: Augmented PyG Data
        """
        if seed is not None:
            random.seed(seed)
        num_atoms = data.x.size(0)
        num_mask_nodes = max([1, math.floor(self.p * num_atoms)])
        mask_nodes = sorted(random.sample(list(range(num_atoms)), num_mask_nodes))

        mask_data = deepcopy(data)
        # mask_data.x[mask_nodes, :] = torch.ones(len(mask_data.x[mask_nodes, :])) * self.TOKEN_VALUE

        # Graph-MAE apply only to the atom indice
        # Note: Not sure what they set the chirality at 0 for the mask
        mask_data.x[mask_nodes, :] = torch.tensor([self.TOKEN_VALUE, 0])
        mask_data.masked_atom_mask = torch.zeros(num_atoms, dtype=torch.bool)
        mask_data.masked_atom_mask[torch.tensor(mask_nodes)] = 1
        mask_data.node_attr_label = F.one_hot(
            data.x[mask_nodes, 0], num_classes=119
        ).float()

        return mask_data
