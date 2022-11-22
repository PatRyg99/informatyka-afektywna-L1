from typing import Union, List, Optional

import torch.utils.data
from torch_geometric.data import Data, HeteroData, Dataset, Batch

class InterpolateCollater:
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):

        for i in range(len(batch)):
            x1, x2 = batch[i], batch[(i + 1) % len(batch)]

            delta = x2.pos - x1.pos
            x1.pos += 0.2 * delta

        return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)


class InterpolateDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, List[Data], List[HeteroData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=InterpolateCollater(follow_batch, exclude_keys),
            **kwargs,
        )
