from typing import Union, List, Optional, Callable

import torch.utils.data
from torch_geometric.data import Data, HeteroData, Dataset, Batch

class TransformsCollater:
    def __init__(self, follow_batch, exclude_keys, transforms):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.transforms = transforms

    def __call__(self, batch):

        if self.transforms is not None:
            batch = self.transforms(batch)

        return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)


class TransformsDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, List[Data], List[HeteroData]],
        batch_size: int = 1,
        shuffle: bool = False,
        transforms: Callable = None,
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
            collate_fn=TransformsCollater(follow_batch, exclude_keys, transforms),
            **kwargs,
        )
