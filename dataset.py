import numpy as np

import torch

from functools import reduce
from operator import iadd

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.utils import collate_feature_dict


class SberDataset(MemoryMapDataset):
    def __init__(self, data, i_filters, splitter):
        super().__init__(data, i_filters)

        self.col_time = 'event_time'
        self.splitter = splitter

    def __getitem__(self, idx):
        feature_arrays = self.processed_data[idx]
        return self.get_splits(feature_arrays)

    def get_splits(self, feature_arrays):
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        return [{k: v[ix] for k, v in feature_arrays.items() if self.is_seq_feature(k, v)} | {'gender': feature_arrays['gender']} for ix in indexes]

    @staticmethod
    def collate_fn(batch):
        class_labels = [sample['gender'] for i, samples in enumerate(batch) for sample in samples]
        batch = reduce(iadd, batch)
        padded_batch = collate_feature_dict(batch)

        return padded_batch, torch.LongTensor(class_labels)
    
    @staticmethod
    def is_seq_feature(k: str, x):
        """Check is value sequential feature
        Synchronized with ptls.data_load.padded_batch.PaddedBatch.is_seq_feature

        Iterables are:
            np.array
            torch.Tensor

        Not iterable:
            list    - dont supports indexing

        Parameters
        ----------
        k:
            feature_name
        x:
            value for check

        Returns
        -------
            True if value is iterable
        """
        if k == 'event_time':
            return True
        if k.startswith('target'):
            return False
        if type(x) in (np.ndarray, torch.Tensor):
            return True
        return False
