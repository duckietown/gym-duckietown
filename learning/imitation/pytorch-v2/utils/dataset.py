import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Tuple, List



class MemoryMapDataset(Dataset):
    """Dataset to store multiple arrays on disk avoiding saturating the RAM"""
    def __init__(self, size: int, data_size: tuple, target_size: tuple, path: str):
        self.size = size
        self.data_size = data_size
        self.target_size = target_size
        self.path = path

        # Path for each array
        self.data_path = os.path.join(path, 'data.dat')
        self.target_path = os.path.join(path, 'target.dat')

        # Create arrays
        self.data = np.memmap(self.data_path, dtype='float32', mode='w+', shape=(self.size, *self.data_size))
        self.target = np.memmap(self.target_path, dtype='float32', mode='w+', shape=(self.size, *self.target_size))

        # Initialize number of saved records to zero
        self.length = 0

        # keep track of real length in case of bypassing size value
        self.real_length =0

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.tensor(self.data[item, ...])
        target = torch.tensor(self.target[item, ...])

        return sample, target

    def __len__(self) -> int:
        return self.length

    def extend(self, observations: List[np.ndarray], actions: List[np.ndarray]):

        for index, (observation, action) in enumerate(zip(observations, actions)):
            current_data_indx = self.real_length + index
            if self.real_length + index >= self.size:
                # it will be a circular by getting rid of old experiments
                current_data_indx %= self.size 
            self.data[current_data_indx, ...] = observation.astype(np.float32)
            self.target[current_data_indx, ...] = action.astype(np.float32)
        if self.real_length >= self.size:    
            self.length = self.size - 1
        else:
            self.length += len(observations)
        self.real_length += len(observations)

    def save(self):
        # TODO
        pass
