import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Tuple, List


class MemoryMapDataset(Dataset):
    """Dataset to store multiple arrays on disk to avoid saturating the RAM"""
    def __init__(self, size: int, data_size: tuple, target_size: tuple, path: str):
        """
        Parameters
        ----------
        size : int
            Number of arrays to store, will be the first dimension of the resulting tensor dataset.
        data_size : tuple
            Size of data, may be 3D (CxHxW) for images or 2D/1D for features.
        target_size : tuple
            Size of the target, for our case it is a 1D array having, angular and linear speed.
        path : str
            Path where the file will be saved.
        """
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
        self.real_length = 0

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get one pair of training examples from the dataset.

        Parameters
        ----------
        item : int
            Index on the first dimension of the dataset.

        Returns
        -------
        sample, target : tuple
            Training sample consisting of data, label of data_size and target_size, respectively.
        """
        sample = torch.tensor(self.data[item, ...])
        target = torch.tensor(self.target[item, ...])

        return sample, target

    def __len__(self) -> int:
        """Get size (number of saved examples) of the dataset.

        Returns
        -------
        length : int
            Occupied length of the dataset. Note that it returns the number of saved examples rather than the maximum
            size used in the initialization.
        """
        return self.length

    def extend(self, observations: List[np.ndarray], actions: List[np.ndarray]):
        """Saves observations to the dataset. Iterates through the lists containing matching pairs of observations and
        actions. After saving each sample the dataset size is readjusted. If the dataset exceeds its maximum size
        it will start overwriting the firs experiences.

        Parameters
        ----------
        observations : List
            List containing np.ndarray observations of size data_size.
        actions
            List containing np.ndarray actions of size target_size.
        """
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
        """In case of wanting to save the dataset this method should be implemented by flushing anc closing the memory
        map. Note that the files (depending on the size parameter) may occupy considerable amount of memory.
        """
        # TODO
        pass
