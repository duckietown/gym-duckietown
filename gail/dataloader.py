from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ExpertTrajDataset(Dataset):
    
    def __init__(self,args, transform=None):
        
        self.file_dir = args.data_directory
#         self.transform = transform
        self.episodes = args.episodes
        
        
    def __len__(self):
        return self.episodes
    
    def __getitem__(self,idx):
        if torch .is_tensor(idx):
            idx = idx.tolist()
            
        observation = torch.load('{}/data_o_{}.pt'.format(self.file_dir, idx))   
        action = torch.load('{}/data_a_{}.pt'.format(self.file_dir,idx))   

        sample = {'observation': observation, 'action':action}
        
        return sample
    

    
