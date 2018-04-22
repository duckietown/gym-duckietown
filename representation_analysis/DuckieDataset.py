from gym_duckietown.envs import SimpleSimEnv
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch

#num_samples = 10
#for i in range(num_samples):
#    obs = env.reset()
#    img = Image.fromarray(np.flipud((obs* 255)).astype('uint8'))
#    img.save(fp='representation_analysis/trajectories/{}.jpg'.format(i))
#    if i+1 % 1000 == 0:
#        print('got to sample {}'.format(i))
#env.close()


class DuckieDataset(Dataset):
    def __init__(self, len, batch_size):
        self.len = len
        self.batch_size = batch_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        env = SimpleSimEnv(draw_curve=False)
        observs = torch.zeros(self.batch_size, 3, 120, 160)
        for i in range(self.batch_size):
            obs = env.reset()
            observs[i,:,:,:] = torch.FloatTensor(obs.transpose((2, 0, 1)))
        return observs
