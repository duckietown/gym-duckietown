from gym_duckietown.envs import SimpleSimEnv
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

env = SimpleSimEnv(draw_curve=False)

#num_samples = 10
#for i in range(num_samples):
#    obs = env.reset()
#    img = Image.fromarray(np.flipud((obs* 255)).astype('uint8'))
#    img.save(fp='representation_analysis/trajectories/{}.jpg'.format(i))
#    if i+1 % 1000 == 0:
#        print('got to sample {}'.format(i))
#env.close()


class DuckieDataset(Dataset):
    def __init__(self, len):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        obs = env.reset()
        return obs.transpose((2, 0, 1))
