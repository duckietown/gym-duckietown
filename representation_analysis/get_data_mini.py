from gym_duckietown.envs import SimpleSimEnv
from PIL import Image
import numpy as np

env = SimpleSimEnv(draw_curve=False)
num_samples = 10

for i in range(num_samples):
    obs = env.reset()
    img = Image.fromarray(np.flipud((obs* 255)).astype('uint8'))
    img.save(fp='representation_analysis/trajectories/{}.jpg'.format(i))
    if i+1 % 1000 == 0:
        print('got to sample {}'.format(i))
env.close()