from gym_duckietown.envs import SimpleSimEnv
from PIL import Image
import numpy as np

env = SimpleSimEnv(draw_curve=False)
num_samples = 1000
latent_factors = np.zeros(shape=[num_samples, 17])

for i in range(num_samples):
    obs = env.reset()
    latent_factors[i, :] = np.array([env.camAngle, env.camHeight, env.camFovY, env.curAngle,
                                     env.curPos[0], env.curPos[1], env.curPos[2],
                                     env.groundColor[0], env.groundColor[1], env.groundColor[2],
                                     env.horizonColor[0], env.horizonColor[1], env.horizonColor[2],
                                     env.roadColor[0], env.roadColor[1], env.roadColor[2],
                                     env.wheelDist])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/data/test_trajectories/{}.jpg'.format(i))
    if i+1 % 1000 == 0:
        print('got to sample {}'.format(i))
env.close()
np.save('representation_analysis/data/latent_factors', latent_factors)
