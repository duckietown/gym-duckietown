import math
import os
import random

import numpy as np

from gym_duckietown.envs import DuckietownEnv
import math
import numpy as np
from learning.imitation.iil_dagger.teacher.pure_pursuit_policy import PurePursuitPolicy

def seed(seed):
    #torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
seed(random.randint(0, 9999999))

def to_image(np_array):
    from PIL import Image
    img = Image.fromarray(np_array, 'RGB')
    img.show()
    i = 0

os.chdir("./src/gym_duckietown")

environment = DuckietownEnv(
        domain_rand=False,
        max_steps=math.inf,
        randomize_maps_on_reset=False,
        map_name="loop_obstacles"
    )

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

while True:
    obs = environment.reset()
    environment.render()
    rewards = []

    nb_of_steps = 0

    while True:
        action = list(policy.predict(np.array(obs)))
        action[1]*=7

        obs, rew, done, misc = environment.step(action)
        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        #to_image(obs)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
    print("mean episode reward:", np.mean(rewards))

environment.close()