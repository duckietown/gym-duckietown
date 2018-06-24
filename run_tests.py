#!/usr/bin/env python3

import os
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv, MultiMapEnv

env = gym.make('SimpleSim-v0')

first_obs = env.reset()

# Check that the human rendering resembles the agent's view
first_render = env.render('rgb_array')
m0 = first_obs.mean()
m1 = first_render.mean()
assert m0 > 0 and m0 < 255
assert abs(m0 - m1) < 5

# Try stepping a few times
for i in range(0, 10):
    obs, _, _, _ = env.step(np.array([0.1, 0.099]))

# Try loading each of the available map files
for map_file in os.listdir('gym_duckietown/maps'):
    map_name = map_file.split('.')[0]
    env = SimpleSimEnv(map_name=map_name)

# Test the multi-map environment
env = MultiMapEnv()
