#!/usr/bin/env python3

import os
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv, MultiMapEnv

env = gym.make('Duckietown-udem1-v0')

# Check that the human rendering resembles the agent's view
first_obs = env.reset()
first_render = env.render('rgb_array')
m0 = first_obs.mean()
m1 = first_render.mean()
assert m0 > 0 and m0 < 255
assert abs(m0 - m1) < 5

# Try stepping a few times
for i in range(0, 10):
    obs, _, _, _ = env.step(np.array([0.1, 0.1]))

# Try loading each of the available map files
for map_file in os.listdir('gym_duckietown/maps'):
    map_name = map_file.split('.')[0]
    env = DuckietownEnv(map_name=map_name)
    env.reset()

# Test the multi-map environment
env = MultiMapEnv()
for i in range(0, 50):
    env.reset()

# Check that we do not spawn too close to obstacles
env = DuckietownEnv(map_name='loop_obstacles')
for i in range(0, 75):
    obs = env.reset()
    assert not env._collision(), "collision on spawn"
    env.step(np.array([0.05, 0]))
    assert not env._collision(), "collision after one step"

# Test the draw_bbox mode
env = DuckietownEnv(map_name='udem1', draw_bbox=True)
env.render('rgb_array')
env = DuckietownEnv(map_name='loop_obstacles', draw_bbox=True)
env.render('rgb_array')
