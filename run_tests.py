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

# Check that we do not spawn too close to obstacles
env = SimpleSimEnv(map_name='loop_obstacles')
for i in range(0, 75):
    obs = env.reset()
    assert not env._collision(), "collision on spawn"
    env.step(np.array([0.1, 0.1]))
    assert not env._collision(), "collision after one step"

# Test the draw_bbox mode
env = SimpleSimEnv(map_name='udem1', draw_bbox=True)
env.render('rgb_array')
env = SimpleSimEnv(map_name='loop_obstacles', draw_bbox=True)
env.render('rgb_array')
