#!/usr/bin/env python3

import numpy as np
import gym
import gym_duckietown

env = gym.make('SimpleSim-v0')

first_obs = env.reset()

# Check that the human rendering resembles the agent's view
first_render = env.render('rgb_array')
m0 = first_obs.mean()
m1 = first_render.mean()
assert m0 > 0 and m0 < 255
assert abs(m0 - m1) < 5

obs, _, _, _ = env.step(np.array([0.1, 0.1]))
