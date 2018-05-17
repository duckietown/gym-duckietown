#!/usr/bin/env python3

import numpy as np
import gym
import gym_duckietown

env = gym.make('SimpleSim-v0')
env.reset()

obs, _, _, _ = env.step(np.array([0.1, 0.1]))
