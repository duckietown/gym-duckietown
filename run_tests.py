#!/usr/bin/env python3

import os
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv, MultiMapEnv
from gym_duckietown.simulator import get_agent_corners
from gym_duckietown.wrappers import PyTorchObsWrapper

env = gym.make("Duckietown-udem1-v0")
env.reset()
# Try stepping a few times
for i in range(0, 10):
    obs, _, _, _ = env.step(np.array([0.1, 0.1]))

# Check that the human rendering resembles the agent's view
first_obs = env.reset()
first_render = env.render("rgb_array")
m0 = first_obs.mean()
m1 = first_render.mean()
assert 0 < m0 < 255
assert abs(m0 - m1) < 5

# Check that the observation shapes match
second_obs, _, _, _ = env.step([0.0, 0.0])
assert first_obs.shape == env.observation_space.shape
assert first_obs.shape == second_obs.shape

# Test the PyTorch observation wrapper
env = PyTorchObsWrapper(env)
first_obs = env.reset()
second_obs, _, _, _ = env.step([0, 0])
assert first_obs.shape == env.observation_space.shape
assert first_obs.shape == second_obs.shape

# Try loading each of the available map files
for map_name in ["loop_only_duckies", "small_loop_only_duckies"]:
    env = DuckietownEnv(map_name=map_name)
    env.reset()

# Test the multi-map environment
env = MultiMapEnv()
for i in range(0, 50):
    env.reset()

# Check that we do not spawn too close to obstacles
env = DuckietownEnv(map_name="loop_obstacles")
for i in range(0, 75):
    obs = env.reset()
    assert not env._collision(get_agent_corners(env.cur_pos, env.cur_angle)), "collision on spawn"
    env.step(np.array([0.05, 0]))
    assert not env._collision(get_agent_corners(env.cur_pos, env.cur_angle)), "collision after one step"

# Test the draw_bbox mode
env = DuckietownEnv(map_name="udem1", draw_bbox=True)
env.render("rgb_array")
env = DuckietownEnv(map_name="loop_obstacles", draw_bbox=True)
env.render("rgb_array")
