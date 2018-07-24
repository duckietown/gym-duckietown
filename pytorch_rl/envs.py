import os
import random

import numpy as np

import gym
from gym.spaces.box import Box

import gym_duckietown
from gym_duckietown.envs import *

def make_env(env_id, seed, rank, log_dir, start_container):
    def _thunk():
        env = gym.make(env_id)
        env = DiscreteWrapper(env)

        env.seed(seed + rank)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape

        if len(obs_shape) == 3 and obs_shape[2] == 3:
            env = WrapPyTorch(env)

        env = ScaleObservations(env)

        return env

    return _thunk

class ScaleObservations(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ScaleObservations, self).__init__(env)
        self.obs_lo = self.observation_space.low[0,0,0]
        self.obs_hi = self.observation_space.high[0,0,0]
        obs_shape = self.observation_space.shape
        self.observation_space = Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=np.float32
        )

    def observation(self, observation):
        return observation.transpose(2, 1, 0)
