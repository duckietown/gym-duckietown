import cv2
import math
import numpy as np
import gym
from gym import spaces

class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # Turn left
        if action == 0:
            vels = [0.6, +1.0]
        # Turn right
        elif action == 1:
            vels = [0.6, -1.0]
        # Go forward
        elif action == 2:
            vels = [0.7, 0.0]
        else:
            assert False, "unknown action"
        return np.array(vels)

class PyTorchObsWrapper(gym.ObservationWrapper):
    """
    Transpose the observation image tensors for PyTorch
    """

    def __init__(self, env=None):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[1,1,1],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return observation.transpose(2, 1, 0)


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, resize_w=80, resize_h=80):
        super().__init__(env)
        self.resize_h = resize_h
        self.resize_w = resize_w
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[1, 1, 1],
            [obs_shape[0], resize_h, resize_w],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 1, 0)

    def reset(self):
        obs = super().reset()
        return cv2.resize(obs.swapaxes(0,2), dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC).swapaxes(0,2)

    def step(self, actions):
        obs, reward, done, info = super().step(actions)
        return cv2.resize(obs.swapaxes(0,2), dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC).swapaxes(0,2), reward, done, info

