import os
import numpy as np
import gym

from .duckietown_env import DuckietownEnv
from ..utils import get_subdir_path

class MultiMapEnv(gym.Env):
    """
    Environment which samples from multiple environments, for
    multi-taks learning
    """

    def __init__(self, **kwargs):
        self.env_list = []

        maps_dir = get_subdir_path('maps')

        self.window = None

        # Try loading each of the available map files
        for map_file in os.listdir(maps_dir):
            map_name = map_file.split('.')[0]

            # Do not load the regression test maps
            if map_name.startswith('regress'):
                continue

            env = DuckietownEnv(map_name=map_name, **kwargs)

            self.action_space = env.action_space
            self.observation_space = env.observation_space
            self.reward_range = env.reward_range

            self.env_list.append(env)

        assert len(self.env_list) > 0

        self.cur_env_idx = 0
        self.cur_reward_sum = 0
        self.cur_num_steps = 0

    def seed(self, seed):
        for env in self.env_list:
            env.seed(seed)

        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        return [seed]

    def reset(self):
        #self.cur_env_idx = self.np_random.randint(0, len(self.env_list))
        self.cur_env_idx = (self.cur_env_idx + 1) % len(self.env_list)

        env = self.env_list[self.cur_env_idx]
        return env.reset()

    def step(self, action):
        env = self.env_list[self.cur_env_idx]

        obs, reward, done, info = env.step(action)

        # Keep track of the total reward for this episode
        self.cur_reward_sum += reward
        self.cur_num_steps += 1

        # If the episode is done, sample a new environment
        if done:
            self.cur_reward_sum = 0
            self.cur_num_steps = 0

        return obs, reward, done, info

    def render(self, mode='human', close=False):
        env = self.env_list[self.cur_env_idx]

        # Make all environments use the same rendering window
        if self.window is None:
            ret = env.render(mode, close)
            self.window = env.window
        else:
            env.window = self.window
            ret = env.render(mode, close)

        return ret

    def close(self):
        for env in self.env_list:
            env.close()

        self.cur_env_idx = 0
        self.env_names = None
        self.env_list = None

    @property
    def step_count(self):
        env = self.env_list[self.cur_env_idx]
        return env.step_count
