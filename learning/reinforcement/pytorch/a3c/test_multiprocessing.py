#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse

import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


class Worker(mp.Process):
    def __init__(self, s):
        super(Worker, self).__init__()
        self.s = s

    def run(self):
        import pyglet
        import numpy as np
        import gym

        print('Starting thread: ', self.s)
        env = gym.make('Duckietown-udem1-v0')
        env.reset()
        env.render()

        # def update(dt):
        #     action = np.array([0.5, 0.0])
        #
        #     obs, reward, done, info = env.step(action)
        #     print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))
        #
        #     env.render()
        #
        #     if done:
        #         print('done!')
        #         #env.reset()
        #         #env.close()
        #         sys.exit(0)
        #
        # pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

        done = False
        while not done:
            action = np.array([0.5, 0.0])

            obs, reward, done, info = env.step(action)
            print('Process = %s, step_count = %s, reward=%.3f' % (self.s, env.unwrapped.step_count, reward))

            env.render()

            if done:
                print('done!')
                # env.reset()
                # env.close()
                sys.exit(0)

        env.close()


if __name__ == '__main__':
    num_workers = 2
    workers = [Worker(i) for i in range(num_workers)]
    [w.start() for w in workers]

    [w.join() for w in workers]
