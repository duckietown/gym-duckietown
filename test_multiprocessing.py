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

        def update(dt):
            action = np.array([0.5, 0.0])

            obs, reward, done, info = env.step(action)
            print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

            env.render()

            if done:
                print('done!')
                #env.reset()
                #env.close()
                sys.exit(0)

        pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

        # Enter main event loop
        pyglet.app.run()

        env.close()


if __name__ == '__main__':
    num_workers = 2
    workers = [Worker(i) for i in range(num_workers)]
    [w.start() for w in workers]

    [w.join() for w in workers]

"""
env = gym.make(args.env_name)
env.reset()
env.render()


def update(dt):
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
"""
