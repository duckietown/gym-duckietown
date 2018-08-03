#!/usr/bin/env python3

"""
Control the simulator or Duckiebot using a model trained with imitation
learning, and visualize the result.
"""

import time
import sys
import argparse
import math
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--no-random', action='store_true', help='disable domain randomization')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        domain_rand = not args.no_random
    )
    env.max_steps = 500
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

while True:

    # Positive distance means right side of the lane
    # Positive angle means turning right
    dist, dot_dir, angle = env.unwrapped.get_lane_pos()
    abs_dist = abs(dist)
    abs_angle = abs(angle)

    print(dist, dot_dir, angle)

    #steering = 0


    steering = 0
    if dist < -0.2:
        steering = 1




    velocity = 0.6
    obs, reward, done, info = env.step([velocity, steering])
    #print('stepCount = %s, reward=%.3f' % (env.stepCount, reward))

    env.render()

    if done:
        if reward < 0:
            print('*** FAILED ***')
            if not args.no_pause:
                time.sleep(0.7)
        obs = env.reset()
        env.render()
