#!/usr/bin/env python3

"""
Exercise: Build a controller to control the Duckiebot in simulation using the ground truth pose
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
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        domain_rand = False,
        draw_bbox = False
    )
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

total_reward=0

while True:

    lane_pose = env.get_lane_pos(env.cur_pos, env.cur_angle)
    distance_to_road_center = lane_pose.dist
    angle_from_straight_in_rads = lane_pose.angle_rad

    ###### START FILL IN CODE HERE.
    # TODO: Figure out how to compute the velocity and steering

    # Velicty is a value between 0 and 1 (corresponds to actual speed of 0 to 1.2m/s)
    velocity = 0.1 # You should overwrite this value
    # the angle of the steering wheel i.e. the change in angle of the car in rads/s
    steering = 0.1 # You should overwrite this value

    ###### END FILL IN CODE HERE
    
    obs, reward, done, info = env.step([velocity, steering])
    total_reward += reward
    
    print('step number = %s, one step reward=%.3f, accumulated reward=%.3f' % (env.step_count, reward, total_reward))

    env.render()

    if done:
        if reward < 0:
            print('*** CRASHED ***')
        print ('final reward = %.3f' % total_reward)
        break;
