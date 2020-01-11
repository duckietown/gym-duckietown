#!/usr/bin/env python3

"""
Simple exercise to construct a controller that controls the simulated Duckiebot using pose. 
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

total_reward = 0

while True:

        pos=env.cur_pos;
        angle=env.cur_angle;
        
        closest_curve_point = env.unwrapped.closest_curve_point
        
        # Find the curve point closest to the agent, and the tangent at that point
        closest_point, closest_tangent = closest_curve_point(pos, angle)

        iterations = 0
        
        lookup_distance = 0.25
        multiplier = 0.5
        curve_point = None
        
        while iterations < 10:            
            ########
            #
            #TODO 1: Modify follow_point so that it is a function of closest_point, closest_tangent, and lookup_distance
            #
            ########
            follow_point = closest_point + closest_tangent * lookup_distance
            
            curve_point, _ = closest_curve_point(follow_point, angle)
            
            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break
            iterations += 1
            lookup_distance *= multiplier
        ########
        #
        #TODO 2: Modify omega
        #
        ########
        L = curve_point - pos
        cos = np.cos(angle)
        sin = np.sin(angle)
        rot = np.array([[cos, -sin], [sin, cos]])
        L_r = np.dot([L[2], L[0]], rot)
        direction = np.arctan2(L_r[0] / np.linalg.norm(L_r), L_r[1] / np.linalg.norm(L_r))
        
        speed = 0.5
        steering = -2 * speed * np.sin(direction) / np.linalg.norm(L)

        obs, reward, done, info = env.step([speed, steering])

        total_reward += reward
        
        print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

        env.render()

        if done:
            if reward < 0:
                print('*** CRASHED ***')
                print ('Final Reward = %.3f' % total_reward)
            break
