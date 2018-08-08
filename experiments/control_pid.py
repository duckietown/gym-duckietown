#!/usr/bin/env python3

"""
Control the simulator or Duckiebot using a a PID controller (heuristic).
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

while True:

    follow_dist = 0.4

    # Find the curve point closest to the agent, and the tangent at that point
    closest_point, closest_tangent = env.closest_curve_point(env.cur_pos)

    while True:
        # Project a point ahead along the curve tangent,
        # then find the closest point to to that
        follow_point = closest_point + closest_tangent * follow_dist
        curve_point, _ = env.closest_curve_point(follow_point)

        # If we have a valid point on the curve, stop
        if curve_point is not None:
            break

        follow_dist *= 0.5

    # Compute a normalized vector to the curve point
    point_vec = curve_point - env.cur_pos
    point_vec /= np.linalg.norm(point_vec)

    dot = np.dot(env.get_right_vec(), point_vec)
    velocity = 0.35
    steering = 2 * -dot

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
