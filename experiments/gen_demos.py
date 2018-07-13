#!/usr/bin/env python3

"""
This script generates fake/synthetic demonstrations (trajectories) by
mutating sequences of actions to maximize the total reward obtained.
The resulting sequences of actions are saved to a JSON file.
"""

import time
import random
import argparse
import math
import json
import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--map-name', required=True)
parser.add_argument('--demo-len', default=30, type=int, help='length of demonstrations to be generated')
parser.add_argument('--tail-len', default=30, type=int, help='extra actions at the end of trajectories, cut out of demonstrations')
parser.add_argument('--num-itrs', default=750, type=int)
args = parser.parse_args()

def gen_actions(seq_len):
    actions = []

    for i in range(0, seq_len):
        vels = np.random.uniform([0.4, -1.0], [1.0, 1.0])
        actions.append(vels)

    return actions

# TODO: try biasing mutations to end
# TODO: try mutating by small adjustments
def mutate_actions(actions):
    actions = actions[:]

    for i in range(0, len(actions)):
        if np.random.uniform(0, 1) < (1 / len(actions)):
            vels = np.random.uniform([0.4, -1.0], [1.0, 1.0])
            actions[i] = vels

        if np.random.uniform(0, 1) < (1 / len(actions)):
            vels = actions[i] + np.random.uniform(low=-0.1, high=0.1, size=(2,))
            vels = vels.clip(0.4, 0.9)
            actions[i] = vels

    return actions

def eval_actions(env, seed, actions):
    env.seed(seed)
    env.reset()

    total_reward = 0

    positions = []

    for i in range(0, len(actions)):
        vels = actions[i]

        positions.append((env.unwrapped.cur_pos, env.unwrapped.cur_angle))

        env.unwrapped.graphics = False
        obs, reward, done, info = env.step(vels)
        env.unwrapped.graphics = True

        total_reward += reward

        if done:
            break

    return positions, total_reward

def gen_trajectory(env, seed, num_actions, num_itrs):
    """
    Generate a sequence of actions trying to maximize the
    total reward obtained
    """

    best_r = -math.inf
    best_len = 0
    best_actions = None
    best_positions = None

    # Do an initial search to try to find a trajectory
    # along which the agent doesn't fail
    for i in range(0, 150):
        new_actions = gen_actions(num_actions)
        positions, r = eval_actions(env, seed, new_actions)
        t_len = len(positions)

        if t_len > best_len or (t_len == best_len and r > best_r):
            best_r = r
            best_len = t_len
            best_actions = new_actions
            best_positions = positions

    # If we couldn't find a working trajectory from this seed, abort
    if best_len < num_actions:
        print('abort')
        return best_positions, best_actions

    # Iteratively mutate and improve the trajectory
    for itr in range(1, num_itrs+1):
        new_actions = mutate_actions(best_actions)
        positions, r = eval_actions(env, seed, new_actions)

        if r > best_r:
            best_r = r
            best_actions = new_actions
            best_positions = positions
            print('iteration #%d, r=%f' % (itr, r))

    print('r=%f' % best_r)

    return best_positions, best_actions

env = SimpleSimEnv(map_name=args.map_name)
env = HeadingWrapper(env)

demos = []
total_steps = 0

while True:
    seed = random.randint(0, 0xFFFFFFFF)
    p, a = gen_trajectory(env, seed, args.demo_len + args.tail_len, args.num_itrs)

    print('trajectory length: %d' % len(p))

    # If the agent did not fall off the road
    if len(p) == args.demo_len + args.tail_len:
        # Drop the last few actions, because the agent behaves more
        # greedily in the last steps (doesn't maximize future reward)
        p = p[:-args.tail_len]
        a = a[:-args.tail_len]

        # Convert numpy array to plain Python lists so we can store
        # the data in a JSON file
        #
        # Each position has the form [ [x,y,z], angle ]
        # Each action has the form [v0,v1]
        p = list(map(lambda p: [ p[0].tolist(), p[1] ], p))
        a = list(map(lambda a: a.tolist(), a))

        demo = {
            'positions': p,
            'actions': a
        }

        demos.append(demo)
        total_steps += len(p)

        # Store the trajectories in a JSON file
        with open('experiments/demos_%s.json' % args.map_name, 'w') as outfile:
            json.dump({ 'demos': demos }, outfile)

    print('total num steps: %d' % total_steps)
