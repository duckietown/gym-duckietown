#!/usr/bin/env python3

"""
Control the simulator or Duckiebot using a model trained with imitation
learning, and visualize the result.
"""

import time
import sys
import argparse
import math

import torch

import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

from train_imitation import Model
from utils import make_var

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='SimpleSim-v0')
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--no-random', action='store_true', help='disable domain randomization')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
args = parser.parse_args()

if args.env_name == 'SimpleSim-v0':
    env = SimpleSimEnv(
        map_name = args.map_name,
        domain_rand = not args.no_random
    )
    #env.max_steps = math.inf
    env.max_steps = 500
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

avg_frame_time = 0
max_frame_time = 0

def load_model():
    global model
    model = Model()

    try:
        model.load_state_dict(torch.load('trained_models/imitate.pt'))
    except:
        print('failed to load model')

    model.eval()
    model.cuda()

load_model()

while True:
    start_time = time.time()

    obs = obs.transpose(2, 0, 1)
    obs = make_var(obs).unsqueeze(0)

    vels = model(obs)
    vels = vels.squeeze().data.cpu().numpy()
    print(vels)

    #vels[1] *= 0.85

    obs, reward, done, info = env.step(vels)
    #print('stepCount = %s, reward=%.3f' % (env.stepCount, reward))

    env.render()

    end_time = time.time()
    frame_time = 1000 * (end_time - start_time)
    avg_frame_time = avg_frame_time * 0.95 + frame_time * 0.05
    max_frame_time = 0.99 * max(max_frame_time, frame_time) + 0.01 * frame_time
    fps = 1 / (frame_time / 1000)

    print('avg frame time: %d' % int(avg_frame_time))
    print('max frame time: %d' % int(max_frame_time))
    print('fps: %.1f' % fps)

    if done:
        if reward < 0:
            print('*** FAILED ***')
            if not args.no_pause:
                time.sleep(0.8)
        obs = env.reset()
        env.render()

        load_model()

    #time.sleep(0.1)
    #time.sleep(0.015)
