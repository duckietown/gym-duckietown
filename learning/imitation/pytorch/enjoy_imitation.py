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
from gym_duckietown.envs import DuckietownEnv

from imitation.pytorch.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model
    model = Model()

    try:
        state_dict = torch.load('trained_models/imitate.pt', map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
    except:
        print('failed to load model')

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

load_model()

    def _enjoy(args):

    while True:
        start_time = time.time()

        obs = obs.transpose(2, 0, 1)
        obs = make_var(obs).unsqueeze(0)

        vels = model(obs)
        vels = vels.squeeze().data.cpu().numpy()
        print(vels)


        obs, reward, done, info = env.step(vels)

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
                    time.sleep(0.7)
            obs = env.reset()
            env.render()

if __name__ == '__main__':
    _enjoy(args)