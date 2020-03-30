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
from gail.models import *

# from learning.imitation.basic.model import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _enjoy(args):

    from learning.utils.env import launch_env
    from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
        DtRewardWrapper, ActionWrapper, ResizeWrapper
    from learning.utils.teacher import PurePursuitExpert
    # model = Model(action_dim=2, max_action=1.)
    model = Generator(action_dim=2)

    try:
        # state_dict = torch.load('models/imitate.pt', map_location=device)
        state_dict = torch.load('models/G_{}.pt'.format(args.training_name), map_location=device)

        model.load_state_dict(state_dict)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print('failed to load model')
        exit()

    model.eval().to(device)

    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)

    obs = env.reset()

    # max_count = 0
    while True:
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

        action = model(obs)
        action = action.squeeze().data.cpu().numpy()
        # print("\nAction taken::", action, "\n")
        obs, reward, done, info = env.step(action)
        env.render()
        

        # if max_count > 50:
        #     max_count = 0
        #     obs = env.reset()

        if done:
            if reward < 0:
                print('*** FAILED ***')
                time.sleep(0.7)
            # max_count += 1
            obs = env.reset()
            env.render()
            # if max_count > 10:
            #     break

if __name__ == '__main__':
    _enjoy()